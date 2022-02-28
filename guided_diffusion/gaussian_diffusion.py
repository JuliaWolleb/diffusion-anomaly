"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
from PIL import Image
from torch.autograd import Variable
import enum
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import math
from visdom import Visdom
viz = Visdom(port=8850)
import numpy as np
import torch as th
from .train_util import visualize
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from scipy import ndimage
from torchvision import transforms
import matplotlib.pyplot as plt
def standardize(img):
    mean = th.mean(img)
    std = th.std(img)
    img = (img - mean) / std
    return img


def standardizetensor(img):
    mean = img.mean()
    std = img.std()
    img = (img - mean) / std
    return img


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        print('self.model_meantype', self.model_mean_type)
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        print('self.num_timesteps', self.num_timesteps)

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        x_t = x_t[:, :4, ...]
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        C=4
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        x = x[:, :4, ...]
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    def p_mean_variance2(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.


        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        C=4

        assert t.shape == (B,)
        Model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        x = x[:, :4, ...]
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert Model_output.shape == (B, C * 3, *x.shape[2:])
            model_output, model_var_values, update = th.split(Model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
                print('LEARNED VAR VALUES')
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
         #   model_variance, model_log_variance, update = {
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "modeloutput": Model_output,
            "update": update
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        x_t = x_t[:, :4, ...]
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            print('tresc',t )
            print('self.num_timesteps222', self.num_timesteps)
            print('scaledtimsetsep', t.float() * (1000.0 / self.num_timesteps))
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, update=None, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """


      #  viz.image(visualize(gradient.cpu()[0, ...]), opts=dict(caption="gradient"))
        if update is not None:
            print('CONDITION MEAN UPDATE NOT NONE')
            stdupdate=standardizetensor(update)

            new_mean = (
                p_mean_var["mean"].detach().float() + p_mean_var["variance"].detach() * update.float()
                )
           # else:
            #    new_mean=p_mean_var["mean"].float()
            a=update
        else:
           a, gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
           new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
            )

        return a, new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t,  model_kwargs=None):

       #
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        print('self._scale_timesteps(t)',self._scale_timesteps(t))
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
      #  eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
       #     x, self._scale_timesteps(t).long(), **model_kwargs)
        print('eps0', eps.max(), eps.min())
      #  print('update', p_mean_var["update"].max(), p_mean_var["update"].min())

      #  eps = eps.detach() - (1 - alpha_bar).sqrt() *p_mean_var["update"]*0
        print('eps1', eps.max(), eps.min())
        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x.detach(), t.detach(), eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out#, eps


    def condition_score2(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        t=t.long()
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        a, cfn= cond_fn(
            x, self._scale_timesteps(t).long(), **model_kwargs
        )
        eps = eps - (1 - alpha_bar).sqrt() * cfn

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out, cfn

    def sample_known(self, img, batch_size = 1):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop_known(model,(batch_size, channels, image_size, image_size), img)

    def p_sample2(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=True,
        org=None,
        model_kwargs=None,
        update_eps=False
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance2(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x[:,:4,...])
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        update = out['update']
        print('updatemaxmim', out['update'].max(), out['update'].min())

        if cond_fn == True:
            print('UUSSEE cond_fn')
            a, out["mean"] = self.condition_mean(cond_fn, out, x, t, update=update,
                                                  model_kwargs=model_kwargs)

        else:
            print('NOO use cond_fn')
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape)

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"].detach()) * noise

           # print('used cond_fn', a.shape, out["mean"].shape)
          #  viz.image(visualize(out["mean"].cpu()[0, ...]), opts=dict(caption="outmean"))
          #  viz.image(visualize(a.cpu()[0, ...]), opts=dict(caption="a"))



        return {"sample": sample, "pred_xstart": out["pred_xstart"], "saliency": out["update"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,

    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x[:, :4, ...])
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            a, out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        else:
            a=0*noise
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "saliency": a}

    def p_sample_loop_known(
        self,
        model,
        shape,
        img,
        org=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conditioning=False,
        conditioner = None,
        classifier=None,
        cyclic=False
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]


        t = th.randint(499,500, (b,), device=device).long().to(device)

        org=img[0].to(device)
        img=img[0].to(device)
        indices = list(range(t))[::-1]
        noise = th.randn_like(img[:, :4, ...]).to(device)
        x_noisy = self.q_sample(x_start=img[:, :4, ...], t=t, noise=noise).to(device)
        x_noisy = torch.cat((x_noisy, img[:, 4:, ...]), dim=1)
        print('xnoisy', x_noisy.shape)
        #viz.image(visualize(x_noisy.cpu()[0, ...]), opts=dict(caption="xnoisy" ))

        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=x_noisy,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            org=org,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            conditioning=conditioning,
            conditioner=conditioner,
            classifier=classifier,
            cyclic=cyclic
        ):
            final = sample
        if conditioning:
            return final["sample"], x_noisy, org#, final["conditioner"]
        else:

            return final["sample"], x_noisy, img

    def p_sample_loop_interpolation(
        self,
        model,
        shape,
        img1,
        img2,
        lambdaint,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(299,300, (b,), device=device).long().to(device)
        img1=torch.tensor(img1).to(device)
        img2 = torch.tensor(img2).to(device)
        noise = th.randn_like(img1).to(device)
        x_noisy1 = self.q_sample(x_start=img1, t=t, noise=noise).to(device)
        x_noisy2 = self.q_sample(x_start=img2, t=t, noise=noise).to(device)
        interpol=lambdaint*x_noisy1+(1-lambdaint)*x_noisy2
        print('interpol', interpol.shape)
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=interpol,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"], interpol, img1, img2


    def p_sample_loop_progressive(
        self,
        model,
        shape,
        time=1000,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        org=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conditioning=False,
        conditioner=None, classifier=None, cyclic=False):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
       # indices = list(range(self.num_timesteps))[::-1]
        orghigh = img[:, 4:, ...]
        print('orghigh', orghigh.shape)

        indices = list(range(time))[::-1]
        print('indices', indices)
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        if cyclic==True:
            for i in indices:
                t = th.tensor([i] * shape[0], device=device)

                with th.no_grad():
                    # if img.shape != (1, 8, 256, 256):
                    #     number = np.random.randint(2, size=1)
                    #     number = torch.tensor(np.float32(number)).to(device)
                    #     img = torch.cat((img, orghigh*number), dim=1)
                    #     gshapeafter', img.shape)
                    if i % 50 == 0:
                        viz.image(visualize(img[0, 0, ...]), opts=dict(caption="sampledi"))

                    out = self.p_sample2(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        # org=org,
                        model_kwargs=model_kwargs,
                        #   update_eps=False
                    )
                    yield out

                    img = out["sample"]
                    update=out['saliency']

                    # print('img', img.shape)
                    # r, g, b = img[0,0, :, :], img[0,1, :, :], img[0,2, :, :]
                    if i % 10 == 0:
                        print(i)
                    if i %100 == 0:
                        viz.image(visualize(img[0, 0, ...]), opts=dict(caption=str(i)))
                        viz.image(visualize(img[0, 1, ...]), opts=dict(caption=str(i)))
                        viz.image(visualize(img[0, 2, ...]), opts=dict(caption=str(i)))
                        viz.image(visualize(img[0, 3, ...]), opts=dict(caption=str(i)))
                        viz.image(visualize(out["saliency"][0, 0, ...]), opts=dict(caption='update0'))
                        viz.image(visualize(out["saliency"][0, 3, ...]), opts=dict(caption='update3'))
                #    number=abs(i-1000)
                #  name=(str(number).zfill(5))
        # save_image(gray, './fake_images_classcond/k/krank/'+str(k)+'.png')

        else:
         for k in range(0,1):
           print('k0', k)
           #img = th.randn(*shape, device=device)
           for i in indices:
                t = th.tensor([i] * shape[0], device=device)

                with th.no_grad():
                    if img.shape != (1, 8, 256, 256):
                        number = np.random.randint(2, size=1)
                        number = torch.tensor(np.float32(number)).to(device)
                        img = torch.cat((img, orghigh), dim=1)
                    if i % 50 == 0:
                        viz.image(visualize(img[0, 0, ...]), opts=dict(caption="sampledi"))


                    out = self.p_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        #org=org,
                        model_kwargs=model_kwargs,
                     #   update_eps=False
                    )
                    yield out
                    img = out["sample"]

                   # print('img', img.shape)
                   # r, g, b = img[0,0, :, :], img[0,1, :, :], img[0,2, :, :]
                    if i%10==0:
                        print(i)
                    if i%100==0:
                     viz.image(visualize(img[0,0,...]), opts=dict(caption=str(i)))
                     viz.image(visualize(img[0, 1,...]), opts=dict(caption=str(i)))
                     viz.image(visualize(img[0, 2,...]), opts=dict(caption=str(i)))
                     viz.image(visualize(img[0, 3,...]), opts=dict(caption=str(i)))
                     viz.image(visualize(out["saliency"][0,0,...]), opts=dict(caption='saliency'))
                 #    number=abs(i-1000)
                   #  name=(str(number).zfill(5))
          # save_image(gray, './fake_images_classcond/k/krank/'+str(k)+'.png')

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        if cond_fn is not None:
            out, saliency = self.condition_score2(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "saliency": saliency }


    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}



    def ddim_sample_loop_interpolation(
        self,
        model,
        shape,
        img1,
        img2,
        lambdaint,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(199,200, (b,), device=device).long().to(device)
        img1=torch.tensor(img1).to(device)
        img2 = torch.tensor(img2).to(device)
        noise = th.randn_like(img1).to(device)
        x_noisy1 = self.q_sample(x_start=img1, t=t, noise=noise).to(device)
        x_noisy2 = self.q_sample(x_start=img2, t=t, noise=noise).to(device)
        interpol=lambdaint*x_noisy1+(1-lambdaint)*x_noisy2
        print('interpol', interpol.shape)
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=interpol,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"], interpol, img1, img2

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(0,1, (b,), device=device).long().to(device)
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):

            final = sample
        viz.image(visualize(final["sample"].cpu()[0, ...]), opts=dict(caption="sample"+ str(10) ))
        return final["sample"]



    def ddim_sample_loop_known(
            self,
            model,
            shape,
            img,
            org=None,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            conditioning=False,
            conditioner=None,
            classifier=None,
            eta = 0.0
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(0,1, (b,), device=device).long().to(device)
        org = img[0].to(device)
        img = img[0].to(device)
        r, g, b = img[0, 0, :, :], img[0, 1, :, :], img[0, 2, :, :]
        gray = img[0, ...]
        gray = visualize(gray)
        #  save_image(gray, './movie/org' + '.png')
        indices = list(range(t))[::-1]
        noise = th.randn_like(img).to(device)
        x_noisy = self.q_sample(x_start=img, t=t, noise=noise).to(device)
        print('xnoisy', x_noisy.shape)

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=250,#t,
            noise=x_noisy,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        viz.image(visualize(final["sample"].cpu()[0,0, ...]), opts=dict(caption="final 0" ))
        viz.image(visualize(final["sample"].cpu()[0,1, ...]), opts=dict(caption="final 1" ))
        viz.image(visualize(final["sample"].cpu()[0,2, ...]), opts=dict(caption="final 2" ))
        viz.image(visualize(final["sample"].cpu()[0,3, ...]), opts=dict(caption="final 3" ))


     #   return final["sample"]
        return final["sample"], x_noisy, img


    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        time=1000,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(time-1))[::-1]
        print('indices', indices)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:

            k=abs(249-i)
            if k%20==0:
                print('k',k)

            t = th.tensor([k] * shape[0], device=device)
            with th.no_grad():

                out = self.ddim_reverse_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )

                yield out
                img = out["sample"]
              #  if i % 100 == 0:
                #    viz.image(visualize(out["sample"].cpu()[0, 0,...]), opts=dict(caption="reversesample" + str(i)))
       # # img=standardize(img)
       #  print('transformedimg', img.shape, img.max(), img.min())
        viz.image(visualize(img.cpu()[0,0, ...]), opts=dict(caption="reversesample"))
        for i in indices:
                t = th.tensor([i] * shape[0], device=device)
                with th.no_grad():
                 out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                 )
                yield out
                img = out["sample"]
                saliency=out['saliency']
                if i % 70 == 0:
                    # j=500-i
                     print('i', i)
                  #   A=visualize(np.array(saliency.cpu()[0, 2, ...]))
                     viz.image(visualize(img.cpu()[0, 3,...]), opts=dict(caption="sample3_" + str(i)))
                  #   viz.image(visualize(saliency.cpu()[0, 1, ...]), opts=dict(caption="sample1_" + str(i)))
                   #  viz.image(visualize(saliency.cpu()[0, 2, ...]), opts=dict(caption="sample2_" + str(i)))
               #      im = Image.fromarray(A)
                 #    name=(f"{j:03d}.png")
                 #    plt.imsave('./gradientmovie/'+name, A)

                     #im.save('./gradientmovie/'+name)
                  #   viz.image(visualize(saliency.cpu()[0, 3, ...]), opts=dict(caption="sample3_" + str(i)))


    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model,  x_start, t, classifier=None, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start[:, :4, ...])
        # viz.image(visualize(x_start[0, 0, ...]), opts=dict(caption="x00"))
        # viz.image(visualize(x_start[0, 1, ...]), opts=dict(caption="x00"))
        x_t = self.q_sample(x_start[:, :4, ...], t, noise=noise)
        x_t = torch.cat((x_t, x_start[:, 4:,  ...]), dim=1)
        # viz.image(visualize(x_t[0, 0, ...]), opts=dict(caption="xt0"))
        # viz.image(visualize(x_t[0,4,...]), opts=dict(caption="xt5"))
        # viz.image(visualize(x_t[0, 5, ...]), opts=dict(caption="xt5"))
        # viz.image(visualize(x_t[0, 6, ...]), opts=dict(caption="xt6"))
        # # viz.image(visualize(x_t[0, 7, ...]), opts=dict(caption="xt7"))

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                C=4
                assert model_output.shape == (B, C * 3, *x_t.shape[2:])
                model_output, model_var_values,  update = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values, update.detach()], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start[:, :4, ...],
                    x_t=x_t[:, :4, ...],
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start[:,:4,...], x_t=x_t[:,:4,...], t=t
                )[0],
                ModelMeanType.START_X: x_start[:,:4,...],
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start[:,:4,...].shape
            terms["mse"] = mean_flat((target - model_output) ** 2)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)
        terms["cls2"]=0 ;terms["cls3"] =0; terms["rec1"]=0 ; terms["rec2"]=0
        return (terms, model_output)

    def training_losses_cycle(self, model,  x_start, t, classifier, model_kwargs=None, noise=None, update_eps=True, cond_fn=True, apply_cycle=True):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        device=model.device
        print('device', device)
        t=t.to(device)
        x_start=x_start.float().to(device)
        print('x_start', x_start.shape)
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start[:, :4, ...])

        x_t = self.q_sample(x_start[:, :4, ...], t, noise=noise)
        x_t = torch.cat((x_t, x_start[:, 4:, ...]), dim=1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_start[:,:4,...].shape) - 1)))
        )

        terms = {}
        print('lossttype',self.loss_type)
        sample3=0

        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            print('y0', model_kwargs['y'])
            y=model_kwargs['y']
            y2 = abs(model_kwargs['y'] - 1)

            model_kwargs2 = model_kwargs.copy()
            model_kwargs2['y'] = y2
            model_kwargs['y'] = y
            print('y',  model_kwargs['y'],'y2',  model_kwargs2['y'])

            out = self.p_mean_variance2(
                model, x_t, self._scale_timesteps(t), model_kwargs=model_kwargs)  # change to original class
            M=out["modeloutput"]
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                C=4
                assert M.shape == (B, C * 3, *x_t.shape[2:])
                model_output, model_var_values, update = th.split(M, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values, update.detach()], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start[:, :4, ...],
                    x_t=x_t[:, :4, ...],
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0
            print('selfmodelmeantype',self.model_mean_type)
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start[:,:4,...], x_t=x_t[:,:4,...], t=t
                )[0],
                ModelMeanType.START_X: x_start[:,:4,...],
                ModelMeanType.EPSILON: noise,
            }[ModelMeanType.PREVIOUS_X]#[self.model_mean_type]
            print('mdoeloutput2', )
            assert model_output.shape == target.shape


            if update_eps==True:


                if cond_fn == True:
                    print('use cond_fn')
                    #   out1, eps1 = self.condition_score(cond_fn, Model, x_t, t, model_kwargs=model_kwargs)
                    #  out1, eps1 = self.condition_mean(cond_fn, Model, x_t, t, model_kwargs=model_kwargs)
                    a, out["mean"] = self.condition_mean(cond_fn, out, x_t[:, :4, ...], t, update=update, model_kwargs=model_kwargs)


                else:
                    print('NOO use cond_fn')
                sample1 = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
                terms["mse"] = mean_flat((target - sample1) ** 2)

                #   viz.image(visualize(sample1[0, 0, ...]), opts=dict(caption="sample1" + str(t)))
                print('update', update.max(), update.min())

                out2 = self.p_mean_variance2(
                    model, x_t, self._scale_timesteps(t), model_kwargs=model_kwargs2)  # change class!!!

                if cond_fn==True:
                    print('use cond_fn')
                    a, out2["mean"] = self.condition_mean(cond_fn, out2, x_t, t, update=out2['update'], model_kwargs=model_kwargs2)
                   # out2, eps2 = self.condition_score(cond_fn, Model2, x_t, t, model_kwargs=model_kwargs2)       #change class!
                else:
                    print('NOO use cond_fn')

             #   alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
              #  alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape)
              #  sigma=0*x_t
               # print('eps2', eps2.max(), eps2.min())
                sample2 = out2["mean"] + nonzero_mask * th.exp(0.5 * out2["log_variance"].detach()) * noise
              #  sample2 = (
               #         out2["pred_xstart"] * th.sqrt(alpha_bar_prev)
               #         + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps2)

                logits2 = classifier(sample2, timesteps=t)
                print('logits2', logits2)
                #  losscls =F.cross_entropy(logits, model_kwargs2['y'], reduction="none")
                losscls2 = F.cross_entropy(logits2, model_kwargs2['y'], reduction="none")  #change class!
                terms["cls2"] = losscls2

            if apply_cycle==True:
                sample_j=self.q_sample(sample2, t=t*0, noise=noise)
                terms["rec1"] = 0#mean_flat((sample_j - x_t[:,:4,...]) ** 2)

              #  viz.image(visualize(sample_j[0, ...]), opts=dict(caption="samplej"))
                #diff=abs(sample_j-x_t)

                sample_j = torch.cat(( sample_j, x_start[:, 4:, ...]), dim=1)
                out3 = self.p_mean_variance2(
                    model, sample_j, self._scale_timesteps(t), model_kwargs=model_kwargs)  #change to original class
                if cond_fn == True:
                    print('use cond_fn')
                    a, out3["mean"] = self.condition_mean(cond_fn, out3, x_t, t, update=out3['update'],
                                                          model_kwargs=model_kwargs2)

                else:
                    print('NOO use cond_fn')
               # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape)


                sample3 = out3["mean"] + nonzero_mask * th.exp(0.5 * out3["log_variance"].detach()) * noise
                # sample3 = (
                #         out3["pred_xstart"] * th.sqrt(alpha_bar_prev)
                #         + th.sqrt(1 - alpha_bar_prev) * eps3)
                #

                logits3 = classifier(sample3, timesteps=t)
                losscls3 = F.cross_entropy(logits3, model_kwargs['y'] , reduction="none")
                terms["cls3"] = losscls3
                terms["rec2"] = mean_flat((sample3 - sample1) ** 2)

            #   a, mean3 = self.condition_mean(
            #           self.training_losses.scond_fn, model_output3, x_t,  model_kwargs)

         #   viz.image(visualize(model_output[0, ...]), opts=dict(caption="mdoleoutput"))
        #    viz.image(visualize(target[0, ...]), opts=dict(caption="target"))
         #   viz.image(visualize(sample[0, ...]), opts=dict(caption="sample3"))
          #  viz.image(visualize(update[0, ...]), opts=dict(caption="update"))

            if "vb" in terms:
                terms["loss"] =  100*terms["mse"]+100*terms["vb"]+0.1*terms["cls2"]+0.1*terms["cls3"]+100*terms["rec1"]+100*terms["rec2"]
                print('ALL LOSS TERMS ADDED')
            else:
                terms["loss"] = 100*terms["mse"]+0.1*terms["cls2"]+0.1*terms["cls3"]+100*terms["rec1"]+100*terms["rec2"]
        else:
            raise NotImplementedError(self.loss_type)
      #  a = th.autograd.grad(losscls3[0].sum(), sample,retain_graph=True)
      #  print('a', a[0].sum())
        sample = torch.cat((sample3, update, x_start), dim=1)

       # losscls3[0].backward(retain_graph=True)
       # grad=th.autograd.grad(losscls3[0], sample[0])
      #  grad=sample.grad
        return (terms, sample)  # (terms, org)


    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
       # device = next(model.parameters()).device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        print('numstept',self.num_timesteps )
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            viz.image(visualize(x_t[0, ...]), opts=dict(caption="xt"))

            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bptimestepsd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
