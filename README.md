# Diffusion Models for Medical Anomaly Detection

We provide the Pytorch implementation of our MICCAI 2022 submission "Diffusion Models for Medical Anomaly Detection" (paper 704).


The implementation of Denoising Diffusion Probabilistic Models presented in the paper is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).


## Data

We evaluated our method on the [BRATS2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html), and on the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/).
A mini-example how the data needs to be stored can be found in the folder *data*. To train or evaluate on the desired dataset, set `--dataset brats` or `--dataset chexpert` respectively. 

## Usage

We set the flags as follows:
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"
```
To train the classification model, run
```
python scripts/classifier_train.py --data_dir path_to_traindata --dataset brats_or_chexpert $TRAIN_FLAGS $CLASSIFIER_FLAGS
```
To train the diffusion model, run
```
python scripts/image_train.py --data_dir --data_dir path_to_traindata --datasaet brats_or_chexpert  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
The model will be saved in the *results* folder.

For image-to-image translation to a healthy subject on the test set, run
```
python scripts/classifier_sample_known.py  --data_dir path_to_testdata  --model_path ./results/model.pt --classifier_path ./results/classifier.pt --dataset brats_or_chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS 
```
A visualization of the sampling process is done using [Visdom](https://github.com/fossasia/visdom).


## Comparing Methods

### FixedPoint-GAN

We follow the implementation given in this [repo](https://github.com/mahfuzmohammad/Fixed-Point-GAN). We choose λ<sub>cls</sub>=1, λ<sub>gp</sub>=λ<sub>id</sub>=λ<sub>rec</sub>=10,  and train our model for 150 epochs. The batch size is set to 10, and the learning rate to 10<sup>-4</sup>.

### VAE

We follow the implementation given in this [repo](https://github.com/aubreychen9012/cAAE) and train the model for 500 epochs.  The batch size is set to 10, and the learning rate to 10<sup>-4</sup>.


### DDPM
For sampling using the DDPM approach, run 
```
python scripts/classifier_sample_known.py  --data_dir path_to_testdata  --model_path ./results/model.pt --classifier_path ./results/classifier.pt  --dataset brats_or_chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS 
```


