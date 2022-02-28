import math
import random
from pathlib import Path
from PIL import Image
import blobfile as bf
#from mpi4py import MPI
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .train_util import visualize
from visdom import Visdom
viz = Visdom(port=8850)
blank = np.ones((256, 256))
from scipy import ndimage
#image_window = viz.image(blank)

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)

    classes = None

    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.

        class_names =[path.split("/")[3] for path in all_files] #9 or 3
        print('classnames', class_names)


        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        print('classes',classes)
   # print('mpishard', MPI.COMM_WORLD.Get_rank(), 'numshards', MPI.COMM_WORLD.Get_size())
    dataset = ImageDataset(
        image_size,
        data_dir,
        classes=classes,
        shard=0,#MPI.COMM_WORLD.Get_rank(),
        num_shards=1,#MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    print('lenloader', len(loader))
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
        exts=['jpg', 'jpeg', 'png', 'npy']
    ):
        super().__init__()
        self.resolution = resolution
        #self.local_images = image_paths[p for ext in exts for p in Path(f'{image_paths}').glob(f'**/*.{ext}')]
        self.local_images = [p for ext in exts for p in Path(f'{image_paths}').glob(f'**/*.{ext}')]


        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        print('len',  len(self.local_images))
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]



     #   with bf.BlobFile(path, "rb") as f:
          #  pil_image = Image.open(f)
          #  pil_image.load()
      #  img = Image.open(path)
      #  numpy_img = np.array(img)
        numpy_img = np.load(path)
      #  numpy_img=visualize(numpy_img)
        print('path', path)
      #  numpy_img = (visualize(numpy_img[...,0]) * 255).astype(np.uint8)
       # print('npmi', numpy_img.shape)
        # viz.image(visualize(numpy_img[1, :, :]), opts=dict(caption="input1"))
        # viz.image(visualize(numpy_img[2, :, :]), opts=dict(caption="input2"))
        # viz.image(visualize(numpy_img[3, :, :]), opts=dict(caption="input3"))
        # viz.image(visualize(numpy_img[4, :, :]), opts=dict(caption="input4"))

   #     numpy_img=np.swapaxes(numpy_img, 0,2)
    #    numpy_img = np.swapaxes(numpy_img, 0, 1)
    #    pil_image = Image.fromarray((numpy_img[...,:3] * 255).astype(np.uint8))#.convert('RGB')
        pil_image = Image.fromarray((numpy_img[ ...,0] * 255).astype(np.uint8)).convert('RGB')

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

          #  arr = np.array(pil_image)
        #    arr=zeropatch(pil_image, self.resolution)



        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        lowpass = ndimage.gaussian_filter(arr, 3)
        highpass = arr - lowpass
        # if self.local_classes[idx] == 0:
        #     arr=arr*0
        # #     arr=th.zeros(256,256,3)
        #     print('zero', arr.max(), arr.min())
        # elif self.local_classes[idx] == 1:
        #     arr = th.ones(256, 256, 3)
        #     print('one', arr.max(), arr.min())
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
         #   print(out_dict["y"] , path)
        return np.transpose(arr, [2, 0, 1]), out_dict

# class ImageDataset(Dataset):
#     def __init__(self, image_size, folder, classes=None, shard=0,
#          num_shards=1,         random_crop=False,
#          random_flip=True, exts = ['jpg', 'jpeg', 'png', 'npy']):
#         super().__init__()
#         self.folder = folder
#         print('folder', folder)
#         self.image_size = image_size
#         self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
#         print('paths', self.paths)
#         self.transform = transforms.Compose([
#            # transforms.Resize(image_size),
#           #  transforms.RandomHorizontalFlip(),
#           #  transforms.CenterCrop(image_size),
#            # TransposeNumpy([1, 2, 0]),
#             transforms.ToTensor(),
#          #   transforms.Lambda(lambda t: (t * 2) - 1)
#         ])
#
#     def __len__(self):
#         return len(self.paths)
#
#     def __getitem__(self, index):
#         path = self.paths[index]
#         #img = Image.open(path)
#         img = torch.from_numpy(np.load(path))
#         y=torch.transpose(img,0,2).float()#self.transform(img)
#         y=torch.transpose(y,1,2)
#         return y,0
#


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 3* image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
   # crop_y=64; crop_x=64
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def zeropatch(pil_image, image_size):
    im=np.array(th.zeros(image_size, image_size,3))
    arr = np.array(pil_image)
    crop_x = (-arr.shape[0] + image_size)
    crop_y = abs(arr.shape[1] - image_size) // 2
  #  print('crop', crop_y, crop_x) #crop_y=64; crop_x=64
    im[0:arr.shape[0] , crop_y : crop_y +arr.shape[1],:]=arr

    return im#arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]



def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
