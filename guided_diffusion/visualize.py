from visdom import Visdom
viz = Visdom(port=8850)
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as mpl
from PIL import Image

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def scalar_2_vector(img_src, colormap):
    cm_hot = mpl.cm.get_cmap(colormap)
    img_src.thumbnail((240,240))
    im = np.array(img_src)
    im = cm_hot(im)
    im = np.uint8(im * 255)
    print(im.max(), im.min())
    im = Image.fromarray(im)
    return im

def save_tensor_image_as_png_jet(image, filename, colormap='gray'):
    im = Image.fromarray(image)
    im = scalar_2_vector(im, colormap)
    im.save(filename)
data=np.load('../results/samples_1x256x256x3.npz')
lst = data.files
print(len(lst))
for item in lst:
     print(item)
     print(data[item].shape)
     viz.image(visualize(data[item][0,...,0]))
     viz.image(visualize(data[item][0, ..., 1]))
     viz.image(visualize(data[item][0, ..., 2]))
     break
#viz.image(visualize(data[item][1,...,0]))
#
# PathDicomstripped = "/raid/dbe_summer_school_2021/brats2020/training"
# List = []
# for dirName, subdirList, fileList in os.walk(PathDicomstripped, topdown=True):
#
#     print('subdirlist', subdirList)
#     s = dirName.split("/", -1)
#     print('ID', s[-1])
#     for filename in fileList:
#         if 't1' in filename and 't1ce' not in filename:
#             sample = nib.load(os.path.join(dirName, filename))
#             img = np.asarray(sample.dataobj).astype(dtype='float32')
#             print(img.shape)
#             path= '../brats/'+s[-1]+'.png'
#             print('path', path)
#             save_tensor_image_as_png_jet(visualize(img), path)


