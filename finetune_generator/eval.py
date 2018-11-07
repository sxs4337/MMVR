# This script shows how to reconstruct from Caffenet features
import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import settings
import site
import sys

site.addsitedir(settings.caffe_root)
pycaffe_root = settings.caffe_root # substitute your path here
sys.path.insert(0, pycaffe_root)
print pycaffe_root

import caffe
import numpy as np
import os
import patchShow
import scipy.misc
import scipy.io
import glob

iteration = "-1"

# choose the net
nets =   ['1_conv5', '2_fc6', '3_fc6_eucl', '4_fc7', '5_fc8']
layers = ['pool5', 'fc6', 'fc6',      'fc7', 'fc8']
if len(sys.argv) == 2:
  iteration = int(sys.argv[1])
else:
  raise Exception('Usage: recon_input.py ITERATION_NUMBER')

layer = "fc6"

file_list = glob.glob("test_images/*.jpg")

# set up the inputs for the net:
batch_size = 64 #len(file_list) * 2
image_size = (3,227,227)
images = np.zeros((batch_size,) + image_size, dtype='float32')

# use crops of the cat image as an example
# in_image = scipy.misc.imread('images/cat.jpg')

for ni, f in enumerate(file_list):
  in_image = scipy.misc.imread(f)
  # images[ni] = np.transpose(in_image[ni:ni+image_size[1], ni:ni+image_size[2]], (2,0,1))
  in_image = scipy.misc.imresize(in_image, (image_size[1], image_size[2], image_size[0]))
  images[ni] = np.transpose(in_image, (2,0,1))

# mirror some images to make it a bit more diverse and interesting
#images[::2,:] = images[::2,:,:,::-1]

#for ni in range(images.shape[0]):
#  images[ni] = np.transpose(in_image[ni:ni+image_size[1], ni:ni+image_size[2]], (2,0,1))
## mirror some images to make it a bit more diverse and interesting
#images[::2,:] = images[::2,:,:,::-1]
#
# RGB to BGR, because this is what the net wants as input
data = images[:,::-1]

# subtract the ImageNet mean
image_mean = np.load("data/MSCOCO_train_val_mean.npy")
image_mean = np.transpose(image_mean, (2, 1, 0))
topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
image_mean = image_mean[topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
#print "==========", image_mean.shape
data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR

#initialize the caffenet to extract the features
caffe.set_mode_gpu() # replace by caffe.set_mode_gpu() to run on a GPU
caffe.set_device(0)
encoder_model = 'bvlc_reference_caffenet.caffemodel'
caffenet = caffe.Net('encoder_test.prototxt', encoder_model, caffe.TEST)

# run caffenet and extract the features
caffenet.forward(data=data)
feat = np.copy(caffenet.blobs[layer].data)
del caffenet

# run the reconstruction net
net = caffe.Net('generator_test.prototxt', 'snapshots_ImageNet/%s/generator.caffemodel' % iteration, caffe.TEST)
generated = net.forward(feat=feat)
recon = generated['deconv0'][:,::-1,topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
del net

print images.shape, recon.shape

output_dir = "recon_ImageNet"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

# save results to a file
#collage = patchShow.patchShow(np.concatenate((images[:8], recon[:8]), axis=3), in_range=(-120,120), rows=2, cols=4)
collage = patchShow.patchShow(recon[:8], in_range=(-120,120), rows=1, cols=8)
filename = '%s/recon_' % output_dir + str(iteration).zfill(7) + '.jpg'
scipy.misc.imsave(filename, collage)

print "   Saved to %s" % filename
