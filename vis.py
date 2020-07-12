import cv2
import glob
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
import matplotlib.pyplot as plt
 
# scale an array of images to a new size
def scale_images(images, new_shape):
  images_list = list()
  for image in images:
	  # resize with nearest neighbor interpolation
	  new_image = resize(image, new_shape, 0)
	  # store
	  images_list.append(new_image)
  
  return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):

 # calculate activations
 act1 = model.predict(images1)
 act2 = model.predict(images2)
 # calculate mean and covariance statistics
 mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
 mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
 # calculate sum squared difference between means
 ssdiff = np.sum((mu1 - mu2)**2.0)
 # calculate sqrt of product between cov
 covmean = sqrtm(sigma1.dot(sigma2))
 # check and correct imaginary numbers from sqrt
 if iscomplexobj(covmean):
  covmean = covmean.real
 # calculate score
 fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
 return fid
 

def process(model, path1, path2,limit):

  X_data = []
  files1 = glob.glob (path1)
  module=limit*2
  i=0
  for myFile in files1:
      i+=1
      image1 = cv2.imread (myFile)
      X_data.append (image1)
      if (i%module==0):
        break
      
  Y_data = []
  i=0
  files2 = glob.glob (path2)
  for myFile in files2:
      i+=1
      image2 = cv2.imread (myFile)
      Y_data.append (image2)
      if (i%module==0):
        break
  
  
  
  image1 = image1.reshape(172, 172, 3)
  image2 = image2.reshape(172, 172, 3)
  print('Prepared', image1.shape, image2.shape)
  
  # convert integer to floating point values
  images1 = image1.astype('float32')
  images2 = image2.astype('float32')
  
  # resize images
  images1 = scale_images(image1, (299,299,3))
  images2 = scale_images(image2, (299,299,3))
  print('Scaled', images1.shape, images2.shape)
  
  # pre-process images
  images1 = preprocess_input(images1)
  images2 = preprocess_input(images2)
  fid = calculate_fid(model, images1, images2)
  return fid


# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

fidslist1 =[]
fidslist2 =[]
fidslist3 =[]
steps =[]
count=0
path1="imagest1/*.png"
path2="imagesreal/*.png"
path3="imagest2/*.png"
path4="images/*.png"

for limit in range(1,22):

  fid1 = process(model,path1, path2,limit)
  fid2 = process(model,path3, path2,limit)
  fid3 = process(model,path4, path2,limit)
  # fid between images1 and images1
  # fid = calculate_fid(model, images1, images1)
  # print('FID (same): %.3f' % fid)
  count+=1
  # fid between images1 and images2
  
  fidslist1.append(fid1)  
  fidslist2.append(fid2) 
  fidslist3.append(fid3) 
  steps.append(count)

fidslist1.sort(reverse=True) 
fidslist2.sort(reverse=True)
fidslist3.sort(reverse=True)

print ('Racing-GAN 1 : \n')
print (fidslist1)
print ('\n Racing-GAN 2 : \n')
print (fidslist2)
print ('\n Regular GAN : \n')
print (fidslist3)

plt.plot(steps,fidslist1,color='blue', lw=2,label = 'Racing-GAN images (Generator 1)')
plt.plot(steps,fidslist2,color='red', lw=2,label = 'Racing-GAN images (Generator 2)')
plt.plot(steps,fidslist3,color='green', lw=2,label = 'Standard WGAN images')
plt.legend(loc="upper left")

plt.draw()
plt.xlabel("Iteration")
plt.ylabel("FID Score")
plt.show()
plt.savefig('Result.png')

print('FID (different): %.3f' % fid1)
print('FID (different): %.3f' % fid2)
print('FID (different): %.3f' % fid3)
