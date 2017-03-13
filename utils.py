import numpy as np 
import os, sys
from PIL import Image

def data_augmentation(image, mode):
	if mode == 0:
		# original
		return image
	elif mode == 1:
		# flip up and down
		return np.flipud(image)
	elif mode == 2:
		# rotate counterwise 90 degree
		return np.rot90(image)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		image = np.rot90(image)
		return np.flipud(image)
	elif mode == 4:
		# rotate 180 degree
		return np.rot90(image, k = 2)
	elif mode == 5:
		# rotate 180 degree and flip
		image = np.rot90(image, k = 2)
		return np.flipud(image)
	elif mode == 6:
		# rotate 270 degree
		return np.rot90(image, k = 3)
	elif mode == 7:
		# rotate 270 degree and flip
		image = np.rot90(image, k = 3)
		return np.flipud(image)

def load_data(filepath='./data/image_clean_pat.npy'):
	assert '.npy' in filepath
	if not os.path.exists(filepath):
		print "data file not exists"
		sys.exit(1)

	data = np.load(filepath)
	np.random.shuffle(data)
	return data

def add_noise(data, sigma):
	noise = sigma / 255.0 * sess.run(tf.truncated_normal(data.shape))
	return (data + noise)

def load_images(filelist):
	data = []
	for file in filelist:
		im = Image.open(file).convert('L')
		data.append(np.array(im).reshape(1, im.size[0], im.size[1], 1))
	return data

def save_images(image, filepath):
