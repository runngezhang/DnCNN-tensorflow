'''
DnCNN
paper: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising

a tensorflow version of the network DnCNN
just for personal exercise

author: momogary
'''

import tensorflow as tf 
import numpy as np 
import math, os, cv2
from ops import *
from six.moves import xrange

class DnCNN(object):
	def __init__(self, sess, image_size=40, batch_size=64,
					output_size=40, input_c_dim=1, output_c_dim=1, 
					sigma=25, clip_b=0.025, lr=0.01, epoch=50,
					ckpt_dir='./checkpoint', dataset='BSD400'):
		self.sess = sess
		self.is_gray = (input_c_dim == 1)
		self.batch_size = batch_size
		self.image_size = image_size
		self.output_size = output_size
		self.input_c_dim = input_c_dim
		self.output_c_dim = output_c_dim
		self.sigma = sigma
		self.clip_b = clip_b
		self.lr = lr
		self.numEpoch = epoch
		self.ckpt_dir = ckpt_dir
		self.dataset_name = dataset

		# Adam setting (default setting)
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.alpha = 0.01
		self.epsilon = 1e-8


	def build_model(self):
		# input : [batchsize, image_size, image_size, channel]
		self.X = tf.placeholder(tf.float32, \
					[None, self.image_size, self.image_size, self.input_c_dim], \
					name='noisy_image')
		self.X_ = tf.placeholder(tf.float32, \
					[None, self.image_size, self.image_size, self.input_c_dim], \
					name='clean_image')

		# layer 1
		with tf.variable_scope('conv1'):
			layer_1_output = self.layer(self.X, [3, 3, 1, 64], useBN=False)

		# layer 2 to 16
		with tf.variable_scope('conv2'):
			layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64])
		with tf.variable_scope('conv3'):
			layer_3_output = self.layer(layer_2_output, [3, 3, 64, 64])
		with tf.variable_scope('conv4'):
			layer_4_output = self.layer(layer_3_output, [3, 3, 64, 64])
		with tf.variable_scope('conv5'):
			layer_5_output = self.layer(layer_4_output, [3, 3, 64, 64])
		with tf.variable_scope('conv6'):
			layer_6_output = self.layer(layer_5_output, [3, 3, 64, 64])
		with tf.variable_scope('conv7'):
			layer_7_output = self.layer(layer_6_output, [3, 3, 64, 64])
		with tf.variable_scope('conv8'):
			layer_8_output = self.layer(layer_7_output, [3, 3, 64, 64])
		with tf.variable_scope('conv9'):
			layer_9_output = self.layer(layer_8_output, [3, 3, 64, 64])
		with tf.variable_scope('conv10'):
			layer_10_output = self.layer(layer_9_output, [3, 3, 64, 64])
		with tf.variable_scope('conv11'):
			layer_11_output = self.layer(layer_10_output, [3, 3, 64, 64])
		with tf.variable_scope('conv12'):
			layer_12_output = self.layer(layer_11_output, [3, 3, 64, 64])
		with tf.variable_scope('conv13'):
			layer_13_output = self.layer(layer_12_output, [3, 3, 64, 64])
		with tf.variable_scope('conv14'):
			layer_14_output = self.layer(layer_13_output, [3, 3, 64, 64])
		with tf.variable_scope('conv15'):
			layer_15_output = self.layer(layer_14_output, [3, 3, 64, 64])
		with tf.variable_scope('conv16'):
			layer_16_output = self.layer(layer_15_output, [3, 3, 64, 64])

		# layer 17
		with tf.variable_scope('conv17'):
			self.Y = self.layer(layer_16_output, [3, 3, 64, 1], useBN=False)

		# L2 loss
		self.Y_ = self.X - self.X_ # noisy image - clean image
		self.loss = (1.0 / self.batch_size) * tf.nn.l2_loss(self.Y - self.Y_)

		optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
		self.train_step = optimizer.minimize(self.loss)
		# create this init op after all variables specified
		self.init = tf.global_variables_initializer() 

	def conv_layer(self, inputdata, weightshape, b_init, stridemode):
		# weights
		W = tf.get_variable('weights', weightshape, initializer = \
							tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
		b = tf.get_variable('biases', [1, weightshape[-1]], initializer = \
							tf.constant_initializer(b_init))

		# convolutional layer
		logits = tf.nn.conv2d(inputdata, W, strides=stridemode, padding="SAME") + b # SAME with zero padding
		return logits

	def bn_layer(self, logits, output_dim, b_init = 0.0):
		alpha = tf.get_variable('bn_alpha', [1, output_dim], initializer = \
								tf.constant_initializer(get_bn_weights([1, output_dim], self.clip_b, self.sess)))
		beta = tf.get_variable('bn_beta', [1, output_dim], initializer = \
								tf.constant_initializer(b_init))
		return batch_normalization(logits, alpha, beta, isCovNet = True)

	def layer(self, inputdata, filter_shape, b_init = 0.0, stridemode=[1,1,1,1], useBN = True):
		logits = self.conv_layer(inputdata, filter_shape, b_init, stridemode)
		if useBN:
			output = tf.nn.relu(self.bn_layer(logits, filter_shape[-1]))
		else:
			output = tf.nn.relu(logits)
		return output

	def train(self):
		# init the variables
		sess.run(self.init)

		# get data
		data = load_data(filepath='./data/image_clean_pat.npy')
		numBatch = data.shape[0] / self.batch_size

		counter = 0
		start_time = time.time()
		for epoch in xrange(self.epoch):
			for batch_id in xrange(numBatch):
				batch_images = data[batch_id*self.batch_size:(batch_id+1)*self.batch_size]
				train_images = add_noise(batch_images, self.sigma)
				loss = self.sess.run([self.loss], \
						feed_dict={self.X:train_images, self.X_:batch_images})
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                    % (epoch + 1, batch_id + 1, numBatch,
                        time.time() - start_time, loss))

				counter += 1

				# save the model
				if np.mod(counter, 500) == 0:
					self.save(counter)

	def save(self,counter):
		model_name = "DnCNN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, \
        							self.batch_size, self.image_size)
        checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=counter)

    def reconstruct_model(self, image):
    	# set reuse flag to True
        tf.get_variable_scope().reuse_variables()

        im_h, im_w = image.shape
        self.X = tf.placeholder(tf.float32, \
					[None, im_h, im_w, self.input_c_dim], \
					name='noisy_image')

        # layer 1 (adpat to the input image)
		with tf.variable_scope('conv1'):
			layer_1_output = self.layer(self.X, [3, 3, 1, 64], useBN=False)

		# layer 2 to 16
		with tf.variable_scope('conv2'):
			layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64])
		with tf.variable_scope('conv3'):
			layer_3_output = self.layer(layer_2_output, [3, 3, 64, 64])
		with tf.variable_scope('conv4'):
			layer_4_output = self.layer(layer_3_output, [3, 3, 64, 64])
		with tf.variable_scope('conv5'):
			layer_5_output = self.layer(layer_4_output, [3, 3, 64, 64])
		with tf.variable_scope('conv6'):
			layer_6_output = self.layer(layer_5_output, [3, 3, 64, 64])
		with tf.variable_scope('conv7'):
			layer_7_output = self.layer(layer_6_output, [3, 3, 64, 64])
		with tf.variable_scope('conv8'):
			layer_8_output = self.layer(layer_7_output, [3, 3, 64, 64])
		with tf.variable_scope('conv9'):
			layer_9_output = self.layer(layer_8_output, [3, 3, 64, 64])
		with tf.variable_scope('conv10'):
			layer_10_output = self.layer(layer_9_output, [3, 3, 64, 64])
		with tf.variable_scope('conv11'):
			layer_11_output = self.layer(layer_10_output, [3, 3, 64, 64])
		with tf.variable_scope('conv12'):
			layer_12_output = self.layer(layer_11_output, [3, 3, 64, 64])
		with tf.variable_scope('conv13'):
			layer_13_output = self.layer(layer_12_output, [3, 3, 64, 64])
		with tf.variable_scope('conv14'):
			layer_14_output = self.layer(layer_13_output, [3, 3, 64, 64])
		with tf.variable_scope('conv15'):
			layer_15_output = self.layer(layer_14_output, [3, 3, 64, 64])
		with tf.variable_scope('conv16'):
			layer_16_output = self.layer(layer_15_output, [3, 3, 64, 64])

		# layer 17
		with tf.variable_scope('conv17'):
			self.Y = self.layer(layer_16_output, [3, 3, 64, 1], useBN=False)
		#return self.Y

	def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self):
    	"""Test DnCNN"""
    	# init variables
        tf.initialize_all_variables().run()

        test_files = glob('./data/test/{}/*.png'.format(self.dataset_name))

        # load testing input
        print("Loading testing images ...")
        test_data = load_images(test_files) # list of array of different size

        if self.load(self.ckpt_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        psnr_sum = 0
        for idx in xrange(len(test_files)):
        	noisy_image = add_noise(test_data[idx], self.sigma)
        	self.reconstruct_model(noisy_image)
        	predicted_noise = self.sess.run([self.Y], feed_dict={self.X:noisy_image})
        	output_clean_image = noisy_image - predicted_noise
 			# calculate PSNR
 			psnr = cv2.PSNR(test_data[idx], output_clean_image)
 			psnr_sum += psnr
        	save_images(output_clean_image, \
        				os.path.basename(test_files[idx]).replace('.png', '%.2f.png' % psnr))
        avg_psnr = psnr_sum / len(test_files)
        print("Average PSNR %.2f" % avg_psnr)




