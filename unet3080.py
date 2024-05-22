import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['PYTHONHASHSEED'] = str(42)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Concatenate, Conv2DTranspose,GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Add, Activation, Lambda 
from keras.optimizers import *
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.optimizers import adam_v2
import tensorflow as tf
# gpus= tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(gpus[0], True)
np.random.seed(42)
tf.random.set_seed(42)
# tf.config.experimental_run_functions_eagerly(True)
from keras import backend as K
import pdb
from alc import * 

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0+1e-5))
    return focal_loss_fixed

class myUnet(object):
	def __init__(self, img_rows = 64, img_cols = 64):
		self.img_rows = img_rows
		self.img_cols = img_cols

	def get_unet_lca(self):
		inputs = Input((self.img_rows, self.img_cols,1))
		# 网络结构定义
		conv1 = Conv2D(8*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print ("conv1 shape:",conv1.shape)
		conv1 = Conv2D(8*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print ("conv1 shape:",conv1.shape)
		# conv1 = SeBlock()(conv1)
		mlc1 = Lambda(circ_shift,arguments={'shift':3})(conv1)
		# mlc1 = Lambda(mlc, arguments={'d':[3,5]})(conv1)
		# blam1 = blam_weight()(mlc1)
		x = Conv2D(int(mlc1.shape[-1]) // 4, 1, padding = 'same', kernel_initializer = 'he_normal')(mlc1)
		x = BatchNormalization()(x, training=False)
		x = Activation('relu')(x)
		x = Conv2D(int(x.shape[-1]) * 4, 1, padding = 'same', kernel_initializer = 'he_normal')(x)
		x = BatchNormalization()(x, training=False)
		blam1 = Activation('sigmoid')(x)
		# conv1 = cbam_block(conv1)
		
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print ("pool1 shape:",pool1.shape)

		conv2 = Conv2D(16*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print ("conv2 shape:",conv2.shape)
		conv2 = Conv2D(16*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print ("conv2 shape:",conv2.shape)
		drop2 = BatchNormalization()(conv2, training=False)
		conv2 = Activation('relu')(drop2)
		# conv2 = SeBlock()(conv2)
		# conv2 = cbam_block(conv2)
		mlc2 = Lambda(circ_shift,arguments={'shift':3})(conv2)
		# mlc2 = Lambda(mlc, arguments={'d':[3,5]})(conv2)
		# blam2 = blam_weight()(mlc2)
		x = Conv2D(int(mlc2.shape[-1]) // 4, 1, padding = 'same', kernel_initializer = 'he_normal')(mlc2)
		x = BatchNormalization()(x, training=False)
		x = Activation('relu')(x)
		x = Conv2D(int(x.shape[-1]) * 4, 1, padding = 'same', kernel_initializer = 'he_normal')(x)
		x = BatchNormalization()(x, training=False)
		blam2 = Activation('sigmoid')(x)

		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print ("pool2 shape:",pool2.shape)


		conv3 = Conv2D(32*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print ("conv3 shape:",conv3.shape)
		conv3 = Conv2D(32*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print ("conv3 shape:",conv3.shape)
		# drop3 = BatchNormalization()(conv3, training=False)
		conv3 = Activation('relu')(conv3)
		conv3 = Dropout(0.5)(conv3)
		# conv3=SeBlock()(conv3)
		# conv3 = cbam_block(drop3)
		mlc3 = Lambda(circ_shift,arguments={'shift':3})(conv3)
		# mlc3 = Lambda(mlc,arguments={'d':[3,5]})(conv3)
		# blam3 = blam_weight()(mlc3)
		x = Conv2D(int(mlc3.shape[-1]) // 4, 1, padding = 'same', kernel_initializer = 'he_normal')(mlc3)
		x = BatchNormalization()(x, training=False)
		x = Activation('relu')(x)
		x = Conv2D(int(x.shape[-1]) * 4, 1, padding = 'same', kernel_initializer = 'he_normal')(x)
		x = BatchNormalization()(x, training=False)
		blam3 = Activation('sigmoid')(x)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print ("pool3 shape:",pool3.shape)
		
		conv4 = Conv2D(64*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(64*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
		# drop4 = Dropout(0.5)(conv4)
		# drop4 = BatchNormalization()(conv4, training=False)
		drop4 = Activation('relu')(conv4)
		drop4 = Dropout(0.5)(drop4)

		up7 = Conv2DTranspose(32*2, 4, activation = 'relu', strides=2, padding='same', kernel_initializer='he_normal')(drop4)
		# up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv2DTranspose(size = (2,2))(drop4))
		up7 = Add()([Multiply()([blam3,up7]), mlc3])
		merge7 = Concatenate(axis=3)([conv3,up7])
		# merge7 = Add()([conv3,up7])
		conv7 = Conv2D(32*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(32*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		conv7 = Dropout(0.5)(conv7)
		# conv7 = BatchNormalization()(conv7, training=False)
		# conv7 = Activation('relu')(conv7)

		up8 = Conv2DTranspose(16*2, 4, activation = 'relu', strides=2, padding='same', kernel_initializer='he_normal')(conv7)
		# up8 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv2DTranspose(size = (2,2))(conv7))
		up8 = Add()([Multiply()([blam2,up8]), mlc2])
		merge8 = Concatenate(axis=3)([conv2,up8])
		conv8 = Conv2D(16*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(16*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
		conv8 = BatchNormalization()(conv8, training=False)
		conv8 = Activation('relu')(conv8)

		up9 = Conv2DTranspose(8*2, 4, activation = 'relu', strides=2, padding='same', kernel_initializer='he_normal')(conv8)
		# up9 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv2DTranspose(size = (2,2))(conv8))
		up9 = Add()([Multiply()([blam1,up9]), mlc1])
		merge9 = Concatenate(axis=3)([conv1,up9])
		conv9 = Conv2D(8*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
		model = Model(inputs = inputs, outputs = conv10)
		# model.compile(optimizer = sgd(lr = 1e-4,momentum=0.9, decay=0.01, nesterov=True), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
		# model.compile(optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss = [focal_loss(alpha=.25, gamma=5)], metrics = ['accuracy'])
		model.compile(optimizer = adam_v2.Adam(lr=1e-4), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])#有结果
		# model.run_eagerly = True
		# model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		return model