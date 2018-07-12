import os
import numpy as np
from skimage import io as img_io
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from skimage.transform import resize
from phoc import build_phoc_descriptor, get_most_common_n_grams
from image_size import check_size
from homography_augmentation import HomographyAugmentation

class DataGenerator(Dataset):
	
	def __init__(self, gw_root_dir, type='train',image_extension='.png',
				 embedding='phoc',
				 phoc_unigram_levels=(1, 2, 4, 8),
				 use_bigrams = False,
				 fixed_image_size=None,
				 min_image_width_height=30):
		'''
		Constructor
		'''
		# class members
		self.word_list = None
		self.word_string_embeddings = None
		self.query_list = None
		self.label_encoder = None

		self.fixed_image_size = fixed_image_size

		self.path = gw_root_dir

		# get all files
		read_file = type+'_files.txt'
		_dir = '../words/splits/'
		f = open(dir+read_file, 'rb')
		all_files = f.readlines()
		all_files = [x.replace('\n', '') for x in all_files]
		f.close()

		#just do numpy read
		for i in all_files:
			np.load('../words/original_images/nopad/original_images_nopad_'+i+'.tiff.npy')

	# fixed sized image
	@staticmethod
	def _image_resize(word_img, fixed_img_size):
		if fixed_img_size is not None:
			if len(fixed_img_size) == 1:
				scale = float(fixed_img_size[0]) / float(word_img.shape[0])
				new_shape = (int(scale * word_img.shape[0]), int(scale * word_img.shape[1]))
			if len(fixed_img_size) == 2:
				new_shape = (fixed_img_size[0], fixed_img_size[1])
			word_img = resize(image=word_img, output_shape=new_shape).astype(np.float32)
		return word_img