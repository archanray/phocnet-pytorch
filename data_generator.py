import os
import numpy as np
from skimage import io as img_io
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from skimage.transform import resize

class DataGenerator(Dataset):
	
	def __init__(self, gw_root_dir, image_extension='.png',
		embedding='phoc',
		phoc_unigram_levels=(1, 2, 4, 8),
		use_bigrams = False,
		fixed_image_size=None,
		min_image_width_height=30):
	
