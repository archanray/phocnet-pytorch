import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spatial_pyramid_layers.gpp import GPP

class PHOCNet(nn.Module):
	'''
	This class will define the network architecture for the PHOCNET archirtecture
	'''
	def __init__(self, D_in, D_out):
		super(PHOCNet, self).__init__()
		self.conv1 = nn.Conv2d(D_in, 64, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
		self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv7 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv8 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv9 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv10 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv11 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
		self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv13 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.pooling_layer_fn = GPP(gpp_type='spp', levels=pooling_levels, pool_type=pool_type)
		pooling_output_size = self.pooling_layer_fn.pooling_output_size
		self.fc5 = nn.Linear(pooling_output_size, 4096)
		self.fc6 = nn.Linear(4096, 4096)
		self.fc7 = nn.Linear(4096, D_out)

	def forward(self, x):
		y = F.relu(self.conv1(x))
		y = F.relu(self.conv2(y))
		y = F.max_pool2d(y, 2, stride=2, padding=0)
		y = F.relu(self.conv3(y))
		y = F.relu(self.conv4(y))
		y = F.max_pool2d(y, 2, stride=2, padding=0)
		y = F.relu(self.conv5(y))
		y = F.relu(self.conv6(y))
		y = F.relu(self.conv7(y))
		y = F.relu(self.conv8(y))
		y = F.relu(self.conv9(y))
		y = F.relu(self.conv10(y))
		y = F.relu(self.conv11(y))
		y = F.relu(self.conv12(y))
		y = F.relu(self.conv13(y))

		y = self.pooling_layer_fn.forward(y)
		y = F.relu(self.fc5(y))
		y = F.dropout(y, p=0.5, training=self.training)
		y = F.relu(self.fc6(y))
		y = F.dropout(y, p=0.5, training=self.training)
		y = self.fc7(y)
		return y