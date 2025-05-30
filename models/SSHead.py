from torch import nn
import math
import copy
import torch
import torch.nn as nn

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

def extractor_from_layer2(net):
	"""Extract features from layer2 of the network"""
	return lambda x: net.layer2(net.layer1(net.conv1(x)))

def extractor_from_layer3(net):
	"""Extract features from layer3 of the network"""
	return lambda x: net.layer3(net.layer2(net.layer1(net.conv1(x))))

def head_on_layer2(net, width, num_classes):
	"""Create head for self-supervised learning on layer2 features"""
	return nn.Sequential(
		net.layer3,
		net.bn,
		net.relu,
		net.avgpool,
		nn.Flatten(),
		nn.Linear(64 * width, num_classes)
	)

class ExtractorHead(nn.Module):
	"""Combines feature extractor and head for self-supervised learning"""
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head
	
	def forward(self, x):
		return self.head(self.ext(x))
