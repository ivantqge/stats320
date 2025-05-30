import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np

# Define normalization constants first
NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# Add mouse-specific transforms
mouse_transforms = transforms.Compose([
	# Keep original size since we need to map coordinates correctly
	transforms.ToTensor(),
	transforms.Normalize(*NORM)
])

class MouseDataset(torch.utils.data.Dataset):
	def __init__(self, root_dir, frames_dir, targets_dir, transform=None):
		"""
		Args:
			root_dir (str): Root directory containing the dataset
			frames_dir (str): Directory with mouse frames (3, 373, 416)
			targets_dir (str): Directory with binary masks (21x21 dilated blobs)
			transform (callable, optional): Optional transform to be applied on frames
		"""
		self.root_dir = root_dir
		self.frames_dir = os.path.join(root_dir, frames_dir)
		self.targets_dir = os.path.join(root_dir, targets_dir)
		self.transform = transform
		self.target_types = ['kp0', 'kp1', 'kp2', 'kp3', 'kp4']  # keypoints 0-5
		
		# Get all frame files
		self.frame_files = sorted([f for f in os.listdir(self.frames_dir) 
								 if f.endswith(('.png', '.jpg', '.pt'))])
		
		# Verify that for each frame, we have all keypoint target types
		for frame_file in self.frame_files:
			frame_base = os.path.splitext(frame_file)[0]
			for target_type in self.target_types:
				target_file = f"{frame_base}_{target_type}{os.path.splitext(frame_file)[1]}"
				assert os.path.exists(os.path.join(self.targets_dir, target_file)), \
					f"Missing target file {target_file} for frame {frame_file}"

	def __len__(self):
		return len(self.frame_files)

	def __getitem__(self, idx):
		frame_path = os.path.join(self.frames_dir, self.frame_files[idx])
		frame_base = os.path.splitext(self.frame_files[idx])[0]
		
		# Load frame
		if frame_path.endswith('.pt'):
			frame = torch.load(frame_path)
		else:
			frame = Image.open(frame_path).convert('RGB')
			if self.transform:
				frame = self.transform(frame)
			else:
				frame = transforms.ToTensor()(frame)
		
		# Load all keypoint targets and get their coordinates
		target_coords = []
		for target_type in self.target_types:
			target_file = f"{frame_base}_{target_type}{os.path.splitext(self.frame_files[idx])[1]}"
			target_path = os.path.join(self.targets_dir, target_file)
			
			# Load target mask
			if target_path.endswith('.pt'):
				mask = torch.load(target_path)
				if isinstance(mask, torch.Tensor):
					mask = mask.numpy()
			else:
				mask = np.array(Image.open(target_path))
			
			# Find center of the blob (average of all 1s coordinates)
			y_coords, x_coords = np.where(mask > 0)
			if len(y_coords) > 0:
				y = int(np.mean(y_coords))
				x = int(np.mean(x_coords))
			else:
				y, x = 0, 0  # Default if no target found
				
			# Normalize coordinates to [0, 1] range for better training
			x = x / mask.shape[1]  # normalize by width
			y = y / mask.shape[0]  # normalize by height
			target_coords.extend([x, y])
			
		return frame, torch.tensor(target_coords, dtype=torch.float32)  # Returns 12 coordinates (x,y for each keypoint)

# Other transforms using the same NORM constant
te_transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(*NORM)
])

tr_transforms = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(*NORM)
])

mnist_transforms = transforms.Compose([
	transforms.Resize((32, 32)),
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
])

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
					'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
					'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def prepare_test_data(args):
	if args.dataset == 'mouse':
		print('Preparing mouse validation data...')
		valset = MouseDataset(
			root_dir=args.dataroot,
			frames_dir=args.val_frames_dir,
			targets_dir=args.val_masks_dir,
			transform=mouse_transforms
		)
		if not hasattr(args, 'workers'):
			args.workers = 1
		valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
											shuffle=False, num_workers=args.workers)
		return valset, valloader
	elif args.dataset == 'cifar10':
		tesize = 10000
		if not hasattr(args, 'corruption') or args.corruption == 'original':
			print('Test on the original test set')
			teset = torchvision.datasets.CIFAR10(root=args.dataroot,
												train=False, download=True, transform=te_transforms)
		elif args.corruption in common_corruptions:
			print('Test on %s level %d' %(args.corruption, args.level))
			teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
			teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
			teset = torchvision.datasets.CIFAR10(root=args.dataroot,
												train=False, download=True, transform=te_transforms)
			teset.data = teset_raw

		elif args.corruption == 'cifar_new':
			from utils.cifar_new import CIFAR_New
			print('Test on CIFAR-10.1')
			teset = CIFAR_New(root=args.dataroot + 'CIFAR-10.1/datasets/', transform=te_transforms)
			permute = False
		else:
			raise Exception('Corruption not found!')
	else:
		raise Exception('Dataset not found!')

	if not hasattr(args, 'workers'):
		args.workers = 1
	teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
											shuffle=False, num_workers=args.workers)
	return teset, teloader

def prepare_train_data(args):
	print('Preparing data...')
	if args.dataset == 'mouse':
		trset = MouseDataset(
			root_dir=args.dataroot,
			frames_dir=args.train_frames_dir,
			targets_dir=args.train_masks_dir,
			transform=mouse_transforms
		)
	elif args.dataset == 'cifar10':
		trset = torchvision.datasets.CIFAR10(root=args.dataroot,
										train=True, download=True, transform=tr_transforms)
	else:
		raise Exception('Dataset not found!')

	if not hasattr(args, 'workers'):
		args.workers = 1
	trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
											shuffle=True, num_workers=args.workers)
	return trset, trloader
	