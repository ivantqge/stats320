from __future__ import print_function
import argparse
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from utils.rotation import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--dataroot', default='/data/yusun/datasets/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--fix_bn', action='store_true')
parser.add_argument('--fix_ssh', action='store_true')
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--dset_size', default=0, type=int)
########################################################################
parser.add_argument('--outf', default='.')
parser.add_argument('--resume', default=None)

args = parser.parse_args()
args.threshold += 0.001		# to correct for numeric errors
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)
teset, teloader = prepare_test_data(args)

print('Resuming from %s...' %(args.resume))
ckpt = torch.load(args.resume + '/ckpt.pth')
if args.online:
	net.load_state_dict(ckpt['net'])
	head.load_state_dict(ckpt['head'])

criterion_ssh = nn.CrossEntropyLoss().cuda()
if args.fix_ssh:
	optimizer_ssh = optim.SGD(ext.parameters(), lr=args.lr)
else:
	optimizer_ssh = optim.SGD(ssh.parameters(), lr=args.lr)

def adapt_single(image):
	if args.fix_bn:
		ssh.eval()
	elif args.fix_ssh:
		ssh.eval()
		ext.train()
	else:
		ssh.train()
		
	for iteration in range(args.niter):
		# Create augmented versions of the test image
		inputs = []
		for _ in range(args.batch_size):
			# Apply random augmentations suitable for mouse frames
			aug_image = tr_transforms(image)
			inputs.append(aug_image)
		inputs = torch.stack(inputs)
		
		# For mouse tracking, we'll use frame patches as self-supervision
		# Split each frame into 4 quadrants and predict their relative positions
		B, C, H, W = inputs.shape
		patches = []
		patch_labels = []
		
		for b in range(B):
			# Split into quadrants
			q1 = inputs[b, :, :H//2, :W//2]
			q2 = inputs[b, :, :H//2, W//2:]
			q3 = inputs[b, :, H//2:, :W//2]
			q4 = inputs[b, :, H//2:, W//2:]
			
			# Create patches with random ordering
			quadrants = [q1, q2, q3, q4]
			order = torch.randperm(4)
			
			for idx in order:
				patches.append(quadrants[idx])
				patch_labels.append(idx)
		
		patches = torch.stack(patches).cuda()
		patch_labels = torch.tensor(patch_labels).cuda()
		
		# Update model using patch position prediction
		optimizer_ssh.zero_grad()
		outputs_ssh = ssh(patches)
		loss_ssh = criterion_ssh(outputs_ssh, patch_labels)
		loss_ssh.backward()
		optimizer_ssh.step()

def test_single(model, image, target):
	model.eval()
	inputs = te_transforms(image).unsqueeze(0)
	with torch.no_grad():
		outputs = model(inputs.cuda())
		# For coordinate prediction, calculate L2 distance
		distance = torch.norm(outputs - target.cuda(), p=2)
	return distance.item()

print('Running...')
distances = []
sshconf = []
if args.dset_size == 0:
	args.dset_size = len(teset)
for i in tqdm(range(1, args.dset_size+1)):
	if not args.online:
		net.load_state_dict(ckpt['net'])
		head.load_state_dict(ckpt['head'])

	frame, target = teset[i-1]
	
	# Convert frame to PIL Image for transformations
	if isinstance(frame, torch.Tensor):
		frame = transforms.ToPILImage()(frame)
	
	# Test current performance
	distance = test_single(net, frame, target)
	distances.append(distance)
	
	# Adapt if confidence is low (distance is high)
	if distance > args.threshold:
		adapt_single(frame)

rdict = {
	'distances': np.asarray(distances),
	'mean_distance': np.mean(distances),
	'median_distance': np.median(distances)
}
torch.save(rdict, args.outf + '/%s_%d_ada.pth' %(args.corruption, args.level))
