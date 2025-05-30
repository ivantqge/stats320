from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from models.ResNet import ResNetMouse
from models.SSHead import extractor_from_layer2, head_on_layer2, ExtractorHead

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mouse')
parser.add_argument('--dataroot', default='./data/mouse_dataset/')
parser.add_argument('--train_frames_dir', default='train/images')
parser.add_argument('--train_masks_dir', default='train/masks')
parser.add_argument('--val_frames_dir', default='val/images')
parser.add_argument('--val_masks_dir', default='val/masks')
parser.add_argument('--shared', default='layer2')
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--group_norm', default=8, type=int)
########################################################################
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--nepoch', default=100, type=int)
parser.add_argument('--milestone_1', default=30, type=int)
parser.add_argument('--milestone_2', default=60, type=int)
########################################################################
parser.add_argument('--outf', default='./results/mouse_tracking')

args = parser.parse_args()
my_makedir(args.outf)

# Build model
net = ResNetMouse(args.depth, args.width, output_dim=10).cuda()
ext = extractor_from_layer2(net)  # Extract features from layer2
head = head_on_layer2(net, args.width, 4)  # 4 classes for quadrant prediction
ssh = ExtractorHead(ext, head).cuda()

# Prepare data
_, teloader = prepare_test_data(args)
_, trloader = prepare_train_data(args)

# Setup training
optimizer = optim.Adam(list(net.parameters()) + list(head.parameters()), 
                      lr=0.005, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [args.milestone_1, args.milestone_2], gamma=0.2)

# Loss functions
criterion_coord = nn.SmoothL1Loss().cuda()  # Huber loss for better coordinate regression
criterion_ssh = nn.CrossEntropyLoss().cuda()

all_losses = []
print('Training...')
print('Epoch\tCoord Loss\tSSH Loss\tLR')

for epoch in range(1, args.nepoch + 1):
    net.train()
    ssh.train()
    epoch_losses = []

    for batch_idx, (inputs, targets) in enumerate(trloader):
        optimizer.zero_grad()
        
        # Main task: coordinate prediction
        inputs_coord, targets_coord = inputs.cuda(), targets.cuda()
        outputs_coord = net(inputs_coord)
        loss_coord = criterion_coord(outputs_coord, targets_coord)
        
        # Self-supervised task: quadrant prediction
        B, C, H, W = inputs.shape
        # Ensure even dimensions for splitting
        H_half = H // 2
        W_half = W // 2
        
        patches = []
        patch_labels = []
        
        for b in range(B):
            # Split into quadrants with adaptive pooling to ensure consistent sizes
            q1 = inputs[b:b+1, :, :H_half, :W_half]
            q2 = inputs[b:b+1, :, :H_half, W_half:]
            q3 = inputs[b:b+1, :, H_half:, :W_half]
            q4 = inputs[b:b+1, :, H_half:, W_half:]
            
            # Find minimum height and width across quadrants
            min_H = min(q1.size(2), q2.size(2), q3.size(2), q4.size(2))
            min_W = min(q1.size(3), q2.size(3), q3.size(3), q4.size(3))
            
            # Resize all quadrants to the minimum size
            q1 = nn.functional.adaptive_avg_pool2d(q1, (min_H, min_W))
            q2 = nn.functional.adaptive_avg_pool2d(q2, (min_H, min_W))
            q3 = nn.functional.adaptive_avg_pool2d(q3, (min_H, min_W))
            q4 = nn.functional.adaptive_avg_pool2d(q4, (min_H, min_W))
            
            quadrants = [q1, q2, q3, q4]
            order = torch.randperm(4)
            
            for idx in order:
                patches.append(quadrants[idx].squeeze(0))
                patch_labels.append(idx)
        
        patches = torch.stack(patches).cuda()
        patch_labels = torch.tensor(patch_labels).cuda()
        
        outputs_ssh = ssh(patches)
        loss_ssh = criterion_ssh(outputs_ssh, patch_labels)
        
        # Combined loss with dynamic weighting
        loss = loss_coord + 0.1 * loss_ssh  # Reduced SSH loss weight
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        
        optimizer.step()
        
        epoch_losses.append([loss_coord.item(), loss_ssh.item()])
    
    # Calculate average losses
    avg_losses = np.mean(epoch_losses, axis=0)
    all_losses.append(avg_losses)
    scheduler.step()
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f'{epoch}/{args.nepoch}\t{avg_losses[0]:.4f}\t{avg_losses[1]:.4f}\t{current_lr:.6f}')
    
    # Save checkpoint
    if epoch % 10 == 0:
        state = {
            'net': net.state_dict(),
            'head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'losses': all_losses
        }
        torch.save(state, f'{args.outf}/checkpoint_epoch_{epoch}.pth')

# Save final model
state = {
    'net': net.state_dict(),
    'head': head.state_dict(),
    'optimizer': optimizer.state_dict(),
    'losses': all_losses
}
torch.save(state, f'{args.outf}/model_final.pth') 