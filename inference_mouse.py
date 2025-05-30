import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models.ResNet import ResNetMouse
import os

def load_model(model_path, depth=26, width=1):
    # Initialize model
    model = ResNetMouse(depth=depth, width=width, output_dim=10).cuda()
    
    # Load trained weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    return model

def process_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(image).unsqueeze(0).cuda()
    return img_tensor, image, original_size

def visualize_keypoints(image, keypoints, save_path=None):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Define colors for different keypoints
    colors = ['r', 'g', 'b', 'y', 'c']
    labels = ['kp0', 'kp1', 'kp2', 'kp3', 'kp4']
    
    # Plot each keypoint
    for i in range(0, len(keypoints), 2):
        x, y = keypoints[i:i+2]
        plt.scatter(x, y, c=colors[i//2], s=100, label=labels[i//2])
    
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./results/mouse_tracking/model_final.pth')
    parser.add_argument('--image_path', required=True, help='Path to input image')
    parser.add_argument('--output_dir', default='./results/predictions')
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, args.depth, args.width)
    
    # Process image
    print("Processing image...")
    img_tensor, original_image, (orig_width, orig_height) = process_image(args.image_path)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(img_tensor)
        
    # Convert normalized predictions back to image coordinates
    keypoints = predictions[0].cpu().numpy()
    keypoints_denorm = []
    for i in range(0, len(keypoints), 2):
        x = keypoints[i] * orig_width
        y = keypoints[i+1] * orig_height
        keypoints_denorm.extend([x, y])
    
    # Save visualization
    output_path = os.path.join(args.output_dir, 
                              f'pred_{os.path.basename(args.image_path)}')
    print("Saving visualization to:", output_path)
    visualize_keypoints(original_image, keypoints_denorm, output_path)
    
    # Print keypoint coordinates
    print("\nPredicted keypoint coordinates (x, y):")
    for i in range(0, len(keypoints_denorm), 2):
        print(f"Keypoint {i//2}: ({keypoints_denorm[i]:.2f}, {keypoints_denorm[i+1]:.2f})")

if __name__ == '__main__':
    main() 