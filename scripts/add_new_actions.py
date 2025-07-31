#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
from action_rec.ar.ar import ActionRecognizer
import pickle as pkl
from action_rec.hpe.hpe import HumanPoseEstimator

def extract_skeleton_from_frames(folder_path):
    """Extract skeleton data from a folder containing image frames"""
    frames = []
    # Initialize the pose estimator
    pose_estimator = HumanPoseEstimator(
        input_type="rgb",
        device="cuda:0",
        final_ckpt_path="action_rec/hpe/weights/epoch_500_49_26.pth",
        n_joints=30
    )

    # Get all image files
    frame_files = sorted([f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not frame_files:
        print(f"No image files found in {folder_path}")
        return None

    # Process each frame
    skeletons = []
    for frame_file in frame_files:
        # Read image
        image_path = os.path.join(folder_path, frame_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get skeleton
        with torch.no_grad():
            skeleton = pose_estimator.inference(image)
            if skeleton is not None:
                skeletons.append(skeleton.reshape(-1))  # Flatten to match expected format
    
    return np.array(skeletons) if skeletons else None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create a new support set from skeleton data')
    parser.add_argument('--output', type=str, required=True, 
                      help='Path where to save the new support set (e.g., action_rec/ar/saved/my_new_support.pkl)')
    parser.add_argument('--input-base', type=str, required=True,
                      help='Base directory containing action folders')
    args = parser.parse_args()

    # Initialize the Action Recognizer with the same parameters as in your main application
    ar = ActionRecognizer(
        input_type="skeleton",
        device="cuda:0",
        add_hook=False,
        final_ckpt_path="action_rec/ar/weights/raws/skeleton/5-w-5-s.pth",
        seq_len=16,
        way=5,
        shot=5,
        n_joints=30
    )

    # Get all action folders
    action_folders = [f for f in os.listdir(args.input_base) if os.path.isdir(os.path.join(args.input_base, f))]
    if not action_folders:
        print(f"No action folders found in {args.input_base}")
        return

    print(f"Found action folders: {action_folders}")
    
    # Dictionary of actions to add
    new_actions = {
        action_name: os.path.join(args.input_base, action_name)
        for action_name in action_folders
    }

    # Add each new action
    for action_name, folder_path in new_actions.items():
        print(f"Processing action: {action_name}")
        
        # Extract skeleton data from the frames
        skeleton_data = extract_skeleton_from_frames(folder_path)
        if skeleton_data is None:
            print(f"Failed to extract skeletons from {action_name}")
            continue
            
        # Ensure we have enough frames
        if len(skeleton_data) < 16:
            print(f"Warning: {action_name} has less than 16 frames ({len(skeleton_data)} found)")
            continue
            
        # Take the first 16 frames
        skeleton_data = skeleton_data[:16]
        
        # Create input dictionary for training
        train_input = {
            "flag": action_name,
            "data": {
                "sk": skeleton_data
            }
        }
        
        # Add to support set
        result = ar.train(train_input)
        print(result)

    # Save the new support set
    print(f"\nSaving support set to {args.output}")
    ar.save(args.output)
    
    # Generate visualization
    print("\nGenerating visualization...")
    ar.save_ss_image()
    
    print("\nDone! To use this support set:")
    print(f"1. Edit configs/action_rec_config.py to use: {args.output}")
    print("   or")
    print(f"2. Use the 'load' command in human console: load {args.output}")

if __name__ == "__main__":
    main()
