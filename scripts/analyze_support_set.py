#!/usr/bin/env python3

import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from action_rec.ar.ar import ActionRecognizer

def analyze_support_set(support_set_path):
    """Analyze the contents of a support set pickle file"""
    print(f"\nAnalyzing support set: {support_set_path}\n")
    
    # Load the support set
    with open(support_set_path, 'rb') as f:
        support_set = pkl.load(f)
    
    # Overall statistics
    n_actions = len(support_set)
    total_demonstrations = sum(len(data['sk']) for data in support_set.values())
    
    print("=== Overall Statistics ===")
    print(f"Number of actions: {n_actions}")
    print(f"Total demonstrations: {total_demonstrations}")
    print("\n=== Per Action Details ===")
    
    # Analyze each action
    for action_name, data in support_set.items():
        n_demos = len(data['sk'])
        sk_shape = data['sk'][0].shape if n_demos > 0 else "No demonstrations"
        n_features = len(data['features'])
        
        print(f"\nAction: {action_name}")
        print(f"  Number of demonstrations: {n_demos}")
        print(f"  Skeleton data shape: {sk_shape}")
        print(f"  Number of features: {n_features}")
        
        if n_demos > 0:
            # Basic statistics of skeleton data
            sk_data = np.array(data['sk'])
            print(f"  Skeleton data statistics:")
            print(f"    Mean: {sk_data.mean():.4f}")
            print(f"    Std: {sk_data.std():.4f}")
            print(f"    Min: {sk_data.min():.4f}")
            print(f"    Max: {sk_data.max():.4f}")
    
    # Plot distribution of demonstrations per action
    demo_counts = [len(data['sk']) for data in support_set.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(support_set.keys(), demo_counts)
    plt.title('Number of Demonstrations per Action')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Demonstrations')
    plt.tight_layout()
    plt.savefig('support_set_distribution.png')
    print(f"\nPlot saved as 'support_set_distribution.png'")

def main():
    # You can specify the path to your support set here
    support_set_path = input("Enter the path to your support set pickle file: ")
    
    if not os.path.exists(support_set_path):
        print(f"Error: File not found: {support_set_path}")
        return
    
    try:
        analyze_support_set(support_set_path)
    except Exception as e:
        print(f"Error analyzing support set: {str(e)}")

if __name__ == "__main__":
    main()
