#!/usr/bin/env python3

import os
import time
import yarp
import cv2
import numpy as np
from glob import glob

def main():
    # Initialize YARP
    yarp.Network.init()
    
    if not yarp.Network.checkNetwork():
        print("[ERROR] YARP network not found. Please start yarpserver")
        return

    # Create ports
    rgb_port = yarp.Port()
    depth_port = yarp.Port()
    rgb_port_name = "/depthCamera/rgbImage:o"
    depth_port_name = "/depthCamera/depthImage:o"
    rgb_port.open(rgb_port_name)
    depth_port.open(depth_port_name)

    # Prepare yarp images
    yarp_rgb = yarp.ImageRgb()
    yarp_depth = yarp.ImageMono()
    
    # Get list of images
    rgb_dir = r"/home/mnasato-iit.local/DATA_processed/DATA_completed/66a1f98e3200d72210ae4dba/1/FeedbackPositive2_Start/[_depthCamera_rgbImage_o]"
    
    # Use os.listdir instead of glob to avoid issues with square brackets
    rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.lower().endswith('.jpg')])
    
    if not rgb_files:
        print(f"[ERROR] No images found in:\n{rgb_dir}")
        return
    
    print(f"Found {len(rgb_files)} images to stream")
    
    try:
        while True:
            for rgb_path in rgb_files:
                # Read RGB image
                rgb_frame = cv2.imread(rgb_path)
                if rgb_frame is None:
                    print(f"[WARNING] Could not read RGB image: {rgb_path}")
                    continue
                
                # Convert BGR to RGB for RGB image
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to grayscale for depth stream
                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                
                # Prepare and send RGB image through YARP
                yarp_rgb.resize(rgb_frame.shape[1], rgb_frame.shape[0])
                yarp_rgb.setExternal(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0])
                rgb_port.write(yarp_rgb)
                
                # Prepare and send grayscale image as depth through YARP
                yarp_depth.resize(gray_frame.shape[1], gray_frame.shape[0])
                yarp_depth.setExternal(gray_frame.data, gray_frame.shape[1], gray_frame.shape[0])
                depth_port.write(yarp_depth)
                
                # Add a small delay to control streaming rate (adjust as needed)
                time.sleep(0.033)  # ~30fps
                print(f"Sent image pair: {os.path.basename(rgb_path)}")
                
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user")
    finally:
        # Cleanup
        rgb_port.close()
        depth_port.close()
        yarp.Network.fini()

if __name__ == "__main__":
    main()
