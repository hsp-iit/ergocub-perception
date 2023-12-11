#!/usr/bin/bash

# Change name also in stop.sh
TMUX_NAME=perception-tmux
DOCKER_CONTAINER_NAME=ergocub_perception_container

echo "Start this script inside the ergoCub visual perception root folder"
usage() { echo "Usage: $0" 1>&2; exit 1; }

# Start the container with the right options
docker run --gpus=all -v "$(pwd)":/home/ergocub/perception -itd --rm \
--gpus=all \
--env DISPLAY=$DISPLAY \
--env PYTHONPATH=/home/ergocub/perception \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--ipc=host \
--network=host --name $DOCKER_CONTAINER_NAME ar0s/ergocub-perception-image bash

# Create tmux session
tmux new-session -d -s $TMUX_NAME
tmux set-option -t $TMUX_NAME status-left-length 140
tmux set -t $TMUX_NAME -g pane-border-status top
tmux set -t $TMUX_NAME -g mouse on

# GRASPING ###############################################
tmux rename-window -t $TMUX_NAME grasping

# 1: RANSAC
tmux select-pane -T "grasp_detection"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "mkdir -p grasping/grasp_detection/ransac_gd/trt/assets/" Enter
tmux send-keys -t $TMUX_NAME "cp onnxs/ransac200_10000.onnx grasping/grasp_detection/ransac_gd/trt/assets/" Enter
tmux send-keys -t $TMUX_NAME "python grasping/grasp_detection/ransac_gd/trt/build/onnx2trt.py" Enter

tmux split-window -h -t $TMUX_NAME

# 1: SEGMENTATION
tmux select-pane -T "segmentation"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "mkdir -p grasping/segmentation/fcn/trt/assets/" Enter
tmux send-keys -t $TMUX_NAME "cp onnxs/segmentation.onnx grasping/segmentation/fcn/trt/assets/" Enter
tmux send-keys -t $TMUX_NAME "python grasping/segmentation/fcn/trt/build/onnx2trt.py" Enter

tmux split-window -h -t $TMUX_NAME

# 1: SHAPE COMPLETION
tmux select-pane -T "shape_completion"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "mkdir -p grasping/shape_completion/confidence_pcr/trt/assets/" Enter
tmux send-keys -t $TMUX_NAME "cp onnxs/pcr.onnx grasping/shape_completion/confidence_pcr/trt/assets/" Enter
tmux send-keys -t $TMUX_NAME "python grasping/shape_completion/confidence_pcr/trt/build/onnx2trt.py" Enter

# HUMAN #################################################
tmux select-layout -t $TMUX_NAME tiled
tmux new-window -t $TMUX_NAME
tmux rename-window -t $TMUX_NAME action_rec

# HUMAN DETECTION
tmux select-pane -T "human_detection"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "mkdir -p action_rec/hd/weights/onnxs/" Enter
tmux send-keys -t $TMUX_NAME "cp onnxs/yolo.onnx action_rec/hd/weights/onnxs/" Enter
tmux send-keys -t $TMUX_NAME "python action_rec/hd/setup/7_create_engines.py" Enter

tmux split-window -h -t $TMUX_NAME

# IMAGE TRANSFORMATION (HOMOGRAPHY)
tmux select-pane -T "image_transformation"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "mkdir -p action_rec/hpe/weights/onnxs/" Enter
tmux send-keys -t $TMUX_NAME "cp onnxs/image_transformation1.onnx action_rec/hpe/weights/onnxs/" Enter
tmux send-keys -t $TMUX_NAME "python action_rec/hpe/setup/7_create_engines_image_transformation.py" Enter

tmux split-window -h -t $TMUX_NAME

# BBONE
tmux select-pane -T "backbone"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "mkdir -p  action_rec/hpe/weights/onnxs/" Enter
tmux send-keys -t $TMUX_NAME "cp onnxs/bbone1.onnx action_rec/hpe/weights/onnxs/" Enter
tmux send-keys -t $TMUX_NAME "python action_rec/hpe/setup/7_create_engines_bbone.py" Enter

tmux split-window -h -t $TMUX_NAME

# HEADS
tmux select-pane -T "heads"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "mkdir -p action_rec/hpe/weights/onnxs/" Enter
tmux send-keys -t $TMUX_NAME "cp onnxs/heads1.onnx action_rec/hpe/weights/onnxs/" Enter
tmux send-keys -t $TMUX_NAME "python action_rec/hpe/setup/7_create_engines_heads.py" Enter

# CONNECT ###############################################
tmux a -t $TMUX_NAME
