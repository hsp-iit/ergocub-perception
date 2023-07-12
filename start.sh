#!/usr/bin/bash

# Change name also in stop.sh
TMUX_NAME=perception-tmux
DOCKER_CONTAINER_NAME=ergocub_perception_container

echo "Start this script inside the ergoCub visual perception root folder"
usage() { echo "Usage: $0 [-i ip_address] [-n nameserver] [-y (to start yarp server] [-s (to start source)] [-r repeater] [-b (just bash)]" 1>&2; exit 1; }

while getopts i:ysbhn:r flag
do
    case "${flag}" in
        i) SERVER_IP=${OPTARG};;
        n) YARP_NAMESERVER=${OPTARG};;
        y) START_YARP_SERVER='1';;
        r) REPEATER='1';;
        s) START_SOURCE='1';;
        b) JUST_BASH='1';;
        h) usage;;
        *) usage;;
    esac
done

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

# Just bash?
if [ -n "$JUST_BASH" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
  tmux a -t $TMUX_NAME
  exit 0
fi

# Human Detection
tmux rename-window -t $TMUX_NAME components
tmux select-pane -T "Human Detection"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/human_detection.py" Enter

tmux split-window -h -t $TMUX_NAME

# Human Pose Estimation
tmux select-pane -T "Human Pose Estimation"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/human_pose_estimation.py" Enter

tmux split-window -h -t $TMUX_NAME

# Action Recognition Pipeline
tmux select-pane -T "Action Recognition"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/action_recognition.py" Enter

tmux split-window -h -t $TMUX_NAME

# Focus Detector
tmux select-pane -T "Focus Detection"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/focus_detection.py" Enter

tmux split-window -h -t $TMUX_NAME

# Grasping Pipeline
tmux select-pane -T "Grasping Pipeline"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/grasping_pipeline.py" Enter

tmux split-window -h -t $TMUX_NAME

# Segmentation
tmux select-pane -T "Segmentation"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/segmentation.py" Enter

tmux select-layout -t $TMUX_NAME tiled
tmux new-window -t $TMUX_NAME
tmux rename-window -t $TMUX_NAME input/output

# Source
tmux select-pane -T "Source"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
if [ -n "$START_SOURCE" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "python scripts/source.py" Enter
fi
tmux split-window -h -t $TMUX_NAME

# Sink
tmux select-pane -T "Sink"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/sink.py" Enter

tmux split-window -h -t $TMUX_NAME

# Recorder
tmux select-pane -T "Recorder"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/recorder.py"

tmux split-window -h -t $TMUX_NAME

# 3D Shape Completion Visualizer
tmux select-pane -T "3D Shape Completion Visualizer"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/od3dviz.py"

tmux split-window -h -t $TMUX_NAME

# Human Console
tmux select-pane -T "Human Console"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/human_console2.py"

tmux split-window -h -t $TMUX_NAME

# REALSENSE GUI
tmux select-pane -T "RealSense GUI"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/realsense_gui.py"

tmux select-layout -t $TMUX_NAME tiled
tmux new-window -t $TMUX_NAME
tmux rename-window -t $TMUX_NAME communication

# Manager
tmux select-pane -T "Manager"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/manager.py" Enter
tmux split-window -h -t $TMUX_NAME

# YarpManager
tmux select-pane -T "Bash"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux split-window -h -t $TMUX_NAME

# Yarp Server
tmux select-pane -T "Yarp Server"

# Set Yarp Server Configurations
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter

if [ -n "$YARP_NAMESERVER" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "yarp namespace $YARP_NAMESERVER" Enter
fi

if [ -n "$SERVER_IP" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "yarp conf $SERVER_IP 10000" Enter
fi

if [ -n "$REPEATER" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "yarp repeat /depthCamera/rgbImage:r" Enter
fi

if [ -n "$START_YARP_SERVER" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "yarpserver --write" Enter
else
  tmux send-keys -t $TMUX_NAME "yarp detect --write" Enter
fi

if [ -n "$REPEATER" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "yarp repeat /depthCamera/depthImage:r" Enter
  tmux split-window -h -t $TMUX_NAME
  tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
  tmux send-keys -t $TMUX_NAME "yarp repeat /depthCamera/depthImage:r" Enter
  tmux send-keys -t $TMUX_NAME "./connect_camera.sh" Enter
fi

tmux split-window -h -t $TMUX_NAME

# RPC server
tmux select-pane -T "RPC Server"
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "cd perception" Enter
tmux send-keys -t $TMUX_NAME "python scripts/rpc_server.py" Enter
tmux split-window -h -t $TMUX_NAME

tmux select-layout -t $TMUX_NAME tiled
# tmux new-window -t $TMUX_NAME
# tmux rename-window -t $TMUX_NAME gaze

# YarpManager
# tmux select-pane -T "iKinGazeCtrl"
# tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
# tmux send-keys -t $TMUX_NAME "iKinGazeCtrl --from .yarp/gaze_config.ini"
# tmux split-window -h -t $TMUX_NAME

# tmux select-pane -T "GazeController"
# tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
# tmux send-keys -t $TMUX_NAME "python scripts/gaze_controller.py"

tmux select-window -t "components"
# Attach
tmux a -t $TMUX_NAME
