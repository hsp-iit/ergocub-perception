#!/usr/bin/bash

# Change name also in stop.sh
TMUX_NAME=perception-tmux
DOCKER_CONTAINER_NAME=ergocub_perception_container

echo "Start this script inside the ergoCub visual perception rooot folder"
usage() { echo "Usage: $0 [-i ip_address] [-n nameserver] [-y (to start yarp server] [-s (to start source)] [-r repeater]" 1>&2; exit 1; }

while getopts i:yshn:r flag
do
    case "${flag}" in
        i) SERVER_IP=${OPTARG};;
        n) YARP_NAMESERVER=${OPTARG};;
        y) START_YARP_SERVER='1';;
        r) REPEATER='1';;
        s) START_SOURCE='1';;
        h) usage;;
        *) usage;;
    esac
done

# Start the container with the right options
docker run --gpus=all -v "$(pwd)":/home/ecub -itd --rm \
--gpus=all \
--env DISPLAY=:0 \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--ipc=host \
--network=host --name $DOCKER_CONTAINER_NAME ar0s/ergocub-perception bash

# Create tmux session
tmux new-session -d -s $TMUX_NAME

# Set server
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter

if [ -n "$YARP_NAMESERVER" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "yarp namespace $YARP_NAMESERVER" Enter
fi

if [ -n "$SERVER_IP" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "yarp conf $SERVER_IP 10000" Enter
else
  tmux send-keys -t $TMUX_NAME "yarp detect --write" Enter
fi

if [ -n "$REPEATER" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "yarp repeat /depthCamera/rgbImage:r" Enter
fi

# Source
echo $START_SOURCE
if [ -n "$START_SOURCE" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "python scripts/source.py" Enter
fi
tmux split-window -v -t $TMUX_NAME

# Action Recognition Pipeline
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "python scripts/action_recognition_pipeline.py" Enter
tmux split-window -v -t $TMUX_NAME

# Yarp Server
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
if [ -n "$START_YARP_SERVER" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "yarpserver --write" Enter
fi

if [ -n "$REPEATER" ] # Variable is non-null
then
  tmux send-keys -t $TMUX_NAME "yarp repeat /depthCamera/depthImage:r" Enter
fi

tmux split-window -h -t $TMUX_NAME

# Manager
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "python scripts/manager.py" Enter
tmux split-window -h -t $TMUX_NAME

# Action Recognition RPC
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "sleep 5" Enter  # TODO TEST
tmux send-keys -t $TMUX_NAME "python scripts/action_recognition_rpc.py" Enter
tmux split-window -h -t $TMUX_NAME

# Object Detection RPC
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "sleep 5" Enter  # TODO TEST
tmux send-keys -t $TMUX_NAME "python scripts/object_detection_rpc.py" Enter
tmux split-window -h -t $TMUX_NAME

# Source to Sink
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "sleep 5" Enter  # TODO TEST
tmux send-keys -t $TMUX_NAME "python scripts/source_to_sink.py" Enter
tmux select-pane -t $TMUX_NAME:0.0
tmux split-window -h -t $TMUX_NAME

# Sink
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "python scripts/sink.py" Enter
tmux select-pane -t $TMUX_NAME:0.2
tmux split-window -h -t $TMUX_NAME

# Grasping Pipeline
tmux send-keys -t $TMUX_NAME "docker exec -it $DOCKER_CONTAINER_NAME bash" Enter
tmux send-keys -t $TMUX_NAME "python scripts/grasping_pipeline.py" Enter

# Attach
tmux a -t $TMUX_NAME
