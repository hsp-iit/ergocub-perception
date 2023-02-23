#!/bin/bash

TMUX_NAME=perception-tmux
DOCKER_IMAGE_NAME=ergocub_perception_container

docker rm -f $DOCKER_IMAGE_NAME
tmux kill-session -t $TMUX_NAME
