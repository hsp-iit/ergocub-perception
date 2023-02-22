#!/bin/bash

docker rm -f ergocub_perception_container
tmux kill-session -t perception-tmux
