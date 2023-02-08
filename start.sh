#!/usr/bin/bash
echo "Start this script from inside the Ergocub-Visual-Perception directory!"

# Stop all existing docker containers and remove them
docker stop ergocub_container && docker rm ergocub_container

# Start the container with the right options
docker run --gpus=all -v $(pwd):/home/ecub -itd --rm --network host --name ergocub_container andrewr96/ecub-env:yarp bash

# Create tmux session
tmux new-session -d -s ecub-tmux

# Source

# 0
tmux send-keys -t ecub-tmux "/home/ergocub/miniconda3/envs/ecub/bin/python scripts/source.py" Enter
tmux split-window -v -t ecub-tmux

# 2
tmux send-keys -t ecub-tmux "docker exec -it ergocub_container bash" Enter
tmux send-keys -t ecub-tmux "python scripts/action_recognition_pipeline.py" Enter
tmux split-window -v -t ecub-tmux

#4
tmux send-keys -t ecub-tmux "/home/ergocub/miniconda3/envs/ecub/bin/yarpserver --write" Enter
tmux split-window -h -t ecub-tmux

# 5
tmux send-keys -t ecub-tmux "docker exec -it ergocub_container bash" Enter
tmux send-keys -t ecub-tmux "python scripts/manager.py" Enter

tmux split-window -h -t ecub-tmux

# 6
tmux send-keys -t ecub-tmux "docker exec -it ergocub_container bash" Enter
tmux send-keys -t ecub-tmux "python scripts/action_recognition_rpc.py" Enter
tmux split-window -h -t ecub-tmux
#7
tmux send-keys -t ecub-tmux "docker exec -it ergocub_container bash" Enter
tmux send-keys -t ecub-tmux "python scripts/object_detection_rpc.py" Enter
tmux split-window -h -t ecub-tmux


tmux select-pane -t ecub-tmux:0.0
tmux split-window -h -t ecub-tmux
# 1
tmux send-keys -t ecub-tmux "/home/ergocub/miniconda3/envs/ecub/bin/python scripts/sink.py" Enter

tmux select-pane -t ecub-tmux:0.2
tmux split-window -h -t ecub-tmux
#3
tmux send-keys -t ecub-tmux "docker exec -it ergocub_container bash" Enter
tmux send-keys -t ecub-tmux "python scripts/grasping_pipeline.py" Enter

#
#tmux split-window -v -p 66 -t ecub-tmux

#
#tmux split-window -v -p 66 -t ecub-tmux
#tmux send-keys -t ecub-tmux "cd ErgoCub-Visual-Perception && /home/ergocub/miniconda3/envs/ecub/bin/python scripts/sink.py" Enter
#
#tmux select-pane -t ecub-tmux:0.0
#tmux split-window -v -p 66 -t ecub-tmux
#tmux send-keys -t ecub-tmux "docker exec -it ergocub_container bash" Enter
#
#tmux split-window -v -p 66 -t ecub-tmux
#tmux send-keys -t ecub-tmux "docker exec -it ergocub_container bash" Enter
#
#tmux split-window -v -p 66 -t ecub-tmux
#tmux send-keys -t ecub-tmux "docker exec -it ergocub_container bash" Enter

tmux a -t ecub-tmux
