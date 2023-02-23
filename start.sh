#!/usr/bin/bash
echo "Start this script from inside the Ergocub-Visual-Perception directory!"

# Stop all existing docker containers and remove them

# Start the container with the right options
docker run --gpus=all -v $(pwd):/home/ecub -itd --rm \
--gpus=all \
--env DISPLAY=:0 \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="/etc/resolv.conf:/etc/resolv.conf" \
--ipc=host \
--network=host --name ergocub_perception_container andrewr96/ecub-env:yarp bash

# Create tmux session
tmux new-session -d -s perception-tmux

# Source

# 0
tmux send-keys -t perception-tmux "docker exec -it ergocub_perception_container bash" Enter

#tmux send-keys -t perception-tmux "yarp conf 10.0.0.150 10000" Enter
tmux send-keys -t perception-tmux "python scripts/source.py" Enter
tmux split-window -v -t perception-tmux

# 2
tmux send-keys -t perception-tmux "docker exec -it ergocub_perception_container bash" Enter
tmux send-keys -t perception-tmux "yarp detect --write" Enter
tmux send-keys -t perception-tmux "python scripts/action_recognition_pipeline.py" Enter
tmux split-window -v -t perception-tmux

#4
tmux send-keys -t perception-tmux "docker exec -it ergocub_perception_container bash" Enter
tmux send-keys -t perception-tmux "yarpserver --write" Enter
tmux split-window -h -t perception-tmux

# 5
tmux send-keys -t perception-tmux "docker exec -it ergocub_perception_container bash" Enter
tmux send-keys -t perception-tmux "python scripts/manager.py" Enter
tmux split-window -h -t perception-tmux

# 6
tmux send-keys -t perception-tmux "docker exec -it ergocub_perception_container bash" Enter
tmux send-keys -t perception-tmux "python scripts/action_recognition_rpc.py" Enter
tmux split-window -h -t perception-tmux
#7
tmux send-keys -t perception-tmux "docker exec -it ergocub_perception_container bash" Enter
tmux send-keys -t perception-tmux "python scripts/object_detection_rpc.py" Enter
tmux split-window -h -t perception-tmux

tmux send-keys -t perception-tmux "docker exec -it ergocub_perception_container bash" Enter
tmux send-keys -t perception-tmux "python scripts/source_to_sink.py" Enter

tmux select-pane -t perception-tmux:0.0
tmux split-window -h -t perception-tmux

# 1
tmux send-keys -t perception-tmux "docker exec -it ergocub_perception_container bash" Enter
tmux send-keys -t perception-tmux "python scripts/sink.py" Enter

tmux select-pane -t perception-tmux:0.2
tmux split-window -h -t perception-tmux
#3
tmux send-keys -t perception-tmux "docker exec -it ergocub_perception_container bash" Enter
tmux send-keys -t perception-tmux "python scripts/grasping_pipeline.py" Enter


tmux a -t perception-tmux
