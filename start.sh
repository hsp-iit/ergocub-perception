#!/usr/bin/bash
echo "Start this script from inside the Ergocub-Visual-Perception directory!"

# Stop all existing docker containers and remove them
docker stop ecub-evp-docker && docker rm ecub-evp-docker

# Start the container with the right options (-icp=host removed)
docker run \
--gpus=all \
-itd \
--network=host \
-v $(pwd)/../:/home/ecub \
--env DISPLAY=:0 \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="/etc/resolv.conf:/etc/resolv.conf" \
--rm \
--name ecub-evp-docker \
andrewr96/ecub-env:yarp \
bash

# Create tmux session
tmux new-session -d -s ecub-evp-tmux

# 0
tmux send-keys -t ecub-evp-tmux "docker exec -it ecub-evp-docker bash" Enter
tmux send-keys -t ecub-evp-tmux "yarp detect --write" Enter  # NOTE here we set yarp detect --write
tmux send-keys -t ecub-evp-tmux "cd Ergocub-Visual-Perception" Enter
tmux send-keys -t ecub-evp-tmux "python scripts/source.py" Enter
tmux split-window -v -t ecub-evp-tmux

# 2
tmux send-keys -t ecub-evp-tmux "docker exec -it ecub-evp-docker bash" Enter
tmux send-keys -t ecub-evp-tmux "cd Ergocub-Visual-Perception" Enter
tmux send-keys -t ecub-evp-tmux "python scripts/action_recognition_pipeline.py" Enter
tmux split-window -v -t ecub-evp-tmux

#4
tmux send-keys -t ecub-evp-tmux "docker exec -it ecub-evp-docker bash" Enter
tmux send-keys -t ecub-evp-tmux "cd Ergocub-Visual-Perception" Enter
tmux send-keys -t ecub-evp-tmux "yarpserver --write" Enter
tmux split-window -h -t ecub-evp-tmux

# 5
tmux send-keys -t ecub-evp-tmux "docker exec -it ecub-evp-docker bash" Enter
tmux send-keys -t ecub-evp-tmux "cd Ergocub-Visual-Perception" Enter
tmux send-keys -t ecub-evp-tmux "python scripts/manager.py" Enter
tmux split-window -h -t ecub-evp-tmux

# 6
tmux send-keys -t ecub-evp-tmux "docker exec -it ecub-evp-docker bash" Enter
tmux send-keys -t ecub-evp-tmux "cd Ergocub-Visual-Perception" Enter
tmux send-keys -t ecub-evp-tmux "python scripts/action_recognition_rpc.py" Enter
tmux split-window -h -t ecub-evp-tmux
#7
tmux send-keys -t ecub-evp-tmux "docker exec -it ecub-evp-docker bash" Enter
tmux send-keys -t ecub-evp-tmux "cd Ergocub-Visual-Perception" Enter
tmux send-keys -t ecub-evp-tmux "python scripts/object_detection_rpc.py" Enter
tmux split-window -h -t ecub-evp-tmux

tmux send-keys -t ecub-evp-tmux "docker exec -it ecub-evp-docker bash" Enter
tmux send-keys -t ecub-evp-tmux "cd Ergocub-Visual-Perception" Enter
tmux send-keys -t ecub-evp-tmux "python scripts/source_to_sink.py" Enter

tmux select-pane -t ecub-evp-tmux:0.0
tmux split-window -h -t ecub-evp-tmux

# 1
tmux send-keys -t ecub-evp-tmux "docker exec -it ecub-evp-docker bash" Enter
tmux send-keys -t ecub-evp-tmux "cd Ergocub-Visual-Perception" Enter
tmux send-keys -t ecub-evp-tmux "python scripts/sink.py" Enter

tmux select-pane -t ecub-evp-tmux:0.2
tmux split-window -h -t ecub-evp-tmux
#3
tmux send-keys -t ecub-evp-tmux "docker exec -it ecub-evp-docker bash" Enter
tmux send-keys -t ecub-evp-tmux "cd Ergocub-Visual-Perception" Enter
tmux send-keys -t ecub-evp-tmux "python scripts/grasping_pipeline.py" Enter


tmux a -t ecub-evp-tmux
