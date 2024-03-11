# ErgoCub-Visual-Perception

## Installation
### Environment
The best way to manage the environment is to use Docker.
Build the image with:
```
cd docker
docker build -t ar0s/ergocub-perception-image .
```
### TensorRT engine creation
This application contains several machine learning modules that are accelerated with TensorRT.
Since the TensorRT engines are hardware-specific, it is necessary to create them everytime a new machine is used.
Each TensorRT engine is built from an ONNX file.
You can download the ONNX files from [here](https://drive.google.com/file/d/1rThILyh1tdpEpAkcqzGSaG3I1_BeEvMV/view?usp=sharing), extract the "onnxs" folder to the root directory of the project.
To build the engines, just launch:
```
./build_engines.sh
```
and wait untill each engine is built (last line should be "[I] Finished engine building in 2.615 seconds" or "Inference succedded!").
Run
```
./stop.sh
```
to remove the container and the tmux session.
You can then erase the "onnxs" folder in the root directory.

### Other files
Other files are needed, you can find them [here](https://drive.google.com/file/d/1iJc_2rBkVeslFHP0jOLU87y6tR2fGLfZ/view?usp=sharing).

### Test installation
To test the installation, launch:
```
./start.sh -ys
```

### Run with webcam
In order to test the pipeline with the webcam, follow [.](https://askubuntu.com/questions/1405903/capturing-webcam-video-with-opencv-in-wsl2).
