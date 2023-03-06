yarp connect /depthCamera/rgbImage:o /depthCamera/rgbImage:r mjpeg
yarp connect /depthCamera/depthImage:o /depthCamera/depthImage:r  tcp+send.portmonitor+file.depthimage_compression_zlib+recv.portmonitor+file.depthimage_compression_zlib+type.dll


