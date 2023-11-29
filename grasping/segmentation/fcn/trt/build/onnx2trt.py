from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, CreateConfig
import tensorrt as trt

def main():
    onnx_file = 'grasping/segmentation/fcn/trt/assets/segmentation.onnx'
    engine_file = 'grasping/segmentation/fcn/trt/assets/seg_fp16_docker.engine'

    config = CreateConfig(profiling_verbosity=trt.ProfilingVerbosity.DETAILED, fp16=True, tf32=True)  # , max_workspace_size=10000 << 40

    build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config)
    engine = build_engine()

    with open(engine_file, 'wb') as f:
        f.write(engine.serialize())


if __name__ == '__main__':
    main()