from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, CreateConfig

import tensorrt as trt

if __name__ == '__main__':

    onnx_file = 'grasping/shape_completion/confidence_pcr/trt/assets/pcr.onnx'
    engine_file = 'grasping/shape_completion/confidence_pcr/trt/assets/pcr_docker.engine'

    # config = CreateConfig(fp16=True, tf32=True)
    config = CreateConfig(profiling_verbosity=trt.ProfilingVerbosity.DETAILED)  # max_workspace_size=10000 << 40, 

    build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config)
    engine = build_engine()

    with open(engine_file, 'wb') as f:
        f.write(engine.serialize())