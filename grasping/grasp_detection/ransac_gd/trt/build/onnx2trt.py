from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner, EngineFromBytes, CreateConfig
from polygraphy.common import TensorMetadata
from polygraphy.comparator import DataLoader, Comparator
import numpy as np

if __name__ == '__main__':

    onnx_file = 'grasping/grasp_detection/ransac_gd/trt/assets/ransac200_10000.onnx'
    engine_file = 'grasping/grasp_detection/ransac_gd/trt/assets/ransac200_10000_docker.engine'

    # data_loader = DataLoader(iterations=100,
    #                          val_range=(-0.5, 0.5),
    #                          input_metadata=TensorMetadata.from_feed_dict({'input': np.zeros([1, 3, 192, 256], dtype=np.float32)}))

    # config = CreateConfig(fp16=True, tf32=True)
    config = CreateConfig(fp16=True, tf32=True, max_workspace_size=10000 << 40)

    build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config)
    engine = build_engine()

    with open(engine_file, 'wb') as f:
        f.write(engine.serialize())
