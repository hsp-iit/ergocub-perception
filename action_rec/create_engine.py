import numpy as np
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, SaveEngine, TrtRunner

BATCH_SIZE = 1


def create_engine(in_path, out_path, inputs):
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(in_path),
        config=CreateConfig(fp16=True)  # ,profiles=profiles  # max_workspace_size=10000 << 20
    )
    build_engine = SaveEngine(build_engine, path=out_path)
    with TrtRunner(build_engine) as runner:
        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict=inputs)

        print("Inference succeeded!")
