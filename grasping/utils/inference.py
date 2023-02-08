from pathlib import Path

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from loguru import logger


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTRunner:
    def __init__(self, engine_path):
        logger.info(f'Loading {Path(engine_path).stem} engine...')

        G_LOGGER = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(G_LOGGER, '')
        runtime = trt.Runtime(G_LOGGER)

        with open(engine_path, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)

        # prepare buffer
        inputs = []
        outputs = []
        bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # 256 x 256 x 3 ( x 1 )
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)  # (256 x 256 x 3 ) x (32 / 4)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        # store
        self.stream = cuda.Stream()
        self.context = None
        self.context = engine.create_execution_context()
        self.engine = engine

        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings

        self.warmup()
        logger.success(f'{Path(engine_path).stem} engine loaded')

    def warmup(self):
        args = [np.random.rand(*inp.host.shape) for inp in self.inputs]
        self(*args)

    def __call__(self, *args):

        for i, x in enumerate(args):
            if isinstance(x, np.ndarray):
                x = x.astype(trt.nptype(trt.float32)).ravel()
            np.copyto(self.inputs[i].host, x)

        # pc = pc.astype(trt.nptype(trt.float32)).ravel()
        # np.copyto(self.inputs[0].host, pc)

        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()

        res = [out.host for out in self.outputs]

        return res


class Refiner:
    def __init__(self, engine):
        G_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(G_LOGGER, '')
        runtime = trt.Runtime(G_LOGGER)

        with open(engine, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)

        # prepare buffer
        inputs = []
        outputs = []
        bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # 256 x 256 x 3 ( x 1 )
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)  # (256 x 256 x 3 ) x (32 / 4)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        # store
        self.stream = cuda.Stream()
        self.context = engine.create_execution_context()
        self.engine = engine

        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings

    def __call__(self, points, sdf):

        points = points.astype(trt.nptype(trt.float32)).ravel()
        np.copyto(self.inputs[0].host, points)

        i = 1
        for layer in sdf:
            for weight in layer:
                weight = weight.detach().cpu().numpy().astype(trt.nptype(trt.float32)).ravel()
                np.copyto(self.inputs[i].host, weight)
                i += 1

        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()

        res = [out.host for out in self.outputs]

        return res[0], res[1]


if __name__ == '__main__':
    from configs import server_config
    from model.PCRNetwork import PCRNetwork
    import torch

    refiner = Refiner('refiner.engine')

    model = PCRNetwork.load_from_checkpoint('./checkpoint/final', config=server_config.ModelConfig)
    model.cuda()
    model.eval()

    fast_weights, _ = model.backbone(torch.randn((1, 2024, 3)).cuda())
    x = torch.tensor(torch.randn(1, 10000, 3).cpu().detach().numpy() * 1, requires_grad=True, device='cuda')

    out = refiner(x.cpu().numpy(), fast_weights)
