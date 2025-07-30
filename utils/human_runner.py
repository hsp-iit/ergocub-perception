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


# class Runner:
#     def __init__(self, engine_path):
#         logger.info(f'Loading {Path(engine_path).stem} engine...')

#         G_LOGGER = trt.Logger(trt.Logger.ERROR)  # TODO PUT ERROR
#         trt.init_libnvinfer_plugins(G_LOGGER, '')
#         runtime = trt.Runtime(G_LOGGER)

#         with open(engine_path, 'rb') as f:
#             buf = f.read()
#             engine = runtime.deserialize_cuda_engine(buf)

#         # prepare buffer
#         inputs = []
#         outputs = []
#         bindings = []
#         for binding in engine:
#             # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # 256 x 256 x 3 ( x 1 )
#             # dtype = trt.nptype(engine.get_binding_dtype(binding))
#             # size = trt.volume(engine.get_tensor_shape(binding)) * engine.max_batch_size  # 256 x 256 x 3 ( x 1 )
#             shape = engine.get_tensor_shape(binding)
#             size = trt.volume(shape)
#             dtype = trt.nptype(engine.get_tensor_dtype(binding))
#             host_mem = cuda.pagelocked_empty(size, dtype)
#             device_mem = cuda.mem_alloc(host_mem.nbytes)  # (256 x 256 x 3 ) x (32 / 4)
#             bindings.append(int(device_mem))
#             # if engine.binding_is_input(binding):
#             index = engine.get_binding_index(binding)
#             if engine.binding_is_input(index):
#                 inputs.append(HostDeviceMem(host_mem, device_mem))
#             else:
#                 outputs.append(HostDeviceMem(host_mem, device_mem))

#         # store
#         self.stream = cuda.Stream()
#         self.context = None
#         self.context = engine.create_execution_context()
#         self.engine = engine

#         self.inputs = inputs
#         self.outputs = outputs
#         self.bindings = bindings

#         self.warmup()
#         logger.success(f'{Path(engine_path).stem} engine loaded')

#     def warmup(self):
#         args = [np.random.rand(*inp.host.shape).astype(inp.host.dtype) for inp in self.inputs]
#         self(*args)

#     def __call__(self, *args):

#         for i, x in enumerate(args):
#             x = x.ravel()
#             np.copyto(self.inputs[i].host, x)

#         [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
#         self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
#         [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
#         self.stream.synchronize()

#         res = [out.host for out in self.outputs]

#         return res
class Runner:
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

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append((name, HostDeviceMem(host_mem, device_mem)))
            else:
                outputs.append((name, HostDeviceMem(host_mem, device_mem)))

        self.stream = cuda.Stream()
        self.context = engine.create_execution_context()
        self.engine = engine
        self.inputs = inputs
        self.outputs = outputs
        self.warmup()
        logger.success(f'{Path(engine_path).stem} engine loaded')

    def warmup(self):
        args = [np.random.rand(*inp[1].host.shape).astype(inp[1].host.dtype) for inp in self.inputs]
        self(*args)

    def __call__(self, *args):
        # Copia input
        for i, x in enumerate(args):
            x = x.astype(self.inputs[i][1].host.dtype).ravel()
            np.copyto(self.inputs[i][1].host, x)
            cuda.memcpy_htod_async(self.inputs[i][1].device, self.inputs[i][1].host, self.stream)
            self.context.set_tensor_address(self.inputs[i][0], int(self.inputs[i][1].device))

        # Imposta output
        for name, out in self.outputs:
            self.context.set_tensor_address(name, int(out.device))

        # Inference
        self.context.execute_async_v3(self.stream.handle)

        # Copia output
        for _, out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        return [out.host for _, out in self.outputs]
