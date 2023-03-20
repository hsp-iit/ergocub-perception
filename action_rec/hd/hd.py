import copy
from action_rec.hpe.utils.misc import postprocess_yolo_output
import numpy as np
from utils.human_runner import Runner
from tqdm import tqdm
import cv2
from action_rec.hpe.utils.matplotlib_visualizer import MPLPosePrinter


class HumanDetector:
    def __init__(self, yolo_thresh=None, nms_thresh=None, yolo_engine_path=None):

        self.yolo_thresh = yolo_thresh
        self.nms_thresh = nms_thresh
        self.yolo = Runner(yolo_engine_path)  # model_config.yolo_engine_path

    def estimate(self, rgb):

        # Preprocess for yolo
        square_img = cv2.resize(rgb, (256, 256), fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
        yolo_in = copy.deepcopy(square_img)
        yolo_in = cv2.cvtColor(yolo_in, cv2.COLOR_BGR2RGB)
        yolo_in = np.transpose(yolo_in, (2, 0, 1)).astype(np.float32)
        yolo_in = np.expand_dims(yolo_in, axis=0)
        yolo_in = yolo_in / 255.0

        # Yolo
        outputs = self.yolo(yolo_in)
        boxes, confidences = outputs[0].reshape(1, 4032, 1, 4), outputs[1].reshape(1, 4032, 80)
        bboxes_batch = postprocess_yolo_output(boxes, confidences, self.yolo_thresh, self.nms_thresh)

        # Get only the bounding box with the human with highest probability
        box = bboxes_batch[0]  # Remove batch dimension
        humans = []
        for e in box:  # For each object in the image
            if e[5] == 0:  # If it is a human
                humans.append(e)
        if len(humans) > 0:
            # humans.sort(key=lambda x: x[4], reverse=True)  # Sort with decreasing probability
            humans.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)  # Sort with decreasing area  # TODO TEST
            human = humans[0]
        else:
            return {"bbox": None}

        # Preprocess for BackBone
        x1 = int(human[0] * rgb.shape[1]) if int(human[0] * rgb.shape[1]) > 0 else 0
        y1 = int(human[1] * rgb.shape[0]) if int(human[1] * rgb.shape[0]) > 0 else 0
        x2 = int(human[2] * rgb.shape[1]) if int(human[2] * rgb.shape[1]) > 0 else 0
        y2 = int(human[3] * rgb.shape[0]) if int(human[3] * rgb.shape[0]) > 0 else 0

        return {"rgb": rgb, "bbox": (x1, y1, x2, y2)}


if __name__ == "__main__":
    from configs.action_rec_config import HPE
    from multiprocessing.managers import BaseManager
    import pycuda.autoinit
    from utils.concurrency.pypy_node import connect

    # Connect to realsense
    BaseManager.register('get_queue')
    manager = BaseManager(address=('172.27.192.1', 5000), authkey=b'abracadabra')
    connect(manager)
    send_out = manager.get_queue('windows_out')

    vis = MPLPosePrinter()

    h = HumanDetection(**HPE.Args.to_dict())

    for _ in tqdm(range(10000)):
        img = send_out.get()["rgb"]
        # cv2.imwrite('test1.jpg', img)
        # img = cv2.imread('test1.jpg')
        r = h.estimate(img)

        if r is not None:

            p = r["pose"]
            e = r["edges"]
            b = r["bbox"]

            if p is not None:
                print(np.sqrt(np.sum(np.square(np.array([0, 0, 0]) - np.array(p[0])))))
                p = p - p[0]
                vis.clear()
                vis.print_pose(p*5, e)
                vis.sleep(0.001)

            if b is not None:
                x1_, x2_, y1_, y2_ = b
                xm = int((x1_ + x2_) / 2)
                ym = int((y1_ + y2_) / 2)
                l = max(xm - x1_, ym - y1_)
                img = img[(ym - l if ym - l > 0 else 0):(ym + l), (xm - l if xm - l > 0 else 0):(xm + l)]
                img = cv2.resize(img, (224, 224))

        cv2.imshow("", img)
        cv2.waitKey(1)
