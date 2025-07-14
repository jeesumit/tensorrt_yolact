import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import threading
import queue
import cv2
import time
# Constants
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)
INPUT_SIZE = 550

COLORS = ((244, 67, 54), (233, 30, 99), (156, 39, 176), (103, 58, 183),
          (63, 81, 181), (33, 150, 243), (3, 169, 244), (0, 188, 212),
          (0, 150, 136), (76, 175, 80), (139, 195, 74), (205, 220, 57),
          (255, 235, 59), (255, 193, 7), (255, 152, 0), (255, 87, 34),
          (121, 85, 72), (158, 158, 158), (96, 125, 139))

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

# Flags
stop_event = threading.Event()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def preprocess(img):
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    img = (img - MEANS) / STD
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0).astype(np.float32)
    return img

def generate_priors():
    feature_map_sizes = [[69, 69], [35, 35], [18, 18], [9, 9], [5, 5]]
    aspect_ratios = [[1, 0.5, 2]] * len(feature_map_sizes)
    scales = [24, 48, 96, 192, 384]
    priors = []

    for idx, fsize in enumerate(feature_map_sizes):
        scale = scales[idx]
        for y in range(fsize[0]):
            for x in range(fsize[1]):
                cx = (x + 0.5) / fsize[1]
                cy = (y + 0.5) / fsize[0]
                for ratio in aspect_ratios[idx]:
                    r = np.sqrt(ratio)
                    w = scale / INPUT_SIZE * r
                    h = scale / INPUT_SIZE / r
                    priors.append([cx, cy, w, h])

    return np.array(priors, dtype=np.float32)

def decode(loc, priors, variances=[0.1, 0.2]):
    boxes = np.zeros_like(loc)
    boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    boxes[:, 2:] = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def nms(boxes, scores, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )
    return np.array(indices).flatten() if len(indices) > 0 else np.array([], dtype=int)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess(output, original_shape):
    proto, loc, mask, _, conf = output
    loc = np.squeeze(loc, axis=0)
    conf = np.squeeze(conf, axis=0)
    mask = np.squeeze(mask, axis=0)
    proto = np.squeeze(proto, axis=0)

    scores = np.max(conf[:, 1:], axis=1)
    classes = np.argmax(conf[:, 1:], axis=1)
    keep = scores > 0.5

    if not np.any(keep):
        return [], [], [], []

    scores = scores[keep]
    classes = classes[keep]
    mask = mask[keep]
    loc = loc[keep]
    priors = generate_priors()[keep]
    boxes = decode(loc, priors)
    keep_nms = nms(boxes, scores)

    boxes = boxes[keep_nms]
    scores = scores[keep_nms]
    classes = classes[keep_nms]
    mask = mask[keep_nms]

    # masks = sigmoid(proto @ mask.T).transpose(2, 0, 1)

    # resized_masks = []
    # for m in masks:
    #     resized = cv2.resize(m, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    #     resized_masks.append(resized > 0.5)

    masks = sigmoid(proto @ mask.T).transpose(2, 0, 1)
    masks = np.array([
    cv2.resize(m, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
    for m in masks
], dtype=bool)

    #masks = np.array(resized_masks, dtype=bool)
    return masks, classes, scores, boxes

class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = [None] * self.engine.num_io_tensors
        self.device_buffers = {}
        self.host_outputs = {}
        self.input_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)

        self.allocate_buffers()

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape)

            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.device_buffers[name] = device_mem
            self.bindings[i] = int(device_mem)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.host_outputs[name] = np.empty(size, dtype=dtype)
                

    def infer(self, image):
        input_data = preprocess(image)
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        original_shape = image.shape[:2]

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, self.input_shape)
                cuda.memcpy_htod_async(self.device_buffers[name], input_data, self.stream)
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        self.context.execute_async_v3(self.stream.handle)

        for name, host_out in self.host_outputs.items():
            cuda.memcpy_dtoh_async(host_out, self.device_buffers[name], self.stream)

        self.stream.synchronize()

        outputs = []
        for name in sorted(self.host_outputs.keys()):
            shape = self.context.get_tensor_shape(name)
            outputs.append(self.host_outputs[name].reshape(shape))

        return outputs, original_shape

# Main script
if __name__ == "__main__":
    engine_path = "yolact.engine"
    trt_infer = TRTInference(engine_path)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Could not open camera")
        exit()

    while True:
        ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        outputs, orig_shape = trt_infer.infer(frame)
        masks, classes, scores, boxes = postprocess(outputs, orig_shape)

        inference_time = (time.time() - start_time) * 1000
        fps = 1.0 / ((time.time() - start_time) + 1e-6)

        for i, mask in enumerate(masks):
            cls = int(classes[i])
            color = COLORS[cls % len(COLORS)]
            frame[mask] = frame[mask] * 0.5 + np.array(color, dtype=np.float32) * 0.5

            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = map(int, [x1 * orig_shape[1], y1 * orig_shape[0], x2 * orig_shape[1], y2 * orig_shape[0]])
            label = f"{class_names[cls]} {scores[i]:.2f}"

            cv2.putText(frame, label, (x1 + 5, max(y1 - 5, 0) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Inference: {inference_time:.2f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLACT TensorRT - Webcam", frame.astype(np.uint8))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
