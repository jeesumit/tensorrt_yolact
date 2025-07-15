from data import get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize

from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse

import os
from collections import defaultdict


import matplotlib.pyplot as plt
import cv2

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    
crp_mask = True
thresh = 0.6
top_kval = 5

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} 
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    for i in dets_out:
        print(i.shape)
    
    keys=["mask","box","class","score","proto"]
    values=[dets_out[2],dets_out[1],dets_out[3],dets_out[4],dets_out[0]]
    
    
    det=dict(zip(keys,values))

    for key, value in det.items():
        print(key, value.shape)

    dets={"detection":det}
    out=[]
    out.append(dets)
    save = cfg.rescore_bbox
    cfg.rescore_bbox = True
    t = postprocess(out, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = 0.6)
    cfg.rescore_bbox = save
    
    idx = t[1].argsort(0, descending=True)[:5]
        
    if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
        masks = t[3][idx]
    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_kval, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < 0.6:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1

    text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

    img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()


    text_pt = (4, text_h + 2)
    text_color = [255, 255, 255]

    cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        return img_numpy
    

    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        color = get_color(j)
        score = scores[j]


        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

        _class = cfg.dataset.class_names[classes[j]]
        text_str = '%s: %.2f' % (_class, score) 

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

        text_pt = (x1, y1 - 3)
        text_color = [255, 255, 255]

        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
    
    return img_numpy

def evalimage(engine, path:str, save_path:str=None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    # preds = net(batch)

    preds= infer(engine, batch)[0]
    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
    
    img_numpy = img_numpy[:, :, (2, 1, 0)]

    
    plt.imshow(img_numpy)
    plt.title(path)
    plt.show()
   
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)
def preprocess(img):
    img = cv2.resize(img, (550,550)).astype(np.float32)
    img = (img - MEANS) / STD
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0).astype(np.float32)
    return img


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, image):
    #input_data = preprocess(image)
    
    input_data=image.cpu().numpy()
    input_data = np.ascontiguousarray(input_data, dtype=np.float32) 
    original_shape = image.shape[:2]

    with engine.create_execution_context() as context:
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_shape = (1, 3, 550,550)

        # Allocate memory for inputs and outputs
        bindings = [None] * len(tensor_names)
        device_buffers = {}
        host_outputs = {}

        stream = cuda.Stream()

        for i, name in enumerate(tensor_names):
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = context.get_tensor_shape(name)
            size = trt.volume(shape)

            # Allocate device memory
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            device_buffers[name] = device_mem
            bindings[i] = int(device_mem)

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                context.set_input_shape(name, input_shape)
                cuda.memcpy_htod_async(device_mem, input_data, stream)
            else:
                host_outputs[name] = np.empty(size, dtype=dtype)

        # Set tensor addresses
        for name, device_mem in device_buffers.items():
            context.set_tensor_address(name, int(device_mem))

        context.execute_async_v3(stream.handle)


        for name in host_outputs:
            cuda.memcpy_dtoh_async(host_outputs[name], device_buffers[name], stream)

        stream.synchronize()
        
        outputs = []
        for name in sorted(host_outputs.keys()):  
            shape = context.get_tensor_shape(name)
            outputs.append(host_outputs[name].reshape(shape))

        return outputs, original_shape

if __name__ == '__main__':
    engine_path = "yolact.engine"
    image_path = "1.jpg"

    engine = load_engine(engine_path)
    evalimage(engine,image_path )

    
  
