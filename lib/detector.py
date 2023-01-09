from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import matplotlib.pyplot as plt

from lib.models import get_resnet, get_resnet_bifpn
from lib.image import get_affine_transform
from lib.decode import ctdet_decode, ctdet_decode_tl
from lib.post_process import ctdet_post_process
from lib.debugger import Debugger
#from external.nms import soft_nms
from lib.manager import load_model
from lib.image import transform_preds

tl_label = {0: 'NA',
            1: 'gre',
            2: 'red',
            3: 'yel',
            4: 'gga',
            5: 'gra',
            6: 'gya',
            7: 'rga',
            8: 'rra',
            9: 'rya',
            10: 'oga',
            11: 'ora',            
            12: 'oya',
            13: 'off',            
            14: 'yga',
            15: 'yra',
            16: 'yyr'
            }
            
nb_label = {0: 'NA',
            1: '3',
            2: '4',
            3: '5'
            }

def nms_np(boxes, score, cls, nb, threshold):
    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_cls = []
    picked_nb = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(boxes[index])
        picked_score.append(score[index])
        picked_cls.append(cls[index])
        picked_nb.append(nb[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    picked_boxes = np.array(picked_boxes)
    picked_score = np.array(picked_score).reshape((-1, 1))
    picked_cls = np.array(picked_cls).reshape((-1, 1))
    picked_nb = np.array(picked_nb).reshape((-1, 1))

    return np.hstack((picked_boxes, picked_score, picked_cls, picked_nb))

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick]

class Detector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')        

        print('Creating model...')
        
        self.model = get_resnet(opt.layers, opt.num_classes, opt.num_features).to(opt.device)            
        # self.model = get_resnet_bifpn(opt.layers, opt.num_classes, opt.num_features).to(opt.device)            
        self.model = load_model(self.model, opt.load_model)        
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

        self.rois = []
        self.rois.append(np.array([0,    0,     1000,    500]))
        self.rois.append(np.array([920,  0,     1000,    500]))
        self.rois.append(np.array([30,   320,    480,    240]))
        self.rois.append(np.array([490,  320,    480,    240]))
        self.rois.append(np.array([950,  320,    480,    240]))
        self.rois.append(np.array([1410, 320,    480,    240]))
        self.rois.append(np.array([55,   515,    210,    105]))
        self.rois.append(np.array([255,  515,    210,    105]))
        self.rois.append(np.array([455,  515,    210,    105]))
        self.rois.append(np.array([655,  515,    210,    105]))
        self.rois.append(np.array([855,  515,    210,    105]))
        self.rois.append(np.array([1055,  515,    210,    105]))
        self.rois.append(np.array([1255,  515,    210,    105]))
        self.rois.append(np.array([1455,  515,    210,    105]))
        self.rois.append(np.array([1655,  515,    210,    105]))
        self.rois = np.array(self.rois)
        self.rois[:,2] = self.rois[:,0] + self.rois[:,2]
        self.rois[:,3] = self.rois[:,1] + self.rois[:,3]
        self.resize_size = (320,160)

    def pre_process(self, image, meta=None):
        width, height = self.resize_size
        inp_height = height + (self.opt.pad + 1)
        inp_width = width + (self.opt.pad + 1)
        # inp_height = (height | self.opt.pad) + 1
        # inp_width = (width | self.opt.pad) + 1
        half_pad = int((self.opt.pad + 1) / 2)
        c = np.array([width // 2, height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)
        
        imgs = []
        for roi in self.rois:
            img = np.copy(image[roi[1]:roi[3],roi[0]:roi[2],:])
            img = cv2.resize(img, self.resize_size)

            # trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])            
            # inp_image = cv2.warpAffine(img, trans_input, (inp_width, inp_height),flags=cv2.INTER_LINEAR)            

            inp_image = np.zeros((inp_height, inp_width, 3), dtype=np.uint8)
            inp_image[half_pad:half_pad+height, half_pad:half_pad+width, :] = img
            inp_image = (inp_image.astype(np.float32) / 255.)

            inp_image = inp_image.transpose(2, 0, 1)
            imgs.append(inp_image)

        images = np.array(imgs)
        images = torch.from_numpy(images)
        
        meta = {'c': c, 's': s, 
                'out_height': inp_height // self.opt.down_ratio, 
                'out_width': inp_width // self.opt.down_ratio}

        return images, meta

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            forward_time = time.time()
            
            hm = output['hm']
            wh = output['wh']
            reg = output['reg']
            cls = output['cls']
            nb = output['nb']

            b, c, h, w = hm.shape                
            cls = torch.argmax(cls[:,1:,:,:], dim=1)
            nb = torch.argmax(nb[:,1:,:,:], dim=1)
            cls = cls.reshape((b,1,h,w))
            nb = nb.reshape((b,1,h,w))
                
            cls_nb = torch.cat((cls, nb), dim=1)
            hm = hm.sigmoid_()

            dets = ctdet_decode_tl(hm, wh, cls_nb, reg=reg, cat_spec_wh=False, K=self.opt.K)            
            
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta):

        outputs = []  
        outputs_nms = []      
        for det, roi in zip(dets, self.rois):
            resize_ratio = self.resize_size[0] / float(roi[2] - roi[0])

            det = det.detach().cpu().numpy()
            det[:, :2]  = transform_preds(det[:, 0:2], meta['c'], meta['s'], (meta['out_width'], meta['out_height']))
            det[:, 2:4] = transform_preds(det[:, 2:4], meta['c'], meta['s'], (meta['out_width'], meta['out_height']))

            det[:, :4] /= resize_ratio
            det[:,[0,2]] += roi[0]
            det[:,[1,3]] += roi[1]
            det[:,5] += 1 # avoid bg class for tl state
            det[:,6] += 1 # avoid bg class for tl number of bulb
            
            outputs.append(det)
        
        outputs = np.vstack(outputs)
        
        
        if len(outputs) != 0:
            outputs_nms = nms_np(outputs[:,:4], outputs[:,4], outputs[:,5], outputs[:,6], 0.1)               

        return outputs_nms

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)
        if len(self.scales) > 1 or self.opt.nms:
            soft_nms(results[j], Nt=0.5, method=2)
        
        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
        
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, image_name, scale=1) :
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                            detection[i, k, 4], 
                                            img_id='out_pred_{:.1f}'.format(scale))
        if self.opt.debug == 4:
            debugger.save_all_imgs(self.opt.debug_dir, prefix=image_name)

    def show_results(self, debugger, image, results, prefix):        
        debugger.add_img(image, img_id='ctdet_{}'.format(prefix))

        for bbox in results:
            if bbox[4] > self.opt.vis_thresh:
                debugger.add_coco_bbox(bbox[:4], 2, bbox[4], img_id='ctdet_{}'.format(prefix))

        debugger.show_all_imgs(pause=self.pause)

    def run(self, image_or_path_or_tensor, prefix=None, meta=None):
        # init timer and debugger
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),theme=self.opt.debugger_theme)
        
        # - time stamp
        start_time = time.time()

        # input image        
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type (''): 
            image = cv2.imread(image_or_path_or_tensor)
            image_name = image_or_path_or_tensor.split('/')[-1].split('.')[0]        

        # - time stamp for load_time
        loaded_time = time.time()
        load_time += (loaded_time - start_time)
        
        # - time stamp for pre_process_time
        pre_process_start_time = time.time()

        # pre process
        images, meta = self.pre_process(image, meta)
        images = images.to(self.opt.device)
        torch.cuda.synchronize()

        # - time stamp for pre_process_time
        pre_process_time = time.time()
        pre_time += pre_process_time - pre_process_start_time
        
        # process
        detections = []
        output, dets, forward_time = self.process(images, return_time=True)        
        torch.cuda.synchronize()
        
        # - time stamp for net_time
        net_time += forward_time - pre_process_time
        decode_time = time.time()
        # dec_time += decode_time - forward_time
        
        # if self.opt.debug >= 2:
        #     self.debug(debugger, images, dets, output, image_name, 1)
        
        # post process
        dets = self.post_process(dets, meta)
        torch.cuda.synchronize()

        # - time stamp for post_time
        post_process_time = time.time()
        post_time += post_process_time - decode_time

        # detections.append(dets)
        # results = self.merge_outputs(detections)
        results = dets
        
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time
        
        if self.opt.debug >= 1:            
            for dt in dets:
                bb = dt[:4].astype(np.int32)
                score = dt[4]
                tl_cls = int(dt[5])
                tl_nb = int(dt[6])
                if tl_nb < 0 or tl_cls < 0:
                    continue
                if score >= self.opt.vis_thresh:
                    cv2.rectangle(image, (bb[0],bb[1]), (bb[2],bb[3]), (0,255,0), 2)
                    cv2.putText(image, tl_label[tl_cls], (bb[0],bb[1]-10), 2, 0.5, (0,255,0), 1)
                    cv2.putText(image, nb_label[tl_nb], (bb[0],bb[1]-20), 2, 0.5, (0,255,0), 1)
                    cv2.putText(image, '{:.2f}'.format(score), (bb[0],bb[1]-30), 2, 0.5, (0,255,0), 1)
                    cv2.putText(image, '{}x{}'.format(bb[2]-bb[0],bb[3]-bb[1]), (bb[0],bb[1]-40), 2, 0.5, (0,255,0), 1)

            cv2.namedWindow('tlr',0)        
            cv2.imshow('tlr', image)
            if cv2.waitKey(1) == 27:
                import sys
                sys.exit(0)        

        return {'results': results, 'image': image, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}