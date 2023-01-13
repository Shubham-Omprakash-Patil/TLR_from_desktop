import os
import cv2
import numpy as np
import pickle
import json
from tqdm import tqdm

from pycocotools import mask as maskUtils
import _init_paths

from lib.opts import opts
from lib import coco_tools
from lib.detector import Detector
from lib.data_generation import MagnaAdTlDataset

color_gt_map = {0: (255, 128, 0),
             1: (255, 0, 128),
             2: (128, 255, 0),
             3: (128, 0, 255),
             4: (0, 128, 255),
             5: (0, 255, 128)}
color_dt_map = {0: (255, 128, 150),
             1: (255, 150, 128),
             2: (128, 255, 150),
             3: (128, 150, 255),
             4: (150, 128, 255),
             5: (150, 255, 128)}
label_map = {1: 'tl'}
nbs_categories = {0: '3', 1: '4', 2: '5'}
cls_categories = {0: 'gre', 1: 'red', 2: 'yel',
                  3: 'gga', 4: 'gra', 5: 'gya',
                  6: 'rga', 7: 'rra', 8: 'rya',
                  9: 'oga', 10: 'ora', 11: 'oya',
                 12: 'off'}

def run(detector, dataset, eval_path):
    outputs = {}
    outputs['gt_image_ids'] = []
    outputs['dt_image_ids'] = []
    outputs['tl_categories']   = [{'id': 1, 'name': 'tl'}]
    outputs['cls_categories']  = [{'id': 1, 'name': 'gre'},
                                  {'id': 2, 'name': 'red'},
                                  {'id': 3, 'name': 'yel'},
                                  {'id': 4, 'name': 'gga'},
                                  {'id': 5, 'name': 'gra'},
                                  {'id': 6, 'name': 'gya'},
                                  {'id': 7, 'name': 'rga'},
                                  {'id': 8, 'name': 'rra'},
                                  {'id': 9, 'name': 'rya'},
                                  {'id': 10, 'name': 'oga'},
                                  {'id': 11, 'name': 'ora'},
                                  {'id': 12, 'name': 'oya'},
                                  {'id': 13, 'name': 'off'}]
                                #   {'id': 14, 'name': 'yga'},
                                #   {'id': 15, 'name': 'yra'},
                                #   {'id': 16, 'name': 'yyr'}]
    outputs['nbs_categories']  = [{'id': 1, 'name': '3'},
                                  {'id': 2, 'name': '4'},
                                  {'id': 3, 'name': '5'}]
            
    outputs['groundtruth_boxes'] = []
    outputs['groundtruth_tls']   = []
    outputs['groundtruth_cls']   = []
    outputs['groundtruth_nbs']   = []
    
    outputs['detections_boxes']  = []
    outputs['detections_tls']    = []
    outputs['detections_cls']    = []
    outputs['detections_nbs']    = []
    outputs['detections_scores'] = []

    outputs['confusion_matrix_cls'] = np.zeros((len(outputs['cls_categories']), len(outputs['cls_categories'])), dtype=np.int32)
    outputs['confusion_matrix_nbs'] = np.zeros((len(outputs['nbs_categories']), len(outputs['nbs_categories'])), dtype=np.int32)

    count = 0    
    labels = []
    for i in tqdm(range(len(dataset))):
        count += 1

        img, gt_dets = dataset.get_item_single(i)
        if len(gt_dets) == 0:        
            print('no gt: {}'.format(i))
            continue    

        image_id = "image_{}".format(count)        
        
        ret = detector.run(img)

        # collect data for od evaluation
        dets = ret['results']
                   
        if not len(dets) == 0:
            dt_scores = dets[:,4].reshape((-1,)).astype(np.float32)
            dt_tls = np.ones_like(dt_scores).astype(np.int32) * 1
            dt_cls = dets[:,5].reshape((-1,)).astype(np.int32) - 1 # to start with 0
            dt_nbs = dets[:,6].reshape((-1,)).astype(np.int32) - 1 # to start with 0
            dt_boxes = dets[:,:4].astype(np.float32)  

            outputs['dt_image_ids'].append(image_id)
            outputs['detections_boxes'].append(np.copy(dt_boxes))
            outputs['detections_tls'].append(dt_tls)
            outputs['detections_cls'].append(dt_cls)
            outputs['detections_scores'].append(dt_scores)
            outputs['detections_nbs'].append(dt_nbs)
        
        if not len(dets) == 0:
            gt_boxes = gt_dets[:,3:].astype(np.float32)
            gt_tls = gt_dets[:,0].reshape((-1,)).astype(np.int32)
            gt_cls = gt_dets[:,1].reshape((-1,)).astype(np.int32) - 1 # to start with 0
            gt_nbs = gt_dets[:,2].reshape((-1,)).astype(np.int32) - 1 # to start with 0
            outputs['gt_image_ids'].append(image_id)
            outputs['groundtruth_boxes'].append(np.copy(gt_boxes))
            outputs['groundtruth_tls'].append(gt_tls)
            outputs['groundtruth_cls'].append(gt_cls)
            outputs['groundtruth_nbs'].append(gt_nbs)

        # calculate ious
        if len(dets) == 0 or len(gt_dets) == 0:
            continue
        
        dt_boxes[:,3] = dt_boxes[:,3] - dt_boxes[:,1]
        dt_boxes[:,2] = dt_boxes[:,2] - dt_boxes[:,0]

        gt_boxes = gt_dets[:,3:]
        gt_boxes[:,3] = gt_boxes[:,3] - gt_boxes[:,1]
        gt_boxes[:,2] = gt_boxes[:,2] - gt_boxes[:,0]

        iscrowd = [0 for i in range(len(gt_dets))]
        ious = maskUtils.iou(dt_boxes,gt_boxes,iscrowd)

        # fill out confusion matrix
        for d in range(len(dt_boxes)):
            iou = ious[d, :]
            g = np.argmax(iou)
            iou_max = iou[g]

            if iou_max >= 0.5:
                # print('[dt] cls: {} - nb: {}'.format(dt_cls[d], dt_nbs[d]))
                # print('[gt] cls: {} - nb: {}'.format(gt_cls[d], gt_nbs[d]))
                try:
                    outputs['confusion_matrix_cls'][dt_cls[d],gt_cls[g]] = outputs['confusion_matrix_cls'][dt_cls[d],gt_cls[g]] + 1
                    if dt_cls[d] != gt_cls[g]:
                        gt_name = cls_categories[gt_cls[g]]
                        dt_name = cls_categories[dt_cls[d]]
                        dt_box = dt_boxes[d].astype(np.int32)
                        dt_img = img[dt_box[1]:dt_box[1]+dt_box[3], dt_box[0]:dt_box[0]+dt_box[2], :]
                        img_name = os.path.basename(dataset.images[i])
                        img_name = img_name.replace('.jpg', '_{}_{}.jpg'.format(dt_name, d))                        
                        img_path = os.path.join(eval_path, 'cls', gt_name, img_name)
                        cv2.imwrite(img_path, dt_img)
                except:
                    print('Error: ', dt_cls[d],gt_cls[g])
                try:
                    outputs['confusion_matrix_nbs'][dt_nbs[d],gt_nbs[g]] = outputs['confusion_matrix_nbs'][dt_nbs[d],gt_nbs[g]] + 1
                    if dt_nbs[d] != gt_nbs[g]:
                        gt_name = nbs_categories[gt_nbs[g]]
                        dt_name = nbs_categories[dt_nbs[d]]
                        dt_box = dt_boxes[d].astype(np.int32)
                        dt_img = img[dt_box[1]:dt_box[1]+dt_box[3], dt_box[0]:dt_box[0]+dt_box[2], :]
                        img_name = os.path.basename(dataset.images[i])
                        img_name = img_name.replace('.jpg', '_{}_{}.jpg'.format(dt_name, d))                        
                        img_path = os.path.join(eval_path, 'nbs', gt_name, img_name)                        
                        cv2.imwrite(img_path, dt_img)
                except:
                    print('Error: ', dt_nbs[d],gt_nbs[g])
        # if count == 100:
        #     break

    return outputs
            
def eval(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str    
    detector = Detector(opt)
    dataset = MagnaAdTlDataset(opt.db_root, opt.test_image_path, aug=False, debug=False)

    # save
    model_name = opt.load_model.split('/')[-1].split('.')[0]
    saved_name = os.path.join(opt.load_model.split('model')[0], 'eval', model_name)
    print(' saved in {}'.format(saved_name))
    if os.path.exists(saved_name) == False:
        os.makedirs(saved_name)

    for k in nbs_categories.keys():
        p = os.path.join(saved_name, 'nbs', nbs_categories[k])
        if os.path.exists(p) == False:
            os.makedirs(p)

    for k in cls_categories.keys():
        p = os.path.join(saved_name, 'cls', cls_categories[k])
        if os.path.exists(p) == False:
            os.makedirs(p)

    outputs = run(detector, dataset, saved_name)

    cocoGt = coco_tools.ExportGroundtruthToCOCO(
                        outputs['gt_image_ids'],
                        outputs['groundtruth_boxes'],
                        outputs['groundtruth_tls'],
                        outputs['tl_categories'],
                        output_path=None)
                        
    cocoDt = coco_tools.ExportDetectionsToCOCO(
                        outputs['dt_image_ids'],
                        outputs['detections_boxes'],
                        outputs['detections_scores'],
                        outputs['detections_tls'],
                        outputs['tl_categories'],
                        output_path=None)

    coco_wrapped_groundtruth = coco_tools.COCOWrapper(cocoGt)
    coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(cocoDt)
    box_evaluator = coco_tools.COCOEvalWrapper(coco_wrapped_groundtruth, coco_wrapped_detections, agnostic_mode=False)
                                           
    box_evaluator.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 7*7*3], [7*7*3, 10*10*3], [10*10*3, 1e5 ** 2]] 
    box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(include_metrics_per_category=False, all_metrics_per_category=False)
    box_metrics.update(box_per_category_ap)
    box_metrics = {'DetectionBoxes_'+ key: value
                   for key, value in iter(box_metrics.items())}
    
    with open('{}/gts.p'.format(saved_name), 'wb') as fid:       
        pickle.dump(box_evaluator._gts, fid)
          
    with open('{}/dts.p'.format(saved_name), 'wb') as fid:       
        pickle.dump(box_evaluator._dts, fid)
        
    with open('{}/eval_imgs.p'.format(saved_name), 'wb') as fid:       
        pickle.dump(box_evaluator.evalImgs, fid)
    
    with open('{}/eval.txt'.format(saved_name),'w') as f:
        f.write("object detection\n")
        for k in box_metrics.keys():
            f.write(" {}: {:.4f}\n".format(k, box_metrics[k]))

        # print out confusion matrix for number of bulb
        f.write('\nconfusion matrix for number of bulb\n')
        confusion_matrix_nbs = outputs['confusion_matrix_nbs']
        confusion_matrix_nbs_p = np.copy(confusion_matrix_nbs).astype(np.float32)

        for j in range(confusion_matrix_nbs_p.shape[1]):
            confusion_matrix_nbs_p[:,j] /= confusion_matrix_nbs_p[:,j].sum()

        _str = '   '
        for k in nbs_categories.keys():
            _str = _str + '{}       '.format(nbs_categories[k])
        f.write(_str)
        f.write('\n')

        for i in range(confusion_matrix_nbs.shape[0]):
            _str = ''
            for v in confusion_matrix_nbs[i]:
                if v != 0:
                    _str = _str + '{:05d}   '.format(v)
                else:
                    _str = _str + '-'.ljust(8)                
            f.write('{}: {}'.format(nbs_categories[i], _str))
            f.write('\n')

        f.write('\n')
        _str = '   '
        for k in nbs_categories.keys():
            _str = _str + '{}       '.format(nbs_categories[k])
        f.write(_str)
        f.write('\n')

        for i in range(confusion_matrix_nbs_p.shape[0]):
            _str = ''
            for v in confusion_matrix_nbs_p[i]:
                if v != 0:
                    _str = _str + '{:02.4f}  '.format(v)
                else:
                    _str = _str + '-'.ljust(8)
                
            f.write('{}: {}'.format(nbs_categories[i], _str))             
            f.write('\n')

        # print out confusion matrix for bulb state
        f.write('\nconfusion matrix for bulb state\n')
        

        confusion_matrix_cls = outputs['confusion_matrix_cls']
        confusion_matrix_cls_p = np.copy(confusion_matrix_cls).astype(np.float32)

        for j in range(confusion_matrix_cls_p.shape[1]):
            if confusion_matrix_cls_p[:,j].sum() != 0:
                confusion_matrix_cls_p[:,j] /= confusion_matrix_cls_p[:,j].sum()

        _str = '     '
        for k in cls_categories.keys():
            _str = _str + '{}     '.format(cls_categories[k])
        f.write(_str)
        f.write('\n')

        for i in range(confusion_matrix_cls.shape[0]):
            _str = ''
            for v in confusion_matrix_cls[i]:
                if v != 0:
                    _str = _str + '{:05d}   '.format(v)
                else:
                    _str = _str + '-'.ljust(8)
            f.write('{}: {}'.format(cls_categories[i], _str))
            f.write('\n')

        f.write('\n')
        _str = '     '
        for k in cls_categories.keys():
            _str = _str + '{}     '.format(cls_categories[k])
        f.write(_str)
        f.write('\n')

        for i in range(confusion_matrix_cls.shape[0]):
            _str = ''
            for v in confusion_matrix_cls_p[i]:
                if v != 0:
                    _str = _str + '{:.4f}  '.format(v)
                else:
                    _str = _str + '-'.ljust(8)                
            f.write('{}: {}'.format(cls_categories[i], _str))
            f.write('\n')


if __name__ == '__main__':    
    opt = opts().init()
    eval(opt)
