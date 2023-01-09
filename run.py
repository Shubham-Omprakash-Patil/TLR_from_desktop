import os
import cv2
import numpy as np

import _init_paths

from lib.opts import opts
from lib.detector import Detector
from lib.debugger import Debugger

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv', 'm4v']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def run(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)    
    detector = Detector(opt)

    b_record = opt.record   
    b_record_init = False    

    if opt.demo == 'webcam' or \
        opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        
        while True:
            _, img = cam.read()            

            if b_record and b_record_init == False:
                _fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                videorecorder = cv2.VideoWriter("out.mp4", _fourcc, 30, (img.shape[1],img.shape[0]))
                b_record_init = True

            ret = detector.run(img)
            
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            
            # debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3),theme=opt.debugger_theme)
            # debugger.add_img(ret['image'], img_id='tlr')
            
            # for j in range(1, opt.num_classes + 1):
            #     for bbox in ret['results'][j]:
        
            #         if bbox[4] > opt.vis_thresh:
            #             debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='tlr')
            # out = debugger.imgs['tlr']

            if b_record:
                videorecorder.write(out)
            
            if cv2.waitKey(1) == 27:
                return  # esc to quit
        if b_record:
            videorecorder.release()
            import sys
            sys.exit(0)

    else:
        if os.path.isdir(opt.demo):            
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        elif opt.demo.endswith('.txt'):
            image_names = []
            with open(opt.demo, 'r') as f:            
                image_path = opt.demo.split('samples')[0] + 'rgb_images'
                for l in f:
                    image_names.append(os.path.join(image_path, l.split('\n')[0]))
        else:
            image_names = [opt.demo]

        if b_record:
            _fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            videorecorder = cv2.VideoWriter("out.mp4", _fourcc, 30, (1920,1080))

        for image_name in image_names:
            print(image_name)

            ret = detector.run(image_name)

            if b_record:
                videorecorder.write(ret['image'])

            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
        
        if b_record:
            videorecorder.release()

if __name__ == '__main__':
    opt = opts().init()    
    run(opt)   