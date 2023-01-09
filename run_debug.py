import pickle
import json
import numpy as np
# from lib import magna_db, drawers
# from lib import tlr_wrapper
import cv2
import os 
import argparse
import matplotlib.pyplot as plt

tl_idx_map = {None: 'all',
              1: 'tl',
              }

def export_DtGtMatches(root, iou_thresh):  
    gts = pickle.load(open(os.path.join(root,'gts.p'),'rb'))
    dts = pickle.load(open(os.path.join(root,'dts.p'),'rb'))
    eval_imgs = pickle.load(open(os.path.join(root,'eval_imgs.p'),'rb'))
    
    iou_threshold_idx = {'0.5':0,'0.55':1,
                         '0.6':2,'0.65':3,
                         '0.7':4,'0.75':5,
                         '0.8':6,'0.85':7,
                         '0.9':8,'0.95':9}

    DtGtMatches = {}
    count = 0
    for k in eval_imgs:
        if k is not None:
            if k['aRng'][0] == 0 and k['aRng'][1] == 1e5 ** 2:            
                key_idx = (k['image_id'], k['category_id'])
                
                if not k['image_id'] in DtGtMatches:
                    DtGtMatches[k['image_id']] = []

                # detection
                dtMatches = k['dtMatches'][iou_threshold_idx[iou_thresh]]
                for i in range(len(k['dtIds'])):
                    dt_id = k['dtIds'][i]
                    if dtMatches[i] != 0.:
                        gt_id =  dtMatches[i]
                        
                        for d in dts[key_idx]:
                            if d['id'] == dt_id:
                                break
                        for g in gts[key_idx]:
                            if g['id'] == gt_id:
                                break
                            
                        #print('- dt: {}, gt: {}'.format(d['bbox'],g['bbox']))
                        
                        dt_gt = np.hstack((k['category_id'], np.array(d['bbox']), np.array(g['bbox']))).astype(np.int)                    
                        DtGtMatches[k['image_id']].append(dt_gt)                    
                        
                    else:
                        for d in dts[key_idx]:
                            if d['id'] == dt_id:
                                dt_gt = np.hstack((k['category_id'], np.array(d['bbox']), np.zeros((4,)))).astype(np.int)
                                DtGtMatches[k['image_id']].append(dt_gt)                    
                
                # ground truth
                gtMatches = k['gtMatches'][iou_threshold_idx[iou_thresh]]                            
                for i in range(len(k['gtIds'])):
                    gt_id = k['gtIds'][i]
                    if gtMatches[i] == 0.:
                        for g in gts[key_idx]:
                            if g['id'] == gt_id:
                                dt_gt = np.hstack((k['category_id'], np.zeros((4,)), np.array(g['bbox']))).astype(np.int)
                                DtGtMatches[k['image_id']].append(dt_gt)
                                
            #count += 1
            #if count == 11:
            #    break

    for k in DtGtMatches.keys():
        dt_gt = DtGtMatches[k]
        dt_gt = np.array(dt_gt)   
        dt_gt[:,[1,2,3,4,5,6,7,8]] = dt_gt[:,[2,1,4,3,6,5,8,7]] # y,x,h,w > x,y,w,h
        DtGtMatches[k] = dt_gt

    with open(os.path.join(root, 'DtGtMatches_iou_{}.p'.format(iou_thresh)), 'wb') as fid: 
        pickle.dump(DtGtMatches,fid)
            
def visualize_DtGtMatches(root):
    image_ids_path = 'eval/test_image_ids.p'
    matched_file_path = os.path.join(root, 'DtGtMatches_iou_0.5.p')
    dataset_root = '/home/chulhoon/Desktop/mag-ad-db'

    DtGtMatches = pickle.load(open(matched_file_path,'rb'))
    image_list = pickle.load(open(image_ids_path,'rb'))

    for image_id in DtGtMatches.keys():
        img_path = os.path.join(dataset_root,image_list[image_id])
        print(image_list[image_id])
        img = cv2.imread(img_path)

        for dt_gt in DtGtMatches[image_id]:
            cls = dt_gt[0]
            dt = dt_gt[1:5]
            gt = dt_gt[5:]
            
            cv2.rectangle(img, (gt[0],gt[1]), (gt[0]+gt[2],gt[1]+gt[3]), (0,128,255), 1)
            cv2.rectangle(img, (dt[0],dt[1]), (dt[0]+dt[2],dt[1]+dt[3]), (0,255,128), 1)           

        cv2.namedWindow('img',0)   
        cv2.imshow('img',img)
        if 27 == cv2.waitKey():
            break

def visualize_fp_fn(root, debug_fp=True, debug_fn=True, export=False, min_w=0, max_w=100):
    image_ids_path = 'eval/test_image_ids.p'
    matched_file_path = os.path.join(root, 'DtGtMatches_iou_0.5.p')    
    dataset_root = '/home/chulhoon/Desktop/mag-ad-db'

    DtGtMatches = pickle.load(open(matched_file_path,'rb'))
    image_list = pickle.load(open(image_ids_path,'rb'))
        
    export_list = []
    for image_id in DtGtMatches.keys():
        dts_gts = DtGtMatches[image_id]
        gts = dts_gts[:,5:]
        
        fp = dts_gts[np.sum(gts,axis=1) == 0]
        fp = fp[fp[:,3]>=min_w]
        fp = fp[fp[:,3]<=max_w]
        gts = dts_gts[np.sum(gts,axis=1) != 0]
        gts = gts[gts[:,7]>min_w]
        gts = gts[gts[:,7]<max_w]
        fn = dts_gts[np.sum(dts_gts[:,1:5],axis=1) == 0]
        fn = fn[fn[:,7]>=min_w]
        fn = fn[fn[:,7]<=max_w]                
        
        if (len(fp) != 0 and debug_fp) or (len(fn) != 0 and debug_fn):        
            img_path = os.path.join(dataset_root,image_list[image_id])
            
            if export:
               export_list.append(img_path)
               continue
            
            print(image_list[image_id])
            img = cv2.imread(img_path)

            for dt_gt in DtGtMatches[image_id]:
                cls = dt_gt[0]
                dt = dt_gt[1:5]
                gt = dt_gt[5:]
                
                img = cv2.putText(img, '{}'.format(tl_idx_map[cls]), (gt[0], gt[1]-5), 1, 1.0, color=(0,255,128), thickness=2)
                cv2.rectangle(img, (dt[0],dt[1]), (dt[0]+dt[2],dt[1]+dt[3]), (0,255,128), 1)    
                cv2.rectangle(img, (gt[0],gt[1]), (gt[0]+gt[2],gt[1]+gt[3]), (0,128,255), 1)                

            for dt_gt in fp:
                cls = dt_gt[0]
                dt = dt_gt[1:5]
                
                img = cv2.putText(img, '{}'.format(tl_idx_map[cls]), (dt[0], dt[1]-5), 1, 1.0, color=(0,0,255), thickness=2)
                cv2.rectangle(img, (dt[0],dt[1]), (dt[0]+dt[2],dt[1]+dt[3]), (0,0,255), 1)
            
            for dt_gt in fn:
                cls = dt_gt[0]
                gt = dt_gt[5:]
                
                img = cv2.putText(img, '{}'.format(tl_idx_map[cls]), (gt[0], gt[1]-5), 1, 1.0, color=(255,128,0), thickness=2)
                cv2.rectangle(img, (gt[0],gt[1]), (gt[0]+gt[2],gt[1]+gt[3]), (255,128,0), 1)
                
            cv2.namedWindow('img',0)   
            cv2.imshow('img',img)
            if 27 == cv2.waitKey():
                break    
        
    if export:
        print('fp and fp images are exported')
        pickle.dump(export_list, open(os.path.join(root, 'fp_fn_list.p'),'wb'))
        
def debug_fp_fn(root, model_path):
    path_fpfn = os.path.join(root, 'fp_fn_list.p')
    image_list = pickle.load(open(path_fpfn,'rb'))

    if '.pb' in model_path:    
        traffic_light_recognizer = tlr_wrapper.Detector(model_path, True, 0)
    elif '.h5' in model_path:    
        traffic_light_recognizer = tlr_wrapper.Detector(model_path, False, 0)        

    
    i = 0
    while True:
        f = image_list[i]
        print(f)
        
        img = cv2.imread(f)
        dets = traffic_light_recognizer.Run(img)        
        img = drawers.draw_tls(img, dets)
        
        cv2.putText(img, "[{}/{}]".format(i, len(image_list)), (5,30), 1, 2.5, (0,255,0), 2) 
        cv2.namedWindow('tlr',0)
        cv2.imshow('tlr',img)        
        k = cv2.waitKey()
        if k == 27:
            break
        elif k == 32: # space: +1
            i += 1
        elif k == 54: # num 6: +10        
            i += 10
        elif k == 57: # num 9: +100        
            i += 100
        elif k == 122: # z: -1
            i -= 1
        elif k == 52: # num 4: -10
            i -= 10
        elif k == 55: # num 7: -100
            i -= 100            
        else:
            i += 1
        
        if i >= len(image_list):
            i = len(image_list)-1
        elif i < 0:
            i = 0
    
    
def export_image_ids(dataset_pickle, output_path):    
    annos = pickle.load(open(dataset_pickle,'rb'))
    
    image_list = {}    
    count = 0
    for img_path in annos.keys():
        count += 1    
        image_id = "image_{}".format(count)
        image_list[image_id] = img_path
    
    with open(output_path,'wb') as fid:
        pickle.dump(image_list,fid)

def plot_wrt_width(root, class_interested=None):
    matched_file_path = os.path.join(root, 'DtGtMatches_iou_0.5.p') 
    DtGtMatches = pickle.load(open(matched_file_path,'rb'))    
    
    data = []
    for image_id in DtGtMatches.keys():
        data.append(DtGtMatches[image_id])

    data = np.concatenate(data)    
    if class_interested is not None:
        data = data[data[:,0]==class_interested]

    fig = plt.figure(figsize=(12,10))    
    plt.subplot(211)    
    
    gts = data[np.sum(data[:,5:],axis=1) != 0]
    w = gts[:,7]
    w = w[w>0]  
    plt.hist(w,np.arange(w.min(),w.max(),1), edgecolor='black', facecolor='green', alpha=0.75, align = 'left', label='gt')
    
    dtgt_matches = data[np.logical_and(np.sum(data[:,5:],axis=1) != 0, np.sum(data[:,1:5],axis=1) != 0)]
    w = dtgt_matches[:,7]
    w = w[w>0]  
    plt.hist(w,np.arange(w.min(),w.max(),1), edgecolor='black', facecolor='red', alpha=0.75, align = 'left', label='dt')
        
    plt.title('[{}] gt and dt w.r.t width'.format(tl_idx_map[class_interested]),fontsize=12)
    plt.xticks(np.arange(0,60,2), fontsize=7)    
    plt.xlim([4,60])
    plt.grid(True)
    plt.legend()
    
    # calculate recall
    recalls = []
    for p in range(4,61):
        gt = gts[gts[:,7]==p]
        dt = dtgt_matches[dtgt_matches[:,7]==p]
        r = len(dt)/float(len(gt)+1e-6)
        recalls.append(np.array([p,r],dtype=np.float))

    recalls = np.array(recalls)   
    plt.subplot(212)    
    plt.bar(recalls[:,0], recalls[:,1], width=1, edgecolor='black', color='orange', alpha=0.75, label='recall')
    plt.title('recall w.r.t width',fontsize=12)
    plt.xlabel('width',fontsize=10)    
    plt.xticks(np.arange(0,60,2), fontsize=7)    
    plt.yticks(np.arange(0,1.01,0.05), fontsize=7)    
    plt.xlim([4,60])
    plt.grid(True)
    plt.savefig(os.path.join(root, 'plot_gt_dt_wrt_width.png'))   

def plot_wrt_height(root, class_interested=None):
    matched_file_path = os.path.join(root, 'DtGtMatches_iou_0.5.p') 
    DtGtMatches = pickle.load(open(matched_file_path,'rb'))    
    
    data = []
    for image_id in DtGtMatches.keys():
        data.append(DtGtMatches[image_id])

    data = np.concatenate(data)    
    if class_interested is not None:
        data = data[data[:,0]==class_interested]

    fig = plt.figure(figsize=(12,10))
    plt.subplot(211)    
    
    gts = data[np.sum(data[:,5:],axis=1) != 0]
    h = gts[:,8]
    h = h[h>0]  
    plt.hist(h,np.arange(h.min(),h.max(),1), edgecolor='black', facecolor='green', alpha=0.75, align = 'left', label='gt')
    
    dtgt_matches = data[np.logical_and(np.sum(data[:,5:],axis=1) != 0, np.sum(data[:,1:5],axis=1) != 0)]
    h = dtgt_matches[:,8]
    h = h[h>0]  
    plt.hist(h,np.arange(h.min(),h.max(),1), edgecolor='black', facecolor='red', alpha=0.75, align = 'left', label='dt')
        
    plt.title('[{}] gt and dt w.r.t height'.format(tl_idx_map[class_interested]),fontsize=12)
    plt.xticks(np.arange(0,141,2), fontsize=7)    
    plt.xlim([4,140])
    plt.grid(True)
    plt.legend()
    
    # calculate recall
    recalls = []
    for p in range(4,141):
        gt = gts[gts[:,8]==p]
        dt = dtgt_matches[dtgt_matches[:,8]==p]
        r = len(dt)/float(len(gt)+1e-6)
        recalls.append(np.array([p,r],dtype=np.float))

    recalls = np.array(recalls)   
    plt.subplot(212)
       
    plt.bar(recalls[:,0], recalls[:,1], width=1, edgecolor='black', color='orange', alpha=0.75, label='recall')
    plt.title('recall w.r.t height',fontsize=12)
    plt.xlabel('height',fontsize=10)    
    plt.xticks(np.arange(0,141,2), fontsize=7)    
    plt.yticks(np.arange(0,1.01,0.05), fontsize=7)    
    plt.xlim([4,140])
    plt.grid(True)
    plt.savefig(os.path.join(root, 'plot_gt_dt_wrt_height.png'))   


def plot_gt(root, class_interested=None):
    matched_file_path = os.path.join(root, 'DtGtMatches_iou_0.5.p') 
    DtGtMatches = pickle.load(open(matched_file_path,'rb'))    
    
    data = []
    for image_id in DtGtMatches.keys():
        data.append(DtGtMatches[image_id])

    data = np.concatenate(data)    
    if class_interested is not None:
        data = data[data[:,0]==class_interested]
        
    fig = plt.figure(figsize=(12,8))    
    
    gts = data[np.sum(data[:,5:],axis=1) != 0]
    h = gts[:,8]
    h = h[h>0]  
    plt.hist(h,np.arange(h.min(),h.max(),1), edgecolor='black', facecolor='green', alpha=0.75, align = 'left', label='gt')
            
    plt.title('Ground Truth'.format(tl_idx_map),fontsize=12)
    plt.xticks(np.arange(0,141,2), fontsize=7)    
    plt.xlim([4,140])
    plt.ylim([0,1450])
    plt.grid(True)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(root, 'plot_gt.png'))   

def plot_recall(root, class_interested=None):
    matched_file_path = os.path.join(root, 'DtGtMatches_iou_0.5.p') 
    DtGtMatches = pickle.load(open(matched_file_path,'rb'))    
    
    data = []
    for image_id in DtGtMatches.keys():
        data.append(DtGtMatches[image_id])

    data = np.concatenate(data)    
    if class_interested is not None:
        data = data[data[:,0]==class_interested]

    fig = plt.figure(figsize=(12,8))
    gts = data[np.sum(data[:,5:],axis=1) != 0]
    dtgt_matches = data[np.logical_and(np.sum(data[:,5:],axis=1) != 0, np.sum(data[:,1:5],axis=1) != 0)]
    
    # calculate recall
    recalls = []
    for p in range(4,141):
        gt = gts[gts[:,8]==p]
        dt = dtgt_matches[dtgt_matches[:,8]==p]
        r = len(dt)/float(len(gt)+1e-6)
        recalls.append(np.array([p,r],dtype=np.float))

    recalls = np.array(recalls)   
       
    plt.bar(recalls[:,0], recalls[:,1], width=1, edgecolor='black', color='orange', alpha=0.75, label='recall')
    plt.title('recall w.r.t height',fontsize=12)
    plt.xlabel('height',fontsize=10)    
    plt.xticks(np.arange(0,141,2), fontsize=7)    
    plt.yticks(np.arange(0,1.01,0.05), fontsize=7)    
    plt.xlim([4,140])
    plt.grid(True)
    plt.savefig(os.path.join(root, 'plot_recall.png')) 

def parse_args():    
    parser = argparse.ArgumentParser(description='debug tlr')    
    parser.add_argument('--f', dest='root', help='folder path that contains gts.p, dts.p, and eval_imgs.p', default='', type=str)
    parser.add_argument('--e_dtgt', dest='e_dtgt', help='export dts and gts', default=False, type=bool)        
    parser.add_argument('--v_dtgt', dest='v_dtgt', help='visualize dts and gts', default=False, type=bool)        
    parser.add_argument('--e_img_id', dest='e_img_id', help='[IMAGE_ID] export image_ids', default=False, type=bool)        
    parser.add_argument('--img_id_dataset', dest='img_id_dataset', help='[IMAGE_ID] path to the dataset', default='', type=str)
    parser.add_argument('--img_id_dst', dest='img_id_dst', help='[IMAGE_ID] path for saving image_id.p', default='', type=str)                    
    parser.add_argument('--v_fp', dest='v_fp', help='visualize fp', default=False, type=bool)
    parser.add_argument('--v_fn', dest='v_fn', help='visualize fn', default=False, type=bool)
    parser.add_argument('--e_fpfn', dest='e_fpfn', help='export fp and fn', default=False, type=bool)
    parser.add_argument('--min_w', dest='min_w', help='minimum width for visualization', default=0, type=int)
    parser.add_argument('--max_w', dest='max_w', help='minimum width for visualization', default=100, type=int)
    parser.add_argument('--debug', dest='debug', help='debug fp and fn', default=False, type=bool)
    parser.add_argument('--plot', dest='plot', help='plot wrt width', default=False, type=bool)
    parser.add_argument('--m', dest='model_path', help='model path', default='', type=str)
    
    args = parser.parse_args()

    return args
    
if __name__ == '__main__':
    args = parse_args()
    if args.e_dtgt:
        export_DtGtMatches(args.root, '0.5')
    if args.v_dtgt:
        visualize_DtGtMatches(args.root)
    if args.e_img_id:
        export_image_ids(args.img_id_dataset, args.img_id_dst)    
    if args.v_fp or args.v_fn:
        visualize_fp_fn(args.root, debug_fp=args.v_fp, debug_fn=args.v_fn, export=args.e_fpfn, min_w=args.min_w, max_w=args.max_w)
    if args.debug:
        debug_fp_fn(args.root, args.model_path)
    if args.plot:
        plot_wrt_height(args.root)
        plot_wrt_width(args.root)
        plot_gt(args.root)
        plot_recall(args.root)
    


