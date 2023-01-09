import os
import cv2
import numpy as np
from image import flip, color_aug
from image import get_affine_transform, affine_transform
from image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from image import draw_dense_reg, draw_dense_reg_cls_nb
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class MagnaAdTlDataset(Dataset):
    def __init__(self, image_path, anno_path, aug=True, debug=False):
        self.image_path = image_path
        self.anno_path = anno_path        

        self.images = {}
        idx = 0
        if self.image_path.endswith('.txt'):
            with open(self.image_path, 'r') as f:
                for l in f:
                    self.images[idx] = l.split('\n')[0]
                    idx += 1
            self.image_path = self.image_path.split('samples')[0] + 'rgb_images'            
        else:
            for l in os.listdir(image_path):
                self.images[idx] = l 
                idx += 1

        self.total_num_images = idx

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

        self.seq = iaa.Sequential([
                                    iaa.MultiplySaturation((0.2, 1.2), from_colorspace=iaa.CSPACE_BGR),
                                    # apply the following augmenters to most images
                                    iaa.Fliplr(0.5),  # horizontally flip 50% of all images                                    
                                    # crop images by -5% to 10% of their height/width
                                    iaa.Sometimes(0.5, iaa.CropAndPad(
                                        percent=(-0.1, 0.1),
                                        pad_mode='constant',
                                        pad_cval=(0)
                                    )),                                    
                                    iaa.Sometimes(0.5, iaa.Affine(
                                        # scale images to 100-140% of their size, individually per axis
                                        scale={"x": (1.0, 1.4), "y": (1.0, 1.4)},        
                                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},        
                                        #rotate=(-45, 45),  # rotate by -15 to +15 degrees
                                        # use nearest neighbour or bilinear interpolation (fast)
                                        order=[0, 1],
                                        # if mode is constant, use a cval between 0 and 255
                                        cval=(0),
                                        # # use any of scikit-image's warping modes
                                        # # (see 2nd image from the top for examples)
                                        mode='constant'
                                    )),            
                                    # execute 0 to 5 of the following (less important) augmenters per
                                    # image don't execute all of them, as that would often be way too
                                    # strong
                                    iaa.SomeOf((0, 5),
                                            [
                                            iaa.OneOf([
                                            # blur images with a sigma between 0 and 3.0
                                            iaa.GaussianBlur((0, 1.5)),
                                            # blur image using local means with kernel sizes
                                            # between 2 and 7
                                            iaa.AverageBlur(k=(3, 5)),
                                            # blur image using local medians with kernel sizes
                                            # between 2 and 7
                                            iaa.MedianBlur(k=(3, 5)),
                                        ]),
                                        iaa.Sharpen(alpha=(0, 1.0), lightness=(
                                                    0.75, 1.5)),  # sharpen images                
                                        # search either for all edges or for directed edges,
                                        # blend the result with the original image using a blobby mask
                                        
                                        # # add gaussian noise to images
                                        iaa.AdditiveGaussianNoise(loc=0, scale=(
                                            0.0, 0.05*255), per_channel=0.5),                                        
                                        # # change brightness of images (by -10 to 10 of original value)
                                        iaa.Add((-10, 10), per_channel=0.5),
                                    ],
                                    random_order=True
                                    )            
                                ],
                                random_order=True)        

        # options
        self.pad = 31        
        self.scale = 0.4        
        self.down_ratio = 4
        self.num_classes = 2        
        self.max_objs = 20
        self.dense_wh = True
        self.reg_offset = True        
        self.radius_scale = 3.0

        # debug
        self.aug = aug
        self.debug = debug

    def create_new_bbox(self, img, roi, tl_box, resize_size, min_width=6):
        img_roi = np.copy(img[roi[1]:roi[3],roi[0]:roi[2],:])        
        roi_h, roi_w, _ = img_roi.shape

        if len(tl_box) != 0:
            new_tl_box = np.copy(tl_box).astype(np.float)            
            new_tl_box[:,[3,5]] = tl_box[:,[3,5]] - roi[0]
            new_tl_box[:,[4,6]] = tl_box[:,[4,6]] - roi[1]

            new_tl_box = new_tl_box[new_tl_box[:,3] >= 0   , :]
            new_tl_box = new_tl_box[new_tl_box[:,4] >= 0   , :]
            new_tl_box = new_tl_box[new_tl_box[:,5] < roi_w, :]
            new_tl_box = new_tl_box[new_tl_box[:,6] < roi_h, :]
            
            resize_ratio = resize_size[0]/float(roi_w)
            new_bbox = new_tl_box[:,3:].astype(np.float) * resize_ratio            
            new_bbox_w = new_bbox[:,2]-new_bbox[:,0]
            new_bbox = new_bbox[new_bbox_w>=min_width]
            new_tl_box = new_tl_box[new_bbox_w>=min_width]
            new_tl_box[:,3:] = new_bbox            
        else:
            new_tl_box = []

        new_img = cv2.resize(img_roi, resize_size)

        return new_img, new_tl_box

    def get_item_single(self, idx):        
        img = cv2.imread(os.path.join(self.image_path, self.images[idx]))
        anno = self.get_annotations(self.images[idx])

        return img, anno

    def get_item(self, idx):        
        img = cv2.imread(os.path.join(self.image_path, self.images[idx]))
        anno = self.get_annotations(self.images[idx])

        imgs = []
        annos = []        
        none_imgs = []
        none_annos = []

        for roi in self.rois:
            new_img, new_anno = self.create_new_bbox(img, roi, anno, (320, 160), 4)     
            imgs.append(new_img)
            annos.append(new_anno)

        return imgs, annos
        #     if len(new_anno) != 0:
        #         imgs.append(new_img)
        #         annos.append(new_anno)
        #     else:
        #         none_imgs.append(new_img)
        #         none_annos.append(new_anno)

        # if len(annos) != 0:
        #     return imgs, annos
        
        # else:
        #     return [none_imgs[-1]], [none_annos[-1]]

    def get_annotations(self, image_name):
        file_name = image_name.split('.')[0] + '.txt'
        anno_file = open(os.path.join(self.anno_path, file_name), 'r')
        
        bbs = []
        for l in anno_file.readlines():
            if 'traffic_light' in l:
                items = l.split(',')
                try:
                    cls = int(items[4])
                    nb = int(items[5])
                    x1 = int(items[6])
                    y1 = int(items[7])
                    x2 = int(items[8])
                    y2 = int(items[9])

                    if cls >= 13:
                        cls = 13

                    bbs.append(np.array([1, cls, nb, x1, y1, x2, y2]))       
                except:
                    print(' error in sample retrieval: {}'.format(os.path.join(self.anno_path, file_name)))
                    continue
        
        return np.array(bbs)

    def __len__(self):
        return len(self.images)

    def encode(self, img, anno):
        # convert numpy to BoundingBox for data augmentation        
        ann_bbs = []
        for ann in anno:
            label = '{}_{}_{}'.format(int(ann[0]),int(ann[1]),int(ann[2]))
            ann_bbs.append(BoundingBox(x1=ann[3], y1=ann[4], x2=ann[5], y2=ann[6], label=label))

        ann_bbs_oi = BoundingBoxesOnImage(ann_bbs, shape=img.shape)
        if self.aug:
            img_aug, ann_bbs_oi_aug = self.seq(image=img, bounding_boxes=ann_bbs_oi)
        else:
            img_aug = img
            ann_bbs_oi_aug = ann_bbs_oi

        # remove samples outside image
        ann_bbs_oi_aug = ann_bbs_oi_aug.remove_out_of_image(partly=True)

        num_objs = min(len(ann_bbs_oi_aug), self.max_objs)

        # center 
        height, width = img_aug.shape[0], img_aug.shape[1]        
        c = np.array([width / 2, height / 2], dtype=np.float32) 

        # input size
        input_h = (height | self.pad) + 1
        input_w = (width | self.pad) + 1
        s = np.array([input_w, input_h], dtype=np.float32)               

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])        
        inp = cv2.warpAffine(img_aug, trans_input, (input_w, input_h),flags=cv2.INTER_LINEAR)

        if self.debug: 
            inp_debug = inp.copy()

        inp = (inp.astype(np.float32) / 255.)
        inp = inp.transpose(2, 0, 1) # channel, height, width

        # output size
        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])        

        num_classes = self.num_classes
        
        # initalize outputs
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        output_tl_cls = np.zeros((output_h, output_w), dtype=np.uint8)
        output_tl_nb = np.zeros((output_h, output_w), dtype=np.uint8)
        
        draw_gaussian = draw_umich_gaussian

        gt_det = []

        if self.debug:
            output_debug = cv2.warpAffine(img_aug, trans_output, (output_w, output_h), flags=cv2.INTER_LINEAR)

        for k in range(num_objs):
            ann = ann_bbs_oi_aug[k]
            bbox = np.array([ann.x1, ann.y1, ann.x2, ann.y2])            
            cls_id = int(ann.label.split('_')[0]) # object: 0 - no object 1 - tl 
            tl_cls = int(ann.label.split('_')[1]) # traffic light status: green, red, yellow, ...
            tl_nb = int(ann.label.split('_')[2]) # number of bulbs: three, four, and five
                    
            if self.debug: 
                img_aug = img_aug.copy()
                bbox_int = bbox.astype(np.int)
                cv2.rectangle(img_aug, (bbox_int[0],bbox_int[1]), (bbox_int[2],bbox_int[3]), (0,255,0), 1)

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)            

            if self.debug: 
                bbox_int = bbox.astype(np.int)
                cv2.rectangle(output_debug, (bbox_int[0],bbox_int[1]), (bbox_int[2],bbox_int[3]), (0,255,0), 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]            
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h)*self.radius_scale, math.ceil(w)*self.radius_scale))
                radius = max(0, int(radius))
                radius = radius
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)                
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.dense_wh:
                    #draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                    dense_wh, output_tl_cls, output_tl_nb = draw_dense_reg_cls_nb(dense_wh, output_tl_cls, output_tl_nb, hm.max(axis=0), ct_int, wh[k], tl_cls, tl_nb, radius)  
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                            ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])        

        if self.debug:         
            cv2.namedWindow('original image',0)
            cv2.imshow('original image', img_aug)
            cv2.namedWindow('input_debug',0)
            cv2.imshow('input_debug', inp_debug)
            cv2.namedWindow('output_debug',0)            
            cv2.imshow('output_debug', output_debug)
            
            cv2.namedWindow('hm', 0)
            cv2.imshow('hm', hm.transpose((1,2,0))[:,:,1])
            print('hm')
            print(np.unique(hm))
                        
            cv2.namedWindow('wh', 0)
            cv2.imshow('wh', dense_wh.transpose((1,2,0))[:,:,1])
            print('wh')
            print(np.unique(wh))

            cv2.namedWindow('cls', 0)
            cv2.imshow('cls', output_tl_cls)
            print('cls')
            print(np.unique(output_tl_cls))

            cv2.namedWindow('nb', 0)
            cv2.imshow('nb', output_tl_nb)
            print('nb')
            print(np.unique(output_tl_nb))
            
            cv2.waitKey()

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'cls': output_tl_cls, 'nb': output_tl_nb}
        
        if self.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']

        if self.reg_offset:
            ret.update({'reg': reg})
            
        return ret

    def __getitem__(self, idx):
        imgs, annos = self.get_item(idx)
        
        rets = []
        for img, anno in zip(imgs, annos):
            ret = self.encode(img, anno)
            rets.append(ret)

        return rets