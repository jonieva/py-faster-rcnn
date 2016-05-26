#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'pectoralis')

NETS = {
    'zf': ('ZF', 'bodyComposition.caffemodel'),
    'vgg16': ('VGG16', 'VGG16_faster_rcnn_final.caffemodel'),
    }


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print("No boxes found with a threshold >= {}. Max threshold={}".format(thresh, np.max(dets[:, -1])))
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[3] - bbox[1],
                          bbox[2] - bbox[0], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[1], bbox[0] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, "BodyCompositionDevKit/BodyComposition2016/JPEGImages/", image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    print("***** DEBUG: Processing file {}".format(im_file))
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    #print ("Classes:", CLASSES)
    #print ("Boxes ({}): ".format(len(boxes)), boxes)
    #print ("Boxes [0]: ", boxes[0])
   # print ("Scores ({}): ".format(len(scores)), scores)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        # Boxes:
        # [[CL0BB0_0, CL0BB0_1, CL0BB0_2, CL0BB0_3, CL1BB0_0, CL1BB0_1, CL1BB0_2, CL1BB0_3],
        #  [CL0BB1_0, CL0BB1_1, CL0BB1_2, CL0BB1_3, CL1BB1_0, CL1BB1_1, CL1BB1_2, CL1BB1_3]]
        cls_ind += 1 # because we skipped background
        # cls_boxes (for cls_ind=1):
        # [[CL1BB0_0, CL1BB0_1, CL1BB0_2, CL1BB0_3],
        #  [CL1BB1_0, CL1BB1_1, CL1BB1_2, CL1BB1_3]]
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        # cls_scores (for cls_ind=1):
        # [CL1BB0_SCORE, CL1BB1_SCORE]
        cls_scores = scores[:, cls_ind]
        # This will give us, for every class, the 4 coordinates of every box plus the score.
        # Ex (for class 1: [[CL1BB0_0, CL1BB0_1, CL1BB0_2, CL1BB0_3, CL1BB0_SCORE],
        #                   [CL1BB1_0, CL1BB1_1, CL1BB1_2, CL1BB1_3, CL1BB1_SCORE]],
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # Use NMS (Non Maximum Supression) to ignore redundant overlapping bounding boxes
        keep = nms(dets, NMS_THRESH)
        # print("Keep ({}): ".format(len(keep)), keep)
        dets = dets[keep, :]
        # print("Dets2 ({}): ".format(len(dets)), dets)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, "ZF", 'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output',"bodyComposition", "BodyComposition_2016_trainval", "current.caffemodel")

    if not os.path.exists(caffemodel):
        raise IOError(('{} not found.\nDid you train your system?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _= im_detect(net, im)

    im_names = ['000300.png']

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo(net, im_name)

    plt.show()
