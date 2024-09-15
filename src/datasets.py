import sys,os,json

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from utils.utils import *

class KAISTPed(data.Dataset):
    """KAIST Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'KAIST')
        condition (string, optional): load condition
            (default: 'Reasonabel')
    """
    def __init__(self, args, condition='train'):
        self.args = args
        assert condition in args.dataset.OBJ_LOAD_CONDITIONS
        
        self.mode = condition
        self.image_set = args[condition].img_set
        self.img_transform = args[condition].img_transform
        self.co_transform = args[condition].co_transform        
        self.cond = args.dataset.OBJ_LOAD_CONDITIONS[condition]
        self.annotation = args[condition].annotation
        self._parser = LoadBox()        

        self._annopath = os.path.join('%s', 'annotations_paired', '%s', '%s', '%s', '%s.txt')

        self._imgpath = os.path.join('%s', 'images', '%s', '%s', '%s', '%s.jpg')  
        
        self.ids = list()
        for line in open(os.path.join('./imageSets', self.image_set)):
            self.ids.append((self.args.path.DB_ROOT, line.strip().split('/')))

    def __str__(self):
        return self.__class__.__name__ + '_' + self.image_set

    def __getitem__(self, index): 

        vis, lwir, boxes_vis, boxes_lwir, labels = self.pull_item(index)
        return vis, lwir, boxes_vis, boxes_lwir, labels, torch.ones(1,dtype=torch.int)*index

    def pull_item(self, index):
        
        frame_id = self.ids[index]
        set_id, vid_id, img_id = frame_id[-1]

        vis = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ))
        lwir = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id ) ).convert('L')
    
        width, height = lwir.size

        # paired annotation
        if self.mode == 'train': 
            vis_boxes = list()
            lwir_boxes = list()

            for line in open(self._annopath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id )) :
                vis_boxes.append(line.strip().split(' '))
            for line in open(self._annopath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id)) :
                lwir_boxes.append(line.strip().split(' '))

            vis_boxes = vis_boxes[1:]
            lwir_boxes = lwir_boxes[1:]

            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]
            boxes_vis_unpaired = []
            boxes_lwir_unpaired = []

            ### add unpaired situation ###
            for i in range(len(vis_boxes)) :
                name = vis_boxes[i][0]

                bndbox = [int(i) for i in vis_boxes[i][1:5]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]

                if name == 'unpaired':
                    bndbox.append(1)
                    boxes_vis_unpaired += [bndbox]
                    boxes_lwir_unpaired += [bndbox]
                else:
                    bndbox.append(3)
                    boxes_vis += [bndbox]

            for i in range(len(lwir_boxes)) :
                name = lwir_boxes[i][0]

                bndbox = [int(i) for i in lwir_boxes[i][1:5]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]

                if name == 'unpaired':
                    bndbox.append(2)
                    boxes_vis_unpaired += [bndbox]
                    boxes_lwir_unpaired += [bndbox]
                else:
                    bndbox.append(3)
                    boxes_lwir += [bndbox]

            boxes_vis += boxes_vis_unpaired
            boxes_lwir += boxes_lwir_unpaired

            boxes_vis = np.array(boxes_vis, dtype=np.float)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float)

        else :
            vis_boxes = list()
            lwir_boxes = list()

            for line in open(self._annopath % (*frame_id[:-1], set_id, vid_id, 'visible', img_id)):
                vis_boxes.append(line.strip().split(' '))
            for line in open(self._annopath % (*frame_id[:-1], set_id, vid_id, 'lwir', img_id)):
                lwir_boxes.append(line.strip().split(' '))

            vis_boxes = vis_boxes[1:]
            lwir_boxes = lwir_boxes[1:]

            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]

            for i in range(len(vis_boxes)):
                name = vis_boxes[i][0]

                bndbox = [int(i) for i in vis_boxes[i][1:5]]
                bndbox[2] = min(bndbox[2] + bndbox[0], width)
                bndbox[3] = min(bndbox[3] + bndbox[1], height)
                bndbox = [cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox)]

                bndbox.append(1)
                boxes_vis += [bndbox]

            for i in range(len(lwir_boxes)):
                name = lwir_boxes[i][0]

                bndbox = [int(i) for i in lwir_boxes[i][1:5]]
                bndbox[2] = min(bndbox[2] + bndbox[0], width)
                bndbox[3] = min(bndbox[3] + bndbox[1], height)
                bndbox = [cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox)]

                bndbox.append(2)
                boxes_lwir += [bndbox]

            boxes_vis = np.array(boxes_vis, dtype=np.float)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float)

        ## Apply transforms
        if self.img_transform is not None:
            vis, lwir, boxes_vis , boxes_lwir, _ = self.img_transform(vis, lwir, boxes_vis, boxes_lwir)

        if self.co_transform is not None:

            ### Follow MLPD semi-unpaired augmentation ###
            pair = 1

            vis, lwir, boxes_vis, boxes_lwir, pair = self.co_transform(vis, lwir, boxes_vis, boxes_lwir, pair)

            if pair != 1:
                if len(boxes_vis.shape) != 1 and len(boxes_lwir.shape) != 1:
                    ### copy all bb pairs, then [0,1,0] for first half, [0,0,1] for second half
                    boxes_vis[1:,4] = 1
                    boxes_vis_neg = boxes_vis[1:].detach().clone()
                    boxes_vis_neg[:,4] = 2
                    # boxes_vis.append(boxes_vis_neg)
                    boxes_vis = torch.cat((boxes_vis, boxes_vis_neg), dim=0)

                    boxes_lwir[1:, 4] = 1
                    boxes_lwir_pos = boxes_lwir[1:].detach().clone()
                    boxes_lwir_pos[:,4] = 2
                    # boxes_lwir.append(boxes_lwir_pos)
                    boxes_lwir = torch.cat((boxes_lwir, boxes_lwir_pos), dim=0)

            boxes_vis = torch.tensor(list(map(list, boxes_vis)))
            boxes_lwir = torch.tensor(list(map(list, boxes_lwir)))

        ## Set ignore flags
        ignore = torch.zeros(boxes_vis.size(0), dtype=torch.bool)
               
        for ii, (box_vis, box_lwir) in enumerate(zip(boxes_vis, boxes_lwir)):
                        
            x_vis = box_vis[0] * width
            y_vis = box_vis[1] * height
            w_vis = ( box_vis[2] - box_vis[0] ) * width
            h_vis = ( box_vis[3] - box_vis[1] ) * height

            x_lwir = box_lwir[0] * width
            y_lwir = box_lwir[1] * height
            w_lwir = ( box_lwir[2] - box_lwir[0] ) * width
            h_lwir = ( box_lwir[3] - box_lwir[1] ) * height

            if  x_vis < self.cond['xRng'][0] or \
                y_vis < self.cond['xRng'][0] or \
                x_vis + w_vis > self.cond['xRng'][1] or \
                y_vis + h_vis > self.cond['xRng'][1] or \
                w_vis < self.cond['wRng'][0] or \
                w_vis > self.cond['wRng'][1] or \
                h_vis < self.cond['hRng'][0] or \
                h_vis > self.cond['hRng'][1] or \
                x_lwir < self.cond['xRng'][0] or \
                y_lwir < self.cond['xRng'][0] or \
                x_lwir + w_lwir > self.cond['xRng'][1] or \
                y_lwir + h_lwir > self.cond['xRng'][1] or \
                w_lwir < self.cond['wRng'][0] or \
                w_lwir > self.cond['wRng'][1] or \
                h_lwir < self.cond['hRng'][0] or \
                h_lwir > self.cond['hRng'][1]:

                ignore[ii] = 1
        
        boxes_vis[ignore, 4] = -1
        boxes_lwir[ignore, 4] = -1

        labels = boxes_vis[:,4] ### boxes_vis and boxes_lwir have the same labels. ###
        boxes_t_vis = boxes_vis[:, 0:4]
        boxes_t_lwir = boxes_lwir[:, 0:4]
        
        return vis, lwir, boxes_t_vis, boxes_t_lwir, labels

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        vis = list()
        lwir = list()
        boxes_vis = list()
        boxes_lwir = list()
        labels = list()
        index = list()

        for b in batch:
            vis.append(b[0])
            lwir.append(b[1])
            boxes_vis.append(b[2])
            boxes_lwir.append(b[3])
            labels.append(b[4])
            index.append(b[5])

        vis = torch.stack(vis, dim=0)
        lwir = torch.stack(lwir, dim=0)
  
        return vis, lwir, boxes_vis, boxes_lwir, labels, index

class LoadBox(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, bbs_format='xyxy'):
        assert bbs_format in ['xyxy', 'xywh']                
        self.bbs_format = bbs_format
        self.pts = ['x', 'y', 'w', 'h']

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """                
        res = [ [0, 0, 0, 0, -1] ]

        for obj in target.iter('object'):           
            name = obj.find('name').text.lower().strip()            
            bbox = obj.find('bndbox')
            bndbox = [ int(bbox.find(pt).text) for pt in self.pts ]

            if self.bbs_format in ['xyxy']:
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )

            bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
            
            bndbox.append(1)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind, occ]
            
        return np.array(res, dtype=np.float)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


if __name__ == '__main__':
    """Debug KAISTPed class"""
    from matplotlib import patches
    from matplotlib import pyplot as plt
    from utils.functional import to_pil_image, unnormalize
    import config

    def draw_boxes(axes, boxes, labels, target_label, color):
        for x1, y1, x2, y2 in boxes[labels == target_label]:
            w, h = x2 - x1 + 1, y2 - y1 + 1
            axes[0].add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=1))
            axes[1].add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=1))

    args = config.args
    test = config.test

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    dataset = KAISTPed(args, condition='test')

    # HACK(sohwang): KAISTPed always returns empty boxes in test mode
    dataset.mode = 'train'

    vis, lwir, boxes_vis, boxes_lwir, labels, indices = dataset[1300]

    vis_mean = dataset.co_transform.transforms[-2].mean
    vis_std = dataset.co_transform.transforms[-2].std

    lwir_mean = dataset.co_transform.transforms[-1].mean
    lwir_std = dataset.co_transform.transforms[-1].std

    # C x H x W -> H X W x C
    vis_np = np.array(to_pil_image(unnormalize(vis, vis_mean, vis_std)))
    lwir_np = np.array(to_pil_image(unnormalize(lwir, lwir_mean, lwir_std)))

    # Draw images
    axes[0].imshow(vis_np)
    axes[1].imshow(lwir_np)
    axes[0].axis('off')
    axes[1].axis('off')

    # Draw boxes on images
    input_h, input_w = test.input_size
    xyxy_scaler_np = np.array([[input_w, input_h, input_w, input_h]], dtype=np.float32)
    boxes_vis = boxes_vis * xyxy_scaler_np
    boxes_lwir = boxes_lwir * xyxy_scaler_np

    draw_boxes(axes[0], boxes_vis, labels, 3, 'blue')
    draw_boxes(axes[0], boxes_vis, labels, 1, 'red')
    draw_boxes(axes[0], boxes_vis, labels, 2, 'green')

    draw_boxes(axes[1], boxes_lwir, labels, 3, 'blue')
    draw_boxes(axes[1], boxes_lwir, labels, 1, 'red')
    draw_boxes(axes[1], boxes_lwir, labels, 2, 'green')

    frame_id = dataset.ids[indices.item()]
    set_id, vid_id, img_id = frame_id[-1]
    fig.savefig(f'{set_id}_{vid_id}_{img_id}.png')
