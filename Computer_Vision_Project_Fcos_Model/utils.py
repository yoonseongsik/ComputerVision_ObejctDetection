import numpy as np
import copy
from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.ops import nms

import wandb


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(
                image, target)

        return image, target


class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target):
        w, h = image.size
        image = image.resize(self.size)
        
        target['boxes'][:, [0, 2]] *= self.size[0] / w
        target['boxes'][:, [1, 3]] *= self.size[1] / h
        
        return image, target

class Normal:
    def __call__(self, image, target):
        normalize = torchvision.transforms.Normalize(mean=[0.4854, 0.4562, 0.4065], std=[0.2293, 0.2244, 0.2250])

        image = normalize(image)
        return image, target



class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def replace_dataset_classes(dataset, class_map):
    """ Replaces classes of dataset based on a dictionary"""
    class_new_names = list(set(class_map.values()))
    class_new_names.sort()
    class_originals = copy.deepcopy(dataset['categories'])
    dataset['categories'] = []
    class_ids_map = {}  # map from old id to new id

    # Assign background id 0
    has_background = False
    if 'Background' in class_new_names:
        if class_new_names.index('Background') != 0:
            class_new_names.remove('Background')
            class_new_names.insert(0, 'Background')
        has_background = True

    # Replace categories
    for id_new, class_new_name in enumerate(class_new_names):

        # Make sure id:0 is reserved for background
        id_rectified = id_new
        if not has_background:
            id_rectified += 1

        category = {
            'supercategory': '',
            'id': id_rectified,  # Background has id=0
            'name': class_new_name,
        }
        dataset['categories'].append(category)
        # Map class names
        for class_original in class_originals:
            if class_map[class_original['name']] == class_new_name:
                class_ids_map[class_original['id']] = id_rectified

    # Update annotations category id tag
    for ann in dataset['annotations']:
        ann['category_id'] = class_ids_map[ann['category_id']]

def calculate_loss(losses):
    loss = sum(loss for loss in losses.values())
    return loss

def decode_output(output):
    'convert tensors to numpy arrays'
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16) # output의 bounding box
    labels = np.array([i for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy() # output bounding box 의 confidence score

    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.3) # nms 수행

    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels= [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

def make_wandb_visual(image, bbs, confs, labels):
    wandb_results = {}

    class_labels = {
            0 : 'background',
            1 : 'WO-01',
            2 : 'WO-02',
            3 : 'WO-03',
            4 : 'WO-04',
            5 : 'WO-05',
            6 : 'WO-06',
            7 : 'WO-07',
            8 : 'WO-08',
            9 : 'SO-01',
            10 : 'SO-02',
            11 : 'SO-03',
            12 : 'SO-04',
            13 : 'SO-05',
            14 : 'SO-06',
            15 : 'SO-07',
            16 : 'SO-08',
            17 : 'SO-09',
            18 : 'SO-10',
            19 : 'SO-11',
            20 : 'SO-12',
            21 : 'SO-13',
            22 : 'SO-14',
            23 : 'SO-15',
            24 : 'SO-16',
            25 : 'SO-17',
            26 : 'SO-18',
            27 : 'SO-19',
            28 : 'SO-20',
            29 : 'SO-21',
            30 : 'SO-22',
            31 : 'SO-23',
        }

    img_np = torch.permute(image.to('cpu'), (1, 2, 0)).numpy().astype(np.float64)

    # for bbox
    bbs = [[a/400 for a in single] for single in bbs]
    wandb_results['predictions'] = {
        'box_data': [
                {
                'position': {
                    "minX": bb[0],
                    "maxX": bb[1],
                    "minY": bb[2],
                    "maxY": bb[3]
                },
                'class_id' : label,
                'box_caption': class_labels[label],
                'scores' : {
                    'conf': conf,
                    }
                } for bb, conf, label in zip(bbs, confs, labels)],
        'class_labes': class_labels
        } 
    

    img = wandb.Image(img_np, boxes = wandb_results)

    wandb.log({"Inference": img})

