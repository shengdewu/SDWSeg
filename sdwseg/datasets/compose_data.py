import cv2
import numpy as np
import random
import os
import engine.transforms.functional as F
from engine.data.build import BUILD_DATASET_REGISTRY
from engine.data.dataset import EngineDataSet
from typing import List, Tuple

__all__ = [
    'ComposeDataSet'
]


def parse_file_path(root_path: str, txt_names: List[str]) -> List[Tuple[str, str]]:
    file_names = list()
    for txt_name in txt_names:
        h = open(os.path.join(root_path, txt_name), mode='r')
        file_names.extend([line.strip('\n') for line in h.readlines()])
        h.close()

    file_paths = set()

    for name in file_names:
        img_name_arr = name.split(',')
        img_name = img_name_arr[0]
        msk_name = img_name_arr[1]

        image_path = os.path.join(root_path, img_name)
        mask_path = os.path.join(root_path, msk_name)
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue

        file_paths.add((image_path, mask_path))

    return list(file_paths)


@BUILD_DATASET_REGISTRY.register()
class ComposeDataSet(EngineDataSet):
    def __init__(self, data_root_path, require_txt, transformer: List, select_nums=0):
        """
        :param data_root_path:
        :param require_txt:
        :param transformer:
        :param select_nums:
        """
        super(ComposeDataSet, self).__init__(transformer)

        self.file_names = parse_file_path(data_root_path, require_txt)
        random.shuffle(self.file_names)

        if len(self.file_names) > select_nums > 0:
            self.file_names = random.sample(self.file_names, k=select_nums)
        return

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += f'{super(ComposeDataSet, self).__repr__()}\n'
        format_string += 'total data={})'.format(len(self.file_names))
        return format_string

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_path, mask_path = self.file_names[idx]

        img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        result = dict(
            img=img,
            gt_semantic_seg=mask,
            img_fields=['img', 'gt_semantic_seg'],
            color_fields=['img'],
            interpolation=dict(gt_semantic_seg='INTER_NEAREST'),
        )

        result = self.data_pipeline(result)

        result['img'] = F.to_tensor(result['img'])
        result['gt_semantic_seg'] = F.to_tensor((result['gt_semantic_seg']).astype(np.int64))

        return dict(img=result['img'], gt_semantic_seg=result['gt_semantic_seg'])
