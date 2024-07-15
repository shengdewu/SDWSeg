import os
import time
import cv2
import torch
import tqdm
import numpy as np
import sys
import argparse
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import engine.transforms.functional as F
from engine.transforms import Resize, Normalize


class Segment:
    def __init__(self, img_size, out_path, weight:str, config:str, device='cpu',show_mask=False, keep_ratio=False):
        resize_cfg = dict(interpolation='INTER_LINEAR',
                          target_size=img_size,
                          keep_ratio=keep_ratio,
                          is_padding=False)

        normalize_cfg = dict(
            mean=(0, 0, 0),
            std=(255, 255, 255),
        )

        self.transform = [
            Resize(**resize_cfg),
            Normalize(**normalize_cfg)
        ]

        self.out_path = out_path
        self.show_mask = show_mask
        self.device = device
        os.makedirs(out_path, exist_ok=True)

        self.model = self.load_model(weight, config)
        return

    def load_model(self, weights: str, config: Optional[str] = None):
        if weights.lower().endswith('onnx'):
            from tool.py_onnx import PyOnnx
            model = PyOnnx(weights)
        elif weights.lower().endswith('pt'):
            model = torch.load(weights)
        else:
            from tool.create_encoder_decoder import create_encoder_decoder
            assert config is not None
            model = create_encoder_decoder(config, weights)
        return model

    def is_jpg(self, name: str, fformat: Tuple[str]):
        flag = [1 if name.lower().endswith(f) else 0 for f in fformat]
        return sum(flag) > 0

    @torch.no_grad()
    def run_function(self, file_names):
        cost_time = list()
        for file_name in tqdm.tqdm(file_names):
            img_name = file_name.split('/')[-1]

            img_bgr = cv2.imread(file_name, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            result = dict(
                img=img_rgb,
                img_shape=img_rgb.shape[:2],
                img_fields=['img'],
                color_fields=['img'],
            )

            for t in self.transform:
                result = t(result)

            img = F.to_tensor(result['img']).to(self.device)

            start_time = time.time() * 1000
            logits = self.model(img.unsqueeze(0))

            cost_time.append(time.time() * 1000 - start_time)

            if isinstance(logits, list):
                logits = logits[0]

            assert logits.shape[0] == logits.shape[1] == 1
            mask = logits[0][0].mul(255).clamp_(0, 255).to('cpu').detach().numpy().astype(np.uint8)

            h, w, c = img_rgb.shape
            mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            if self.show_mask:
                arr = img_name.split('.')
                cv2.imwrite(f'{self.out_path}/{arr[0]}_1.png', mask)
                continue

            weight = np.zeros_like(img_bgr)
            weight[:, :, 0] = mask
            overlapping = cv2.addWeighted(img_bgr, 0.65, weight, 0.35, 0)
            weight[:, :, 1] = mask
            weight[:, :, 2] = mask
            overlapping = np.where(weight > 0, overlapping, img_bgr)
            cv2.imwrite(f'{self.out_path}/{img_name}', overlapping)

        print('cost time {}'.format(sum(cost_time) / (len(cost_time) + 1)))
        return

    def __call__(self, source:str, fformat:Tuple[str]):

        if self.is_jpg(source, fformat):
            file_names = [source]
        else:
            file_names = [f'{source}/{name}' for name in os.listdir(source) if self.is_jpg(name, fformat) and not name.startswith('face')]

        self.run_function(file_names)
        return

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, help='model path')
    parser.add_argument('--source', type=str, help='file/dir')
    parser.add_argument('--config', type=str, default='', help='如果是onnx 则这个可以不填')
    parser.add_argument('--imgsz', type=int, default=608, help='inference size h,w')
    parser.add_argument('--out-path', type=str, help='out path')
    parser.add_argument('--fformat', type=str, nargs='+', default=['jpg', 'png'], help='out path')
    parser.add_argument('--show-mask', action='store_true', help='where to save mask')
    parser.add_argument('--num_classes', type=int, help='segment class', default=10)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    container = Segment(img_size=opt.imgsz, out_path=opt.out_path, weight=opt.weight, config=opt.config, show_mask=opt.show_mask)
    container(source=opt.source,  fformat=opt.fformat)

