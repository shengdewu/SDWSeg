import os
import time
import cv2
import torch
import tqdm
import numpy as np
import sys


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import engine.transforms.functional as F
from tool.segment import Segment, parse_opt
from util.colors import colors


class MultSegment(Segment):
    def __init__(self, img_size, out_path, num_classes, weight:str, config:str, device='cpu',show_mask=False, keep_ratio=False):
        super(MultSegment, self).__init__(img_size, out_path, weight, config, device,show_mask, keep_ratio)
        self.idx = [i for i in range(num_classes)]
        return

    @torch.no_grad()
    def run_function(self, file_names):
        cost_times = list()
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

            start_time = time.time() * 1000
            img = F.to_tensor(result['img']).to(self.device)
            logits = self.model(img.unsqueeze(0))
            cost_times.append(time.time() * 1000 - start_time)

            if isinstance(logits, list):
                logits = logits[0]

            assert logits.shape[0] == logits.shape[1] == 1
            mask = logits[0][0].to('cpu').detach().numpy().astype(np.uint8)

            h, w, c = img_rgb.shape
            mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

            mask_mult = np.zeros_like(img_rgb)

            for c in self.idx:
                if c == 0:
                    continue
                r, g, b = colors[c]['rgb']
                mask_mult[..., 2][mask == c] = r
                mask_mult[..., 1][mask == c] = g
                mask_mult[..., 0][mask == c] = b

            if self.show_mask:
                arr = img_name.split('.')
                cv2.imwrite(f'{self.out_path}/{arr[0]}_1.png', mask_mult)
                continue

            overlapping = cv2.addWeighted(img_bgr, 0.65, mask_mult, 0.35, 0)
            overlapping = np.where(mask_mult > 0, overlapping, img_bgr)
            cv2.imwrite(f'{self.out_path}/{img_name}', overlapping)

        print('cost time {}'.format(sum(cost_times) / (len(cost_times) + 1)))
        return


if __name__ == '__main__':
    opt = parse_opt()
    container = MultSegment(opt.imgsz, opt.out_path, opt.num_classes, weight=opt.weight, config=opt.config, show_mask=opt.show_mask)
    container(source=opt.source, fformat=opt.fformat)

