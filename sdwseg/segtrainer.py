from engine.trainer.trainer import BaseTrainer
import logging
from engine.trainer.build import BUILD_TRAINER_REGISTRY
import torch
import cv2
import numpy as np
import torchvision
from util import colors


@BUILD_TRAINER_REGISTRY.register()
class SegTrainer(BaseTrainer):

    def __init__(self, cfg):
        super(SegTrainer, self).__init__(cfg)
        return

    def after_loop(self):
        self.model.disable_train()

        for i, batch in enumerate(self.test_data_loader):
            result = self.model(batch)
            device = batch['img'].device
            seg_pred = result['seg_pred'].to(device)
            gt_semantic_seg = batch['gt_semantic_seg']
            seg_acc = result['seg_acc'].item()
            assert seg_pred.shape[1] == 1 and gt_semantic_seg.shape[1] == 1
            if result['classes'] == 1:
                seg_pred = result['seg_pred'].to(device)
                img_sample = torch.cat((batch['img'], seg_pred.repeat(1, 3, 1, 1), gt_semantic_seg.repeat(1, 3, 1, 1)), -1)
                self.save_image(img_sample, '{}/{}.jpg'.format(self.output, i), nrow=1, normalize=False)
            else:
                gt_semantic_seg = gt_semantic_seg[:, 0, ...]
                seg_pred = seg_pred[:, 0, ...]
                fake_seg = torch.zeros_like(batch['img'])
                gt_seg = torch.zeros_like(batch['img'])
                gt_idx = torch.unique(gt_semantic_seg).detach().cpu().tolist()
                fake_idx = torch.unique(seg_pred).detach().cpu().tolist()
                idx = set(gt_idx + fake_idx)
                for c in idx:
                    if c == 0:
                        continue
                    r, g, b = colors[c]['rgb']
                    gt_seg[:, 0, ...][gt_semantic_seg == c] = r
                    fake_seg[:, 0, ...][seg_pred == c] = r
                    gt_seg[:, 1, ...][gt_semantic_seg == c] = g
                    fake_seg[:, 1, ...][seg_pred == c] = g
                    gt_seg[:, 2, ...][gt_semantic_seg == c] = b
                    fake_seg[:, 2, ...][seg_pred == c] = b
                img_sample = torch.cat((batch['img'].mul(255).add_(0.5).clamp_(0, 255), fake_seg, gt_seg), -1)
                self.save_image(img_sample, '{}/{}.jpg'.format(self.output, i), False, nrow=1, normalize=False)
            logging.getLogger(self.default_log_name).info('test acc = {}'.format(seg_acc))
        return

    def iterate_after(self, epoch):
        if int(epoch + 0.5) % self.checkpoint.check_period == 0:
            acc = [0.]
            self.model.disable_train()
            with torch.no_grad():
                for i, batch in enumerate(self.test_data_loader):
                    acc.append(self.model(batch)['seg_acc'].item())
            self.model.enable_train()

            logging.getLogger(self.default_log_name).info('trainer run step {} acc = {}'.format(epoch, sum(acc) / len(acc)))
        return

    @torch.no_grad()
    def save_image(self, tensor: torch.Tensor, fp: str, to_int=True, **kwargs):
        grid = torchvision.utils.make_grid(tensor, **kwargs)
        # Add 0.5 after unnormalizing to [0, unnormalizing_value] to round to nearest integer
        if to_int:
            grid = grid.mul(255).add_(0.5).clamp_(0, 255)
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy().astype(np.uint8)
        cv2.imwrite(fp, ndarr[:, :, ::-1])
        return
