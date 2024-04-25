from engine.model.build import BUILD_NETWORK_REGISTRY
from engine.model.build import build_network
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Union, List, Tuple
from sdwseg.model.utils.accuracy import accuracy

__all__ = [
    'EncoderDecoder'
]


@BUILD_NETWORK_REGISTRY.register()
class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Dict,
                 decoder: Dict,
                 necker: Optional[Dict] = None,
                 auxiliary: Optional[Union[List[Dict], Dict]] = None,
                 ignore_index: int = None):
        super(EncoderDecoder, self).__init__()
        self.encoder = build_network(encoder)
        self.decoder = build_network(decoder)
        self.with_neck = False
        if necker is not None:
            self.necker = build_network(necker)
            self.with_neck = True
        self.with_auxiliary = False
        if auxiliary is not None:
            self.with_auxiliary = True
            if isinstance(auxiliary, list):
                self.auxiliary = nn.ModuleList()
                for config in auxiliary:
                    self.auxiliary.append(build_network(config))
            else:
                self.auxiliary = build_network(auxiliary)
        self.ignore_index = ignore_index
        return

    @property
    def model_name(self):
        return self.encoder.__class__.__name__ + '-' + self.decoder.__class__.__name__

    def extract_feature(self, img: Tensor):
        x = self.encoder(img)
        if self.with_neck:
            x = self.necker(x)
        return x

    def forward_train(self, img: Tensor, gt_semantic_seg: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.extract_feature(img)
        loss, acc = self.decoder.forward_train(x, img.shape[2:], gt_semantic_seg)
        if self.with_auxiliary:
            if isinstance(self.auxiliary, nn.ModuleList):
                for idx, aux in enumerate(self.auxiliary):
                    loss_aux, _ = aux.forward_train(x, img.shape[2:], gt_semantic_seg)
                    loss += loss_aux
            else:
                loss_aux, _ = self.auxiliary.forward_train(x, img.shape[2:], gt_semantic_seg)
                loss += loss_aux
        return loss, acc

    def forward_test(self, img: Tensor, gt_semantic_seg: Tensor) -> Dict:
        x = self.extract_feature(img)
        seg_pred = self.decoder(x, img.shape[2:])

        result = dict()
        if seg_pred.shape[1] == 1:
            result['seg_pred'] = F.sigmoid(seg_pred)
            result['classes'] = 1
        else:
            result['seg_pred'] = seg_pred.argmax(dim=1).unsqueeze(1)
            result['classes'] = seg_pred.shape[1]

        gt_semantic_seg = gt_semantic_seg.squeeze(1)
        result['seg_acc'] = accuracy(
            seg_pred, gt_semantic_seg, ignore_index=self.ignore_index)

        return result

    def forward(self, img: Tensor):
        x = self.extract_feature(img)
        seg_pred = self.decoder(x, img.shape[2:])

        if seg_pred.shape[1] == 1:
            seg_pred = F.sigmoid(seg_pred)
        else:
            seg_pred = seg_pred.argmax(dim=1).unsqueeze(1)
        return seg_pred

    def preparate_deploy(self):
        if self.with_auxiliary:
            delattr(self, 'auxiliary')
            self.with_auxiliary = False
        self.decoder.preparate_deploy()
        return
