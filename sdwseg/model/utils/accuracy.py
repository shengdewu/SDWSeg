import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple, Optional


def compute_miou(p_tensor: Tensor,
                 t_tensor: Tensor,
                 ignore_index: Optional[int] = None,
                 thresh: float = 0.5):
    """
    iou = tp / tp + fn + fp : 真阳 / 真阳 + 假负(正确的预测成错误的) + 假阳(错误的预测成正确的)
    :param p_tensor: 语义分割模型的原始值，没有经过 argmax or sigmoid, shape (N, num_class, ...)
    :param t_tensor: 标签, shape (N, 1, ...) or (N, ...)
    :param ignore_index: 标签中需要忽略的值
    :param thresh:当预测类别执行sigmoid时使用
    :return float | tuple[float]
    """
    batch, num_classes, w, h = p_tensor.shape
    if num_classes > 1:
        predicts = p_tensor.argmax(dim=1)
    else:
        predicts = p_tensor[:, 0, ...].sigmoid()
        predicts = torch.where(predicts >= thresh, 1, 0)

    if t_tensor.ndim == p_tensor.ndim:
        assert t_tensor.shape[1] == 1
        targets = t_tensor[:, 0, ...]
    else:
        assert p_tensor.ndim == t_tensor.ndim + 1
        targets = t_tensor

    if ignore_index is not None:
        valid_mask = (targets != ignore_index).float()
        predicts = predicts * valid_mask
        targets = targets * valid_mask

    num_classes = num_classes + 1 if num_classes == 1 else num_classes
    predicts = predicts.long()
    targets = targets.long()

    confusion_matrix = torch.zeros((num_classes, num_classes), device=predicts.device)

    for b in range(targets.shape[0]):
        predict = predicts[b].flatten()
        target = targets[b].flatten()
        indices = num_classes * target + predict
        confusion_matrix += torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    iou = list()
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        union = confusion_matrix[i, :].sum() + confusion_matrix[:, i].sum() - intersection
        if union == 0:
            iou.append(torch.tensor(0.0, device=predicts.device))
        else:
            iou.append(intersection / union)

    return torch.mean(torch.tensor(iou))


def accuracy_multi_class(pred: Tensor, target: Tensor,
                         topk: Union[int, Tuple[int]] = 1,
                         thresh: Optional[float] = None,
                         ignore_index: Optional[int] = None) -> Union[float, Tuple[float]]:
    """
    :param pred: 语义分割模型的原始值，没有经过 argmax or sigmoid, shape (N, num_class, ...)
    :param target: 标签, shape (N, 1, ...) or (N, ...)
    :param topk:
    :param thresh: 预测结果的阈值， 低于这个值的认为不正确
    :param ignore_index: 标签中需要忽略的值
    :return float | tuple[float]
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu

    if target.ndim == pred.ndim:
        assert target.shape[1] == 1
        target = target[:, 0, ...]
    else:
        assert pred.ndim == target.ndim + 1

    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'

    pred = torch.softmax(pred, dim=1)

    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # 仅当预测值大于阈值thresh 被认为时正确
        correct = correct & (pred_value > thresh).t()
    if ignore_index is not None:
        correct = correct[:, target != ignore_index]
    res = []
    eps = torch.finfo(torch.float32).eps
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        if ignore_index is not None:
            total_num = target[target != ignore_index].numel() + eps
        else:
            total_num = target.numel() + eps
        res.append(correct_k.mul_(100.0 / total_num))
    return res[0] if return_single else res


def accuracy_single_class(pred: Tensor, target: Tensor,
                          ignore_index: Optional[int] = None) -> Union[float, Tuple[float]]:
    """
    :param pred: 语义分割模型的原始值，没有经过 argmax or sigmoid, shape (N, num_class, ...)
    :param target: 标签, shape (N, 1, ...) or (N, ...)
    :param ignore_index: 标签中需要忽略的值
    :return float | tuple[float]
    """

    if pred.size(0) == 0:
        return pred.new_tensor(0.)

    if target.shape == pred.shape:
        assert target.shape[1] == 1 and pred.shape[1] == 1
        target = target[:, 0, ...]
        pred = pred[:, 0, ...]
    else:
        assert pred.ndim == target.ndim + 1
        assert pred.shape[1] == 1
        pred = pred[:, 0, ...]

    assert pred.size(0) == target.size(0)

    pred = torch.sigmoid(pred)

    if ignore_index is not None:
        valid_mask = (target != ignore_index).float()
        pred = pred * valid_mask
        target = target * valid_mask

    tp = torch.sum(pred * target)
    tpfp1 = torch.sum(target, dtype=torch.float64)
    tpfp2 = torch.sum(pred, dtype=torch.float64)
    return (tp * 2 / (tpfp1 + tpfp2)).type_as(pred) * 100


def accuracy(pred: Tensor, target: Tensor,
             topk: Union[int, Tuple[int]] = 1,
             thresh: Optional[float] = None,
             ignore_index: Optional[int] = None):
    if pred.shape[1] > 1:
        return accuracy_multi_class(pred, target, topk, thresh,
                                    ignore_index)
    else:
        return accuracy_single_class(pred, target, ignore_index)


class Accuracy(nn.Module):
    """Accuracy calculation module."""

    def __init__(self, topk=(1,), thresh=None, ignore_index=None):
        super().__init__()
        self.topk = topk
        self.thresh = thresh
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        if pred.shape[1] > 1:
            return accuracy_multi_class(pred, target, self.topk, self.thresh,
                                        self.ignore_index)
        else:
            return accuracy_single_class(pred, target, self.ignore_index)


if __name__ == '__main__':
    # test for empty pred
    pred = torch.empty(0, 4)
    label = torch.empty(0)
    accuracy_fn = Accuracy(topk=1)
    acc = accuracy_fn(pred, label)
    assert acc.item() == 0

    pred = torch.Tensor([[0.2, 0.3, 0.6, 0.5], [0.1, 0.1, 0.2, 0.6],
                         [0.9, 0.0, 0.0, 0.1], [0.4, 0.7, 0.1, 0.1],
                         [0.0, 0.0, 0.99, 0]])
    # test for ignore_index
    true_label = torch.Tensor([2, 3, 0, 1, 2]).long()
    accuracy_fn = Accuracy(topk=1, ignore_index=None)
    acc = accuracy_fn(pred, true_label)
    assert torch.allclose(acc, torch.tensor(100.0))

    # test for ignore_index 1 with a wrong prediction of other index
    true_label = torch.Tensor([2, 0, 0, 1, 2]).long()
    accuracy_fn = Accuracy(topk=1, ignore_index=1)
    acc = accuracy_fn(pred, true_label)
    assert torch.allclose(acc, torch.tensor(75.0))

    # test for top1 with score thresh=0.8
    true_label = torch.Tensor([2, 3, 0, 1, 2]).long()
    accuracy_fn = Accuracy(topk=1, thresh=0.8)
    acc = accuracy_fn(pred, true_label)
    assert torch.allclose(acc, torch.tensor(40.0))

    # test for top2
    accuracy_fn = Accuracy(topk=2)
    label = torch.Tensor([3, 2, 0, 0, 2]).long()
    acc = accuracy_fn(pred, label)
    assert torch.allclose(acc, torch.tensor(100.0))

    # test for both top1 and top2
    accuracy_fn = Accuracy(topk=(1, 2))
    true_label = torch.Tensor([2, 3, 0, 1, 2]).long()
    acc = accuracy_fn(pred, true_label)
    for a in acc:
        assert torch.allclose(a, torch.tensor(100.0))

    # topk is larger than pred class number
    accuracy_fn = Accuracy(topk=5)
    accuracy_fn(pred, true_label)

    # wrong topk type
    accuracy_fn = Accuracy(topk='wrong type')
    accuracy_fn(pred, true_label)