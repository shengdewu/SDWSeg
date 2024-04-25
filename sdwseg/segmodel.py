from engine.model.base_model import BaseModel
from engine.model.build import BUILD_MODEL_REGISTRY


@BUILD_MODEL_REGISTRY.register()
class SegModel(BaseModel):
    def __init__(self, cfg):
        super(SegModel, self).__init__(cfg)
        return

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        此方法必须实现
        """
        img = data['img'].to(self.device, non_blocking=True)
        gt_semantic_seg = data['gt_semantic_seg'].to(self.device, non_blocking=True)
        loss, acc = self.g_model.forward_train(img, gt_semantic_seg)
        return dict(loss=loss, acc=acc)

    def generator(self, data):
        """
        此方法必须实现
        """
        img = data['img'].to(self.device, non_blocking=True)
        gt_semantic_seg = data['gt_semantic_seg'].to(self.device, non_blocking=True)
        result = self.g_model.forward_test(img, gt_semantic_seg)
        return result
