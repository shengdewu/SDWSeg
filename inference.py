import cv2
import numpy as np

import onnxruntime
from typing import Tuple, Sequence, Union


class PyOnnx:
    def __init__(self, onnx_path):
        # model = onnx.load(onnx_path)
        # onnx.checker.check_model(model)
        # print(onnx.helper.printable_graph(model.graph))

        self.ort_session = onnxruntime.InferenceSession(onnx_path,
                                                        providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        self.onnx_input_names = [o_input.name for o_input in self.ort_session.get_inputs()]
        self.onnx_output_names = [o_output.name for o_output in self.ort_session.get_outputs()]
        return

    def __call__(self, input_buffer: Union[Tuple[np.ndarray], np.ndarray]) -> Sequence[np.ndarray]:
        input_param = dict()
        if isinstance(input_buffer, np.ndarray):
            assert len(self.onnx_input_names) == 1
            input_param[self.onnx_input_names[0]] = input_buffer
        else:
            assert len(input_buffer) == len(self.onnx_input_names)
            for i in range(len(self.onnx_input_names)):
                input_param[self.onnx_input_names[i]] = input_buffer[i]

        outputs = self.ort_session.run(None, input_param)
        return outputs


class SegmentInference:
    """分割推理接口"""

    def __init__(self, onnx_path: str, img_size: int, n_classes: int):
        """
        初始化推理接口

        Args:
            onnx_path: 模型onnx文件路径
            img_size: 模型的输入大小
            n_classes:类别数量
        """

        self.model = PyOnnx(onnx_path)
        self.img_size = img_size
        self.n_classes = n_classes
        return

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理, 图像只支持不保持比率缩放

        Args:
            image: 图像 bgr

        Returns:
            处理后的张量
        """
        array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        array = cv2.resize(array, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        array = array.astype(np.float32) / 255

        array = np.stack([array])
        array = np.transpose(array, (0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        return np.ascontiguousarray(array)

    @staticmethod
    def softmax(x: np.ndarray, axis=None) -> np.ndarray:
        """
        numpy 实现softmax
        Args:
            x: 输入数组
            axis: 执行操作的维度
        Returns:
            处理后的数组
        """
        # 为了数值稳定性，减去最大值
        x_max = np.max(x, axis=axis, keepdims=True)
        x = x - x_max

        # 计算指数
        exp_x = np.exp(x)

        # 计算分母
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)

        # 返回结果
        return exp_x / sum_exp

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        对输入图像进行分割预测

        Args:
            image: 图像数组bgr

        Returns:
            分割掩码（numpy数组）
        """
        array = self.preprocess(image)
        output = self.model(array)[0]

        return self.postprocess(output, image.shape[:2])

    def postprocess(self, mask: np.ndarray, shape: Tuple) -> np.ndarray:
        """
        对分割掩码进行后处理

        Args:
            mask: 模型输出的分割掩码
            shape: 图像原始大小 (h, w)

        Returns:
            处理后的分割掩码
        """
        assert mask.shape[0] == 1 and mask.shape[1] == 1
        mask = mask[0][0]
        # 调整大小到原始图像尺寸
        if self.n_classes < 2:
            mask = np.where(mask > 1e-3, mask, 0)

        return cv2.resize(mask, shape[::-1], interpolation=cv2.INTER_NEAREST)

    def visualize(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        可视化分割结果

        Args:
            image: 原始图像路径或numpy数组 [h, w, c]
            mask: 分割掩码 [h, w]

        Returns:
            可视化结果（numpy数组）
        """
        assert self.n_classes < 255
        mask = np.clip(mask * (255 // self.n_classes), 0, 255).astype(np.uint8)

        # 创建彩色掩码
        weight = np.zeros_like(image)
        weight[:, :, 0] = mask
        overlay = cv2.addWeighted(image, 0.65, weight, 0.35, 0)
        weight[:, :, 1] = mask
        weight[:, :, 2] = mask
        overlay = np.where(weight > 0, overlay, image)

        return overlay


# 使用示例
if __name__ == "__main__":
    # 初始化推理接口
    inference = SegmentInference('/mnt/sda/train.output/sdwseg/pdxlk-false/RegSegEncoder-RegSegDecoder_final.onnx', 576, 1)

    # 对单张图像进行推理
    image_path = "/mnt/sda/datasets/pidai-xlk-seg/75.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = inference.predict(image)

    # 可视化结果
    result = inference.visualize(image, mask)

    # 保存结果
    cv2.imwrite("segmentation_result.jpg", result)

    print("分割完成，结果已保存")
