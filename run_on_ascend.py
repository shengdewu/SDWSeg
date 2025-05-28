import acl
import cv2
import numpy as np
import sys


# 封装acl模型推理过程为OMNet类
class OMNet:
    ACL_MEM_MALLOC_HUGE_FIRST = 0
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self, model_path, device_id, out_type):
        # 初始化函数
        self.device_id = device_id

        # step1: 初始化
        ret = acl.init()
        assert ret == 0, f"初始化失败: {ret}"
        # 指定运算的Device
        ret = acl.rt.set_device(self.device_id)
        assert ret == 0, f"设置device{self.device_id}失败: {ret}"

        # step2: 加载模型
        # 加载离线模型文件，返回标识模型的ID        
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        assert ret == 0, f"加载模型失败 {model_path}: {ret}"

        # 创建空白模型描述信息，获取模型描述信息的指针地址
        self.model_desc = acl.mdl.create_desc()
        # 通过模型的ID，将模型的描述信息填充到model_desc
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        # step3：创建输入输出数据集
        # 创建输入数据集
        self.input_dataset, self.input_data = self.prepare_dataset('input')
        # 创建输出数据集
        self.output_dataset, self.output_data = self.prepare_dataset('output')

        self.out_type = out_type
        return

    def prepare_dataset(self, io_type):
        # 准备数据集
        if io_type == "input":
            # 获得模型输入的个数
            io_num = acl.mdl.get_num_inputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_input_size_by_index
        else:
            # 获得模型输出的个数
            io_num = acl.mdl.get_num_outputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_output_size_by_index
        assert io_num > 0, f"获取模型{io_type}输入个数失败: {io_num}"
        # 创建aclmdlDataset类型的数据，描述模型推理的输入。
        dataset = acl.mdl.create_dataset()
        datas = []
        for i in range(io_num):
            # 获取所需的buffer内存大小
            buffer_size = acl_mdl_get_size_by_index(self.model_desc, i)
            # 申请buffer内存
            buffer, ret = acl.rt.malloc(buffer_size, self.ACL_MEM_MALLOC_HUGE_FIRST)
            assert ret == 0, f"申请{io_type}buffer内存出错: {ret}"
            # 从内存创建buffer数据
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            # 将buffer数据添加到数据集
            _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
            assert ret == 0, f"将{io_type}buffer添加到数据集出错: {ret}"

            datas.append({"buffer": buffer, "data": data_buffer, "size": buffer_size})
        return dataset, datas

    def forward(self, inputs):
        # 执行推理任务
        # 遍历所有输入，拷贝到对应的buffer内存中
        copy_success = True
        input_num = len(inputs)
        for i in range(input_num):
            bytes_data = inputs[i].tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            if len(bytes_data) != self.input_data[i]["size"]:
                print(f"输入数据的大小: {len(bytes_data)} 和模型的输入大小 {self.input_data[i]['size']} 不匹配")
                copy_success = False
                break

            # 先清空buffer
            ret = acl.rt.memset(self.input_data[i]["buffer"], self.input_data[i]["size"], 0, 2)
            if ret != 0:
                print(f"清空输入buffer失败: {ret}")
                copy_success = False
                break

            # 将数据从Host传输到Device。
            ret = acl.rt.memcpy(self.input_data[i]["buffer"],  # 目标地址 device
                                self.input_data[i]["size"],  # 目标地址大小
                                bytes_ptr,  # 源地址 host
                                len(bytes_data),  # 源地址大小
                                self.ACL_MEMCPY_HOST_TO_DEVICE)  # 模式:从host到device
            if ret != 0:
                print(f"拷贝输入数据到device失败: {ret}")
                copy_success = False
                break

        inference_result = []

        if not copy_success:
            return inference_result

        # 执行模型推理。
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            print(f"模型推理失败: {ret}")
            return inference_result

        # 处理模型推理的输出数据。
        inference_result = []
        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            if ret != 0:
                print(f"分配host memory 失败: {ret}")
                continue

            # 将推理输出数据从Device传输到Host。
            ret = acl.rt.memcpy(buffer_host,  # 目标地址 host
                                self.output_data[i]["size"],  # 目标地址大小
                                self.output_data[i]["buffer"],  # 源地址 device
                                self.output_data[i]["size"],  # 源地址大小
                                self.ACL_MEMCPY_DEVICE_TO_HOST)  # 模式：从device到host
            if ret != 0:
                print(f"拷贝数据到host失败: {ret}")
                continue

            # 从内存地址获取bytes对象
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            if len(bytes_out) != self.output_data[i]["size"]:
                print(f"输入数据的大小({len(bytes_out)})和模型定义输出数据的大小({self.output_data[i]['size']})不匹配")
                continue
            # 按照设定的dtype格式将数据转为numpy数组
            data = np.frombuffer(bytes_out, dtype=self.out_type)
            inference_result.append(data)
            # 释放内存
            ret = acl.rt.free_host(buffer_host)
            if ret != 0:
                print(f"释放host内存失败: {ret}")
        # vals = np.array(inference_result).flatten()
        return inference_result

    def __del__(self):
        # 析构函数 按照初始化资源的相反顺序释放资源。
        # 销毁输入输出数据集
        for dataset in [self.input_data, self.output_data]:
            while dataset:
                item = dataset.pop()
                ret = acl.destroy_data_buffer(item["data"])  # 销毁buffer数据
                ret = acl.rt.free(item["buffer"])  # 释放buffer内存
        ret = acl.mdl.destroy_dataset(self.input_dataset)  # 销毁输入数据集
        ret = acl.mdl.destroy_dataset(self.output_dataset)  # 销毁输出数据集
        # 销毁模型描述
        ret = acl.mdl.destroy_desc(self.model_desc)
        # 卸载模型
        ret = acl.mdl.unload(self.model_id)
        # 释放device
        ret = acl.rt.reset_device(self.device_id)
        # acl去初始化
        ret = acl.finalize()
        return
    

if __name__ == '__main__':

    mode_path = "/home/ckp/seg_bs1.om" 

    net = OMNet(mode_path, device_id=0, out_type=np.int64)

    file_name = sys.argv[1] #"./JinLiDupic71.jpg"

    img_bgr = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, dsize=(800, 800), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255

    in_np = np.stack([img])
    in_np = in_np.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    in_np = np.ascontiguousarray(in_np)
    
    logits = net.forward(in_np)
    if len(logits) == 0:
        print('输出为空')
        raise RuntimeError('输出为空')

    num_classes = 8
    logits = np.array(logits, dtype=np.int64)
    logits = logits.reshape(1, 1, 800, 800)
    
    mask = np.clip(logits[0][0] * (255 // num_classes), 0, 255).astype(np.uint8)

    h, w, c = img_rgb.shape
    mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite('/output/mask.png', mask)

    weight = np.zeros_like(img_bgr)
    weight[:, :, 0] = mask
    overlapping = cv2.addWeighted(img_bgr, 0.65, weight, 0.35, 0)
    weight[:, :, 1] = mask
    weight[:, :, 2] = mask
    overlapping = np.where(weight > 0, overlapping, img_bgr)
    cv2.imwrite('/output/demo.png', overlapping)
