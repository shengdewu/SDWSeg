# 语义分割框架  

- 基于 [TrainFramework](https://github.com/shengdewu/TrainFramework.git)实现的分割框架  

- 配置遵循组合原则  

- 本文的重点在[模型设计与实现](doc/network.md)    

<br>  

# 目标  

 提供通用的使用方式，让用户专注于数据的处理，使用用户能够快速的输出结果  

 <br>

# 使用

**第一步** 按照[TrainFramework](https://github.com/shengdewu/TrainFramework.git)编译训练引擎

**第二步** 
- `docker`  
    - 安装 [docker](https://docs.docker.com/engine/install/ubuntu/)  

    - 按照[TrainFramework](https://github.com/shengdewu/TrainFramework.git)编译基础镜像  

    - 拷贝基础镜像到 工程目录 whl 中  

        ```none
        cp TranFramework/dist/engine_frame-xx.whl SDWSeg/whl
        ``` 

-  `ubuntu`  
    - 使用python 按照[TrainFramework](https://github.com/shengdewu/TrainFramework.git)中的`依赖`安装依赖  

        ```none
        pip3 install -r docker/requirements.txt
        ```  

    - 安装训练引擎  

        ```none
        pip3 install engine_frame-xx.whl
        ```        

**第三步**  

- [准备数据](doc/data.md)   

- [准备网络](doc/network.md)  

- 准备模型 根据TrainFramework的[简单的使用](https://github.com/shengdewu/TrainFramework.git/blob/master/README.md) 自定义模型, 不要在构造函数里创建损失函数, 在`create_model`里创建`网络`  

- 准备优化器、学习率调度器, 配置参考[base](config/base/schedule/sgd_poly.py), 详情参考[pytorch](https://pytorch.org/docs/1.10/optim.html) 


**第四步**  

- `docker`  

    - 编译训练镜像  

        ```none
        docker build  ./ -f docker/Dockerfile -t seg:1.0
        ```
    - 训练  根据 [TrainFramework](https://github.com/shengdewu/TrainFramework.git/blob/master/doc/config.md) 配置中的运行参数指定需要的参数  

        ```none
            docker run --gpus='"device=0"' --shm-size=20g -v /mnt:/mnt -t train --config-file /mnt/config/train.py --num-gpus 1
        ``` 

- `ubuntu`   

    - 训练  根据 [TrainFramework](https://github.com/shengdewu/TrainFramework.git/blob/master/doc/config.md) 配置中的运行参数指定需要的参数  

        ```none
            python3 train_tool.py --config-file /mnt/config/train.py --num-gpus 1
        ```

<br>  

# 部署

## `量化`  

- 离线量化、 在线量化  

- 加快快推理速度  
访问一次 32 位浮点型可以访问四次 int8 整型，整型运算比浮点型运算更快；CPU 用 int8 计算的速度更快  

- 减小模型大小  
如 int8 量化可减少 75% 的模型大小，int8 量化模型大小一般为 32 位浮点模型大小的 1/4  

    - 减少存储空间：在端侧存储空间不足时更具备意义  

    - 减少内存占用：更小的模型当然就意味着不需要更多的内存空间  

    - 减少设备功耗：内存耗用少了推理速度快了自然减少了设备功耗  

- 量化又可以分为权重量化，权重和激活量化  

- 量化公式  

    假设`x_f`的数据类型的取值范围是`[f_min, f_max]`,则值域是 `f_range = f_max - f_min`  

    量化`x_q`的的位数是`N`, 则量化的范围是  
    - 不对称: `[0, 2^N - 1], l_bound=0, u_bound=2^N-1` 
    - 对称: `[-2^(N-1), 2^(N-1) - 1], l_bound=-2^(N-1), u_bound=2^(N-1) - 1`  

    则值域是 `q_range = 2 ^ N - 1`   

    `scale = f_range / q_range`

    则偏移 `offset = round(0 - f_min / scale)`  

    x_int =  round(x_f / scale) + offset  

    则量化结果 `x_q = clamp(x_int, l_bound, u_bound)`   

- `本文采用离线量化`   

<br>  

## 剪枝  

依赖`模型拥有很大的参数冗余` 

- 结构化剪枝`[本文实现]`  

- 非结构化剪枝  

    裁剪粒大更大，但是依赖特殊硬件 

- 实现遍历网络的拓扑结构方法  

- 通过BN层或者Conv层权重的幅度来裁剪掉不重要的权重  

## 模型打包 

#### 打包成`so`文件  
 
- 使用 __Cython__

1. 把文件的后缀替换成 `.pyx`  
  
2. 文件结构如下  

    ```text
    |---package
           |---- my.py
           |---- my.pyx
    
    ```  
   ```python
       class MyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            stem = ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
            head = ConvBNReLU(in_channels=32, out_channels=num_classes, kernel_size=3, stride=2, padding=1)
            return
    
        def forward(self, x):
            x = self.stem(x)
            x = self.head(x)
            return x
    
        def __repr__(self):
            return self.__class__.__name__ 
   
       class Excute:
        def __init__(self, model_path:str):
            self.img_size = 608
            self.model = MyModel(1)
            self.model.eval()    
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
    
            return
    
        def __call__(self, img_rgb:np.ndarray):
            if img_rgb.ndim != 3:
                raise RuntimeError('the image shape must by [h, w, c]')
    
            if img_rgb.dtype != np.dtype("uint8"):
                raise RuntimeError('the image type [np.uint8]')
    
            infer_img = cv2.resize(img_rgb, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            infer_img = torch.from_numpy(infer_img.transpose((2, 0, 1))).contiguous()
            infer_img = infer_img.to(dtype=torch.float32).div(255)
    
            logits = self.model(infer_img.unsqueeze(0))
            logits = logits[0][0].float().mul(255).clamp_(0, 255).to('cpu').detach().numpy().astype(np.uint8)
            fake_mask = cv2.resize(logits, dsize=(img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            return fake_mask
   
   
    ```

3. 创建`setup.py`文件  

    ```python
    
    from setuptools import setup, Extension
    from Cython.Build import cythonize
    
    
    ext_modules = [
        Extension('package', sources=['package/my.pyx'])
    ]
    
    setup(
        name='package_cython',
        ext_modules=cythonize(ext_modules),
    )
    
    ```  

4. 执行命令  
    ```python
    
     python3 setup.py build_ext  --inplace
    
    ```   
5. 生成文件目录 `build`  

    ```text
    
    |----build
          |---- lib.linux-x86_64-3xx
                    |---- package.cpython-3xm-x86_64-linux-gnu.so
          |---- temp.linux-x86_64-3xx
    ```      

6. 使用   
    
    ```text
    
    |--project
       |---package.cpython-3xm-x86_64-linux-gnu.so
       |--- test.py
       |--- model.pth
       |--- test.jpg
    ```
   ```python

    import cv2

    if __name__ == '__main__':
    
        #1. 初始化模型
        import package
        model = package.Excute('model.pth')
    
        # 2. 图片转换成rgb格式
        img_bgr = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
        # 3. 模型推理
        mask = model(img_rgb)
        cv2.imwrite('test-mask.jpg', mask)
    ```
