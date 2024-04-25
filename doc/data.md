#  配置数据集 

## 组织自己的数据结构  

如下是一个数据的文件结构
```none
|---data
    |---image1
    |   |---xxxx[img_suffix]
    |   |---yyyy[img_suffix]
    |   |---zzzz[img_suffix]
    |
    |---image2
    |   |---aaaa[img_suffix]
    |   |---bbbb[img_suffix]
    |   |---cccc[img_suffix] 
    |   
    |---image3
    |   |---eeee[img_suffix]
    |   |---ffff[img_suffix]
    |   |---jjjj[img_suffix]  
    |   
    |---mask1
    |   |---xxxx[img_suffix]
    |   |---yyyy[img_suffix]
    |   |---zzzz[img_suffix]
    |
    |---mask2
    |   |---aaaa[img_suffix]
    |   |---bbbb[img_suffix]
    |   |---cccc[img_suffix] 
    |
    |---mask3
    |   |---eeee[img_suffix]
    |   |---ffff[img_suffix]
    |   |---jjjj[img_suffix] 
    |     
    |---1.txt
    |   
    |---2.txt
    |
    |---3.txt
```   
其中 1.txt的内容是
```none
image1/xxxx[img_suffix],mask1/xxxx[img_suffix]
image1/yyyy[img_suffix],mask1/yyyy[img_suffix]
image1/zzzz[img_suffix],mask1/zzzz[img_suffix]
```   
其中 2.txt的内容是
```none
image2/xxxx[img_suffix],mask2/xxxx[img_suffix]
image2/yyyy[img_suffix],mask2/yyyy[img_suffix]
image2/zzzz[img_suffix],mask2/zzzz[img_suffix]
```   
注意: 标注跟图片是同样的形状`(H, W)`， 像素值是 `[0, num_classes]`, 其中0表示背景， num_classes >= 1  

<br>  

## 定义数据  
- 根据[TrainFramework](https://github.com/shengdewu/TrainFramework/blob/master/README.md)中`定义自己的数据`, 创建 `ComposeDataSet`

- 继承 [EngineDataSet](https://github.com/shengdewu/TrainFramework/blob/master/engine/data/dataset.py)  


- 示例  

    ```python
    from engine.data.dataset import EngineDataSet
    from engine.data.build import BUILD_DATASET_REGISTRY
    import engine.transforms.functional as F
    import cv2


    __all__ = [
        'ComposeDataSet'
    ]


    @BUILD_DATASET_REGISTRY.register()
    class ComposeDataSet(EngineDataSet):
    
        def __init__(self, my_param, transformer:List):
            super(ComposeDataSet, self).__init__(transformer)
            return

        def __getitem__(self, index):
            img_path, target_data = self.dataset[index]
            img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            results = dict()
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = results['img_shape']
            results['img_fields'] = ['img']
            results['color_fields'] = ['img']

            results = self.data_pipeline(results)

            return {'input_data': F.to_tensor(results['img']), 'target_data': target_data}

        def __len__(self):
            return len(self.dataset)

    ```


<br>  

## 配置数据  

#### 第一步 选择[数据增强](https://github.com/shengdewu/TrainFramework/blob/master/doc/data_aug.md)  

- 训练增强
```python
t_transformer = [
    dict(
        name='RandomFlip',
        direction=['horizontal', 'vertical'],
        p=0.3,
    ),
    dict(
        name='RandomAffine',
        rotate_degree_range=[degree for degree in range(-90, 90, 10)],
        rotate_range=False,
        border_val=0,
        p=0.1,
    ),
    dict(name='RandomCrop',
         min_crop_ratio=0.05,
         max_crop_ratio=0.25,
         crop_step=0.05,
         p=0.8),
    dict(name='Resize',
         interpolation='INTER_LINEAR',
         target_size=img_size,
         keep_ratio=False,
         is_padding=False),
    dict(
        name='RandomCompress',
        quality_lower=75,
        quality_upper=99,
        quality_step=5,
        p=0.5),
    dict(name='RandomColorJitter',
         brightness_limit=[0.5, 1.2],
         brightness_p=0.6,
         contrast_limit=[0.7, 1.3],
         contrast_p=0.6,
         saturation_limit=[0.7, 1.4],
         saturation_p=0.6,
         hue_limit=0.1,
         hue_p=0.3,
         blur_limit=[3, 5],
         sigma_limit=0,
         blur_p=0.2,
         gamma_limit=[0.3, 2.0],
         gamma_p=0.2,
         clahe_limit=4,
         clahe_p=0.2),
    dict(name='ToGray',
         p=0.05),
]
```   
- 验证增强
```python
v_transformer = [
    dict(name='Resize',
         interpolation='INTER_LINEAR',
         target_size=img_size,
         keep_ratio=False,
         is_padding=False),
    dict(
        name='RandomCompress',
        quality_lower=75,
        quality_upper=99,
        quality_step=5,
        p=0.5),
]
```

#### 第二步 配置加载的数据集

- 训练数据
```python
train_data_set=dict(
    name='ComposeDataSet',
    data_root_path='/data',
    require_txt=['1.txt', '2.txt'],
    transformer=t_transformer,
    num_classes=num_classes,
)

```  
- 验证数据  
    - select_nums 可以指定使用验证的数据的最大数量
```python
val_data_set=dict(
    name='ComposeDataSet',
    select_nums=800,
    data_root_path='/data',
    require_txt=['3.txt'],
    transformer=v_transformer,
    num_classes=num_classes,
)
```

#### 第三步 最终配置   

- num_workers 数据处理的进程数  

```python
dataloader = dict(
    num_workers=num_workers,
    train_data_set=train_data_set,
    val_data_set=val_data_set
)
```  
