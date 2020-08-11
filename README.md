# 使用MTCNN进行人脸检测

1.使用前先在`main.py`的目录下创建`test_Imgs`文件夹，并将图片放在该文件夹下。

2.使用`train_pnet.py`,`train_rnet.py`,`train_onet.py`对三个网络进行训练。并修改如下变量值。

```python
 save_path="saved_models/pnet.pt" #模型保存位置
 dataset_path=r"path\to\data" #数据保存的位置
 sum_path='./pnet_logs' #tensorboard数据保存位置
```

3.训练数据使用`CelebA`数据集，并使用`ImageCrop.py`生成，使用时需要对文件路径进行修改。

4.使用时执行`main.py`即可

5.使用的结果

**![结果](https://github.com/Lizi2hua/MTCNN/blob/master/test_Imgs/u%3D123579553%2C2548320036%26fm%3D26%26gp%3D0.jpg)**

