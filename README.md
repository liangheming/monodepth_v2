# Monodepth2
注:不想写英文了,就用中文.如果看不懂中文的请移步GOOGLE翻译(if you can't read chinese,please refer to Google Translation).
## 1.简介
一种基于自监督方式的深度估计算法(论文 [Monodepth2](https://arxiv.org/abs/1806.01260) ),支持单目图像序列或者双目图像,训练过程不需要深度真值,但是仅基于单目图像序列的单目深度估计结果将缺少尺度信息.

## 2.环境

- Pillow 9.5.0
- pytorch 2.0.0
- torchvision 0.15.1
- tensorboardX 2.6
- pytorch-lightning 2.0.1
- opencv-python 4.7.0.72 (仅用于显示)

## 3. KITTI数据集
请首先参考 [monodepth2](https://github.com/nianticlabs/monodepth2) 进行数据准备,后面我会上传一个缩略版本的数据集,敬请期待.

## API

- 训练
```shell script
python train.py --model_name M_640x192 --kitti_dir 'your kitti img dir' --split eigen_zhou --strategy M --frame_ids -1,0,1 --width 640 --height 192 #单目
python train.py --model_name S_640x192 --kitti_dir 'your kitti img dir' --split eigen_full --strategy S --frame_ids -1,0,1 --width 640 --height 192 #双目
python train.py --model_name MS_640x192 --kitti_dir 'your kitti img dir' --split eigen_zhou --strategy MS --frame_ids -1,0,1 --width 640 --height 192 #单目和双目
```
**如果有深度真值,可以将真值文件 gt_depths.npz 放在 split/eigen_zhou或者eigen_full目录下,这样在训练过程中,验证过程将会以评估指标值(默认为损失值)进行输出**

- 验证
```shell script
python evaluation.py --model_name M_640x192 --kitti_dir 'your kitti img dir' --strategy M --width 640 --height 192 --cuda 1 --ckpt 'your weight path' --split_path 'test txt path' --gt_path 'gt npy path' #单目
python evaluation.py --model_name M_640x192 --kitti_dir 'your kitti img dir' --strategy S --width 640 --height 192 --cuda 1 --ckpt 'your weight path' --split_path 'test txt path' --gt_path 'gt npy path' #双目
python evaluation.py --model_name M_640x192 --kitti_dir 'your kitti img dir' --strategy MS --width 640 --height 192 --cuda 1 --ckpt 'your weight path' --split_path 'test txt path' --gt_path 'gt npy path' #双目
```

## 4.复现指标

论文中指标

| 模型 | 输出尺寸 | Abs Rel | Sq Rel | RMSE | RMSE log | δ<1.25 | δ<1.25^2 | δ<1.25^3 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|M | 640x192 | 0.115 | 0.903 | 4.863 | 0.193 | 0.877 | 0.959 | 0.981 |
|S | 640x192 | 0.109 | 0.873 | 4.960 | 0.209 | 0.864 | 0.948 | 0.975 |
|MS | 640x192 | 0.106 | 0.818 | 4.750 | 0.196 | 0.874 | 0.958 | 0.980 |

该项目复现指标

| 模型 | 输出尺寸 | Abs Rel | Sq Rel | RMSE | RMSE log | δ<1.25 | δ<1.25^2 | δ<1.25^3 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|M | 640x192 | 0.117 | 0.847 | 4.841 | 0.195 | 0.871 | 0.959 | 0.981 |
|S | 640x192 | 0.109 | 0.882 | 4.972 | 0.207 | 0.867 | 0.950 | 0.975 |
|MS | 640x192 | 0.109 | 0.836 | 4.814 | 0.198 | 0.868 | 0.956 | 0.980 |

官方开源项目 [monodepth2](https://github.com/nianticlabs/monodepth2)

| 模型 | 输出尺寸 | Abs Rel | Sq Rel | RMSE | RMSE log | δ<1.25 | δ<1.25^2 | δ<1.25^3 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|M | 640x192 | 0.118 | 0.884 | 4.898 | 0.197 | 0.870 | 0.958 | 0.980 |

其他开源项目 [Paddle-MonoDept2](https://aistudio.baidu.com/aistudio/projectdetail/3399869)

| 模型 | 输出尺寸 | Abs Rel | Sq Rel | RMSE | RMSE log | δ<1.25 | δ<1.25^2 | δ<1.25^3 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|MS | 640x192 | 0.109 | 0.880 | 4.903 | 0.200 | 0.868 | 0.955 | 0.979 |


**注: 与δ相关的项目越高越好**

- 论文中提供的开源代码无法完全复现论文中的指标,即便完全基于开源代码训练,性能评估结果整体上都要比论文中的低(甚至低于该项目的结果)
- 该项目主要使用3080训练, 单双目融合的训练由于显存问题只能用batch_size=8进行训练(不知道batch_size是否会对性能产生本质影响)

## 5.参考
```text
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
```

