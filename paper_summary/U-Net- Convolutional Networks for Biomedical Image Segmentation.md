







# U-Net: Convolutional Networks for Biomedical Image Segmentation



### Abstract

- 给出了新的 end-end 网络结构，针对**小量数据集**的训练，可以获得**更加优异**的表现。而且，网络训练的**速度很快。**
- 轻松拿下  ISBI cell tracking challenge 2015

- 给出了 Caffe 代码：https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net





### Introduction

在 biomedical image processing 作为研究对象时，localization 应该尤为想得到的信息。也就是说：a class label is supposed to be assigned to each pixel.  Ciresan et al. (2012) 的文章虽然赢了2012年的 ISBI，但是有两个明显缺陷：慢+在准确性定位与容量的使用（the use of context） 之间的妥协。



### Network Architecture

网络呈U 型：

![](https://i.loli.net/2018/08/29/5b860105668e6.png)

























