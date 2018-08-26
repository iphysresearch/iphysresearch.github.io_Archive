---
title: Very deep convolutional networks for large-scale image recognition
date: 2018-08-24
---

[返回到上一页](./index.html)

---

![](https://i.loli.net/2018/08/24/5b7fffecd8d1d.png)

# Very deep convolutional networks for large-scale image recognition (2014)

> Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.



<iframe src="https://arxiv.org/pdf/1409.1556.pdf" style="width:1000px; height:1000px;" width="100%" height=100%>This browser does not support PDFs. Please download the PDF to view it: <a href="https://arxiv.org/pdf/1409.1556.pdf">Download PDF</a></iframe>



> FYI：
>
> - [VGG论文翻译——中英文对照](http://noahsnail.com/2017/08/17/2017-8-17-VGG论文翻译——中英文对照/)



[TOC]

## Introduction

文章谈到自从 [Keizhevsky et al. (2012)](./ImageNet Classification with Deep Convolutional Neural Networks.html) （ILSVRC-2012 的冠军）的卷积神经网络发布后，有两类在该模型基础上的改进方向：

1. (Zeiler & Fergus, 2013; Sermanet et al., 2014): 在第一卷积层使用了更小的感受野窗口和更小的步长；（ILSVRC-2013 的冠军）
2. (Sermanet et al., 2014; Howard, 2014):  dealt with training and testing the networks densely over the whole image and over multiple scales.

而该文目标是一个新的改进方向：**深度**！

这个模型不仅把 ILSVRC 数据集的准确率(acc)调到了最好，还发现模型可以直接适配于其他图像识别数据集，而且性能还比传统的最好机器学习模型还要好。[代码地址](http://www.robots.ox.ac.uk/ ̃vgg/research/very_deep/)



## ConvNet Configurations

为了测量深度对 ConvNet 的影响，所有的卷积层都是用相同的 principles。



### Architecture & Configurations

- 预处理：只做了**图像去均值**（subtracting the mean RGB value），给所有图片减去一个均值图像。

这两节的全部内容，都是在细致的说清楚下面网络结构图表：

![](https://i.loli.net/2018/08/25/5b81787a0e337.png)

- 构造了5个网络模型（每一列）。它们都有着相同的 Input，即 224x224x3 尺寸的图像，都有着相同的三层全连接网络构造（4096-4096-1000），最后用 softmax 计算输出。
- 这5个模型设计的差异性仅仅在于**深度**（除了 A-LRN）。即在不同的 Configurations 基础上都有一对模型可以相互对照。（如 A 与 A-LRN、A 与 B、B 与 C、C 与 D、B 与 D、A 与 D 和 D 与 E）
- 所有模型的卷积核都是 3x3 或 1x1 ，步长都为1。其中卷积核为 3x3 的 padding 设定为1。共5个最大池化层，核为 2x2，步长为 2.
- 所有的卷积层分组后，特征图数目（通道数）按照从64起2的倍数直到512。
- 作者特别的强调了：**LRN 归一化压根毛用没有！**还占我的内存，浪费了我的时间！
- 最后，作者提及到他的网络模型参数比那些浅层的网络还要省。



### Discussion

- 作者提及到：两个 3x3 卷积计算层堆叠的效果与一个 5x5 的卷积计算层是等效的。（需要验证这件事情！写一篇感受野的文章支撑一下！）



















---

[返回到上一页](./index.html) | [返回到顶部](./Very deep convolutional networks for large-scale image recognition.html)

---
<br>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
<br>
<script type="application/json" class="js-hypothesis-config">
  {
    "openSidebar": false,
    "showHighlights": true,
    "theme": classic,
    "enableExperimentalNewNoteButton": true
  }
</script>
<script async src="https://hypothes.is/embed.js"></script>