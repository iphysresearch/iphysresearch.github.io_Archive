

[TOC]



# 关于感受野 (Receptive field) 你该知道的事

**Receptive field** 可中译为“感受野”，是卷积神经网络中非常重要的概念之一。

我个人最早看到这个词的描述是在 2012 年 Krizhevsky 的 paper 中就有提到过，当时是各种不明白的，事实上各种网络教学课程也都并没有仔细的讲清楚“感受野”是怎么一回事，有什么用等等。直到我某天看了 UiO 的博士生 [Dang Ha The Hien](https://medium.com/@nikasa1889?source=post_header_lockup) 写了一篇非常流传甚广的博文：[**A guide to receptive field arithmetic for Convolutional Neural Networks**](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)，才大彻大悟，世界变得好了，人生都变得有意义了，正如博主自己谈到的写作动机：

> *This post fills in the **gap** by introducing a new way to visualize feature maps in a CNN that exposes the receptive field information, accompanied by a **complete receptive field calculation** that can be used for **any CNN** architecture.*

此文算是上述博文的一个精要版笔记，再加上个人的理解与计算过程。和其他所有博文一样，写作的目的是给未来的自己看。

> FYI：读者已经熟悉 CNN 的基本概念，特别是卷积和池化操作。一个非常好的细致概述相关计算细节的 paper 是：[A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf)。



## 感受野可视化

我们知晓某一层的卷积核大小对应于在上一层输出的“图像”上的“视野”大小。比如，某层有 3x3 的卷积核，那就是一个 3x3 大小的滑动窗口在该层的输入“图像”上去扫描，我们就可以谈相对于上一层，说该层下特征图（feature map）当中任一特征点（feature）的“感受野”大小只有 3x3.（打引号说明术语引用不够严谨）。

那么，于是就有这样的问题了。对于较深层中高度抽象的特征图中某一特征点，它们在真实图像（Input）上究竟看到了多少？一个特征点的感受野大小可以影响到整个图片范围么？在感受视野范围内有聚焦点、还是处处平权关注的呢

- 先看个感受野的较严格定义：

  > The **receptive field** is defined as the region in the input space that **a particular CNN’s feature** is looking at (i.e. be affected by).

一个特征点的感受野可以用其所在的**中心点位置（center location）**和**大小（size）**来描述。然而，某卷积特征点所对应的感受野上并不是所有像素都是同等重要的，就好比人的眼睛所在的有限视野范围内，总有要  focus 的焦点。对于感受野来说，距离中心点越近的像素肯定对未来输出特征图的贡献就越大。换句话说，一个特征点在输入图像（Input） 上所关注的特定区域（也就是其对应的感受野）会在该区域的中心处聚焦，并以指数变化向周边扩展（*need more explanation*）。

废话不多说，我们直接先算起来。



首先假定我们所考虑的 CNN 架构是对称的，并且输入图像也是方形的。这样的话，我们就忽略掉不同长宽所造成的维度不同。

Way1 对应为通常的一种理解感受野的方式。在下方左侧的上图中，是在 5x5 的图像(蓝色)上做一个 3x3 卷积核的卷积计算操作，步长为2，padding 为1，所以输出为 3x3 的特征图(绿色)。那么该特征图上的每个特征(1x1)对应的感受野，就是 3x3。在下方左侧的下图中，是在上述基础上再加了一个完全一样的卷积层。对于经过第二层卷积后其上的一个特征(如红色圈)在上一层特征图上“感受”到 3x3 大小，该 3x3 大小的每个特征再映射回到图像上，就会发现由 7x7 个像素点与之关联，有所贡献。于是，就可以说第二层卷积后的特征其感受野大小是 7x7（需要自己画个图，好好数一数）。Way2 （下方右侧的图像）是另一种理解的方式，主要的区别仅仅是将两层特征图上的特征不进行“合成”，而是保留其在前一层因“步长”而产生的影响。

![](https://i.loli.net/2018/09/06/5b90c595f0c8b.png)

Way2 的理解方式其实更具有一般性，我们可以无需考虑输入图像的大小对感受野进行计算。如下图：

![](https://i.loli.net/2018/09/06/5b90cf56acf4d.png)

虽然，图上绘制了输入 9x9 的图像（蓝色），但是它的大小情况是无关紧要的，因为我们现在只关注某“无限”大小图像某一像素点为中心的一块区域进行卷积操作。首先，经过一个 3x3 的卷积层（padding=1，stride=2）后，可以得到特征输出（深绿色）部分。其中深绿色的特征分别表示卷积核扫过输入图像时，卷积核中心点所在的相对位置。此时，每个深绿色特征的感受野是 3x3 （浅绿）。这很好理解，每一个绿色特征值的贡献来源是其周围一个 3x3 面积。再叠加一个 3x3 的卷积层（padding=1，stride=2）后，输出得到 3x3 的特征输出（橙色）。此时的中心点的感受野所对应的是黄色区域 7x7，代表的是输入图像在中心点橙色特征所做的贡献。

这就是为何在 VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION 文章中提到：

> It is easy to see that a stack of two 3 × 3 conv. layers (without spatial pooling in between) has an effective receptive field of 5 × 5; three such layers have a 7 × 7 effective receptive field. 
>

也就是说两层 3x3 的卷积层直接堆叠后（无池化）可以算的有感受野是 5x5，三层堆叠后的感受野就是 7x7。





##  感受野计算

直观的感受了感受野之后，究竟该如何定量计算嗯？只要依据 Way2 图像的理解，我们对每一层的特征“顺藤摸瓜”即可。

我们已经发觉到，某一层特征上的感受野大小依赖的要素有：每一层的卷积核大小 k，padding 大小 p，stride s。在推导某层的感受野时，还需要考虑到该层之前各层上特征的的感受野大小 r，以及各层相邻特征之间的距离 j（jump）。

所以对于某一卷积层（卷积核大小 k，padding 大小 p，stride s）的感受野公式为：
$$

$$
































