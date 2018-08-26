---
title: ImageNet Classification with Deep Convolutional Neural Networks
date: 2018-08-24
---

[返回到上一页](./index.html)

---

![](https://i.loli.net/2018/08/24/5b7fffecd8d1d.png)

# ImageNet Classification with Deep Convolutional Neural Networks (2012)

> Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.



<iframe src="./ImageNet Classification with Deep Convolutional Neural Networks.pdf" style="width:1000px; height:1000px;" width="100%" height=100%>This browser does not support PDFs. Please download the PDF to view it: <a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">Download PDF</a></iframe>






> FYI：
>
> - [读《ImageNet Classification with Deep Convolutional Neural Networks》](https://zhuanlan.zhihu.com/p/20324656)
>
> - [AlexNet论文翻译——中英文对照](http://noahsnail.com/2017/07/04/2017-7-4-AlexNet论文翻译/)

[TOC]



## Introduction



在此前之前，对于较简单的目标识别（Object recognition） 任务，用传统机器学习方法可以有效解决主要得益于数据量级（NORB, Caltech-101/256, CIFAR-10/100）还并不算大到难以承受（in tens of thousands）。但是为了应对更真实的图片的可变性（variability），超大数据量是必要的（LabelMe in hundreds of thousands; ImageNet in 15in millions + high-resolution + 22,000 categories）。

用 CNN 网络是基于对先验知识的考虑：

> ... have lots of prior knowledge to compensate for all the data we don’t have.

具体说来，就是基于图像的特征假设：**stationarity of statistics** and **locality of pixel dependencies**.

数据用的是：subsets of ImageNet used in the ILSVRC-2010 and ILSVRC-2012 competitions，CNN 网络用了5层卷积和3层全连接构造，并且发觉去掉卷积层（不超过1%的模型参数量）后模型的效果变差。论文的代码地址可见：http://code.google.com/p/cuda-convnet/ 。 花了5-6天在两块 GTX 580 3GB GPUs 上训练。



## The Dataset

- ImageNet
  - is a dataset of over **15 million** labeled high-resolution images belonging to roughly **22,000** categories
- ImageNet Large-Scale Visual Recognition Challenge (ILSVRC)
  - is a subset of ImageNet with roughly **1000** images in each of **1000** categories
  - which are roughly **1.2 million** training images, **50,000** validation images, and **150,000** testing images.
- ILSVRC-2010
  - is the **only** version of ILSVRC for which the test set labels are available
  - which is used to perform most of experiments (in this paper)
- ILSVRC-2012
  - in which test set labels are unavailable.

- Two error rates: top-1 and top-5
  - Top-5 error rate is the fraction of test images for which the correct label is not among the five labels considered most probable by the model.

- 数据预处理：
  - 由于模型需要固定的图片尺寸，所以都下采样到了 256x256。
  - 对于矩形图片，把短边尺度缩放到 256，然后在中心处裁剪出 256x256。
  - 对训练集图片的所有图片像素值都分别进行了标准化，使得每张图片的像素数值的均值为0。



## The Architecture

这就是文章中著名的AlexNet模型结构图：

![](https://i.loli.net/2018/08/24/5b7fc04ec53c7.png)

### ReLU Nonlinearity（ReLU 的非线性性）

实验显示：在 GD（gradient descent）优化过程中，saturating nonlinearities（如）比 non-saturating nonlinearity 速度慢得多。

> FYI: **what do they mean by "saturating" or "non-saturating" non-linearities**
>
> - Intuition: A saturating activation function squeezes the input.
> - Definitions: 
>   - $f$ is non-saturating iff $(|\lim_{z\rightarrow-\infty}f(z)|=+\infty)\cup(|\lim_{z\rightarrow+\infty} f(z)|=+\infty)$
>   - $ f $ is saturating iff $f$ is not non-saturating.
> - 简单来说，就是看这个函数是否关于自变量是**有界**的。
>
> Ref: [StackExchange](https://stats.stackexchange.com/questions/174295/what-does-the-term-saturating-nonlinearities-mean)

于是，作者用了 Rectified Linear Units (ReLUs) —— [Nair and Hinton 2010]
$$
f(x) = \max(0,x)
$$
作为激活函数。而不是 $f(x)=\tanh(x)$ or $f(x)=(1+e^{-x})^{-1}$。训练效果如下图：

![](https://i.loli.net/2018/08/24/5b7f981f2245c.png)

在达到25%的训练错误率的所花时间，相差六倍。图上两个模型的对比在训练的过程中，**各自的学习率是独立各自取值的**，为了保证最快的训练实验效果。并且，No regularization of any kind was employed。



### Training on Multiple GPUs (在多 GPUs 上训练)

由于一块 GTX 580 GPU 只有 3G 内存，所以把网络平行的分别放在两块 GPU 上训练。

平行（parallelization）的放置使得每层的 kernels 数目分别各占一半在两块 GPU 上，但 GPUs 之间的数据传输仅会在特定的层中进行。比方说，网络中的第二层和第三层之间，第三层的输入会取自第二层两块 GPUs 的全部特征图（feature maps）；在网络的第三层和第四层之间，第四层的输入只读取相同 GPU 传递来的第三层特征图。

该网络在 top-1 和 top-5 error rates 上分别得到 1.7% 和 1.2 %。

这个 two-GPU 网络的训练时间比 one-GPU 网络稍稍快一点。（因为都有着相同的 kernels 的数目，所以这两个网络的模型参数基本差不多，one-GPU 稍稍多一丢丢）





### Local Response Normalization (局部响应归一化)

局部响应归一化（Local Response Normalization）原理是仿造生物学上活跃的神经元对相邻神经元的抑制现象（侧抑制 lateral inhibitio）。 归一化（normalization） 的目的是“抑制”。因为 ReLU 是 unbounded 的，所以文章就用了 LRN 来归一化。

- 好处：LRN 层模仿生物神经系统的侧抑制机制，对局部神经元的活动创建竞争机制，使得相应比较大的值相对更大，**提高模型泛化能力**。根据 Hinton 的描述，将 top-1 和 top-5 error rate 分别降到 1.4% 和 1.2 %。同样地在 CIFAR-10 数据集上一个 4 层 CNN 在没有归一化时得到 13% test error，有归一化时为 11%。

- 计算公式：
  $$
  b^i_{x,y}=a^i_{x,y}/\Big(k+\alpha\sum^{\min(N-1,i+n/2)}_{j=\max(0,i-n/2)}(a^j_{x,y})^2\Big)^\beta
  $$
  公式看上去比较复杂，但理解起来非常简单，这就是一个pixel-wise的操作：

  - $a^i_{x,y}$ 表示每个在传递过非线性的 ReLu 函数之后的神经元激活值，其中 $i$ 表示第 $i$ 个 kernel 在像素位置 (x, y) 。 N 是该层的 kernel 总数。其他参数 $k,\alpha, \beta,n$ 就是超参数，其中$n$ 代表的是该 kernel 的“毗邻” kernel 的数目。 

  - Hinton 等人在验证集上的实验结果是，超参数可以为：$k=2,n=5,\alpha=10^{-4},$ and $\beta=0.75$。

  - 模型中在个别层上运用了 LRN。

  - Flowchart of Local Response Normalization（[source](http://yeephycho.github.io/2016/08/03/Normalizations-in-neural-networks/)）

    ![](https://i.loli.net/2018/08/24/5b7fa6a7d3606.png)

  - 简单的示意图：![](https://i.loli.net/2018/08/24/5b7fb04c607cb.png)

- 代码实现：

  - [tf.nn.local_response_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization)

  - [What the output of LRN looks like？](http://yeephycho.github.io/2016/08/03/Normalizations-in-neural-networks/)

  - ```python
    import numpy as np
    import matplotlib.pyplot as plt
    def lrn(x):
        y = 1 / (2 + (10e-4) * x ** 2) ** 0.75
    	return y
    input = np.arange(0, 1000, 0.2)
    output = lrn(input)
    plt.plot(input, output)
    plt.xlabel('sum(x^2)')
    plt.ylabel('1 / (k + a * sum(x^2))')
    plt.show()
    ```

    ![](https://i.loli.net/2018/08/24/5b7fb21506152.png)

    Since the slope at the beginning is very steep, little difference among the inputs will be significantly enlarged, this is where the competition happens.

- 相关文献：

  - The exact way of doing this was proposed in (but not much extra info here):
    - Kevin Jarrett, Koray Kavukcuoglu, Marc’Aurelio Ranzato and Yann LeCun, What is the best Multi-Stage Architecture for Object Recognition?, ICCV 2009. [pdf](http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf)

  - It was inspired by computational neuroscience:
    - S. Lyu and E. Simoncelli. Nonlinear image representation using divisive normalization. CVPR 2008. [pdf](http://www.cns.nyu.edu/pub/lcv/lyu08b.pdf). This paper goes deeper into the math, and is in accordance with the answer of seanv507.
    - [24] N. Pinto, D. D. Cox, and J. J. DiCarlo. Why is real-world vi- sual object recognition hard? [PLoS Computational Biology, 2008.](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0040027)

- **后期争议**

  - 在2015年著名的 Very Deep Convolutional Networks for Large-Scale Image Recognition. 提到LRN基本没什么用。

    ![](https://i.loli.net/2018/08/24/5b7fa43993218.png)

  - [CS231n](http://cs231n.github.io/convolutional-networks/) 的课程也曾提到这种仿生物学的归一化策略一般没什么鸟用：

    ![](https://i.loli.net/2018/08/24/5b7faf534a2a1.png)


Ref：

1. [What Is Local Response Normalization In Convolutional Neural Networks](https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/)
2. [深度学习的局部响应归一化LRN(Local Response Normalization)理解](https://blog.csdn.net/yangdashi888/article/details/77918311)
3. [【深度学习技术】LRN 局部响应归一化](https://blog.csdn.net/hduxiejun/article/details/70570086)
4. [Importance of local response normalization in CNN](https://stats.stackexchange.com/questions/145768/importance-of-local-response-normalization-in-cnn)
5. [Normalizations in Neural Networks](http://yeephycho.github.io/2016/08/03/Normalizations-in-neural-networks/)





### Overlapping Pooling（部分重叠的池化）

文章强调了可部分重叠的池化对模型的泛化能力总是更有效的。

让步长 s 比池化核的尺寸 z 小，这样池化层的输出之间会有重叠和覆盖，可以提升了特征的丰富性，使得模型在训练过程中更难以过拟合。

文章的实验结果说，s=2，z=3 比 s = z =2 在 top-1 和 top-5 error rate 减少了 0.4% 和 0.3%。





### Overall Architecture

正如上面图中所示的，另外需要提及的是在第一层、第二层和第五层运用 LRN。全部使用最大池化策略。

 **注意！原论文对网络结构的描述有各种错误！应该参照 cs231n 等其他网络资料给出的解释说明版本！**

![](https://i.loli.net/2018/08/24/5b7fbe6560b34.png)

而且结构图中给出的特征图维数也错的不行不行的，比方说只有 48x48x55x2=253440 才能得到文中提到的第一层特征图维度，但这显然不对嘛。

![](https://i.loli.net/2018/08/24/5b7fc09e32aaa.png)

经过一番资料查找，下面列出AlexNet正确的网络结构超参数：

1   'data'     Image Input                   227x227x3 images with 'zerocenter' normalization

---

2   'conv1'    Convolution                   96 11x11x3 convolutions with stride [4  4] and padding [0  0  0  0]
3   'relu1'    ReLU                          ReLU
4   'norm1'    Cross Channel Normalization   cross channel normalization with 5 channels per element
5   'pool1'    Max Pooling                   3x3 max pooling with stride [2  2] and padding [0  0  0  0]

---

6   'conv2'    Convolution                   256 5x5x48 convolutions with stride [1  1] and padding [2  2  2  2]
7   'relu2'    ReLU                          ReLU
8   'norm2'    Cross Channel Normalization   cross channel normalization with 5 channels per element
9   'pool2'    Max Pooling                   3x3 max pooling with stride [2  2] and padding [0  0  0  0]

---

10   'conv3'    Convolution                   384 3x3x256 convolutions with stride [1  1] and padding [1  1  1  1]
11   'relu3'    ReLU                          ReLU

---

12   'conv4'    Convolution                   384 3x3x192 convolutions with stride [1  1] and padding [1  1  1  1]
13   'relu4'    ReLU                          ReLU

---

14   'conv5'    Convolution                   256 3x3x192 convolutions with stride [1  1] and padding [1  1  1  1]
15   'relu5'    ReLU                          ReLU
16   'pool5'    Max Pooling                   3x3 max pooling with stride [2  2] and padding [0  0  0  0]

---

17   'fc6'      Fully Connected               4096 fully connected layer
18   'relu6'    ReLU                          ReLU
19   'drop6'    Dropout                       50% dropout

---

20   'fc7'      Fully Connected               4096 fully connected layer
21   'relu7'    ReLU                          ReLU
22   'drop7'    Dropout                       50% dropout

---

23   'fc8'      Fully Connected               1000 fully connected layer
24   'prob'     Softmax                       softmax

---

25   'output'   Classification Output         crossentropyex with 'tench' and 999 other classes

![](https://i.loli.net/2018/08/24/5b7fc32a1e649.png)

网络优化的目标是最大化 multinomial logistic regression objective：
$$
\begin{align}
l(\mathbf{\omega})=&\sum^n_{j=1}\log P(\mathbf{y}_i|\mathbf{x_j},\mathbf{\omega})\\
=&\sum^n_{j=1}\Big[\sum^m_{i=1}\mathbf{y}^{(i)}_j\mathbf{\omega}^{(i)T}\mathbf{x}_j-\log\sum^m_{i=1}\exp\Big(\omega^{(i)}\mathbf{x}_j\Big)\Big]
\end{align}
$$
上面的公式参考自paper：[Sparse Multinomial Logistic Regression: Fast Algorithms and Generalization Bounds](http://www.stat.columbia.edu/~liam/teaching/neurostat-spr11/papers/optimization/hartemink05.pami.pdf)



Ref：

1. [How does Krizhevsky's '12 CNN get 253,440 neurons in the first layer?](https://stats.stackexchange.com/questions/132897/how-does-krizhevskys-12-cnn-get-253-440-neurons-in-the-first-layer)
2. [alexnet —— MathWorks](https://ww2.mathworks.cn/help/nnet/ref/alexnet.html)
3. [Understanding AlexNet](https://www.learnopencv.com/understanding-alexnet/)
4. [Walkthrough: AlexNet](https://github.com/dmlc/minerva/wiki/Walkthrough:-AlexNet)（有误）
5. [A Walk-through of AlexNet](https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637)





## Reducing Overfitting

network ~ 60 million parameters

### Data Augmentation（数据增广）

关于数据增广有两种方案实施，都仅需很少量的计算而无需占据内存。并且在 CPU 中用 Python 写成。

#### Image translations and horizontal reflections.

- 随机截取 224x224 的小窗口及其相应的水平镜面反射图像。
- 在测试时，每张图片截取四角和中心的5个小窗口以及相应的镜面反射图像（共10个）作为测试图片，取预测均值。

#### Fancy PCA - altering the intensities of the RGB channels in training images.

- 将图片数据转化为 3x3 covariance matrix of RGB

- 通过 PCA 找到3个最优的维度来理解图片 RGB 数据结构

- 在原图的 matrix 基础上增加了 PCA 的这三个维度或 components 后（altering），如下，图片的物体在面对光线和色彩强弱变化时，能更容易被识别。

  ![](https://i.loli.net/2018/08/24/5b7fda196a055.png)

Ref: [2分钟AlexNet PCA是如何帮助CNN处理图片的intensity来降低过拟合](https://zhuanlan.zhihu.com/p/36432137)





### Dropout（随机失活）

模型融合是降低 test errors 都一种有效手段，但对大型网络来说就效率太低了。

然而，对于一个有效模型来说：

> There is a very efficient version of model combination that only costs about a factor of two during training.

所以，我们可以变训练变融合。。。。于是，dropout 技术横天出世！[G.E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R.R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580, 2012.]

这个文章对前两个全连接层都是用了 0.5 概率的 dropout。

过拟合问题显著降低，不过训练迭代时长增加了一倍。





## Details of learning

- 优化方案：

> Stochastic gradient descent (SGD) with a **batch size** of 128 examples, **momentum** of 0.9, and **weight decay** of 0.0005.

	论文中甚至直接给出了参数迭代更新的公式：
$$
\begin{align}
v_{i+1} &:= 0.9\cdot v_i-0.0005\cdot\epsilon\cdot\omega_i-\epsilon\cdot\langle\frac{\partial L}{\partial\omega}\Big|_{\omega_i}\rangle{D_i} \\
\omega_{i+1} &:= \omega_i + v_{i+1}
\end{align}
$$
![](https://i.loli.net/2018/08/24/5b7fde4d15435.png)



- 参数初始化：

  $\omega:$ zero-mean Gaussian distribution with standard de- viation 0.01.

  $b:$ constant 1 for 2nd, 4th, and 5th layers; constant 0 for the remainning layers.

  据称： This initialization accelerates the early stages of learning by providing the ReLUs with positive inputs. 



- 学习率策略：
  - **divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate**. The learning rate was initialized at 0.01 and reduced three times prior to termination.





## Results

- 在 ILSVRC-2010 数据集上：

![](https://i.loli.net/2018/08/24/5b7ff4492a8d2.png)

	CNN 比其他两种算法差不多低过近10%的错误率。

- 在 ILSVRC-2012 和 ImageNet 2011 + ILSVRC-2012 数据集上：

![](https://i.loli.net/2018/08/24/5b7ff53fa8cac.png)

由于没有测试数据集，所以采用了：

> validation and test error rates interchangeably because in our experience they do not differ by more than 0.1%.





### Qualitative Evaluations（定性评估）



- kernel 可视化 —— specialization

  ![](https://i.loli.net/2018/08/24/5b7ff88e19f10.png)

  - A variety of **frequency**- (GPU1) and **orientation**-selective kernels (GPU2), as well as various **colored** blobs (GPU2) are learned.
  - Specialization occurs during every run.
  - which is **independent** of any particular random weight **initialization**.

- asses top-5 predictions（左图）

  ![](https://i.loli.net/2018/08/24/5b7ff9e516c05.png)

  - 即使对象并不自图像的中心，CNN依然可以识别清楚（如 mite）
  - 大多排前 top-5 的标签都比较 reasonable，比如豹子的图中，也较大概率的被识别为猫咪
  - 有些图片中，CNN 网络聚焦的点是很模糊的

- feature activations at the last（右图）

  - 第一列是4张测试图像，剩下的列是6张训练图像
  - 首先明确度量标准：如果两幅图像生成的特征激活向量之间有较小的欧式距离，我们可以认为神经网络的更高层特征认为它们是相似的。
  - 如果基于 L2 度量标准，可以明显发觉在像素级别上，他们每行的距离通常是不接近的（比如狗和牛的各种姿态）

- 为一种新的图像检索方法带俩启示（这优于直接作用到像素上的传统自动编码器）

  - 理由：Computing similarity by using Euclidean distance between two 4096-dimensional, real-valued vectors is inefficient, but it could be made efficient by training an auto-encoder to compress these vectors to short binary codes. 





## Discussion

-  CNN的深度似乎非常重要，因为一旦移除任意一中间层都会损失大概2%的 top-1 性能。
- 本论文的实验没有使用任何无监督的预训练。





---

[返回到上一页](./index.html) | [返回到顶部](./ImageNet Classification with Deep Convolutional Neural Networks.html)

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