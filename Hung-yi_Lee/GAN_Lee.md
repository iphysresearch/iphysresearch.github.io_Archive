

[TOC]



# Introduction of Generative Adversarial Network (GAN)

> 对抗生成网络(GAN)国语教程(2018)
>
> [李宏毅](http://speech.ee.ntu.edu.tw/~tlkagk/)
>
> Hung-yi Lee



https://www.bilibili.com/video/av24011528



https://www.youtube.com/playlist?list=PLJV_el3uVTsMq6JEFPW35BCiOQTsoqwNw



- Generative Adversarial Network (GAN):
  - Introduction [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GAN%20(v2).pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GAN%20(v2).pptx),[video](https://youtu.be/DQNNMiAP5lw) (2018/05/04) 
  - Conditional GAN [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/CGAN.pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/CGAN.pptx),[video](https://youtu.be/LpyL4nZSuqU) (2018/05/11) 
  - Unsupervised Conditional GAN [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/CycleGAN.pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/CycleGAN.pptx),[video](https://youtu.be/-3LgL3NXLtI) (2018/05/18) 
  - Theory [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GANtheory%20(v2).pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GANtheory%20(v2).pptx),[video](https://youtu.be/DMA4MrNieWo) (2018/05/11) 
  - General Framework [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/fGAN.pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/fGAN.pptx),[video](https://youtu.be/av1bqilLsyQ) (2018/05/11) 
  - WGAN, EBGAN [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/WGAN%20(v2).pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/WGAN%20(v2).pptx),[video](https://youtu.be/3JP-xuBJsyc) (2018/05/18) 
  - InfoGAN, VAE-GAN, BiGAN [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GANfeature.pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GANfeature.pptx),[video](https://youtu.be/sU5CG8Z0zgw) (2018/05/18) 
  - Application to Photo Editing [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/PhotoEditing.pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/PhotoEditing.pptx),[video](https://youtu.be/Lhs_Kphd0jg) (2018/05/18) 
  - Application to Sequence Generation [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GANSeqNew.pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GANSeqNew.pptx),[video](https://youtu.be/Xb1x4ZgV6iM) (2018/05/25) 
  - Application to Speech (by Dr. Yu Tsao) [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/ICASSP%202018%20audio.pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/ICASSP%202018%20audio.pptx) (2018/06/01) 
  - Evaluation of GAN [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GANEvaluation.pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GANEvaluation.pptx),[video](https://youtu.be/IB_ADssBomk) (2018/05/25)

- HW3-1: [link](https://docs.google.com/presentation/d/1UdLXHcu-pvvYkNvZIWT7tFbuGO2HzHuAZhcA0Xdrtd8/edit#slide=id.p3) (2018/05/04) 
- HW3-2: [link](https://docs.google.com/presentation/d/1P5ToVdC_FaFzqC-wD6al6RoLseOgzoyaYESyJasef2E/edit#slide=id.p3),[tips](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/CGANtip.pdf) (2018/05/11) 
- HW3-3: [link](https://docs.google.com/presentation/d/1Xc07oRuS2aSENvepSBE_tyuf52ikxFIrUTVOAMA-dZk/edit#slide=id.p3) (2018/05/18)



> 只摘录经典和重要的评论理解、理论语录以及知识点内容。





- All Kinds of GAN ...

https://github.com/hindupuravinash/the-gan-zoo

![](https://i.loli.net/2018/09/06/5b912affd048b.png)







## Basic Idea of GAN

- 我们的目标是：生成一个 **NN Generator**。

![](https://i.loli.net/2018/09/04/5b8e7cc0b0e69.png)

	与此同时，我们会训练出来一个 **NN Discriminator**。
	
	![](https://i.loli.net/2018/09/04/5b8e7d175d956.png)







### Algorithm（在操作意义上）

![](https://i.loli.net/2018/09/04/5b8e831ac6ec2.png)

![](https://i.loli.net/2018/09/04/5b8e8342264c7.png)

- 正式的把算法规则写出来：

  ![](https://i.loli.net/2018/09/04/5b8e842e6c7d9.png)

 





## GAN as structured learning

> 从概念上讲 GAN 的原理。

- **Why Structured Learning Challenging?**
  - **One-shot/Zero-shot Learning:**
    - In classification, each class has some examples.
    - In structured learning,
      - If you cansider each possible output as a "class" ......
      - Since the output space is huge, most "classes" do not have any training data.
      - Machine has to create new stuff during testing.
      - Need more intelligence
  - Machine has to learn to do **planning**
    - Machine generates objects component-by-component, but it should have a big picture in its mind.
    - Because the output components have dependency, they should be considered globally.



- Strutured Learning Approach

  ![](https://i.loli.net/2018/09/04/5b8e87ad58f8f.png)



## Can Generator learn by itself?

> 可以啊，想想**自编码器**！
>
> ![](https://i.loli.net/2018/09/04/5b8e894a1ffd1.png)

> 顺带在想想**变分自编码器（VAE）**！可以比自编码器更加稳定（线性？！）
>
> ![](https://i.loli.net/2018/09/04/5b8e8a335ffeb.png)

- 既然看似有所谓的 Generator learned by itself，那和 GAN 相比，缺陷在哪里？

  （下面的图信息量很大~）

  ![](https://i.loli.net/2018/09/04/5b8e8c38931df.png)

  - 用 Autoencoder 作为 Generator 生成图片，往往需要更深更大的网络才能得到和 GAN 相差不太多的结果。

  - 换做是 VAE 的话，其实也是很有困难的，比如下面的图，蓝色点是 Generator 出来的，绿色点是学习目标：

    ![](https://i.loli.net/2018/09/04/5b8e8ce46aba9.png)



## Can Discriminator generate?

- Discriminator 在不同的文献领域可能有不同的名字：
  - Evaluation function
  - Potential Function
  - Energy Function
  - ...

- 我们其实是可以直接训练一个 Discriminator 的，大致的算法是这样的：

  ![](https://i.loli.net/2018/09/04/5b8e909b0f7a3.png)

  - 随机穷举 x （图片像素）经过 Discriminator 后取得分最高的图片作为 negative example

  - 在每一次迭代更新 Discriminator 的过程中，都会将判定为 negative 的结果放回到下一次迭代中，做作为更新后的新 negative example.

  - 训练的过程中，分布表示是这样的：（红色线代表 D 的 value，分布在高维的空间中）

    ![](https://i.loli.net/2018/09/04/5b8e91f6d23a6.png)

    ![](https://i.loli.net/2018/09/04/5b8e9248de060.png)

    这种训练方式（尤其是正例样本少的情况），还是容易出现过拟合的。

- 这样训练的 Discriminator，其实有很多。。。。都是 Strutured Learning 中的技术，GAN 也是其中之一。

  ![](https://i.loli.net/2018/09/04/5b8e9300e2d29.png)



## Generator v.s. Discriminator

![](https://i.loli.net/2018/09/04/5b8e9389adb4d.png)



## Benefit of GAN

![](https://i.loli.net/2018/09/04/5b8e940090ff5.png)

来看看效果：

![](https://i.loli.net/2018/09/04/5b8e943ebebf6.png)

![](https://i.loli.net/2018/09/04/5b8e945f88063.png)







## A little bit theory

![](https://i.loli.net/2018/09/04/5b8e95f772098.png)

![](https://i.loli.net/2018/09/04/5b8e9608b6c43.png)

![](https://i.loli.net/2018/09/04/5b8e961e6f001.png)

![](https://i.loli.net/2018/09/04/5b8e962e5a9b0.png)

![](https://i.loli.net/2018/09/04/5b8e964b4d1b7.png)

![](https://i.loli.net/2018/09/04/5b8e9660aa828.png)







# Conditional GAN

【Scott Reed, et al, ICML, 2016】

![](https://i.loli.net/2018/09/04/5b8e9b260b165.png)

严格说，算法是这样的：

![](https://i.loli.net/2018/09/04/5b8e9ae154e96.png)



- 现在的人们都怎么设计 GAN 类型的架构呢？

  ![](https://i.loli.net/2018/09/04/5b8e9cb80c2ea.png)



推荐两种架构都可以试一下，看下哪个效果会更好。

现在有所谓的生成高清大图的方式，就是 Stack GAN：

![](https://i.loli.net/2018/09/04/5b8e9da4b1ba2.png)

 

图太大通常训练也很低效，那么可以考虑 Patch GAN:

![](https://i.loli.net/2018/09/04/5b8e9e835c8e8.png)

语音降噪也很帅！

![](https://i.loli.net/2018/09/04/5b8e9f346ec30.png)



其他应用就不提了。。。











# Unsupervised Conditional Generation

- 非监督的G 就是没有 label。

![](https://i.loli.net/2018/09/06/5b912e0141bef.png)



一般有两种方向做法：

![](https://i.loli.net/2018/09/06/5b912e4a5f8e2.png)





## Direct Transformation（CycleGAN, StarGAN）

1. 第一种做法就是忽视掉 Generator 的输入输出之间的相似性要求。

   ![](https://i.loli.net/2018/09/06/5b91300ec1aa3.png)

2. CycleGAN（不忽略）

   ![](https://i.loli.net/2018/09/06/5b9130d3861de.png)



   这里有个例子：Cycle GAN – Silver Hair https://github.com/Aixile/chainer-cyclegan 

   可以将动漫人物的头发转换成银色。。。。。



   - Issue os Cycle Consistency 

     ![](https://i.loli.net/2018/09/06/5b91321d6fee0.png)

   - 好些 GAN 其实都是 Cycle GAN：

3. 还有多个 domain 之间互相转换的 StarGAN

   ![](https://i.loli.net/2018/09/06/5b9132f16362c.png)

   ![](https://i.loli.net/2018/09/06/5b91331821d97.png)



## Projection to Common Space

思路大体如下图，两个自编码器是独立进行的。不过很可能的问题就会出现，那就是中间的 latent space 难以相互对应上。。。

![](https://i.loli.net/2018/09/06/5b91361f2c10c.png)

有些人想了各种办法解决这种问题，比如：

1. 共享一部分参数。

   ![](https://i.loli.net/2018/09/06/5b913678566fc.png)

2. 一种类似 VAE 的办法。

   ![](https://i.loli.net/2018/09/06/5b9136e9ca864.png)

3. 利用 Cycle Consistency

   ![](https://i.loli.net/2018/09/06/5b913763d184b.png)

4. 利用 Semantic Consistency

   ![](https://i.loli.net/2018/09/06/5b9137fd724a5.png)



来看看应用吧：世界二次元化

Using the code: https://github.com/Hi-king/kawaii_creator 

- It is not cycle GAN, Disco GAN 



还有一个应用：Voice Conversion



最后还给你文献：

- Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros, Unpaired Image-to- Image Translation using Cycle-Consistent Adversarial Networks, ICCV, 2017 
- Zili Yi, Hao Zhang, Ping Tan, Minglun Gong, DualGAN: Unsupervised Dual Learning for Image-to-Image Translation, ICCV, 2017 
- Tomer Galanti, Lior Wolf, Sagie Benaim, The Role of Minimal Complexity Functions in Unsupervised Learning of Semantic Mappings, ICLR, 2018 
- Yaniv Taigman, Adam Polyak, Lior Wolf, Unsupervised Cross-Domain Image Generation, ICLR, 2017 
- Asha Anoosheh, Eirikur Agustsson, Radu Timofte, Luc Van Gool, ComboGAN: Unrestrained Scalability for Image Domain Translation, arXiv, 2017 
- Amélie Royer, Konstantinos Bousmalis, Stephan Gouws, Fred Bertsch, Inbar Mosseri, Forrester Cole, Kevin Murphy, XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings, arXiv, 2017 

- Guillaume Lample, Neil Zeghidour, Nicolas Usunier, Antoine Bordes, Ludovic Denoyer, Marc'Aurelio Ranzato, Fader Networks: Manipulating Images by Sliding Attributes, NIPS, 2017 
- Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, Jiwon Kim, Learning to Discover Cross-Domain Relations with Generative Adversarial Networks, ICML, 2017 
- Ming-Yu Liu, Oncel Tuzel, “Coupled Generative Adversarial Networks”, NIPS, 2016 
- Ming-Yu Liu, Thomas Breuel, Jan Kautz, Unsupervised Image-to-Image Translation Networks, NIPS, 2017 
- Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo, StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation, arXiv, 2017 







