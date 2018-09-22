---
title: CS231n
date: 2018-08-22
---

[返回到首页](../index.html)

---

[TOC]

# CS231n: Convolutional Neural Networks for Visual Recognition

> ### Course Description
>
> Computer Vision has become ubiquitous in our society, with applications in search, image understanding, apps, mapping, medicine, drones, and self-driving cars. Core to many of these applications are visual recognition tasks such as image classification, localization and detection. Recent developments in neural network (aka “deep learning”) approaches have greatly advanced the performance of these state-of-the-art visual recognition systems. This course is a deep dive into details of the deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification. During the 10-week course, students will learn to implement, train and debug their own neural networks and gain a detailed understanding of cutting-edge research in computer vision. The final assignment will involve training a multi-million parameter convolutional neural network and applying it on the largest image classification dataset (ImageNet). We will focus on teaching how to set up the problem of image recognition, the learning algorithms (e.g. backpropagation), practical engineering tricks for training and fine-tuning the networks and guide the students through hands-on assignments and a final course project. Much of the background and materials of this course will be drawn from the [ImageNet Challenge](http://image-net.org/challenges/LSVRC/2014/index).
>
> CS231n 课程的官方地址：<http://cs231n.stanford.edu/index.html>



## Original Works

- [一段关于神经网络的故事](./cs231n_story_MLP.html)（**Original**，30671字 + 多图）

---

## Notes from Lecture Video and Slices

- [Lecture 1. Computer vision overview & Historical context](./cs231n_1.html)
- [Lecture 2. Image Classification & K-nearest neighbor](./cs231n_2.html)
- [Lecture 3. Loss Functions and Optimization](./cs231n_3.html)
- [Lecture 4. Introduction to Neural Networks](./cs231n_4.html)
- [Lecture 5. Convolutional Neural Networks](./cs231n_5.html)
- [Lecture 6. Training Neural Networks, part I](./cs231n_6.html)
- [Lecture 7. Training Neural Networks, part II](./cs231n_7.html)
- [Lecture 8. Deep Learning Hardware and Software](./cs231n_8.html)
- [Lecture 9. CNN Architectures](./cs231n_9.html)
- [Lecture 10. Recurrent Neural Networks](./cs231n_10.html)
- [Lecture 11. Detection and Segmentation](./cs231n_11.html)
- [Lecture 12. Generative Models](./cs231n_12.html)
- [Lecture 13. Visualizing and Understanding](./cs231n_13.html)
- [Lecture 14. Deep Reinforcement Learning](./cs231n_14.html)

---

- [Guest Lecture. Efficient Methods and Hardware for Deep Learning](./cs231n_Efficient Methods and Hardware for Deep Learning.html) by [**Song Han**](https://stanford.edu/~songhan/) (Spring 2017)
- [Guest Lecture. Adversarial Examples and Adversarial Training](./cs231n_Guest Lecture. Adversarial Examples and Adversarial Training.html) by [**Ian Goodfellow**](http://www.iangoodfellow.com/) (Spring 2017)

---



## Course Notes

- [图像分类笔记](./CS231n_image_classification_note.html)

  > L1/L2 distances, hyperparameter search, cross-validation

- [线性分类笔记](./CS231n_linear_classification_note.html)

  > parameteric approach, bias trick, hinge loss, cross-entropy loss, L2 regularization, web demo

- [最优化笔记](./CS231n_optimization_note.html)

  > optimization landscapes, local search, learning rate, analytic/numerical gradient

- [反向传播笔记](./CS231n_backprop_notes.html)

  > chain rule interpretation, real-valued circuits, patterns in gradient flow

- [卷积神经网络笔记](./CS231n_ConvNet_notes.html)

  > layers, spatial arrangement, layer patterns, layer sizing patterns, AlexNet/ZFNet/VGGNet case studies, computational considerations

- [神经网络笔记1](./CS231n_Neural_Nets_notes_1.html)

  > model of a biological neuron, activation functions, neural net architecture, representational power

- [神经网络笔记2](./CS231n_Neural_Nets_notes_2.html)

  > preprocessing, weight initialization, batch normalization, regularization (L2/dropout), loss functions

- [神经网络笔记3](./CS231n_Neural_Nets_notes_3.html)

  > gradient checks, sanity checks, babysitting the learning process, momentum (+nesterov), second-order methods, Adagrad/RMSprop, hyperparameter optimization, model ensembles

- [循环神经网络惊人的有效性](./The Unreasonable Effectiveness of Recurrent Neural Networks.html)

  > From: [Andrej Karpathy blog](http://karpathy.github.io/)'s 《The Unreasonable Effectiveness of Recurrent Neural Networks (2015)》

---



## How to Comment

With use of the [hypothes.is](https://hypothes.is/) extension (right-sided), you can highlight, annote any comments and discuss these notes inline*at any pages*and *posts*.

*Please Feel Free* to Let Me Know and *Share* it Here.





> ### Memo to myself
>
> - [ ] 根据英文字幕更新 Spring 2018 视频内的课程内容
> - [ ] 完善和更新所有提及的文献 paper
> - [ ] 尽可能将图片化内容信息改写为文本 markdown
> - [ ] 需要细致，完整的给出插图以及 Slide 等来源或作者信息。







---

[返回到首页](../index.html) | [返回到顶部](./index.html)


<div id="disqus_thread"></div>
<script>
/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://iphysresearch.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>


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