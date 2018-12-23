---
title: Classifying Lensed Gravitational Waves in the Geometrical Optics Limit with Machine Learning
date: 2018-12-23
---

[返回到上一页](./index.html)

---

![](https://i.loli.net/2018/08/24/5b7fffecd8d1d.png)

# Classifying Lensed Gravitational Waves in the Geometrical Optics Limit with Machine Learning (2018)

>Classifying Lensed Gravitational Waves in the Geometrical Optics Limit with Machine Learning. Amit Jit Singh, Ivan S.C. Li, Otto A. Hannuksela, Tjonnie G.F. Li, Kyungmin Kim [The Chinese University of Hong Kong & Imperial College London] (2018) arXiv:1810.07888



<iframe src="./Classifying Lensed Gravitational Waves in the Geometrical Optics Limit with Machine Learning.pdf" style="width:1000px; height:1000px;" width="100%" height=100%>This browser does not support PDFs. Please download the PDF to view it: <a href="https://arxiv.org/pdf/1810.07888.pdf">Download PDF</a></iframe>



> FYI：
>



[TOC]



## My comments

颇有学生之作的味道，因为很多训练细节是不清楚的，同时实验讨论的结果也很难给出决定性的结果。

## The Dataset

数据样本用的是都含有 GW 信号的二维 spectrogram images，label 标签用 lensed or unlensed 来区别，齐总 match-filtered SNR 是不超过 80 的 gaussian noise，所以总共 2000 spectrogram samples of lensed GWs (其中两种模型方法point mass 和 SIS生成的lensed GWs各占1000) 和1000 samples of unlensed GWs，还说所有这3000个samples的 SNR 是一个正态分布(均值41方差19)。

训练集共3000个样本，训练时会根据point mass 和 SIS来区分地输入网络2000个样本，其中正负样本各1000，测试时用了500个样本，其中正负样本各250。





---

[返回到上一页](./index.html) | [返回到顶部](./Classifying Lensed Gravitational Waves in the Geometrical Optics Limit with Machine Learning.html)

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