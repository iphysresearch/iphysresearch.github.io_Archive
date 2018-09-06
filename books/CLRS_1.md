---
title: 第1章 算法在计算中的作用
date: 2018-09-06
---

[返回到首页](./CLRS.html)

---



[TOC]



# 第一部分 基础知识



## 第1章 算法在计算中的作用



### 1.1 算法

- **算法（algorithm）**就是任何良定义的计算过程，该过程取某个值或值的集合作为**输入**并产生某个值或值的集合作为**输出**。

```mermaid
graph LR

    输入 -->|算法| E[输出]
```

- **排序**是计算机科学中一个基本操作。
- “算法”有别于"程序"的一点：算法都以正确的输出**停机**。
- **数据结构**是一种存储和组织数据的方式。
- 没有任何一种单一的**数据结构**对所有用途均有效。



### 1.2 作为一种技术的算法

- 为求解相同问题而设计的不同算法在**效率**方面常常具有显著的区别。这些差别可能比由于硬件和软件造成的差别要重要得多。
- **插入排序**（所花时间：$c_1n^2$）在数据规模较小时，比**归并排序**（所花时间：$c_2n\log n$） 速度要快；但在数据规模较大时，归并排序的相对优势会增大。
- 在较大问题规模时，算法之间效率的差别才变得特别显著。



### 练习

<iframe src="http://nbviewer.jupyter.org/github/iphysresearch/Introduction_to_Algorithms_solution/blob/master/CLRS_1.ipynb" width="850" height="500"></iframe>

Ref: [solutions of Ch1  to "*Introduction to Algorithms*"](http://sites.math.rutgers.edu/~ajl213/CLRS/Ch1.pdf)
<iframe src="http://sites.math.rutgers.edu/~ajl213/CLRS/Ch1.pdf" style="width:1000px; height:500px;" width="100%" height=50%>This browser does not support PDFs. Please download the PDF to view it: <a href="http://sites.math.rutgers.edu/~ajl213/CLRS/Ch1.pdf">Download PDF</a></iframe>







---

[返回到首页](./CLRS.html) | [返回到顶部](./CLRS_1.html)

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


