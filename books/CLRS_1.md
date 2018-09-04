





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



## 第2章 算法基础



### 2.1 插入排序

- 插入排序算法是一种**就地算法**，顾名思义，就是指空间用量是一个常数 $O(1)$
- 我们希望排序的数也称为**关键词**（key），也就是说对一系列 key 进行排序。
- 该算法对小规模数据的效率比较高。
- 伪代码与真码的区别在于：
  1. 伪代码用最清晰、最简洁的表示方法来说明给定的算法；
  2. 伪代码通常不关心软件工程的问题。

- 插入排序的伪代码：【INSERTION-SORT】

  ```pseudocode
  for j = 2 to A.length
  	key = A[j]
  	// Insert A[j] into the sorted sequence A[1..j-1].
  	i = j-1
  	while i > 0 and A[i] > key
  		A[i+1] = A[i]
  		i = i -1
  	A[i+1] = key
  ```

- Python 版本：

  ```python
  for j in range(1, len(A)):
      key = A[j]
      i = j - 1
      while i >= 0 and A[i] > key:
          A[i+1] = A[i]
          i -= 1
      A[i+1] = key
  ```

- **循环不等式**：其主要是用来帮助我们理解算法的正确性。

  - 对于插入排序来说，所谓循环不等式的特点，即【当前已排序】+【保留其他原有数据】，其中【保留其他原有数据】是尚未处理的序列。









