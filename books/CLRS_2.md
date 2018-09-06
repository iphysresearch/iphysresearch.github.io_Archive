---
title: 第2章 算法基础
date: 2018-09-06
---

[返回到首页](./CLRS.html)

---

[TOC]

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

  - *伪代码中的一些规定*：书中P11
    - **缩进表示块结构；**
    - **while、for与repeat-until等循环结构以及if-else等条件结构与C、C++、Java、Python和Pascal中的那些结构具有类似的解释；**
    - **符号“//”表示该行后面部分是个注释；**
    - **形如i=j=e的多重赋值将表达式e的值赋给变量i和j；它被处理成等价于赋值j=e后跟着赋值i=j；**
    - **变量是局部给定过程的；**
    - **数组元素通过“数组名[下标]”这样的形式来访问的；**
    - **复合数据通常被组织成对象，对象又由属性组成。**
    - **我们按值把参数传递给过程：被调用过程接收其参数自身的副本；**
    - **一个return语句立即将控制返回到调用过程的调用点；**
    - **布尔运算符“and”和”or”都是短路的；** 短路表达式(if...) 会先处理第一个表达式。
    - **关键字error表示因为已被调用的过程情况不对而出现一个错误。**

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

- **循环不等式**：其主要是用来帮助我们理解和证明算法的正确性。

  - 对于插入排序来说，所谓循环不等式的特点，即【当前已排序】+【保留其他原有数据】，其中【保留其他原有数据】是尚未处理的序列。
  - 必有证明：（颇为像数学归纳法）
    - **初始化**：第一次迭代的正确性
    - **维护/保持**：某次迭代正确后，下一次迭代也会正确
    - **终结/终止**：最后一次迭代，循环终止。

- **数据结构不变式**。

- 在插入算法中，可以将其中寻找插位的方法改为利用**二分查找**实现快速定位**序位**。





### 2.2 分析算法

- 在分析算法的过程中，通常我们想度量的是**计算时间**。
- 我们算法的分析，都假定一种通用的单处理器计算模型——**随机访问机**（random-access machine, **RAM**），没有并发操作。
- RAM 模型包含真实计算机中常见的指令：算术指令、数据移动指令和控制指令。每条这样的指令所需时间都为常量。

---

- 插入排序算法的分析：

  - 过程 INSERTION-SORT 需要的时间**依赖于输入**，不仅指的是输入的数目（所谓**输入规模**，如排序输入中的项数），也依赖于它们**已被排序的程度**。

  - 一个算法在特定输入上的运行时间是指执行的基本操作数或步数。

  - 我们假定：执行每行伪代码需要常量时间，即第 i 行的每次执行需要时间 $c_i$，其中 $c_i$ 是一个常量。

    ```
    for j = 2 to A.length									c1		n
    	key = A[j]											c2		n-1
    	// Insert A[j] into the sorted sequence A[1..j-1].	0		n-1	
    	i = j-1												c4		n-1
    	while i > 0 and A[i] > key							c5		sum(t_j, j=2, n)
    		A[i+1] = A[i]									c6		sum(t_j-1, j=2, n)
    		i = i -1										c7		sum(t_j-1, j=2, n)
    	A[i+1] = key										c8		n-1
    ```

    所以有运行时间：
    $$
    T(n) = c_1n+c_2(n-1)+c_4(n-1)+c_5\sum^n_{j=2}t_j+c_6\sum^n_{j=2}(t_j-1)+c_7\sum^n_{j=2}(t_j-1)+c_8(n-1)
    $$

  - 最佳情况：（输入数组已排好序）
    $$
    T(n)=(c_1+c_2+c_4+c_5+c_8)n-(c_2+c_4+c_5+c_8)
    $$
    它是 $n$ 的线性函数。

  - 最坏情况：（输入数组已反向排序）
    $$
    T(n)=(c_5+c_6+c_7)n^2/2+(c_1+c_2+c_4+c_5/2-c_6/2-c_7/2+c_8)n-(c_2+c_4+c_5+c_8)
    $$
    它是 $n$ 的二次函数。

- 最坏情况和平均情况分析：往往我们集中于只求**最坏情况运行时间**，理由有三：

  1. 可以给出一个**上界**；
  2. 最坏情况**经常出现**；
  3. “平均情况”往往与最坏情况一致**一样差**。

- 增长量级

  - 我们真正感兴趣的是运行时间的**增长率**或**增长量级。**
  - 如果一个算法的最坏情况运行时间具有比另一个算法更低的增长量级，那么我们通常认为前者比后者更有效。





### 2.3 设计算法

插入排序使用的是**增量法**，而归并排序使用的是**分治法**。



#### 2.3.1 分治法

- 算法在结构上是**递归的**：为了解决一个给定的问题，苏散发一次或多次递归地调用其自身以解决紧密相关的若干子问题。

- 结构上递归的算法典型地遵循**分治法**的思想：将原问题分解为几个规模较小但类似于原问题的子问题，递归地求解这些子问题，然后合并这些子问题的解来建立原问题的解。

- 分治模式在每层递归时都有三个步骤：
  1. 分解；
  2. 解决；
  3. 合并。

- 归并排序算法就是如此三个步骤。

- 归并排序算法的**关键操作**时：**“合并”步骤中两个已排序序列的合并。**
  - 归并排序中子程序的伪代码：【MERGE】

    ```pseudocode
    n1 = q - p + 1
    n2 = r - q
    let L[1..n1+1] and R[1..n_2+1] be new arrays
    for i=1 to n1
    	L[i] = A[p + i -1]
    for j=1 to n2
    	R[i] = A[q + j] 
    L[n1 + 1] = \infinty // 哨兵
    R[n2 + 1] = \infinty  
    i = 1
    j = 1
    for k = p to r
    	if L[i] <= R[j]
    		A[k] = L[i]
    		i = i + 1
    	else A[k] = R[j]
    		j = j + 1
    ```

  - 上面的伪代码我已无力吐槽，还是看 Python 代码吧：

    ```python
    def MERGE(A, p, q, r):
        L = [value for i, value in enumerate(A) if i in range(p, q+1)]
        R = [value for j, value in enumerate(A) if j in range(q+1, r+1)]
        L += [float('inf')]
        R += [float('inf')]
        i, j= 0, 0
        for k in range(p, r+1):
            if L[i] <= R[j]:
                A[k] = L[i]
                i += 1
            else:
                A[k] = R[j]
                j += 1
    >>> A = [2, 4, 5 ,7, 1, 2, 3, 6]
    >>> MERGE(A, 0, 3, 7)
    ```

  - 【MERGE】 的运行时间是 $\Theta(n)$，其中 n=r-p+1。

- 归并排序算法 Python 代码：【MERGE-SORT】

  ```python
  def MERGE_SORT(A, p, r):
  	if p < r:
          q = (p+r)//2
          MERGE_SORT(A, p , q)
          MERGE_SORT(A, q+1, r)
          MERGE(A, p, q, r)
  >>> A = [2, 4, 5 ,7, 1, 2, 3, 6]
  >>> MERGE_SORT(A, 0, len(A)-1)
  ```






#### 2.3.2 分析分治算法

- **递归方程**或**递归式**

  规模为 n 的一个问题的运行时间：
  $$
  T(n) = \Big\{\begin{align}
  &\Theta(1) &n\leq c \\
  & aT(n/b) + D(n) + C(n) & others
  \end{align}
  $$

  - 问题规模足够小 $n\leq c$，直接求解需要常量时间：$\Theta(1)$
  - 把原问题分解为 a 个子问题，每个子问题的规模是原问题的 1/b，求解规模为 n/b 的子问题需要时间 T(n/b)。
  - 分解问题成子问题需要时间 $D(n)$，合并子问题为原问题需要时间 $C(n)$

-  归并算法的递归式（最坏情况）：
  $$
  T(n) = \Big\{\begin{align}
  &\Theta(1) &n=1 \\
  & 2T(n/b) +\Theta(n) & n>1
  \end{align}
  $$

  - 主定理可证明：$T(n) = \Theta(n\lg n)$











Ref: [solutions of Ch2  to "*Introduction to Algorithms*"](http://sites.math.rutgers.edu/~ajl213/CLRS/Ch2.pdf)





---

[返回到首页](./CLRS.html) | [返回到顶部](./CLRS_2.html)

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


