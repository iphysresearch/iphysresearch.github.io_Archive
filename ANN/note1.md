---
title: 前馈性人工神经网络
date: 2018-11-12
---

[返回到首页](../index.html) | [返回到上一页](./index.html)

---


# 人工神经网络

[TOC]



## 第二章 前馈性人工神经网络

### 2.1 线性阈值单元组成的前馈网络

已知神经单元（线性阈值单元）的输入输出的基本关系：
$$
\begin{align}
s &= \sum^n_iw_ix_i+\theta \\
u& =g(s) = s\\
y&=f(u)=Sgn(u)=
\left\{\begin{matrix}
1, & u\geq0\\
0,-1, &  u<0
\end{matrix}\right.
\end{align}
$$
结合在一起就是：
$$
y=Sgn(\sum^n_{i=1}w_ix_i-\theta)
$$
其中输入 $x_i$ 和权 $w_i$ 都是一个 n 维实数向量，阈值 $\theta$ 是一个实数，输出 y 是一个二值变量。

 

#### M-P 模型

M-P 模型由 McCulloch & Pitts 提出，它有固定的结构和权组成，权分为两种类型：兴奋型（1）和抑制型（-1）：
$$
\sum^n_{i=1}w_ix_i=\sum^m_{j=1}x_{ej}-\sum^k_{j=1}x_{ij} \\
y = \left\{\begin{matrix}
1, \sum^m_{j=1}x_{ej}-\sum^k_{j=1}x_{ij} \geq \theta& \\ 
0, \sum^m_{j=1}x_{ej}-\sum^k_{j=1}x_{ij} < \theta& 
\end{matrix}\right.
$$

1. "或"运算：$(w_1, w_2 ,\theta)=(1, 1, 1)$  
2. "与"运算：$(w_1, w_2 ,\theta)=(1, 1, 1.5)
3. $
4. "非"运算：$(w, \theta)=(-1, 0)$
5. "XOR（异或）"运算：M-P 模型无法实现。

> **小结：**M-P 模型或网络的权、输入、输出都是二值变量（0或1），这同用逻辑门组成的逻辑式的实现区别不大，又由于其权无法调节，因而现在很少有人单独使用。



#### 感知机模型与学习算法：

感知机作为 M-P 模型的一种发展与推广，有**优点**：

- 非离散输入
- 非离散权值
- 权可以修正或学习
- 建立了完整的学习算法
- 多层感知机网络可以进行复杂的分类
- 为 BP 网络发展提供了模型和理论基础



1. 单层感知机

   输入向量：$x=[x_1,x_2,\cdots,x_n]^T\in \mathbb{R}^n (or \{\pm1\}^n)$

   输出值：
   $$
   y = Sgn(H(X))=\left\{\begin{matrix}
   1,& \text{if  } H(X) \geq0；\\
   -1,& \text{if  } H(X) < 0,
   \end{matrix}\right.
   $$
   where $H(X) = \sum^n_{i=0}w_ix_i=\sum^n_{i=1}w_ix_i-\theta$ ，即 $w_0=-\theta, x_0=1$。

2. 感知机的功能与线性可分性

   - 通过超平面

   $$
   w_1x_1+w_2x_2+\cdots+w_nx_n+\theta=0
   $$

   ​	划分参数空间 $\mathbb{R}^n$。

   - 若二分类问题，$x^i\rightarrow t_i\in\{\pm1\}, i=1,2,\cdots,N$

   - 若存在权向量 $W=[w_1,\cdots,w_n]^T$ 和阈值 $\theta $ 使得 
     $$
     Sgn(Wx^j-\theta)=Sgn(\sum^n_{i=1}w_ix^j_i-\theta) = t_j,\,\,\,\,\,\,\,\,\,\, j=1,2,\cdots,N
     $$
     则称样本组 $X=\{x^1, x^2,\cdots,x^N\}$ 对于目标 $T=\{t_1,t_2,\cdots,t_N\}$ 是**线性可分的**。 

   - 若线性可分，则线性分割面（超平面）存在无穷多种

3. 感知机学习算法

   > 1. 初始化权值 $w_i$ 和阈值 $\theta$ ；
   >
   > 2. 在第 k 时刻，取得 N 个样本 $x(k)\in\{x^1,x^2,\cdots,x^N\}$，$t(k)$ 是 $x(k)$ 的目标输出；
   >
   > 3. 计算第 k 时刻的实际输出：
   >    $$
   >    y(k) = Sgn(\sum^n_{i=1}w_ix_i(k)-\theta)
   >    $$
   >
   > 4. 对权值和阈值进行修正：
   >    $$
   >    \begin{align}
   >    w_i(k+1) &= w_i(k)+\eta(t(k)-y(k))x_i(k) \\
   >    \theta(k+1) &= \theta(k) - \eta (t(k)-y(k))
   >    \end{align}
   >    $$
   >    其中 $\eta$ 为学习率，一般为 0.1 
   >
   >    - 当令 $\delta(k)=t(x(k))-y(k)\in\{0,\pm2\}$，该算法符合 $\delta$ 学习率。
   >
   > 5. 返回到第2步，直到对所有样本 $W$ 和 $\theta$ 不再改变结束。

   - 注：算法停止时，没有误差，实际输出与目标输出完全一致

   - 注：权和阈值的初始值是选取的小随机数

   - 注：样本选取的顺序可以随机取，也可以按某固定顺序，遍历所有样本

   - 感知机学习算法的统一格式：
     $$
     w_i(k+1)=w_i(k)+\eta(t(x(k))-y(k))x_i(k),\,\,\,\,\, i=0,1,\cdots,n.
     $$
     其中 $w_0(k)=-\theta(k),x_0(k)=-1$。

4.  感知机学习算法的推导

   感知机学习算法实际上是一种最小均方误差的梯度算法（Least Means Square -- LMS） 在感知器上的推广，下面我们来进行推导：

   令 $w_0=\theta, x_0\equiv-1$，则有 $y=Sgn(\sum^n_{i=0}w_ix_i)$。

   对于输入：$x(k)=(x_0(k),x_1(k),\cdots,x_n(k))^T\rightarrow t(k)$，有线性单元的输出为：$\hat{y}(k)=\sum^n_{i=0}w_i(k)x_i(k)$，其与目标值的误差是：$e(k)=(t(k)-\hat{y}(k))^2$。

   采用向量符号，我们可以表示 LMS 算法如下：
   $$
   \begin{align}
   &\frac{\partial e(k)}{\partial W}=-2(t(k)-\hat{y}(k))x(k)\\
   &W(k+1)=W(k)-\eta\frac{\partial e(k)}{\partial W}=W(k)+2\eta(t(k)-\hat{y}(k))x(k)\\
   &w_i(k+1)=w_i(k)+2\eta(t(k)-\hat{y}(k))x_i(k)
   \end{align}
   $$
   因此，感知机学习算法是“线性单元”下的 LMS 算法在感知机学习上的应用。而感知机本身却不能直接推导出 LMS 算法，为什么？然而，感知机算法在一定条件下是收敛的。

5. 感知机学习算法的收敛性

   > 定理
   >
   > 若输入样本集 $X=\{x^1,x^2,\cdots,x^N\}$，对于二元目标集 $T=\{t_1,t_2,\cdots,t_N\}$ 是线性可分的，感知机学习算法能够在有限步内收敛到正确的 $W$ 。

   证明：先将算法进行一些简化和处理：

   （1）若存在 $W$ 使得 $Sgn(W^Tx^i)=t_i,i=1,2,\cdots,N$，则必然存在 $W$ 使得
   $$
   \left\{\begin{matrix}
   W^Tx^i>0 & t^i=1\\ 
   W^Tx^i<0 & t^i=-1
   \end{matrix}\right.
   $$
   ​	因此我们可假定正确的权向量 $W$ 满足上面的严格不等式。

   （2） 样本归一，目标统一化

   ​	对所有样本 $x^i$，令：$||x^i||=1,i=1,2,\cdots,N$。这样与原学习问题是等价的。

   ​	另外，我们将对应于 $t_i=1$ 的样本换为 $-x^i$，并使用 $-x^i$ 进行学习。这样所有学习样本的目标值全部为1，即 $t_i=1$ 。尽管如此，但与原问题具有等价的学习目标。

   （3）假设 $W^*$ 为一个所求的权向量，并令 $||W^*||=1$ 使得 $W^{*T}x^i>\delta>0,\forall i\in\{1,2,\cdots,N\}$ （$W^*$ 满足（1）），并且权向量序列为：$W(0),W(1),\cdots,W(k),W(k+1),\cdots$。

   （4）学习规则（实质性学习）
   $$
   \begin{align}
   W(k+1)&=W(k)+\eta(t(k)-y(k))x(k) \\
   &=W(k)+2\eta x(k)\\
   &=W(k)+\mu x(k) \;\;\;\;\; (\mu=2\eta>0)
   \end{align}
   $$
   下面我们证明：
   $$
   W(k)\rightarrow W^*
   $$
   令：
   $$
   c(k)=\frac{W^*\cdot W(k)}{||W^*||||W(k)||}=\frac{W^*\cdot W(k)}{||W(k||}
   $$
   则
   $$
   \begin{align}
   W^*\cdot W(k+1)&=W^*\cdot(W(k)+\mu x(k))\\
   &=W^*\cdot W(k)+\mu W^*\cdot x(k)\\
   &\geq W^*\cdot W(k)+\mu \delta\\
   &\geq W^*\cdot W(0)+(k+1)\mu\delta\\
   &\geq (k+1)\mu\delta\;\;\;\;\;(W^*\cdot W(0)\geq 0)
   \end{align}
   $$
   另一方面，我们有
   $$
   \begin{align}
   ||W(k+1)||^2
   &=W(k+1)\cdot W(k+1)\\
   &=||W(k)||^2+2\mu W(k)\cdot x(k)+||x(k)||^2\mu^2\\
   &=||W(k)||^2+2\mu W(k)\cdot x(k)+\mu^2\\
   &<||W(k)||^2+\mu^2\\
   &<||W(0)||^2+(k+1)\mu^2\\
   &<1+(k+1)\mu^2
   \end{align}
   $$
   结合上面两个不等式，得：
   $$
   \begin{align}
   &1\geq c(k)=\frac{W^*\cdot W(k)}{||W(k)||}\geq\frac{k\mu\delta}{\sqrt{1+k\mu^2}}\\
   &\Rightarrow 1+k\mu^2>k^2\mu^2\delta^2\\
   &\Rightarrow k\leq\frac{1+\sqrt{1+\frac{4\delta^2}{\mu^2}}}{2\delta^2}
   \end{align}
   $$
    因此，$k$ 是有限的，算法必然在有限步内停止或收敛。证明完毕。 





#### 多层感知机网络

- 非线性分类问题

  ![](https://i.loli.net/2018/10/18/5bc840d2b7c53.png)

- 多层感知机的结构

  ![](https://i.loli.net/2018/10/18/5bc841967b58c.png)

- 非线性分类的多层感知机设计

  考虑二维平面上点的非线性分类，则 $n=2$。

  ![](https://i.loli.net/2018/10/18/5bc842961ace2.png)

  我们一般使用三层网络，即
  $$
  \begin{align}
  h_j&=Sgn(\sum^2_{i=1}w_{ji}x_i-\theta_j), \;\;\;\;j=1,2,\cdots,m\\
  o_p&=Sgn(\sum^m_{j=1}w_jh_j-\theta)
  \end{align}
  $$
  每个隐单元时一个感知机，也就是一个超平面，这些超平面通过结合组成非线性分类区域（超多面体）。

  输入单元依逻辑单元表示：

  ![](https://i.loli.net/2018/10/18/5bc8432a90905.png)

- 感知机网络隐单元数的上限

  $$
  \begin{align}
  h_j&=Sgn(\sum^2_{i=1}w_{ji}x_i-\theta_j), \;\;\;\;j=1,2,\cdots,m\\
  o_p&=Sgn(\sum^m_{j=1}w_jh_j-\theta)
  \end{align}
  $$

  1. 隐单元数 $n_1$ 的上限

     假设样本集合目标为 $X=\{x^1,x^2,\cdots,x^N\}\rightarrow T=\{t_1,t_2,\cdots,t_N\}$。在输入空间上，存在 $N$ 个样 本，如果需要用 $n_1$ 超平面划分输入空间并进行任意二元分类，则所需最大的隐单元数 $n_1$ 满足：$n_1=N-1$.

     下面验证存在这样的三层感知机网络能够实现分类目标。考虑 $N\times N$ 的矩阵 D：

     ![](https://i.loli.net/2018/10/31/5bd9ba08bf872.png)

     显然 D 是满秩的。矩阵中各列中1，-1表示对样本的分类，x 表示分类结果不定。第 i 列对剩下的 N-i+1 个样本，找一个超平面，将一个样本与其他样本区分开，该样本处于正面，其他处于负面。对于以前考虑过的样本则不管它们处于哪面。这样只需 N-1 超平面（隐单元）就得到了 D。

     令 $C=(w_1,w_2,\cdots,w_{N-1},\theta)^T$，则
     $$
     DC= G\Leftrightarrow C=D^{-1}G, \,\,\,\,\,\,\,\,\,\,G=(t_1,t_2,\cdots,t_N)^T
     $$
     故 $n_1=N-1$。

  2. 隐单元数 $n_1$ 的下限 

     隐单元所形成的超平面将输入空间划分成一些区域。例如，在二维空间上，三条直线将空间 划分为一个封闭区域和21个开区域。但独立的区域仅有7个。 

     ![](https://i.loli.net/2018/10/31/5bd9c26388462.png)

     若让每个样本都落入一个独立的区域中，那么在每个样本输入时，对应的隐单元的输出是不同的，从而可以通过“与”、“或”的组合得到正确分类。这样，对 $n_1$ 数的求解，就变成了求在 n 维空间的 $n_1$ 区域分割，能得到最大的区域数。在数学可得到独立区域数为: 
     $$
     P(n_1,n)=\sum^n_{i=0}\binom{n_1}{i},n_1\geq n
     $$
     那么，对 N 个样本进行分类则要求:
     $$
     P(n_1,n)\geq N  \text{   或   } n_1\geq\min\{k,P(k,n)\geq N\}
     $$
     例如，在前面的图中，$n=2,n=3$ ，我们有
     $$
     P(3,2)=\sum^2_{i=0}\binom{3}{i}=1+3+3=7
     $$
     注意：这里的下限是从独立区域数的角度考虑的，未考虑分类结果，因此实际中某个分类问题所
      使用的隐单元数可能比上式得到的下限更小。

  3. 根据隐单元数的上下限进行网络设计

     - 输入单元个数：输入空间的维数

     - 输出单元个数：根据分类的类别数。若是二元分类，只需一个输出单元;若是多个类别，先进行类别二元编码。所用的编码位数即是输出单元的个数。

     - 隐单元数：根据样本个数，按其上限或下限公式确定。

     - 对 XOR 问题进行设计：

       - XOR问题：$N=4$，$(-1,-1),(1,1)\rightarrow-1$，$(-1,1),(1,-1)\rightarrow1$

       - 根据上限：$n_1=N-1=3$

       - 根据下限：$n_1=\min\{k:P(k,2)\geq4\}=2$

         ![](https://i.loli.net/2018/10/31/5bd9c40acb46c.png)





-  总结

  - M-P模型可实现一些基本的逻辑运算。 
  - 感知机可实现线性分类问题。
  - 感知机算法是收敛的。
  - 多层感知机网络可实现非线性分类问题。
  - 多层感知机网络的设计：隐单元数的限 

- 作业

  1. 证明XOR运算不能够由感知机来实现。

  2. 采用感知机学习算法求解下面二元分类问题：
     $$
     \begin{align}
     & x^1 = (0,0,0)\rightarrow1,& x^2 = (1,0,0)\rightarrow1,&\,\,\,\,\, x^3 = (0,1,0)\rightarrow1\\
     & x^4 = (0,0,1)\rightarrow1,& x^5 = (1,1,0)\rightarrow1,&\,\,\,\,\,x^6 = (1,0,1)\rightarrow-1\\
     &x^7 = (0,1,1)\rightarrow-1, &x^8 = (1,1,1)\rightarrow-1
     \end{align}
     $$















































































### 2.2 自适应线性单元与网络（略）



### 2.3 非线性连续变换单元组成的前馈网络

由非线性连续变换单元组成的前馈网络，简称为BP(Back Propagation) 网络。

#### 1. 网络的结构与数学描述

   - 非线性连续变换单元

     对于非线性连续变换单元，其输入、输出变换函数是非线性、单调上升、连续的即可。但在BP网络中，我们采用S型函数：
     $$
     \begin{align}
     u_i = & s_i=\sum^n_{j=1}w_{ij}x_j-\theta_i\\
     y_i=&f(u_i)=\frac{1}{1+e^{-u_i}}=\frac{1}{1+e^{-(\sum^n_{j=1}w_{ij}x_j-\theta_i)}}
     \end{align}
     $$
     函数 $f(u)$ 是可微的，并且
     $$
     f'(u)=\Big(\frac{1}{1+e^{-u}}\Big)'=f(u)(1-f(u))
     $$
     这种函数用来区分类别时，其结果可能是一种模糊的感念。当 $u>0$ 时，其输出不是1， 而是大于0.5的一个数，而当 $u<0$ 时，输出是一个小于0.5的一个数。若用这样一个单元进行分类，当输出是0.8时，我们可认为属于A类的隶属度(或概率)为0.8时，而属于B类的隶属度(或概率)为0.2。  

   - 网络结构与参数

     下面以四层网络为例，来介绍BP网络的结构和参数，一般情况类似。 

     ![](https://i.loli.net/2018/10/31/5bd9c788cca56.png)

     > 网络输入：$x=(x_1,x_2,\cdots,x_n)^T\in\mathbb{R}^n$
     >
     > 第一隐层输出：$x'=(x'_1,x'_2,\cdots,x'_{n_1})^T\in\mathbb{R}^{n_1}$
     >
     > 第二隐层输出：$x''=(x''_1,x''_2,\cdots,x''_{n_2})^T\in\mathbb{R}^{n_2}$
     > 网络输出：$y=(y_1,y_2,\cdots,y_m)^T\in\mathbb{R}^m$
     > 连接权：$w_{ji},w'_{kj},w''_{lk}$
     > 阈值：$\theta_i,\theta'_k,\theta''_l$

     网络的输入输出关系为：
     $$
     \left\{\begin{matrix}
     x'_j=f(\sum^n_{i=1}w_{ji}x_i-\theta_j), &j=1,2,\cdots,n_1 \\ 
      x''_k=f(\sum^{n_1}_{j=1}w'_{kj}x'_j-\theta'_k),& k=1,2,\cdots,n_2 \\ 
      y_l=f(\sum^{n_2}_{j=1}w''_{lk}x''_j-\theta''_l),&l=1,2,\cdots,m 
     \end{matrix}\right.
     $$
     显然可以将阈值归入为特别的权，从而网络的参数可用 $W$ 表示($W$ 为一个集合)。上述网络实现了一个多元连续影射：
     $$
     y=F(x,W): \mathbb{R}^n\rightarrow\mathbb{R}^m
     $$

   - 网络的学习问题

     - 学习的目标：通过网络(或 $F(x,W)$ )来逼近一个连 续系统，即连续变换函数 $G(x)$。
     - 学习的条件：一组样本(对) $S=\{(x^1,y^1),(x^2,y^2),\cdots,(x^N,y^N)\}$

     对于样本对 $(x^i,y^i)$，存在 $\mathbf{W}^i$ 使得
     $$
     y^i=F(x^i,W),W\in\mathbf{W}^i\subset\mathbb{R}^p,p=n\times n_1+n_1+n_1\times n_2+n_2+n_2\times m +m
     $$
     对于所有样本的解空间为：
     $$
     \mathbf{W}=\cap^N_{i=1}\mathbf{W}^i
     $$

   - Kolmogorov 定理

     Kolmogorov定理（映射神经网络存在定理，1950s）给定任何连续函数 $f:[0,1]^n\rightarrow\mathbb{R}^m,y=f(x)$，则 $f$ 能够被一个三层前馈神经网络所实现，其中网络的隐单元数为 $2n+1$。

     ![](https://i.loli.net/2018/11/01/5bda88e6744ff.png)
     $$
     \begin{align}
     z_j&=\sum^n_{i=1}\lambda^j\psi(x_i+j\epsilon)+j,&j=1,2,\cdots,2n+1&\\
     y_k&=\sum^{2n+1}_{j=1}g_k(z_j),&k=1,2,\cdots,m&
     \end{align}
     $$
     其中 $\psi$ 为连续单调递增函数，$g_j$ 为连续函数, $\lambda$ 为常数，$\epsilon$ 为正有理数。

     注意：定理未解决构造问题。

#### 2. BP 学习算法

   - 基本思想

     BP算法属于 $\delta$ 学习律，是一种有监督学习：

     ![](https://i.loli.net/2018/11/01/5bda8a2525045.png)

     对于辅助变量并将阈值归入权参数：
     $$
     x_0\equiv-1,w_{j0}=\theta_j,x'_0\equiv-1,w'_{k0}=\theta'_k,x''_0\equiv-1,w''_{l0}=\theta''_l
     $$

     则有：
     $$
     x'_j=f(\sum^n_{j=0}w_{ji}x_i),x''_k=f(\sum^{n_1}_{j=0}w'_{kj}x'_j),y_l=f(\sum^{n_2}_{k=0}w''_{lk}x''_k)
     $$
     考虑第 $\mu$ 个样本的误差：
     $$
     E_\mu=\frac{1}{2}||t^\mu-y^\mu||^2=\frac{1}{2}\sum^m_{l=1}(t^\mu_l-y^mu_l)^2
     $$
     进一步得总误差：
     $$
     E=\sum^N_{\mu=1}=\frac{1}{2}\sum^N_{\mu=1}||t^\mu-y^\mu||^2=\frac{1}{2}\sum^N_{\mu=1}\sum^m_{l=1}(t^\mu_l-y^\mu_l)^2
     $$
     引入权参数矩阵：
     $$
     \mathbf{W}=(w_{ji})_{n_1\times(n+1)},\mathbf{W}'=(w'_{kj})_{n_2\times(n_1+1)},\mathbf{W}''=(w''_{lk})_{m\times(n_2+1)}
     $$
     和总权参数向量:
     $$
     W=\begin{bmatrix}
      vec[\mathbf{W}]\\ 
      vec[\mathbf{W}']\\ 
      vec[\mathbf{W}'']
      \end{bmatrix}=(w_{10},w_{11},\cdots,w_{sg},\cdots,w_{cd})^T
     $$
     根据总误差得到一般性的梯度算法：
     $$
     \begin{align}
     E & = \sum^N_{\mu=1}E_\mu(\mathbf{W},t^\mu,x^\mu)\\
     \Delta w_{sg}&=-\eta\frac{\partial E}{\partial w_{sg}}=-\eta\sum^N_{\mu=1}\frac{\partial E_\mu(\mathbf{W},t^\mu,x^\mu)}{\partial w_{sg}}\\
     \Delta E & = \sum_{s,g}\frac{\partial E}{\partial w_{sg}}\delta w_{sg}=-\eta\sum_{s,g}\Big(\frac{\partial E}{\partial w_{sg}}\Big)^2
     \end{align}
     $$
     终止规则：
     $$
     \Delta E=0,E\approx0?,E\leq\epsilon(>0)
     $$
     这里用梯度法可以使总的误差向减小的方向变化，直到 $\Delta E$ 或梯度为零结束。这种学习方式使权向量 $\mathbf{W}$ 达到一个稳定解，但无法保证 $E$ 达到全局最优，一般收敛到一个局部极小解。

   - BP算法的推导

     令 $n_0$ 为迭代次数，则得一般性梯度下降法：
     $$
     \left\{\begin{matrix}
     w''_{lk}(n_0+1)=w''_{lk}(n_0)-\eta\frac{\partial E}{\partial w''_{lk}} \\ 
     w'_{kj}(n_0+1)=w'_{kj}(n_0)-\eta\frac{\partial E}{\partial w'_{kj}} \\ 
     w_{ji}(n_0+1)=w_{ji}(n_0)-\eta\frac{\partial E}{\partial w_{ji}} 
     \end{matrix}\right.
     $$
     其中 $\eta$ 为学习率，是一个大于零的较小的实数。 先考虑对于 $w''_{lk}$ 的偏导数：
     $$
     \frac{\partial E}{\partial w''_{lk}}=\sum^N_{\mu=1}\frac{\partial E_\mu}{\partial y^\mu_l}\frac{\partial y^\mu_l}{\partial u''^\mu_l}\frac{u''^\mu_l}{\partial w''_{lk}}=-\sum^N_{mu=1}(t^\mu_l-y^\mu_l)f'(u''^\mu_l)x''^\mu_k
     $$
     where $(u''^\mu_l=\sum^{n_2}_{k=0}w''_{lk}x''^\mu_k)$
     在上式中，$x''^\mu_k$ 为第 $\mu$ 个样本输入网络时，$x''_k$ 的对应值。另外
     $$
      f'(u''^\mu_l)=f(u''^\mu_l)(1-f(u''^\mu_l))=y^\mu_l(1-y^\mu_l)
     $$
     令 $\delta^\mu_l=(t^\mu_l-y^\mu_l)y^\mu_l(1-y^\mu_l)$
     则：
     $$
     w''_{lk}(n_0+1)=w''_{lk}(n_0)-\eta\frac{\partial E}{\partial w''_{lk}}=w''_{lk}(n_0)+\eta\sum^N_{\mu=1}\delta^\mu_lx''^\mu_k
     $$
     为了方便，引入记号：
     $$
     \left\{\begin{matrix}
     y_l=f(u''_l), &u''_l=\sum^{n_2}_{k=0}w''_{lk}x''_k \\ 
     x''_k=f(u'_k), &u'_k=\sum^{n_1}_{j=0}w'_{kj}x'_j \\ 
     x'_j=f(u_j), &u_j=\sum^n_{i=0}w_{ji}x_i 
     \end{matrix}\right.
     $$

     对于 $w'_{kj}$ 的偏导数，我们有：
     $$
     \begin{align}
     \frac{\partial E}{\partial w'_{kj}} 
     &=\sum^N_{\mu=1}\sum^m_{l=1}\frac{\partial E_\mu}{\partial y^\mu_l}\frac{\partial y^\mu_l}{\partial u''^\mu_l}\frac{\partial u''^\mu_l}{x''^\mu_k}\frac{x''^\mu_k}{u'^\mu_k}\frac{u'^\mu_k}{w'_{kj}}\\
     &=-\sum^N_{\mu=1}\sum^m_{l=1}(t^\mu_l-y^\mu_l)f'(u''^\mu_l)w''_{lk}f'(u'^\mu_k)x'^\mu_j\\
     &=-\sum^N_{\mu=1}\sum^m_{l=1}(t^\mu_l-y^\mu_l)y^\mu_l(1-y^\mu_l)w''_{lk}x''^\mu_k(1-x''^\mu_k)x'^\mu_j\\
     &=\sum^N_{\mu=1}\sum^m_{l=1}\delta^\mu_lw''_{lk}x''^\mu_k(1-x''^\mu_k)x'^\mu_j\\
     &=\sum^N_{\mu=1}\dot{\delta}^\mu_kx'^\mu_j
     \end{align}
     $$
     其中
     $$
     \dot{\delta}^\mu_k=\sum^m_{l=1}\delta^\mu_lw''_{lk}x''^\mu_k(1-x''^\mu_k)=x''^\mu_k(1-x''^\mu_k)\sum^m_{l=1}\delta^\mu_lw''_{lk}
     $$
     这样我们有：
     $$
     w'_{kj}(n_0+1)=w'_{kj}(n_0)+\eta\sum^N_{\mu=1}\dot{delta}^\mu_kx'^\mu_j
     $$
     类似的推导可得：
     $$
     w_{ji}(n_0+1)=w_{ji}(n_0)+\eta\sum^N_{\mu=1}\ddot{\delta}^\mu_jx^\mu_i
     $$
     其中 
     $$
     \ddot{\delta}^\mu_j=\sum^{n_1}_{k=1}w'_{kj}\dot{\delta}^\mu_k x'^\mu_j(1-x'^\mu_j)=x'^\mu_j(1-x'^\mu_j)\sum^{n_1}_{k=1}w'_{kj}\dot{\delta}^\mu_k
     $$

   - BP算法

     - Step 1. 赋予初值：$w_{sg}(0)=\alpha(Random(\cdot)-0.5),\alpha>0,(w_{sg}\rightarrow w_{ji},w'_{kj},w''_{lk})$
     - Step 2. 在 $n_0$ 时刻，计算 $x'^\mu_j,x''^\mu_k,y^\mu_l$ 及其广义误差 $\delta^\mu_l,l=1,2,\cdots,m;\dot{\delta}^\mu_k,k=1,2,\cdots,n_2;\ddot{\delta}^\mu_j,j=1,2,\cdots,n_1;$
     - Step 3. 修正权值：
     $$
     \left\{\begin{matrix}
     w''_{lk}(n_0+1)=w''_{lk}(n_0)+\eta\sum^N_{\mu=1}\delta^\mu_lx''^\mu_k \\ 
     w'_{kj}(n_0+1)=w'_{kj}(n_0)+\eta\sum^N_{\mu=1}\dot{\delta}^\mu_kx^\mu_j \\ 
     w_{ji}(n_0+1)=w_{ji}(n_0)+\eta\sum^N_{\mu=1}\ddot{\delta}^\mu_jx^\mu_i 
     \end{matrix}\right.
     $$
     - Step 4. 计算修正后的误差：
     $$
     E(n_0+1) = \sum^N_{\mu+1}E_\mu(\mathbf{W}(n_0+1),t^\mu,x^\mu)
     $$
     ​	若 $E(n_0+1)<\epsilon$ ($\epsilon$,预先给定)或 $|\frac{\partial E}{\partial w_{sg}}|<\epsilon$，算法结束，否则返回到Step 2。

   - BP算法的讨论：

     1. 这里的梯度是对于全部样本 求的，因此是一种批处理算法，即 Batch-way，它符合梯度算法，稳定地收敛到总误差的一个极小点而结束。(注意:按总误差小于 $\epsilon$ 可能导致算法不收敛.) 
     2. 实际中更常用的是对每个样本修改，即自适应算法，当每次样本是随机选取时，可通过随机逼近理论证明该算法也是收敛的。特点是收敛速度快。
     3. 为了使得算法既稳定，又具有快的收敛速度，可以使用批处理与自适应相补充的算法，即选取一组样本 (远小于全部样本)进行计算梯度并进行修正，其它不变。

#### 3. BP网络误差曲面的特性

   BP网络的误差公式为：
$$
   E=\frac{1}{2}\sum^N_{\mu=1}\sum^{m}_{l=1}(t^\mu_l-y^\mu_l)^2
$$
   $y_l=f(u_l)$ 是一种非线性函数，而多层的BP网络中 $u_l$ 又是上一层神经元状态的非线性函数，用 $E_\mu$ 表示其中一个样本对应的误差，则有：
$$
   \begin{align}
   E_\mu &= \frac{1}{2}\sum^N_{\mu=1}\sum^{m}_{l=1}(t^\mu_l-y^\mu_l)^2 = E(\mathbf{W},t^\mu,x^\mu)\\
   E &= \sum^N_{\mu=1}E(\mathbf{W},t^\mu,x^\mu)
   \end{align}
$$
   可见，$E$ 与 $\mathbf{W}$ 有关，同时也与所有样本对有关，即与 $S=\{(x^1,y^1),(x^2,y^2),\cdots,(x^N,y^N)\}$ 有关。

   假定样本集 $S$ 给定，那么 $E$ 是 $\mathbf{W}$ 的函数。在前面考虑的4层网络中，权值参数的总个数为：
$$
   n_\mathbf{W} = n(n_1+1)+n_1(n_2+1)+n_2(m+1)
$$

   那么在加上 $E$ 这一维数，在 $n_\mathbf{W}+1$ 维空间中，$E$ 是一个具有极其复杂形状的曲面。如果在考虑样本，其形状就更为复杂，难于想象。从实践和理论上，人们得出了下面三个性质：

   1. **平滑区域**

      ![](https://i.loli.net/2018/11/01/5bda9b71b0e08.png)
      $$
      \delta^\mu_l = (t^\mu_l-y^\mu_l)y^\mu(1-y^\mu_l)
      $$

      广义误差 $\neq$ 误差

   2. **全局最优解 $\mathbf{W}^*$ 不唯一**

      $\mathbf{W}^*$ 中的某些元素进行置换依然是全局最优解，这从下面的简单模型可以看出。

      ![](https://i.loli.net/2018/11/01/5bda9c735bdff.png)

   3. **局部极小**

      一般情况下，BP算法会收敛到一个局部极小解，即
      $$
      \mathbf{W}(n_0) \rightarrow \mathbf{W}^0
      $$
      当 $E(\mathbf{W}^0)<\epsilon$，算法以希望误差收敛;
      当 $E(\mathbf{W}^0)\geq\epsilon$，算法不以希望误差收敛，但可按 梯度绝对值小于预定值结束。

#### 4. 算法的改进

   - 变步长算法( $\eta$ 是由一维搜索求得)

     Step 1. 赋予初始权值 $\mathbf{W}(0)$ 和允许误差 $\epsilon>0$ ;
     Step 2. 在时刻 $n_0$，计算误差 $E(\mathbf{W}(n_0))$ 的负梯度 (方向)：
     $$
     d^{(n_0)} = -\nabla E(\mathbf{W})|_{\mathbf{W}=\mathbf{W}(n_0)}
     $$
     Step3. 若 $||d^{(n_0)}||$，结束；否则从 $\mathbf{W(n_0)}$ 出发，沿 $d^{(n_0)}$ 做一维搜索，求出最优步长 $\eta(n_0)$：
     $$
     \eta(n_0) = \arg \min_\eta E(\mathbf{W}(n_0)+\eta d^{(n_0)})
     $$
     Step4. $\mathbf{W}(n_0+1) = \mathbf{W}(n_0)+\eta(n_0)d^{(n_0)}$，转Step2。

     步长(学习率) $\eta(n_0)$ 的确定方法：
     (a) 求最优解：对 $\eta$ 求导数，并令其为零，直接求解：
     $$
     \frac{\partial E(\mathbf{W}(n_0)+\eta d^{(n_0)})}{\partial \eta} = 0
     $$
     (b) 迭代修正法：令 $\Delta E = E(\mathbf{W}(n_0)+\eta d^{(n_0)})-E(\mathbf{W}(n_0))$
     $$
     \eta^{new}=
     \begin{cases}
      \eta^{old}\psi& \text{ if } \Delta E<0 \\ 
      \eta^{old}\beta& \text{ if } \Delta E>0 
     \end{cases}
     $$
     其中 $\psi=1+\delta, \beta=1-\delta, \delta >0$ .

   - 加动量项

     为了防止震荡并加速收敛，可采用下述规则：
     $$
     \begin{align}
     \mathbf{W}(n_0+1) 
     &= \mathbf{W}(n_0)+\eta(n_0)d^{(n_0)} + \alpha\Delta\mathbf{W}(n_0) \\
     &= \mathbf{W}(n_0)+\eta(n_0)(d^{(n_0)} +\frac{\alpha\eta(n_0-1)}{\eta(n_0)}d^{(n_0-1)})
     \end{align}
     $$
     其中，$\alpha\Delta\mathbf{W}(n_0)$ 为动量项. $( \Delta\mathbf{W}(n_0)=\mathbf{W}(n_0)-\mathbf(n_0-1),0<\alpha<1 )$

     注意: 上式类似于共轭梯度法的算式，但是这里 $d^{(n_0)},d^{(n_0-1)}$ 不共轭。因此可能出现误差增加的现象，即 $\Delta E>0$，这时可令 $\alpha=0$，即退回到原来的梯度算法。

   - 加入 $\gamma$ 因子

     当算法进入平坦区，即 $(1-y^\mu_l)y^\mu_l\approx 0$，则 $|u''^\mu_l|\rightarrow +\infty$。为了消除或减弱这种现象，引入 $\gamma$ 因子，使得:
     $$
     y^\mu_l = \frac{1}{1+\exp(-u''^\mu_l)}, \\
     u''^\mu_l = \sum^{n_2}_{k=0}w''_{lk}x''^\mu_k/\gamma^\mu_l, \gamma^\mu_l>1.
     $$

   - 模拟退化方法

     在所有权上加一个噪声，改变误差曲面的形状， 使用模拟退火的机制，使算法逃离局部极小点，达到全局最优而结束。

#### 5. BP 网络的设计

   - 输入输出层的设计

     BP网络输入、输出层单元个数是完全根据实际问题来设计的，我们分三种情况讨论：

     1. 系统识别
        $$
        y=F(X):\mathbb{R}^n\rightarrow\mathbb{R}^m
        $$
        这时输入单元个数为 $n$；输出单元个数为 $m$。

     2. 分类问题
        $$
        S = \{(x^1,t^1), (x^2,t^2), \cdots, (x^N,t^N)\}, t^i\in\{C^1,C^2,\cdots,C^m\}
        $$
        
        (a). 若 $t^i\leftrightarrow C^j$，则令 $t^i=\lambda j(\lambda>0)$，这样输出层仅需要一个单元。 
        (b). 若 $t^i\leftrightarrow C^j$，则令: $t^i=(0,\cdots,0,1,0,\cdots,0)^T$ (第 $j$ 个分量为 1，其余分量为 0)
        这样输出层则需要 $m$ 个单元。 
        (c). 二进制编码方法
        对 $C^1,C^2 ,\cdots,C^m$ 进行二进制编码，编码位数为 $\log_2m$，这样输出层仅需 $\log_2m$ 个单元。
        

   - 隐单元数与映射定理

     1989年，R. Hecht-Nielson证明了任何一个闭区 间内的连续函数都可以用一个三层(仅有一个隐层)BP网络来逼近(任意给定精度)。

     **引理2.1** 任意给定一个连续函数 $g\in C(a,b)$ 及精度 $\epsilon>0$，必存在一个多项式 $p(x)$，使得不等式 $|g(x)-p(x)|<\epsilon$ 对任意 $x\in [a,b]$ 成立。

     **引理2.2** 任意给定一个周期为 $2\pi$ 的连续函数 $g\in C_{2\pi}$ 及精度 $\epsilon>0$，必存在一个三角函数多项式 $T(x)$，使得 $|g(x)-T(x)|<\epsilon$ 对于$\forall x\in \mathbb{R}$ 成立。

     在 $n$ 维空间中，任一向量 $x$ 都可表示为
     $$
     x = c_1e_1+c_2e_2+\cdots+c_ne_n
     $$
     其中 $\{e_1,e_2,cdots,e_n\}$ 为 $\mathbb{R}^n$ 的一个正交基。同样考虑连续函数空间 $c[a,b]$ 或 $c_{2\pi}$，必然存在一组正交函数序列 $\{\psi_k(x)^\infty_{k=1}\}$，那么对 $\forall g(x)\in c[a,b]$, 则
     $$
     g(x) = \sum^\infty_{k=1}c_k\psi_k(x) = \sum^N_{k=1}c_k\psi_k(x)+\epsilon_N(x)
     $$
     或对 $\forall g_F(x)\in c_{2\pi}$，则有
     $$
     g(x) = \sum_{k}c_ke^{2\pi i kx} = \sum^{+\infty}_{k=-\infty}c_ke^{2\pi ikx} = \sum^{N}_{k=-N}c_ke^{2\pi ikx}+\epsilon_N(x)
     $$
     其中 $c_k=\int g(x)e^{-2\pi ikx}dx$ 为傅里叶系数。

     当 $N$ 充分大时，对每个 $x$ 成立:
     $$
     \begin{align}
     g^N_F(x) &= \sum^N_{k=-N}c_ke^{2\pi ikx} \\
     |g_F(x) &- g^N_F(x)| < \epsilon (>0)
     \end{align}
     $$

     进一步考虑 $c([0,1]^n)$ 中的多元连续函数：

     $$
     g(x) : [0,1]^n \rightarrow \mathbb{R}, g(x) \in c([0,1]^n)
     $$

     根据傅里叶级数展开理论，若

     $$
     \int_{[0,1]^n}|g(x)|dx_1\cdots dx_n < \infty
     $$

     则同样存在一个 $N$ 步傅里叶级数和函数：

     $$
     g_F(x,N,g) = \sum^N_{k_1=-N}\cdots\sum^N_{k_n=-N}c_{k_1\cdots k_n}e^{2\pi i\sum^n_{j=1}k_ix_i} = \sum^{(N,\cdots,N)}_{K=(-N,\cdots,-N)} c_{k_1\cdots k_n}e^{2\pi iK^Tx}
     $$

     其中系数为：$c_{k_1\cdots k_n} = \int_{[0,1]^n} g(x) e^{-2\pi iK^Tx}dx$，并且当 $N\rightarrow\infty$ 时，满足

     $$
     g_F(x,N,g)\rightarrow g(x)
     $$

     即 $g_F(x,\infty,g)$ 在 $[0,1]^n$ 可以完全收敛达到 $g(x)$。

     现在考虑对一个任意连续映射: $h(x):[0,1]^n\rightarrow \mathbb{R}^m$，其中 $h(x)=[h_1(x),\cdots,h_n(x)], h_j(x)\in c([0,1]^n)$，则 $h(x)$ 的每个分量也都可以用上面的傅立叶级数表示，依此就可以得到下面的影射定理(定理中所考虑的三层网络输出单元为线性单元)。

     > **映射定理(Hecht-Nielsen)**：给定任意精度 $\epsilon>0$，对于一个连续影射 $h(x):[0,1]^n\rightarrow\mathbb{R}^m$，其中: 
     > $$
     > \int_{[0,1]^n} ||h(x)||dx_1\cdots dx_n<\infty
     > $$
     > 那么，必存在一个三层BP神经网络来逼近函数，使得在每点上的误差不超过 $\epsilon$。
     >

     **证明**：由于输出单元是独立的，分别与 $h(x)$ 的每个分量函数相对应，我们仅需要 对单个输出单元和分量函数来证明。

     ![](https://i.loli.net/2018/11/12/5be921841583c.png)

     根据傅立叶级数理论，对于 $h(x)$ 的分量 $h_j (x)$，则

     $$
      |h_j(x)-g_F(x,N,h_j)|<\delta_1(>0), \forall x\in[0,1]^n
     $$

     其中 $g_F(x,N,h_j)$ 是 $h_j(x)$ 的 $N$ 步傅立叶级数和函数:

     $$
      g_F(x,N,h_j) = \sum^{(N,\cdots,N)}_{K=(-N,\cdots,-N)}c_{k_1,k_2,\cdots,k_n}e^{2\pi iK^Tx}, c_{k_1\cdots k_n} = \int_{[0,1]^n} h_j(x) e^{-2\pi iK^Tx}dx
     $$

     下面证明傅立叶级数中任意三角函数可以用三层 BP 子网络来逼近，那么通过傅立叶级数的线性组合就可以保证用三层BP网络来逼近函 $h_j(x)$。

     考虑结构为 $n-n_1-1$ 的三层BP网络，其输出为:

     $$
      y=\sum^{n_1}_{j=1}w_jf(\sum^n_{k=1}w_{jk}x_k-\theta_j)
     $$

     我们来证明输出函数 $y$ 能够逼近任何三角函数: 令 $\sin(2\pi K^Tx)=\sin(u)$ $2\pi K^Tx=2\pi\sum^n_{l=1}k_lx_l\in[d,e]$, $\sum^{n_1}_{k=1}w_{jk}x_k -\theta_j = u'_j-\theta_j = \beta_j(u - \alpha_j)$

     考虑函数 $f(\beta_j(u-\alpha_j))$，当 $\beta_j\rightarrow+\infty$，趋向于单位阶跃函数(见下图)，则

     $$
     S = (\alpha,\beta,\mathbb{W},u) = \sum^{n_1}_{j=1}w_jf(\beta_j(u-\alpha_j))
     $$

     为一些近似单位阶跃函数的线性叠加，故当 $n_1$ 充分大时，我们可将区间 $[d,e]$ 充分的细分，选取 $\alpha_j$ 和 $\beta_j$，使得 $|S(\alpha,\beta,\mathbb{W},u) - \sin(u)|<\delta_2(>0)$，或

     $$
      |\sum^{n_1}_{j=1}w_jf(\sum^n_{k=1}w_{jk}x_k-\theta)j - \sin(2\pi K^Tx)| < \delta_2
     $$

     即得：

     $$
      \sum^{n_1}_{j=1}w_jf(\sum^n_{k=1}w_{jk}x_k-\theta)j \approx \sin(2\pi K^Tx)
     $$

     对于 $h_j(x)$，我们有下面的展开：

     $$
     \begin{align}
      h_j(x) 
      &\approx g_F(x,N,h_j) = \sum^{(N,\dots,N)}_{K=(-N,\cdots,-N)}c_Ke^{2\pi iK^Tx} \\
      &= \sum^{(N,\dots,N)}_{K=(-N,\cdots,-N)} c_K (\sin(2\pi K^T x) +i\cos(2\pi K^Tx)) \\
      &= \sum^{(N,\dots,N)}_{K=(-N,\cdots,-N)} a_K \sin(2\pi K^Tx) + b_K \cos(2\pi K^Tx)  + i \sum^{(N,\dots,N)}_{K=(-N,\cdots,-N)} a'_K\sin(2\pi K^Tx)+ b'_K\cos(2\pi K^Tx)
      \end{align}
     $$

     使用充分多的隐单元，可得

     $$
     y(x) = \sum^{(N,\dots,N)}_{K=(-N,\cdots,-N)} a_K S(\alpha_K,\beta_K,\mathbb{W}_K,u_K) + b_K S(\alpha'_K,\beta'_K,\mathbb{W}'_K,u_K) 
     $$

     令 $h_F(x) = \sum^{(N,\dots,N)}_{K=(-N,\cdots,-N)} a_K \sin(2\pi K^Tx) + b_K \cos(2\pi K^Tx) $

     $$
     \begin{align}
     |h_j(x) - y(x)| 
     &= |h_j(x) - h_F(x)+h_F(x) - y(x)| \\
     &\leq |h_j(x)-h_F(x)| + |h_F(x) -y(x)| \\
     &\leq |h_j(x)-h_F(x)| + \\
     &|\sum^{(N,\dots,N)}_{K=(-N,\cdots,-N)} a_K (\sin(u_K) - S(\alpha_K,\beta_K,\mathbb{W}_K,u_K) ) + b_K (\cos(u_K) - S(\alpha'_K,\beta'_K,\mathbb{W}'_K,u_K)) | \\
     &\leq \delta_1 +\delta_2 \sum^{(N,\dots,N)}_{K=(-N,\cdots,-N)} |a_K|+|b_K| \leq \epsilon ,  (\forall x\in [0,1]^n)
     \end{align}
     $$
     
     证毕。

   - 隐单元数的选择
     
     隐单元数：小，结构简单，逼近能力差，不收敛；大，结构复杂，逼近能力强，收敛慢。
     对于用作分类的三层BP网络，可参照多层感知机网络的情况，得到下面设计方法：
     
     (a). 

     $$
     N < \sum^n_{i=0}\binom{n_1}{i}, (i>n_1,\binom{n_1}{i}=0)
     $$
     其中 $N$ 为样本个数，选取满足上式最小的 $n_1$.
     
     (b). $n_1=\sqrt{n+m}+a$ $a\in\{1,2,\cdots,10\}$ $n_1=\log_2n$

   - 网络参数初始值的选取
     
     初试权：随机，比较小(接近于0)，保证状态值较小，不在平滑区域内。



#### 6. BP 网络的应用

   (i). 模式识别、分类。用于语音、文字、图象的 识别，用于医学图象的分类、诊断等。
   (ii). 函数逼近与系统建模。用于非线性系统的建模，拟合非线性控制曲线，机器人的轨迹控制，金融预测等。

   (iii). 数据压缩。在通信中的编码压缩和恢复，图象数据的压缩和存储及图象特征的抽取等。

   **例1. 手写数字的识别** 

     由于手写数字变化很大，有传统的 统计模式识别或句法识别很难得到高的识别率，BP网络可通过对样本的学习得到较高的学习率。为了克服字体大小不同，我们选取这些数字的一些特征值作为网络输入。(可提取)特征如: 

   - 1，2，3，7 :具有两个端点; 

   - 0，6，8，9:具有圈; 
   - 2: 两个端点前后; 

   ![](https://i.loli.net/2018/11/12/5be92d5128196.png)

   对于一个样本，若具有那个特征，所对应的特征输入单元取值为1，否则为0。我们可选择34个特征，即输入单元个数为34。输出可取10个单元，即1个输出单元对应一个数字(该单元输出为1，其它为0)。如果选取200个人所写的1000个样本进行学习，使用三层BP网络，隐层单元数 $n_1$ 应如何选择呢?

   根据前面的经验公式，可得到下面结果:
$$
   \sum^n_{i=0}\binom{n_1}{i}>1000 \Rightarrow \min n_1=10 \\
   \begin{align}
   n_1 &= \sqrt{m+n}+a = \sqrt{44}+a = 8 \sim 17\\
   n_1 &= \log_234\approx 6
   \end{align}
$$
   在实际中，我们选择 $n_1=14$。通过对1000个样本的学习所得到的网络对6000个手写数字的正确识别率达到95%。 

   **例2.非线性曲线的拟合。**

   在控制中往往希望产生一些非线性的输出输入关系。例如，已知一个机械臂取物的轨迹，根据这个轨迹可计算出机械臂关节的角度 $\theta_1$ 和 $\theta_2$ (两个关节)，按照机械臂的 $\theta$ 要求应该反演计算出驱动马达的力或频率这是一个相当复杂的计算问题。但我们可采用BP网络对一些样本的学习得到这些非线性曲线的拟合，根本无须知道机械臂的动力学模型。

   在一维情况下，就是拟合 $y=g(x)$，其中 $x$ 表示 $\theta$ 角，$y$ 为所对应的马达驱动力。在某些位置，我们容易得到这些对应值，因此可以得到足够的样本。 

   ![](https://i.loli.net/2018/11/12/5be92f017d80e.png)

   **例3.数据压缩**

   ![](https://i.loli.net/2018/11/12/5be92f4e7da2c.png)

   ![](https://i.loli.net/2018/11/12/5be92f25c1a66.png)

   BP网络相当于一个编码、解码器，$n_1$ 越小，压缩率越小，但太小可能达不到唯一译码的要求。





### 作业

1. 推导k层前馈网络的BP算法，并且考虑跨层连接的权值。

2. 采用2-2-1结构的前馈网络通过BP算法求解XOR问题，其中逼近精度 $\epsilon = 0.001$。

3. 采用2-m-1结构的前馈网络通过BP算法来逼近定义于[0,1]连续函数 $y=f(x)=1/(1+\sqrt{x^2_1+x^2_2})$，其中逼近精度 $\epsilon=0.01$。请按均匀格点选择 10000 个样本点，随机选取5000个作为训练样本，且剩余的5000个作检测样本。根据该学习问题，可选取三种不同的m值，并观察所得网络在检测样本上的误差变化。 











---

[返回到首页](../index.html) | [返回到顶部](./note1.html)


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



