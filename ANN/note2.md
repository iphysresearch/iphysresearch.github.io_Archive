



# 反馈型人工神经网络





[TOC]







## 动力系统的稳定性

### A. 网络的模型与分类

反馈式网络的结构如下图：

![](https://i.loli.net/2018/11/15/5bed1157bf7eb.png)

这种网络中的每个神经元的输出都与其它神经元的输入相连接，其输入输出关系如下：
$$
\left\{\begin{align}
&s_i = \sum^n_{j=0}w_{ij}V_j+I_i, &(w_{i0}=\theta_i,V_0\equiv-1)\\ 
&x_i = g(s_i), &(x_i=u_i,\text{为神经元的状态})\\ 
&V_i = f(x_i), &(\text{神经的输出})
\end{align}\right.
$$
注意：在反馈网络中，神经元这一时刻的状态(或输出)与上一时刻的状态(或输出)有关，与前馈网络不同。

- 函数的选取：
  - $x_i=g(s_i)=s_i$, $f(x_i)=Sgn(x_i) \leftrightarrow \text{离散型反馈网络}$
  - 如果通过下列方程给出输入输出关系，则称网络为连续型反馈网络：

  $$
  \left\{\begin{align}
  &\frac{dX_i}{dt} = - \frac{1}{\tau_i}X_i+s_i \\
  &V_i=f(X_i), f(\cdot)\text{为一个连续单调上升的有界函数。}
  \end{align}\right.
  $$

- 演化过程与轨迹：令 $I=(I_1,I_2,\cdots,I_n)^T\in\mathbb{R}^n,X=(x_1,x_2,\cdots,x_n)^T\in\mathbb{R}^n,V=(V_1,V_2,\cdots,V_n)^T\in\mathbb{R}^n$，

  则在 $\mathbb{R}^n$ 空间上状态点 $X(t)$（或输出$V(t)$）随时间则形成一条轨迹，从这一轨迹的趋势可以判断网络系统的稳定性。

  对于离散型网络，轨迹是跳跃变化的；对于连续型网络，轨迹是连续变化的。$(X(t)\leftarrow I(t)\,\,(或 I(t_0)))$



### B. 状态轨迹的分类









## 离散 Hopfield 网络的稳定性







## 离散 Hopfield 网络与联想记忆





## 连续 Hopfield 网络的稳定性





## 连续 Hopfield 网络与组合优化