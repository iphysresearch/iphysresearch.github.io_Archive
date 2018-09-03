---
title: The Unreasonable Effectiveness of Recurrent Neural Networks
date: 2018-09-03
---

[返回到上一页](./index.html)

---



# CS231n课程资料：循环神经网络惊人的有效性

> **自注：**此文是在转载的基础上，通过网络搜集其他相关学习资料，再结合自己理解下，填充并注释了更多的细节和内容，以此详尽的文本资料代替各种视频课程等资料，仅方便自己回头翻查。
>
> 转载请注明本文出处和[原译文](https://zhuanlan.zhihu.com/p/22107715)出处。

> 译者注：版权声明：本文[智能单元](https://zhuanlan.zhihu.com/intelligentunit)首发，本人原创翻译，禁止未授权转载。 
>
> 经知友推荐，将 [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 一文翻译作为CS231n课程无RNN和LSTM笔记的补充，感谢 [@堃堃](https://www.zhihu.com/people/e7fcc05b0cf8a90a3e676d0206f888c9)，@[猴子](https://www.zhihu.com/people/hmonkey) 和[@李艺颖](https://www.zhihu.com/people/f11e78650e8185db2b013af42fd9a481)的校对。

[TOC]

## 目录

- 循环神经网络

- 字母级别的语言模型

- RNN的乐趣

  - Paul Graham生成器

  - 莎士比亚
  - 维基百科
  - 几何代数
  - Linux源码
  - 生成婴儿姓名

- 理解训练过程

  - 训练时输出文本的进化

  - RNN中的预测与神经元激活可视化

- 源代码

- 拓展阅读

- 结论

- 译者反馈

## 原文如下

循环神经网络（RNN）简直像是魔法一样不可思议。我为[图像标注](https://cs.stanford.edu/people/karpathy/deepimagesent/)项目训练第一个循环网络时的情景到现在都还历历在目。当时才对第一个练手模型训练了十几分钟（超参数还都是随手设置的），它就开始生成一些对于图像的描述，描述内容看起来很不错，几乎让人感到语句是通顺的了。有时候你会遇到模型简单，结果的质量却远高预期的情况，这就是其中一次。当时这个结果让我非常惊讶是因为我本以为RNN是非常难以训练的（随着实践的增多，我的结论基本与之相反了）。让我们快进一年：即使现在我成天都在训练RNN，也常常看到它们的能力和鲁棒性，有时候它们那充满魔性的输出还是能够把我给逗乐。这篇博文就是来和你分享RNN中的一些魔法。

> 我们将训练RNN，让它们生成一个又一个字母。同时好好思考这个问题：这怎么可能呢？

顺便说一句，和这篇博文一起，我在Github上发布了一个项目。项目基于多层的LSTM，使得你可以训练字母级别的语言模型。你可以输入一大段文本，然后它能学习并按照一次一个字母的方式生成文本。你也可以用它来复现我下面的实验。但是现在我们要超前一点：RNN到底是什么？



## 循环神经网络

**序列**。基于知识背景，你可能会思考：*是什么让RNN如此独特呢？*普通神经网络和卷积神经网络的一个显而易见的局限就是他们的API都过于限制：他们接收一个固定尺寸的向量作为输入（比如一张图像），并且产生一个固定尺寸的向量作为输出（比如针对不同分类的概率）。不仅如此，这些模型甚至对于上述映射的演算操作的步骤也是固定的（比如模型中的层数）。RNN之所以如此让人兴奋，其核心原因在于其允许我们对向量的序列进行操作：输入可以是序列，输出也可以是序列，在最一般化的情况下输入输出都可以是序列。下面是一些直观的例子：

---

![](https://i.loli.net/2018/09/03/5b8d476496916.png)

上图中每个正方形代表一个向量，箭头代表函数（比如矩阵乘法）。输入向量是红色，输出向量是蓝色，绿色向量装的是RNN的状态（马上具体介绍）。从左至右为：

1. 非RNN的普通过程，从固定尺寸的输入到固定尺寸的输出（比如图像分类）。
2. 输出是序列（例如图像标注：输入是一张图像，输出是单词的序列）。
3. 输入是序列（例如情绪分析：输入是一个句子，输出是对句子属于正面还是负面情绪的分类）。
4. 输入输出都是序列（比如机器翻译：RNN输入一个英文句子输出一个法文句子）。
5. 同步的输入输出序列（比如视频分类中，我们将对视频的每一帧都打标签）。

注意在每个案例中都没有对序列的长度做出预先规定，这是因为循环变换（绿色部分）是固定的，我们想用几次就用几次。

---

如你期望的那样，相较于那些从一开始连计算步骤的都定下的固定网络，序列体制的操作要强大得多。并且对于那些和我们一样希望构建一个更加智能的系统的人来说，这样的网络也更有吸引力。我们后面还会看到，RNN将其输入向量、状态向量和一个固定（可学习的）函数结合起来生成一个新的状态向量。在程序的语境中，这可以理解为运行一个具有某些输入和内部变量的固定程序。从这个角度看，RNN本质上就是在描述程序。实际上RNN是具备[图灵完备性](http://binds.cs.umass.edu/papers/1995_Siegelmann_Science.pdf)的，只要有合适的权重，它们可以模拟任意的程序。然而就像神经网络的通用近似理论一样，你不用过于关注其中细节。实际上，我建议你忘了我刚才说过的话。

> 如果训练普通神经网络是对函数做最优化，那么训练循环网络就是针对程序做最优化。



**无序列也能进行序列化处理**。你可能会想，将序列作为输入或输出的情况是相对少见的，但是需要认识到的重要一点是：即使输入或输出是固定尺寸的向量，依然可以使用这个强大的形式体系以序列化的方式对它们进行处理。例如，下图来自于[DeepMind](http://deepmind.com)的两篇非常不错的论文。左侧动图显示的是一个算法学习到了一个循环网络的策略，该策略能够引导它对图像进行观察；更具体一些，就是它学会了如何从左往右地阅读建筑的门牌号（[Ba et al](https://arxiv.org/abs/1412.7755)）。右边动图显示的是一个循环网络通过学习序列化地向画布上添加颜色，生成了写有数字的图片（[Gregor et al](https://arxiv.org/abs/1502.04623)）。

---

![](http://karpathy.github.io/assets/rnn/house_read.gif)![](http://karpathy.github.io/assets/rnn/house_generate.gif)

左边：RNN学会如何阅读建筑物门牌号。右边：RNN学会绘出建筑门牌号。 

---

必须理解到的一点就是：即使数据不是序列的形式，仍然可以构建并训练出能够进行序列化处理数据的强大模型。换句话说，你是要让模型学习到一个处理固定尺寸数据的分阶段程序。



**RNN的计算**。那么RNN到底是如何工作的呢？在其核心，RNN有一个貌似简单的API：它接收输入向量**x**，返回输出向量**y**。然而这个输出向量的内容不仅被输入数据影响，而且会收到整个历史输入的影响。写成一个类的话，RNN的API只包含了一个**step**方法：

```python
rnn = RNN()
y = rnn.step(x) # x is an input vector, y is the RNN's output vector
```

每当**step**方法被调用的时候，RNN的内部状态就被更新。在最简单情况下，该内部装着仅包含一个内部*隐向量h*。下面是一个普通RNN的step方法的实现：

```python
class RNN:
  # ...
  def step(self, x):
    # update the hidden state
    self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    # compute the output vector
    y = np.dot(self.W_hy, self.h)
    return y
```

上面的代码详细说明了普通RNN的前向传播。该RNN的参数是三个矩阵：**W_hh, W_xh, W_hy**。隐藏状态**self.h**被初始化为零向量。**np.tanh**函数是一个非线性函数，将激活数据挤压到[-1,1]之内。注意代码是如何工作的：在tanh内有两个部分。一个是基于前一个隐藏状态，另一个是基于当前的输入。在numpy中，**np.dot**是进行矩阵乘法。两个中间变量相加，其结果被tanh处理为一个新的状态向量。如果你更喜欢用数学公式理解，那么公式是这样的：![h_t=tanh(W_{hh}h_{t-1}+W_{hx}x_t)](https://www.zhihu.com/equation?tex=h_t%3Dtanh%28W_%7Bhh%7Dh_%7Bt-1%7D%2BW_%7Bhx%7Dx_t%29)。其中tanh是逐元素进行操作的。

我们使用随机数字来初始化RNN的矩阵，进行大量的训练工作来寻找那些能够产生描述行为的矩阵，使用一些损失函数来衡量描述的行为，这些损失函数代表了根据输入**x**，你对于某些输出**y**的偏好。



**更深层网络**。RNN属于神经网络算法，如果你像叠薄饼一样开始对模型进行重叠来进行深度学习，那么算法的性能会单调上升（如果没出岔子的话）。例如，我们可以像下面代码一样构建一个2层的循环网络：

```python
y1 = rnn1.step(x)
y = rnn2.step(y1)
```

换句话说，我们分别有两个RNN：一个RNN接受输入向量，第二个RNN以第一个RNN的输出作为其输入。其实就RNN本身来说，它们并不在乎谁是谁的输入：都是向量的进进出出，都是在反向传播时梯度通过每个模型。



**更好的网络**。需要简要指明的是在实践中通常使用的是一个稍有不同的算法，这就是我在前面提到过的*长短基记忆*网络，简称LSTM。LSTM是循环网络的一种特别类型。由于其更加强大的更新方程和更好的动态反向传播机制，它在实践中效果要更好一些。本文不会进行细节介绍，但是在该算法中，所有本文介绍的关于RNN的内容都不会改变，唯一改变的是状态更新（就是**self.h=...**那行代码）变得更加复杂。从这里开始，我会将术语RNN和LSTM混合使用，但是在本文中的所有实验都是用LSTM完成的。



## 字母级别的语言模型

现在我们已经理解了RNN是什么，它们何以令人兴奋，以及它们是如何工作的。现在通过一个有趣的应用来更深入地加以体会：我们将利用RNN训练一个字母级别的语言模型。也就是说，给RNN输入巨量的文本，然后让其建模并根据一个序列中的前一个字母，给出下一个字母的概率分布。这样就使得我们能够一个字母一个字母地生成新文本了。

在下面的例子中，假设我们的字母表只由4个字母组成“helo”，然后利用训练序列“hello”训练RNN。该训练序列实际上是由4个训练样本组成：1.当h为上文时，下文字母选择的概率应该是e最高。2.l应该是he的下文。3.l应该是hel文本的下文。4.o应该是hell文本的下文。

具体来说，我们将会把每个字母编码进一个1到k的向量（除对应字母为1外其余为0），然后利用**step**方法一次一个地将其输入给RNN。随后将观察到4维向量的序列（一个字母一个维度）。我们将这些输出向量理解为RNN关于序列下一个字母预测的信心程度。下面是流程图：

---

![](https://i.loli.net/2018/09/03/5b8d492e318e5.png)

一个RNN的例子：输入输出是4维的层，隐层神经元数量是3个。该流程图展示了使用hell作为输入时，RNN中激活数据前向传播的过程。输出层包含的是RNN关于下一个字母选择的置信度（字母表是helo）。我们希望绿色数字大，红色数字小。

---

举例如下：在第一步，RNN看到了字母h后，给出下一个字母的置信度分别是h为1，e为2.2，l为-3.0，o为4.1。因为在训练数据（字符串hello）中下一个正确的字母是e，所以我们希望提高它的置信度（绿色）并降低其他字母的置信度（红色）。类似的，在每一步都有一个目标字母，我们希望算法分配给该字母的置信度应该更大。因为RNN包含的整个操作都是可微分的，所以我们可以通过对算法进行反向传播（微积分中链式法则的递归使用）来求得权重调整的正确方向，在正确方向上可以提升正确目标字母的得分（绿色粗体数字）。然后进行*参数更新*，即在该方向上轻微移动权重。如果我们将同样的数据输入给RNN，在参数更新后将会发现正确字母的得分（比如第一步中的e）将会变高（例如从2.2变成2.3），不正确字母的得分将会降低。重复进行一个过程很多次直到网络收敛，其预测与训练数据连贯一致，总是能正确预测下一个字母。

更技术派的解释是我们对输出向量同步使用标准的Softmax分类器（也叫作交叉熵损失）。使用小批量的随机梯度下降来训练RNN，使用[RMSProp](https://arxiv.org/abs/1502.04390)或Adam来让参数稳定更新。

注意当字母l第一次输入时，目标字母是l，但第二次的目标是o。因此RNN不能只靠输入数据，必须使用它的循环连接来保持对上下文的跟踪，以此来完成任务。

在**测试**时，我们向RNN输入一个字母，得到其预测下一个字母的得分分布。我们根据这个分布取出得分最大的字母，然后将其输入给RNN以得到下一个字母。重复这个过程，我们就得到了文本！现在使用不同的数据集训练RNN，看看将会发生什么。

为了更好的进行介绍，我基于教学目的写了代码：[minimal character-level RNN language model in Python/numpy](https://gist.github.com/karpathy/d4dee566867f8291f086)，它只有100多行。如果你更喜欢读代码，那么希望它能给你一个更简洁直观的印象。我们下面介绍实验结果，这些实验是用更高效的Lua/Torch代码实现的。



## RNN的乐趣

下面介绍的5个字母模型我都放在Github上的[项目](https://github.com/karpathy/char-rnn)里了。每个实验中的输入都是一个带有文本的文件，我们训练RNN让它能够预测序列中下一个字母。



### Paul Graham生成器

译者注：中文名一般译为保罗•格雷厄姆，著有《黑客与画家》一书，中文版已面世。在康奈尔大学读完本科，在哈佛大学获得计算机科学博士学位。1995年，创办了Viaweb。1998年，Yahoo!收购了Viaweb，收购价约5000万美元。此后架起了个人网站[http://paulgraham.com](http://paulgraham.com)，在上面撰写关于软件和创业的文章，以深刻的见解和清晰的表达而著称。2005年，创建了风险投资公司Y Combinator，目前已经资助了80多家创业公司。现在，他是公认的互联网创业权威。

让我们先来试一个小的英文数据集来进行正确性检查。我最喜欢的数据集是[Paul Graham的文集](http://www.paulgraham.com/articles.html)。其基本思路是在这些文章中充满智慧，但Paul Graham的写作速度比较慢，要是能根据需求生成富于创业智慧的文章岂不美哉？那么就轮到RNN上场了。

将Paul Graham最近5年的文章收集起来，得到大小约1MB的文本文件，约有1百万个字符（这只算个很小的数据集）。*技术要点*：训练一个2层的LSTM，各含512个隐节点（约350万个参数），每层之后使用0.5的dropout。每个数据批量中含100个样本，时间长度上截断了100个字符进行梯度的反向传播。按照上述设置，每个数据批量在TITAN GPU上的运算耗时为0.46秒（如果仅对50个字符进行BPTT，那么耗时会减半，性能的耗费几乎忽略不计）。*译者注：BPTT即Backpropagation Through Time*。不在啰嗦，让我们看看RNN生成的文本：

*“The surprised in investors weren’t going to raise money. I’m not the company with the time there are all interesting quickly, don’t have to get off the same programmers. There’s a super-angel round fundraising, why do you can do. If you have a different physical investment are become in people who reduced in a startup with the way to argument the acquirer could see them just that you’re also the founders will part of users’ affords that and an alternation to the idea. [2] Don’t work at first member to see the way kids will seem in advance of a bad successful startup. And if you have to act the big company too.”*

好吧，显然生成器暂时还无法替代Paul Graham，但是RNN可是完全从头开始学英语的（包括逗号，撇号和空格），而且数据集又如此的小。我还很喜欢它自己学会了如何进行引用（例如上文中的[2]）。有时候它甚至会说出一些充满智慧的洞见，比如“a company is a meeting to think to investors（公司就是一个琢磨如何让投资者打钱的会议）”。*译者注：RNN你瞎说什么大实话：）*如果你想要查看更多细节，点击[这里](https://cs.stanford.edu/people/karpathy/char-rnn/pg.txt)。



**温度**。在生成文本的时候，我们可以调节Softmax的温度。将温度从1降到更低的数值（比如0.5）可以让RNN更加自信的同时变得更加保守。相反，如果将温度设置的更高，结果会更加多样化，但是代价就是可能会出现错误（比如更多的拼写错误）。如果将温度设置得非常接近于0，我们就会得到最像Paul Graham说的话：

*“is that they were all the same thing that was a startup is that they were all the same thing that was a startup is that they were all the same thing that was a startup is that they were all the same”*

看来我们陷入到连续创业的无限循环中去了。



## 莎士比亚

看起来我们可以让RNN学会如何拼写单词，但是数据中有更多的结构和样式呢？为了回答这个问题，我下载了莎士比亚的所有作品，并将它们装进一个单独的文件中，大小约4.4MB。我们现在可以训练更大的网络，这次试用的是一个3层的RNN，每层的隐节点512个。当我们训练了几个小时候，我们得到了下面的文本输出：

```py
PANDARUS:
Alas, I think he shall be come approached and the day
When little srain would be attain'd into being never fed,
And who is but a chain and subjects of his death,
I should not sleep.

Second Senator:
They are away this miseries, produced upon my soul,
Breaking and strongly should be buried, when I perish
The earth and thoughts of many states.

DUKE VINCENTIO:
Well, your wit is in the care of side and that.

Second Lord:
They would be ruled after this chamber, and
my fair nues begun out of the fact, to be conveyed,
Whose noble souls I'll have the heart of the wars.

Clown:
Come, sir, I will make did behold your worship.

VIOLA:
I'll drink it.
```

记住，RNN只知道字符，所以它学会了说话者的名字和说话内容的格式，有时候我们还会得到类似独白的文字，比如：

```shell
VIOLA:
Why, Salisbury must find his flesh and thought
That which I am not aps, not a man and in fire,
To show the reining of the raven and the wars
To grace my hand reproach within, and not a fair are hand,
That Caesar and my goodly father's world;
When I was heaven of presence and our fleets,
We spare with hours, but cut thy council I am great,
Murdered and by thy master's ready there
My power to give thee but so much as hell:
Some service in the noble bondman here,
Would show him to her wine.

KING LEAR:
O, if you were a feeble sight, the courtesy of your law,
Your sight and several breath, will wear the gods
With his heads, and my hands are wonder'd at the deeds,
So drop upon your lordship's head, and your opinion
Shall be against your honour.
```

我个人是很难把这段话从莎士比亚的原作中分辨出来啦：）如果你喜欢莎士比亚，那可以来看看这篇[长达1000000字符的生成文本](https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt)。当然，你可以使用我提供的代码，在不同的温度设置下来生成不同的文本。



## 维基百科

我们看见LSTM能够拼写单词，复现语法结构。那么现在就提高难度，使用markdown文本对它进行训练。我使用了[Hutter Prize](http://prize.hutter1.net)的100MB的数据集，数据集内容是原始的维基百科内容，然后在LSTM上训练。根据[Graves等的论文](https://arxiv.org/abs/1308.0850)，我使用了其中96MB用于训练，剩下的用做验证集。模型跑了有一晚上，然后可以生成维基百科文章了。下面是一些有趣的文本片段。首先，一些基本的markdown输出：

```markdown
Naturalism and decision for the majority of Arab countries' capitalide was grounded
by the Irish language by [[John Clair]], [[An Imperial Japanese Revolt]], associated 
with Guangzham's sovereignty. His generals were the powerful ruler of the Portugal 
in the [[Protestant Immineners]], which could be said to be directly in Cantonese 
Communication, which followed a ceremony and set inspired prison, training. The 
emperor travelled back to [[Antioch, Perth, October 25|21]] to note, the Kingdom 
of Costa Rica, unsuccessful fashioned the [[Thrales]], [[Cynth's Dajoard]], known 
in western [[Scotland]], near Italy to the conquest of India with the conflict. 
Copyright was the succession of independence in the slop of Syrian influence that 
was a famous German movement based on a more popular servicious, non-doctrinal 
and sexual power post. Many governments recognize the military housing of the 
[[Civil Liberalization and Infantry Resolution 265 National Party in Hungary]], 
that is sympathetic to be to the [[Punjab Resolution]]
(PJS)[http://www.humah.yahoo.com/guardian.
cfm/7754800786d17551963s89.htm Official economics Adjoint for the Nazism, Montgomery 
was swear to advance to the resources for those Socialism's rule, 
was starting to signing a major tripad of aid exile.]]
```

如果你注意到的话，yahoo的那个url是不存在的，是模型生造了它。还有，可以看见模型学会了对于圆括号要成对出现。模型还学会了很多markdown结构，比如标题，列表等：

```markdown
{ { cite journal | id=Cerling Nonforest Department|format=Newlymeslated|none } }
''www.e-complete''.

'''See also''': [[List of ethical consent processing]]

== See also ==
*[[Iender dome of the ED]]
*[[Anti-autism]]

===[[Religion|Religion]]===
*[[French Writings]]
*[[Maria]]
*[[Revelation]]
*[[Mount Agamul]]

== External links==
* [http://www.biblegateway.nih.gov/entrepre/ Website of the World Festival. The labour of India-county defeats at the Ripper of California Road.]

==External links==
* [http://www.romanology.com/ Constitution of the Netherlands and Hispanic Competition for Bilabial and Commonwealth Industry (Republican Constitution of the Extent of the Netherlands)]
```

有时候模型也会生成一些随机但是合法的XML：

```xml
<page>
  <title>Antichrist</title>
  <id>865</id>
  <revision>
    <id>15900676</id>
    <timestamp>2002-08-03T18:14:12Z</timestamp>
    <contributor>
      <username>Paris</username>
      <id>23</id>
    </contributor>
    <minor />
    <comment>Automated conversion</comment>
    <text xml:space="preserve">#REDIRECT [[Christianity]]</text>
  </revision>
</page>
```

模型生成了时间戳，id和其他一些东西。同时模型也能正确地让标示符成对出现，嵌套规则也合乎逻辑。如果你对文本感兴趣，点击[这里](https://cs.stanford.edu/people/karpathy/char-rnn/wiki.txt)。



## 代数几何

上面的结果表明模型确实比较擅长学习复杂的语法结构。收到这些结果的鼓舞，我和同伴[Justin Johnson](https://link.zhihu.com/?target=http%3A//cs.stanford.edu/people/jcjohns/)决定在结构化这一块将研究更加推进一步。我们在网站Stacks上找到了这本关于代数几何的[书](https://link.zhihu.com/?target=http%3A//stacks.math.columbia.edu/)，下载了latex源文件（16MB大小），然后用于训练一个多层的LSTM。令人惊喜的是，模型输出的结果几乎是可以编译的。我们手动解决了一些问题后，就得到了一个看起来像模像样的数学文档，看起来非常惊人：

---

![](https://i.loli.net/2018/09/03/5b8d4a4ee302d.png)

生成的代数几何。这里是[源文件](https://cs.stanford.edu/people/jcjohns/fake-math/4.pdf)。

---

这是另一个例子：

![](https://i.loli.net/2018/09/03/5b8d4a7030ba0.png)

更像代数几何了，右边还出现了图表。

---

由上可见，模型有时候尝试生成latex图表，但是没有成功。我个人还很喜欢它跳过证明的部分（“Proof omitted”，在顶部左边）。当然，需要注意的是latex是相对困难的结构化语法格式，我自己都还没有完全掌握呢。下面是模型生成的一个源文件：

```latex
\begin{proof}
We may assume that $\mathcal{I}$ is an abelian sheaf on $\mathcal{C}$.
\item Given a morphism $\Delta : \mathcal{F} \to \mathcal{I}$
is an injective and let $\mathfrak q$ be an abelian sheaf on $X$.
Let $\mathcal{F}$ be a fibered complex. Let $\mathcal{F}$ be a category.
\begin{enumerate}
\item \hyperref[setain-construction-phantom]{Lemma}
\label{lemma-characterize-quasi-finite}
Let $\mathcal{F}$ be an abelian quasi-coherent sheaf on $\mathcal{C}$.
Let $\mathcal{F}$ be a coherent $\mathcal{O}_X$-module. Then
$\mathcal{F}$ is an abelian catenary over $\mathcal{C}$.
\item The following are equivalent
\begin{enumerate}
\item $\mathcal{F}$ is an $\mathcal{O}_X$-module.
\end{lemma}
```

这份模型输出的文本展示了一些常见错误。比如模型起了**\begin{proof}**的头，结尾却是**\end{lemma}**。这种错误我们就必须手动改正，错误产生的原因可能在于依赖关系过于长了：当模型完成证明后，它已经忘了自己的开头是proof还是lemma了。类似的，模型起了**\begin{enumerate}**的头，却忘了结尾。我们观察到，在使用更大规模或更好的模型后，这些错误变少了。然而，以上这些错误总是存在出现的可能性的。



## Linux源码

我想要把结构化数据的实验推到极限，所以最后一个实验我们决定使用代码。具体说来，就是从Linux在Github上的[项目](https://github.com/torvalds/linux)提取了所有代码和能找到的头文件，把他们装进一个巨大的文件中（474MB的C代码）。我原本计划是只训练Linux的核心文件的，但是其大小只有约16MB。然后我在GPU上使用3层LSTM训练了好几天。这些模型有大约1000万的参数，这对于RNN模型来说还不算多的。实验结果超级有趣：

```c
/*
 * Increment the size file of the new incorrect UI_FILTER group information
 * of the size generatively.
 */
static int indicate_policy(void)
{
  int error;
  if (fd == MARN_EPT) {
    /*
     * The kernel blank will coeld it to userspace.
     */
    if (ss->segment < mem_total)
      unblock_graph_and_set_blocked();
    else
      ret = 1;
    goto bail;
  }
  segaddr = in_SB(in.addr);
  selector = seg / 16;
  setup_works = true;
  for (i = 0; i < blocks; i++) {
    seq = buf[i++];
    bpf = bd->bd.next + i * search;
    if (fd) {
      current = blocked;
    }
  }
  rw->name = "Getjbbregs";
  bprm_self_clearl(&iv->version);
  regs->new = blocks[(BPF_STATS << info->historidac)] | PFMR_CLOBATHINC_SECONDS << 12;
  return segtable;
}
```

这些代码看起来相当酷了。虽然我不认为这些代码能够编译，但是看着这些代码你会感觉是一个巨大的C代码库。注意RNN还不时的给自己的代码加上了注释。代码中也很少有语法错误。比如它合理地使用了字符串，指针标记等。它学会了让花括号和中括号成对出现，让代码有正确的缩进。一个常见的错误是它不能追中变量的名字：它常常使用未声明的变量（比如上面的**rw**），或声明一些永不使用的变量（比如上面的**int error**），或返回不存在的变量。让我们看看更多的例子，下面是一个代码片段，展示了RNN学习到的更多的操作：

```c
/*
 * If this error is set, we will need anything right after that BSD.
 */
static void action_new_function(struct s_stat_info *wb)
{
  unsigned long flags;
  int lel_idx_bit = e->edd, *sys & ~((unsigned long) *FIRST_COMPAT);
  buf[0] = 0xFFFFFFFF & (bit << 4);
  min(inc, slist->bytes);
  printk(KERN_WARNING "Memory allocated %02x/%02x, "
    "original MLL instead\n"),
    min(min(multi_run - s->len, max) * num_data_in),
    frame_pos, sz + first_seg);
  div_u64_w(val, inb_p);
  spin_unlock(&disk->queue_lock);
  mutex_unlock(&s->sock->mutex);
  mutex_unlock(&func->mutex);
  return disassemble(info->pending_bh);
}

static void num_serial_settings(struct tty_struct *tty)
{
  if (tty == tty)
    disable_single_st_p(dev);
  pci_disable_spool(port);
  return 0;
}

static void do_command(struct seq_file *m, void *v)
{
  int column = 32 << (cmd[2] & 0x80);
  if (state)
    cmd = (int)(int_state ^ (in_8(&ch->ch_flags) & Cmd) ? 2 : 1);
  else
    seq = 1;
  for (i = 0; i < 16; i++) {
    if (k & (1 << 1))
      pipe = (in_use & UMXTHREAD_UNCCA) +
        ((count & 0x00000000fffffff8) & 0x000000f) << 8;
    if (count == 0)
      sub(pid, ppc_md.kexec_handle, 0x20000000);
    pipe_set_bytes(i, 0);
  }
  /* Free our user pages pointer to place camera if all dash */
  subsystem_info = &of_changes[PAGE_SIZE];
  rek_controls(offset, idx, &soffset);
  /* Now we want to deliberately put it to device */
  control_check_polarity(&context, val, 0);
  for (i = 0; i < COUNTER; i++)
    seq_puts(s, "policy ");
}
```

注意在第二个函数中，比较了**tty == tty**，这永远为真。但这次至少参数名出现在了函数体中。在最后一个函数中，没有返回值，因为函数的声明有void，所以倒也正确。 然而，前面两个函数声明为void确有返回值，这又是一种因为依赖过长产生的常见错误。

有时候模型决定生成新的文件。这通常是非常让人惊喜的部分：模型首先引用了GUN许可，进行了一些导入操作，生成一些宏，然后开始进入代码：

```c
/*
 *  Copyright (c) 2006-2010, Intel Mobile Communications.  All rights reserved.
 *
 *   This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published by
 * the Free Software Foundation.
 *
 *        This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *
 *  GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software Foundation,
 *  Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include <linux/kexec.h>
#include <linux/errno.h>
#include <linux/io.h>
#include <linux/platform_device.h>
#include <linux/multi.h>
#include <linux/ckevent.h>

#include <asm/io.h>
#include <asm/prom.h>
#include <asm/e820.h>
#include <asm/system_info.h>
#include <asm/setew.h>
#include <asm/pgproto.h>

#define REG_PG    vesa_slot_addr_pack
#define PFM_NOCOMP  AFSR(0, load)
#define STACK_DDR(type)     (func)

#define SWAP_ALLOCATE(nr)     (e)
#define emulate_sigs()  arch_get_unaligned_child()
#define access_rw(TST)  asm volatile("movd %%esp, %0, %3" : : "r" (0));   \
  if (__type & DO_READ)

static void stat_PC_SEC __read_mostly offsetof(struct seq_argsqueue, \
          pC>[1]);

static void
os_prefix(unsigned long sys)
{
#ifdef CONFIG_PREEMPT
  PUT_PARAM_RAID(2, sel) = get_state_state();
  set_pid_sum((unsigned long)state, current_state_str(),
           (unsigned long)-1->lr_full; low;
}
```

这里面有太多有趣的地方可以讨论，我几乎可以写一整个博客，所以我现在还是暂停，感兴趣的可以查看[这里](https://cs.stanford.edu/people/karpathy/char-rnn/linux.txt)。



## 生成婴儿姓名

让我们再试一个。给RNN输入一个包含8000个小孩儿姓名的文本文件，一行只有一个名字。（名字是从[这里](https://link.zhihu.com/?target=http%3A//www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/)获得的）我们可以把这些输入RNN然后生成新的名字。下面是一些名字例子，只展示了那些没有在训练集中出现过的名字：

*Rudi Levette Berice Lussa Hany Mareanne Chrestina Carissy Marylen Hammine Janye Marlise Jacacrie Hendred Romand Charienna Nenotto Ette Dorane Wallen Marly Darine Salina Elvyn Ersia Maralena Minoria Ellia Charmin Antley Nerille Chelon Walmor Evena Jeryly Stachon Charisa Allisa Anatha Cathanie Geetra Alexie Jerin Cassen Herbett Cossie Velen Daurenge Robester Shermond Terisa Licia Roselen Ferine Jayn Lusine Charyanne Sales Sanny Resa Wallon Martine Merus Jelen Candica Wallin Tel Rachene Tarine Ozila Ketia Shanne Arnande Karella Roselina Alessia Chasty Deland Berther Geamar Jackein Mellisand Sagdy Nenc Lessie Rasemy Guen Gavi Milea Anneda Margoris Janin Rodelin Zeanna Elyne Janah Ferzina Susta Pey Castina*

点击[这里](https://cs.stanford.edu/people/karpathy/namesGenUnique.txt)可以查看更多。我个人最喜欢的名字包括“Baby” (哈)， “Killie”，“Char”，“R”，“More”，“Mars”，“Hi”，“Saddie”，“With”和“Ahbort”。这真的蛮有意思，你还可以畅想在写小说或者给创业公司起名字的时候，这个能给你灵感。



## 理解训练过程

我们已经看见训练结束后的结果令人印象深刻，但是它到底是如何运作的呢？现在跑两个小实验来一探究竟。



### 训练时输出文本的进化

首先，观察模型在训练时输出文本的不断进化是很有意思的。例如，我使用托尔斯泰的《战争与和平》来训练LSTM，并在训练过程中每迭代100次就输出一段文本。在第100次迭代时，模型输出的文本是随机排列的：

```text
tyntd-iafhatawiaoihrdemot  lytdws  e ,tfti, astai f ogoh eoase rrranbyne 'nhthnee e 
plia tklrgd t o idoe ns,smtt   h ne etie h,hregtrs nigtike,aoaenns lng
```

但是至少可以看到它学会了单词是被空格所分割的，只是有时候它使用了两个连续空格。它还没学到逗号后面总是有个空格。在迭代到第300次的时候，可以看到模型学会使用引号和句号。

```text
"Tmont thithey" fomesscerliund
Keushey. Thom here
sheulke, anmerenith ol sivh I lalterthend Bleipile shuwy fil on aseterlome
coaniogennc Phe lism thond hon at. MeiDimorotion in ther thize."
```

单词被空格所分割，模型开始知道在句子末尾使用句号。在第500次迭代时：

```text
we counter. He stutn co des. His stanted out one ofler that concossions and was 
to gearang reay Jotrets and with fre colt otf paitt thin wall. Which das stimn 
```

模型开始学会使用最短和最常用的单词，比如“we”、“He”、“His”、“Which”、“and”等。从第700次迭代开始，可以看见更多和英语单词形似的文本：

```text
Aftair fall unsuch that the hall for Prince Velzonski's that me of
her hearly, and behs to so arwage fiving were to it beloge, pavu say falling misfort 
how, and Gogition is so overelical and ofter.
```

在第1200次迭代，我们可以看见使用引号、问好和感叹号，更长的单词也出现了。

```text
"Kite vouch!" he repeated by her
door. "But I would be done and quarts, feeling, then, son is people...."
```

在迭代到2000次的时候，模型开始正确的拼写单词，引用句子和人名。

```text
"Why do what that day," replied Natasha, and wishing to himself the fact the
princess, Princess Mary was easier, fed in had oftened him.
Pierre aking his soul came to the packs and drove up his father-in-law women.
```

从上述结果中可见，模型首先发现的是一般的单词加空格结构，然后开始学习单词；从短单词开始，然后学习更长的单词。由多个单词组成的话题和主题词要到训练后期才会出现。



## RNN中的预测与神经元激活可视化

另一个有趣的实验内容就是将模型对于字符的预测可视化。下面的图示是我们对用维基百科内容训练的RNN模型输入验证集数据（蓝色和绿色的行）。在每个字母下面我们列举了模型预测的概率最高的5个字母，并用深浅不同的红色着色。深红代表模型认为概率很高，白色代表模型认为概率较低。注意有时候模型对于预测的字母是非常有信心的。比如在[http://www](https://link.zhihu.com/?target=http%3A//www/). 序列中就是。

输入字母序列也被着以蓝色或者绿色，这代表的是RNN隐层表达中的某个随机挑选的神经元是否被*激活*。绿色代表非常兴奋，蓝色代表不怎么兴奋。LSTM中细节也与此类似，隐藏状态向量中的值是[-1, 1]，这就是经过各种操作并使用tanh计算后的LSTM细胞状态。直观地说，这就是当RNN阅读输入序列时，它的“大脑”中的某些神经元的激活率。不同的神经元关注的是不同的模式。在下面我们会看到4种不同的神经元，我认为比较有趣和能够直观理解（当然也有很多不能直观理解）。

---

![](https://i.loli.net/2018/09/03/5b8d4b844ea23.png)

本图中高亮的神经元看起来对于URL的开始与结束非常敏感。LSTM看起来是用这个神经元来记忆自己是不是在一个URL中。

---

![](https://i.loli.net/2018/09/03/5b8d4b9e95495.png)

高亮的神经元看起来对于markdown符号[[]]的开始与结束非常敏感。有趣的是，一个[符号不足以激活神经元，必须等到两个[[同时出现。而判断有几个[的任务看起来是由另一个神经元完成的。

---

![](https://i.loli.net/2018/09/03/5b8d4bb62791b.png)

这是一个在[[]]中线性变化的神经元。换句话说，在[[]]中，它的激活是为RNN提供了一个以时间为准的坐标系。RNN可以使用该信息来根据字符在[[]]中出现的早晚来决定其出现的频率（也许？）。

---

![](https://i.loli.net/2018/09/03/5b8d4bc898754.png)

这是一个进行局部动作的神经元：它大部分时候都很安静，直到出现www序列中的第一个w后，就突然关闭了。RNN可能是使用这个神经元来计算www序列有多长，这样它就知道是该输出有一个w呢，还是开始输出URL了。

---

当然，由于RNN的隐藏状态是一个巨大且分散的高维度表达，所以上面这些结论多少有一点手动调整。上面的这些可视化图片是用定制的HTML/CSS/Javascript实现的，如果你想实现类似的，可以查看[这里](https://link.zhihu.com/?target=http%3A//cs.stanford.edu/people/karpathy/viscode.zip)。

我们可以进一步简化可视化效果：不显示预测字符仅仅显示文本，文本的着色代表神经元的激活情况。可以看到大部分的细胞做的事情不是那么直观能理解，但是其中5%看起来是学到了一些有趣并且能理解的算法：

---

![](https://i.loli.net/2018/09/03/5b8d4bfd108ce.png)

![](https://i.loli.net/2018/09/03/5b8d4c3d8db06.png)

在预测下个字符的过程中优雅的一点是：我们不用进行任何的硬编码。比如，不用去实现判断我们到底是不是在一个引号之中。我们只是使用原始数据训练LSTM，然后它自己决定这是个有用的东西于是开始跟踪。换句话说，其中一个单元自己在训练中变成了引号探测单元，只因为这样有助于完成最终任务。这也是深度学习模型（更一般化地说是端到端训练）强大能力的一个简洁有力的证据。



## 源代码

我想这篇博文能够让你认为训练一个字符级别的语言模型是一件有趣的事儿。你可以使用我在Github上的[char rnn代码](https://github.com/karpathy/char-rnn)训练一个自己的模型。它使用一个大文本文件训练一个字符级别的模型，可以输出文本。如果你有GPU，那么会在比CPU上训练快10倍。如果你训练结束得到了有意思的结果，请联系我。如果你看Torch/Lua代码看的头疼，别忘了它们只不过是这个[100行项目](https://gist.github.com/karpathy/d4dee566867f8291f086)的高端版。

*题外话*。代码是用[Torch7](http://torch.ch)写的，它最近变成我最爱的深度学习框架了。我开始学习Torch/LUA有几个月了，这并不简单（花了很多时间学习Github上的原始Torch代码，向项目创建者提问来解决问题），但是一旦你搞懂了，它就会给你带来很大的弹性和加速。之前我使用的是Caffe和Theano，虽然Torch虽然还不完美，但是我相信它的抽象和哲学层次比前两个高。在我看来，一个高效的框架应有以下特性：

- 有丰富函数（例如切片，数组/矩阵操作等）的，对底层CPU/GPU透明的张量库。
- 一整个基于脚本语言（比如Python）的分离的代码库，能够对张量进行操作，实现所有深度学习内容（前向、反向传播，计算图等）。
- 分享预训练模型非常容易（Caffe做得很好，其他的不行）。
- 最关键的：没有编译过程！或者至少不要像Theano现在这样！深度学习的趋势是更大更复杂的网络，这些网络都有随着时间展开的复杂计算流程。编译时间不能太长，不然开发过程将充满痛苦。其次，编译导致开发者放弃解释能力，不能高效地进行调试。如果在流程开发完成后有个*选项*能进行编译，那也可以。



## 拓展阅读

在结束本篇博文前，我想把RNN放到更广的背景中，提供一些当前的研究方向。RNN现在在深度学习领域引起了不小的兴奋。和卷积神经网络一样，它出现已经有十多年了，但是直到最近它的潜力才被逐渐发掘出来，这是因为我们的计算能力日益强大。下面是当前的一些进展（肯定不完整，而且很多工作可以追溯的1990年）：

在NLP/语音领域，RNN将[语音转化为文字](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf)，进行[机器翻译](https://arxiv.org/abs/1409.3215)，生成[手写文本](http://www.cs.toronto.edu/~graves/handwriting.html)，当然也是强大的语言模型 (Sutskever等) (Graves) (Mikolov等)。字符级别和单词级别的模型都有，目前看来是单词级别的模型更领先，但是这只是暂时的。

计算机视觉。RNN迅速地在计算机视觉领域中被广泛运用。比如，使用RNN用于[视频分类](https://arxiv.org/abs/1411.4389)，[图像标注](https://arxiv.org/abs/1411.4555)（其中有我自己的工作和其他一些），[视频标注](https://arxiv.org/abs/1505.00487)和最近的[视觉问答](https://arxiv.org/abs/1505.02074)。在计算机视觉领域，我个人最喜欢的RNN论文是《[Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247)》，之所以推荐它，是因为它高层上的指导方向和底层的建模方法（对图像短时间观察后的序列化处理），和建模难度低（REINFORCE算法规则是增强学习里面策略梯度方法中的一个特例，使得能够用非微分的计算来训练模型（在该文中是对图像四周进行快速查看））。我相信这种用CNN做原始数据感知，RNN在顶层做快速观察策略的混合模型将会在感知领域变得越来越流行，尤其是在那些不单单是对物体简单分类的复杂任务中将更加广泛运用。

归纳推理，记忆和注意力（Inductive Reasoning, Memories and Attention）。另一个令人激动的研究方向是要解决普通循环网络自身的局限。RNN的一个问题是它不具有归纳性：它能够很好地记忆序列，但是从其表现上来看，它不能很好地在正确的方向上对其进行归纳（一会儿会举例让这个更加具体一些）。另一个问题是RNN在运算的每一步都将表达数据的尺寸和计算量联系起来，而这并非必要。比如，假设将隐藏状态向量尺寸扩大为2倍，那么由于矩阵乘法操作，在每一步的浮点运算量就要变成4倍。理想状态下，我们希望保持大量的表达和记忆（比如存储全部维基百科或者很多中间变量），但同时每一步的运算量不变。

在该方向上第一个具有说服力的例子来自于DeepMind的[神经图灵机（Neural Turing Machines）](https://arxiv.org/abs/1410.5401)论文。该论文展示了一条路径：模型可以在巨大的外部存储数组和较小的存储寄存器集（将其看做工作的存储器）之间进行读写操作，而运算是在存储寄存器集中进行。更关键的一点是，神经图灵机论文提出了一个非常有意思的存储解决机制，该机制是通过一个（soft和全部可微分的）注意力模型来实现的。*译者注：这里的soft取自softmax*。基于概率的“软”注意力机制（soft attention）是一个强有力的建模特性，已经在面向机器翻译的《[ Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)》一文和面向问答的《[Memory Networks](https://arxiv.org/abs/1503.08895)》中得以应用。实际上，我想说的是：

> 注意力概念是近期神经网络领域中最有意思的创新。

现在我不想更多地介绍细节，但是软注意力机制存储器寻址是非常方便的，因为它让模型是完全可微的。不好的一点就是牺牲了效率，因为每一个可以关注的地方都被关注了（虽然是“软”式的）。想象一个C指针并不指向一个特定的地址，而是对内存中所有的地址定义一个分布，然后间接引用指针，返回一个与指向内容的权重和（这将非常耗费计算资源）。这让很多研究者都从软注意力模式转向硬注意力模式，而硬注意力模式是指对某一个区域内的内容固定关注（比如，对某些单元进行读写操作而不是所有单元进行读写操作）。这个模型从设计哲学上来说肯定更有吸引力，可扩展且高效，但不幸的是模型就不是可微分的了。这就导致了对于增强学习领域技术的引入（比如REINFORCE算法），因为增强学习领域中的研究者们非常熟悉不可微交互的概念。这项工作现在还在进展中，但是硬注意力模型已经被发展出来了，在《[ Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](https://arxiv.org/abs/1503.01007)》，《[ Reinforcement Learning Neural Turing Machines](https://arxiv.org/abs/1505.00521)》，《[Show Attend and Tell](https://arxiv.org/abs/1502.03044)》三篇文章中均有介绍。

研究者。如果你想在RNN方面继续研究，我推荐[Alex Graves](http://www.cs.toronto.edu/~graves/)，[Ilya Sutskever](http://www.cs.toronto.edu/~ilya/)和[Tomas Mikolov](http://www.rnnlm.org)三位研究者。想要知道更多增强学习和策略梯度方法（REINFORCE算法是其中一个特例），可以学习[David Silver的课程](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Home.html)，或[Pieter Abbeel的课程](http://www.cs.berkeley.edu/~pabbeel/)。

代码。如果你想要继续训练RNN，我听说Theano上的[keras](https://github.com/fchollet/keras)或[passage](https://github.com/IndicoDataSolutions/Passage)还不错。我使用Torch写了一个[项目](https://github.com/karpathy/char-rnn)，也用numpy实现了一个可以前向和后向传播的LSTM。你还可以在Github上看看我的[NeuralTalk](https://github.com/karpathy/neuraltalk)项目，是用RNN/LSTM来进行图像标注。或者看看Jeff Donahue用[Caffe](http://jeffdonahue.com/lrcn/)实现的项目。



## 结论

我们已经学习了RNN，知道了它如何工作，以及为什么它如此重要。我们还利用不同的数据集将RNN训练成字母级别的语言模型，观察了它是如何进行这个过程的。可以预见，在未来将会出现对RNN的巨大创新，我个人认为它们将成为智能系统的关键组成部分。

最后，为了给文章增添一点格调，我使用本篇博文对RNN进行了训练。然而由于博文的长度很短，不足以很好地训练RNN。但是返回的一段文本如下（使用低的温度设置来返回更典型的样本）：

```text
I've the RNN with and works, but the computed with program of the 
RNN with and the computed of the RNN with with and the code
```

是的，这篇博文就是讲RNN和它如何工作的，所以显然模型是有用的：）下次见！



---

[返回到上一页](./index.html) | [返回到顶部](./The Unreasonable Effectiveness of Recurrent Neural Networks.html)

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