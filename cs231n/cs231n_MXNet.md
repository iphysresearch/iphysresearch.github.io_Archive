



# Introduction

所有的代码来源自课程 cs231n 的 **spring1718_assignment[1](http://cs231n.github.io/assignments2018/assignment1/)&[2](http://cs231n.github.io/assignments2018/assignment2/)**，经过重新整理，精简和筛选，并且改制为 MXNet 框架语言下的简洁代码。内容涉及如何从零编写一个神经网络和 CNN 网络，去掉了原课程作业中的 knn 和 svm 作业部分。





# Setup

下载代码 github





# Download data

Once you have the starter code, you will need to download the CIFAR-10 dataset. Run the following from the `assignment2` directory:

```shell
cd cs231n/datasets
./get_datasets.sh
```

如果遇到 `permission denied: ./get_datasets.sh`，就先用下面的代码提高权限，然后继续运行 `get_datasets.sh` 脚本。

```shell
chmod +x get_datasets.sh
```

下载完成后且自动解压后，会得到一个文件夹名为 `cifar-10-batches-py` 的数据文件。





# Q: Implement a Softmax classifier

The IPython Notebook **softmax.ipynb** will walk you through implementing the Softmax classifier.







### Q4: Two-Layer Neural Network (25 points)

The IPython Notebook **two_layer_net.ipynb** will walk you through the implementation of a two-layer neural network classifier.

### Q5: Higher Level Representations: Image Features (10 points)

The IPython Notebook **features.ipynb** will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.

 



# Q: Fully-connected Neural Network

The IPython notebook `FullyConnectedNets.ipynb` 







