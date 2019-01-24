---
title: Ubuntu 16.04/CUDA-9.2/cuDNN-7.3/MXNet-cu92 深度学习环境配置流程
date: 2019-01-22
---

[返回到首页](../index.html)

---



# Ubuntu 16.04/CUDA-9.2/cuDNN-7.3/MXNet-cu92 深度学习环境配置流程



> 此文是自己首次配置 Ubuntu 16.04 和深度学习 GPU 1080ti x4 环境的配置流程，一些容易碰到的坑和参考资料都已注明清楚。
>
> 以下配置流程，仅供参考。



[TOC]



> 硬件配置清单：
>
> - （华硕）Z10PE-D8 WS 主板
>
> - （追风者）PHANTEKS PK-614PC-BK 机箱（[ref](http://item.jd.com/1340847.html#crumb-wrap)）
>
> - （技嘉）GTX 1080 TI TURBO 11G 涡轮风扇GPUx4（[ref](http://gz.zol.com.cn/693/6931437.html)）
>
> - （英特尔）E5-2678 V3 2011 服务器整机CPUx2（[ref](http://detail.zol.com.cn/servercpu/index1235828.shtml)）+ 一体水冷
>
> - （三星）M393A4K40CB1-CRC4Q 32GB 2Rx4 PC4-2400T-RA1 服务器内存x2（[ref](https://info.b2b168.com/s168-85780930.html)）
>
> - （三星） 860 EVO 250G 2.5英寸 固态硬盘（[ref](http://item.jd.com/6287165.html)）
>
>   <img src="https://i.loli.net/2019/01/24/5c4965e138b95.jpeg" style="zoom:50%" />
>



## optional：刷 BIOS

曾刷新过主板的 BIOS  `Z10PE-D8 WS`，从华硕官网[下载](https://www.asus.com.cn/Commercial-Servers-Workstations/Z10PED8_WS/HelpDesk_BIOS/)到 U 盘上，查到主机上后再 POST 时候按 <kbd>Del</kbd>。

<img src="https://i.loli.net/2019/01/24/5c49682418010.jpeg" style="zoom:10%" />

> **注意！**
>
> 刷 BIOS 后，需要在 BIOS 中重新设定开启 `4G Decoding`，才可以带起来4个 GPU！如下图：
>
> <img src="https://i.loli.net/2019/01/24/5c4969675959c.png" style="zoom:50%"/>
>
> 不然会有 PCI 资源不够的问题，如下图：
>
> <img src="https://i.loli.net/2019/01/24/5c496920e57f9.png" style="zoom:50%" />





## 禁用Ubuntu自动更新

禁用系统自动更新，以免显卡驱动出现问题

- 系统设置（System Settings）- Software & Updates - Updates - 修改：

  - "Automatically check for updates" 为 `Never`
  - "Notify me of a new Ubuntu version" 为 `Never`

- 修改配置文件：

  ```bash
  sudo gedit /etc/apt/apt.conf.d/10periodic
  ```

  修改文件中所有选项数值为0：

  ```json
  APT::Periodic::Update-Package-Lists "0";
  APT::Periodic::Download-Upgradeable-Packages "0";
  APT::Periodic::AutocleanInterval "0";
  ```

  





## tuna 源

设置清华的APT源：

```bash
$ sudo apt-get update && sudo apt-get grade
```



## Gdebi

安装 Gdebi

```bash
$ sudo apt-get install gdebi
```





## CUDA 9.2

> REF：
>
> - [CUDA Toolkit 9.2](https://developer.nvidia.com/cuda-92-download-archive) (May 2018)
>
>   可以从我的百度云直接快速下载到：[百度云](https://pan.baidu.com/s/1U8pGv_o62iS1JFyXjK04Tg)
>
> - [Online Documentation](https://docs.nvidia.com/cuda/archive/9.2/)  (From : [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)) 
>
> - [CUDA Installation Guide for Ubuntu](https://gartner.io/cuda-installation-guide/)
>
> - [[SOLVED, solution included] How to install 9.2 patch on ubuntu with deb?](https://devtalk.nvidia.com/default/topic/1038566/cuda-setup-and-installation/-solved-solution-included-how-to-install-9-2-patch-on-ubuntu-with-deb-/)

这次用 Deb 包来安装。先要严格按照官方文档的说明来 pre-intallation：

```bash
$ lspci | grep -i nvidia							# return NVIDIA-GPU信息
$ uname -m && cat /etc/*release						# return x86_64...
$ gcc --version										# 5.4.0
$ uname -r											# 瞅一眼 kernel 的版本
$ sudo apt-get install linux-headers-$(uname -r)	# 安装 kernel headers 和 dev 包
```

上面后两个指令是很重要的，原因是：

> While the Runfile installation performs no package validation, the RPM and **Deb** installations of the driver will make an attempt to install the kernel header and development packages if no version of these packages is currently installed. However, it will install the latest version of these packages, which may or may not match the version of the kernel your system is using. **Therefore, it is best to manually ensure the correct version of the kernel headers and development packages are installed prior to installing the CUDA Drivers, as well as whenever you change the kernel version.**

再如下安装 CUDA：

```bash
$ sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu1604-9-2-148-local-patch-1_1.0-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
$ sudo apt update
$ sudo apt install cuda
```

注：若是先用 `cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64.deb` 已经安装 cuda ，后再安装升级包 `cuda-repo-ubuntu1604-9-2-148-local-patch-1_1.0-1_amd64.deb` 的话，只需：

```bash
$ sudo dpkg -i cuda-repo-ubuntu1604-9-2-148-local-patch-1_1.0-1_amd64.deb
$ sudo apt update
$ sudo apt upgrade cuda
```

运行下面指令查验 CUDA：

```bash
$ cat /proc/driver/nvidia/version 
$ nvcc -V
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2018 NVIDIA Corporation
# Built on Tue_Jun_12_23:07:04_CDT_2018
# Cuda compilation tools, release 9.2, V9.2.148
```

- 把一些可选的第三方库也顺道都装了吧~ （安装办法来自 NVIDIA-CUDA 官方文档）

```bash
$ sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev \
    curl file
```

- 查看 GPU 状态：`$ nvidia-smi`

  ![](https://i.loli.net/2018/10/29/5bd6a3eb29bab.png)

  ```shell
  # 可用如下代码实时监控
  $ watch -n 1 -d nvidia-smi
  ```



## cuDNN-9.2

安装 [CUDA Dependencies](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#cuda-dependencies) ： [cuDNN 7.1.4](https://developer.nvidia.com/cudnn) 。这也可从我的[百度云](https://pan.baidu.com/s/1U8pGv_o62iS1JFyXjK04Tg)下载：

```shell
$ tar xvf cudnn-9.2-linux-x64-v7.3.1.20.tar
$ sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
$ sudo ldconfig
```

（From: [ref](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#cuda-dependencies)）



## Git

> Ref:
>
> - [Download for Linux and Unix](https://git-scm.com/download/linux)
> - [Ubuntu下git的安装与使用](https://www.cnblogs.com/lxm20145215----/p/5905765.html)
> - [Testing your SSH connection](https://help.github.com/articles/testing-your-ssh-connection/#platform-linux)

```bash
$ sudo apt-get install git
```

安装好后，简单配置一下：

```bash
$ git config --global user.name "iphysresearch"  # 换成你的用户名
$ git config --global user.email "hewang@mail.bnu.edu.cn"	# 换成你的邮箱
```

创建验证用的公钥：（因为`git`是通过`ssh`的方式访问资源库的，所以需要在本地创建验证用的文件）

```bash
$ ssh-keygen -C 'hewang@mail.bnu.edu.cn' -t rsa # 换成你的邮箱
```

这样会在用户目录 `~/.ssh/ `下建立相应的密钥文件后，上传到自己的 Github 账号里：

```bash
$ less ~/.ssh/id_rsa.pub
```

复制上面指令后的内容到 https://github.com/settings/keys ，创建 SSH 公钥，保存即可。

最后，查验一下：

```bash
$ ssh -T git@github.com
```

> The authenticity of host 'github.com (52.74.223.119)' can't be established.
> RSA key fingerprint is SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8.
> Are you sure you want to continue connecting (yes/no)? **yes**
> Warning: Permanently added 'github.com,52.74.223.119' (RSA) to the list of known hosts.
> Hi **iphysresearch**! You've successfully authenticated, but GitHub does not provide shell access.





## 坚果云

从官网下载 deb 包：https://www.jianguoyun.com/s/downloads/linux

安装：（用 Gdebi 的图像界面安装也可以）

```bash
$ sudo gdebi nautilus_nutstore_amd64.deb
```

需重启 nautilus：

```bash
$ nautilus -q
```

安装完成！





## 向日葵

官网下载客户端Linux安装包：https://sunlogin.oray.com/zh_CN/download

```bash
$ sudo ./install.sh
```

你懂得。。。



## pip

> Ref:
>
> - [pip 18.1 Installation](https://pip.pypa.io/en/stable/installing/#upgrading-pip)
> - [Installing pip/setuptools/wheel with Linux Package Managers](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers)

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
```

升级 pip

```bash
pip3 install -U pip
pip3 --verison
```

Tip：[升级pip后出现ImportError: cannot import name main](https://blog.csdn.net/accumulate_zhang/article/details/80269313)



- Python 2:

  ```bash
  $ sudo apt install python-pip
  ```

- Python 3:

  ```bash
  $ sudo apt install python3-venv python3-pip
  ```



## Linuxbrew

根据官网安装：https://linuxbrew.sh

```bash
$ sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
```

Add Linuxbrew to your `PATH` and to your bash shell profile script:

```bash
$ test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
$ test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
$ test -r ~/.profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile
$ echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile
```

测试一下：

```bash
$ brew doctor
$ brew install hello
```



## Pyenv

从官网下载：https://github.com/pyenv/pyenv#homebrew-on-macos

```bash
$ brew update
$ brew install pyenv
```

写入环境变量：（[ref](https://amaral.northwestern.edu/resources/guides/pyenv-tutorial)）

```bash
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo 'eval "$(pyenv init -)"' >> ~/.bashrc
$ source ~/.bashrc
```

配置国内七牛的镜像：（[ref](http://ju.outofmemory.cn/entry/82405)）（也可以参考[pyenv 安装本地版本](https://www.cnblogs.com/uangyy/p/6186427.html)）

```bash
$ export PYTHON_BUILD_MIRROR_URL="http://pyenv.qiniudn.com"
```

查看可安装的 Python 版本列表：

```bash
$ pyenv install --list
```

作为例子，选择装 anaconda 下的最新版：

```bash
$ pyenv install anaconda3-2018.12  # -v 参数可以显示完整的安装过程
```

常用操作：

```bash
$ pyenv versions	# 查看目前已经安装的
# system 表示系统安装
# * 表示当前使用的那个版本
$ pyenv rehash		# 更新数据库
$ python -V          	# 查看设置前
$ pyenv global anaconda3-2018.12	# 用 pyenv 变更全局 python 版本
$ pyenv versions		# 用 pyenv 查看已安装的状态
$ python -V				# 查看设置后
$ which python 			# 查看目前 python
```

可以设定某文件目录下的局部 python 环境（use pyenv to define a project-specific, or local, version of Python）

```shell
$ pyenv local anaconda3-2018.12        # 在某目录下执行局部环境的切换
```

Python 的优先级 （[ref](http://einverne.github.io/post/2017/04/pyenv.html)）

- shell > local > global

`pyenv` 会从当前目录开始向上逐级查找 .python-version 文件，直到根目录为止。若找不到，就用 global 版本。

```shell
$ pyenv shell 2.7.3 # 设置面向 shell 的 Python 版本，通过设置当前 shell 的 PYENV_VERSION 环境变量的方式。这个版本的优先级比 local 和 global 都要高。–unset 参数可以用于取消当前 shell 设定的版本。
$ pyenv shell --unset

$ pyenv rehash  # 创建垫片路径（为所有已安装的可执行文件创建 shims，如：~/.pyenv/versions/*/bin/*，因此，每当你增删了 Python 版本或带有可执行文件的包（如 pip）以后，都应该执行一次本命令）
```





## Pipenv

从官网下载：https://pipenv.readthedocs.io/en/latest/install/#homebrew-installation-of-pipenv

```bash
$ brew install pipenv
```





## VS Code

从官网安装：https://code.visualstudio.com/docs/setup/linux

下载 deb 后，选择性安装：

```bash
$ sudo apt install ./<file>.deb

# If you're on an older Linux distribution, you will need to run this instead:
# $ sudo dpkg -i <file>.deb
# $ sudo apt-get install -f # Install dependencies
```



## sudo 免密

Ref: [Ubuntu 设置当前用户sudo免密码](https://www.linuxidc.com/Linux/2016-12/139018.htm)

```bash
# 备份 /etc/sudoers
$ sudo cp /etc/sudoers .
# 编辑 /etc/sudoers
$ sudo vim /etc/sudoers
# 或使用visudo打开sudoers并编辑
# $ sudo visudo
```

文内添加一行：（设置用户hewang免密）

```reStructuredText
hewang		ALL=(ALL:ALL) NOPASSWD: ALL
```

注：sudo 的工作过程如下（[ref](https://blog.csdn.net/a19881029/article/details/18730671)）

1，当用户执行 sudo 时，系统会主动寻找`/etc/sudoers`文件，判断该用户是否有执行 sudo 的权限

2，确认用户具有可执行 sudo 的权限后，让用户输入用户自己的密码确认

3，若密码输入成功，则开始执行 sudo 后续的命令

4，root 执行 sudo 时不需要输入密码(`eudoers`文件中有配置`root ALL=(ALL) ALL`这样一条规则)

5，若欲切换的身份与执行者的身份相同，也不需要输入密码





## Docker

[Install using the repository](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-repository)

安装好后，下载我需要用到的 Images：

```bash
# https://cloud.docker.com/repository/docker/iphysresearch/gwave
$ sudo docker pull iphysresearch/gwave:2.0.0
$ sudo docker pull iphysresearch/gwave:1.0.0
# https://hub.docker.com/_/redis/
$ sudo docker pull redis
```





## Redis-py

The Python interface to the Redis key-value store. https://github.com/andymccurdy/redis-py

```bash
$ pip install redis
```





## Ray

官网下载：https://ray.readthedocs.io/en/latest/installation.html#latest-stable-version

```bash
$ pip install -U ray
```





## Pyinstrument

性能调试工具：https://github.com/joerick/pyinstrument

```bash
$ pip install pyinstrument
```





## MXNet-cu92

在我的项目目录 `ML4GW` 中，局部本地化 anaconda3 环境，并在此环境中安装 `mxnet-cu92`：

```bash
~/ML4GW$ pyenv local anconda3-5.3.1
(anaconda3-5.3.1) ~/ML4GW$ pip install -U --pre mxnet-cu92
(anaconda3-5.3.1) ~/ML4GW$ sudo ldconfig /usr/local/cuda-9.2/lib64
# 启动 anaconda3
(anaconda3-5.3.1) ~/ML4GW$ anaconda-navigator
```

注意：

**这样的安装步骤，此用户的 mxnet 将只能在 `anaconda3-5.3.1` 环境下才能调用成功！（不限于此项目目录）**





（持续更新中。。。。）





---

[返回到首页](../index.html) | [返回到顶部](./Ubuntu16.04_CUDA-9.2_cuDNN-7.3_MXNet-cu92_MyInstallitionNotes.html)


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



