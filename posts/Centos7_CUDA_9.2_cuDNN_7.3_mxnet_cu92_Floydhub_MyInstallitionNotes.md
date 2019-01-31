---
title: Centos7 + CUDA-9.2 + cuDNN-7.3 + mxnet-cu92 + Floydhub 安装记录
date: 2018-10-26
---

[返回到首页](../index.html)

---



# Centos7/CUDA-9.2/cuDNN-7.3/MXNet-cu92/Floydhub 深度学习环境配置手册



> 此文是自己首次配置 Centos7 和深度学习 GPU 环境的配置流程，一些容易碰到的坑和参考资料都已注明清楚。
>
> - Centos7 系统可从自己的百度云盘下载：
>
>   链接:https://pan.baidu.com/s/1xYEdRHhTTOYef4qz5vtpHA  密码:wog4
>
> - GPU 硬件是 `GeForce GTX 960`，非 POWER8/POWER9 类型。
>
> - 以下配置流程，除 emacs 和 zip 等外，都是必要步骤。

---

[TOC]

---



成功安装 Centos7 系统后，用 `root` 账户登录系统。。.

## 1. 配置网卡

```shell
$ nm-connection-editor
```

设置为自动获取 ip，然后校园网登陆 `172.16.202.203`



## 2. 配置 CentOS 镜像源

配置清华的 CentOS 镜像源： https://mirrors.tuna.tsinghua.edu.cn/help/centos   

（网易的CentOS镜像也不错：https://mirrors.163.com/.help/centos.html）



## 3. gcc

```shell
$ yum install gcc    		# version 4.8.5
$ yum install gcc-c++
```



## 4. CUDA-9.2

下载并按照官方文档进行安装：（CUDA-9.2 和 cuDNN-9.2 也都可在我的百度云盘下载：https://pan.baidu.com/s/1q3_TKTOSE7bhnzigdEdfQA  密码:vsg7）

> REF：
>
> - [CUDA Toolkit 9.2](https://developer.nvidia.com/cuda-92-download-archive) (May 2018)
>
> - [Online Documentation](https://docs.nvidia.com/cuda/archive/9.2/)  (From : [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)) 

- `pre-installition `之后，关于 `.run` 的安装步骤中，`Disable the Nouveau drivers` 是需要先查看是否开启中：

  ```shell
  $ lsmod | grep nouveau
  ```

  若开启中，则根据文档所写：

  > 1. Create a file at `/etc/modprobe.d/blacklist-nouveau.conf`with the following contents:
  >
  >    ```bash
  >    blacklist nouveau
  >    options nouveau modeset=0
  >    ```
  >
  > 2. Regenerate the kernel initramfs:
  >
  >    ```shell
  >    $ sudo dracut --force
  >    ```

  根据 [ref](https://www.tecmint.com/install-nvidia-drivers-in-linux/) 所写的，顺道再 create a new **`initramfs`** file and taking backup of existing.

  ```shell
  $ mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r).img.bak  
  $ dracut -v /boot/initramfs-$(uname -r).img $(uname -r)
  ```

  重启 `$ reboot` ，进入 Login into command mode using <kbd>Alt</kbd> +<kbd>F2~F5</kbd> as root。

  可以验证 Nouveau 未开启。

- 步骤 `Reboot into text mode (runlevel 3).` 时，Enter runlevel 3 by typing `init 3`。（[ref](https://askubuntu.com/questions/149206/how-to-install-nvidia-run?answertab=votes#tab-top)）

- 开始安装 `sudo sh cuda_9.2.148_396.37_linux.run` 和 CUDA 9.2 Patch Update `sudo sh cuda_9.2.148.1_linux.run` ， 重启。

- 配置环境变量：`vim /etc/profile`， 键入：

  ```bash
  PATH=$PATH:/usr/local/cuda-9.2/bin
  export PATH
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/local/cuda-9.2/lib64
  ```

  保存，退出，执行：`$ source /etc/profile`。在查看环境变量：`$ echo $PATH` （[ref](https://www.jianshu.com/p/73399a4c9114)）

  （*或许*也可以按照官方的环境变量配置方法 [ref](https://docs.nvidia.com/cuda/archive/9.2/cuda-installation-guide-linux/index.html#environment-setup)：`$ export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}` , `$ export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}` ）

- 验证安装：

  ```shell
  $ cd /root/NVIDIA_CUDA-9.2_Samples/1_Utilities/deviceQuery
  $ make
  $ ./deviceQuery
  ```

  结果显示 `Result = PASS` 为成功，

- 查看版本：`$ nvcc --version`

- 查看 GPU 状态：`$ nvidia-smi`

  ![](https://i.loli.net/2018/10/29/5bd6a3eb29bab.png)

  ```shell
  # 可用如下代码实时监控
  $ watch -n 1 -d nvidia-smi
  ```


## 5. cuDNN-9.2

安装 [CUDA Dependencies](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#cuda-dependencies) ： [cuDNN 7.1.4](https://developer.nvidia.com/cudnn) 。这也可从我的[百度云](https://pan.baidu.com/s/1U8pGv_o62iS1JFyXjK04Tg)下载：

```shell
$ tar xvf cudnn-9.2-linux-x64-v7.3.1.20.tar
$ sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
$ sudo ldconfig
```

（From: [ref](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#cuda-dependencies)）



## 6. pip [PyPi]

```shell
$ sudo yum -y install epel-release  # 安装 EPEL 扩展源
$ sudo yum -y install python-pip   # pip 8.1.2 (python 2.7)

# 先临时使用清华的源更新 pypi 到最新
$ pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
# 把清华的源设为默认
$ pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```





---

> 从此处开始往后，都登陆系统管理员用户配置，非 root 用户。
>

---

## 7. Git

升级 git，参考：[ref](https://www.cnblogs.com/kevingrace/p/8252517.html)

```shell
# 安装依赖软件
$ sudo yum install curl-devel expat-devel gettext-devel openssl-devel zlib-devel asciidoc
$ sudo yum install  gcc perl-ExtUtils-MakeMaker

# 拆卸系统自带的 git 版本
$ git --version
$ sudo yum remove git

# 编译安装最新的 git 版本
$ su
root$ cd /usr/local/src/
root$ wget https://www.kernel.org/pub/software/scm/git/git-2.19.1.tar.xz
root$ tar -vxf git-2.19.1.tar.xz
root$ cd git-2.19.1
root$ make prefix=/usr/local/git all
root$ make prefix=/usr/local/git install
root$ echo "export PATH=$PATH:/usr/local/git/bin" >> /etc/profile
root$ source /etc/profile

root$ git --version

# 如果是非root用户使用git，则需要配置下该用户下的环境变量
$ echo "export PATH=$PATH:/usr/local/git/bin" >> ~/.bashrc
$ source ~/.bashrc
$ git --version
```



## 8. Zip

```shell
$ sudo yum -y install zlib*
```



## 9. Emacs & Spacemacs





## 10. pyenv & pyenv-virtualenv

> 虚拟环境之 `pyenv` 和 `pyenv-virtualenv`：官方 [repo](https://github.com/pyenv/pyenv#installation) + 很棒的 [pyenv tutorial](https://amaral.northwestern.edu/resources/guides/pyenv-tutorial)

- 一键安装 `pyenv` ：`$ curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash`

- 在 `~/.bashrc`  中写入环境变量：

  ```bash
  export PATH="~/.pyenv/bin:$PATH"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"
  ```

   环境激活生效：`$ source ~/.bashrc`

- 查看可安装的 Python 版本列表：`$ pyenv install --list`

- 使用国内镜像加速 `pyenv`，不然下载速度死慢。。。。解决办法如下：（[ref](https://www.jianshu.com/p/228cd025a368)+[ref](https://blog.csdn.net/l1216766050/article/details/77526455)）

  1. 参考国内的镜像地址来下载 python 版本、anaconda3等等：

     - https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/ 
     - https://pypi.tuna.tsinghua.edu.cn/simple 
     - http://mirrors.aliyun.com/pypi/simple/ 
     - http://pypi.douban.com/simple/ 
     - https://mirrors.ustc.edu.cn/pypi/web/simple/
     - http://mirrors.sohu.com/python/

  2. 作为一个例子，用 `pyenv` 安装一个高配的 python3.6 作为系统的全局 python 环境，有两种办法：

     - 一种是从镜像下载相关安装包到 `~/.pyenv/cache/` 下面，如：

     ```shell
     $ wget http://mirrors.sohu.com/python/3.6.7/Python-3.6.7.tar.xz -P ~/.pyenv/cache
     ```

     - 另一种是修改相应的相关安装文档，如将 `$ ~/.pyenv/plugins/python-build/share/python-build/3.6.7` 中的安装包地址改为国内源地址： `http://mirrors.sohu.com/python/3.6.7/Python-3.6.7.tar.xz`

     1. 在正式安装前需要先安装一些相关依赖，不然会解压安装出错！

        ```bash
        $ sudo yum install readline readline-devel readline-static -y
        $ sudo yum install openssl openssl-devel openssl-static -y
        $ sudo yum install sqlite-devel -y
        $ sudo yum install bzip2-devel bzip2-libs -y
        ```

     2. 开始用 `pyenv` 安装！

        ```shell
        $ pyenv install 3.6.7  # -v 参数可以显示完整的安装过程
        $ pyenv versions  # 查看目前已经安装的
        # system 表示系统安装
        # * 表示当前使用的那个版本
        ```

     3. 更新数据库：`$ pyenv rehash`

     4. 把全局系统的 python 环境设置为 3.6.7 版本

        ```shell
        $ python -V          	# 查看设置前
        $ pyenv global 3.6.7	# 用 pyenv 变更全局 python 版本
        $ pyenv versions		# 用 pyenv 查看已安装的状态
        $ python -V				# 查看设置后
        $ which python 			# 查看目前 python
        ```

  3. 可以设定某文件目录下的局部 python 环境（use pyenv to define a project-specific, or local, version of Python）

     ```shell
     $ pyenv local 3.6.7        # 在某目录下执行局部环境的切换
     ```

  4. Python 的优先级 （[ref](http://einverne.github.io/post/2017/04/pyenv.html)）

     - shell > local > global

     `pyenv` 会从当前目录开始向上逐级查找 .python-version 文件，直到根目录为止。若找不到，就用 global 版本。

     ```shell
     $ pyenv shell 2.7.3 # 设置面向 shell 的 Python 版本，通过设置当前 shell 的 PYENV_VERSION 环境变量的方式。这个版本的优先级比 local 和 global 都要高。–unset 参数可以用于取消当前 shell 设定的版本。
     $ pyenv shell --unset
     
     $ pyenv rehash  # 创建垫片路径（为所有已安装的可执行文件创建 shims，如：~/.pyenv/versions/*/bin/*，因此，每当你增删了 Python 版本或带有可执行文件的包（如 pip）以后，都应该执行一次本命令）
     ```

- 使用 `pyenv-virtualenv` （官方 [repo](https://github.com/pyenv/pyenv-virtualenv), [ref](https://www.jianshu.com/p/861f9a474f70), [ref](https://amaral.northwestern.edu/resources/guides/pyenv-tutorial)）

  本来这是一个单独的软件用来虚拟一个python版本环境，让每个工作环境都有一套独立的python各自的第三方插件互不影响。然而在 pyenv 下有一个插件 pyenv-virtualenv 他可以在 pyenv 的环境下担负起 virtualenv 的事情。（如果使用的是原生python可以用这个工具，如果用的是anaconda则不用这个，用conda工具来完成虚拟环境）

  1. 安装：

     ```shell
     $ git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
     $ source ~/.bashrc
     ```

  2. 使用：(creat a virtualenv based on Python 3.6.7 under `/root/.pyenv/versions/` in a folder called `venv`)

     ```shell
     ~$ mkdir virtual_env
     ~$ cd virtual_env/
     ~/virtual_env$ pyenv virtualenv 3.6.7 venv
     ~/virtual_env$ pyenv versions
     ```

  3. 查看：`pyenv virtualenvs`

  4. 激活/不激活：`pyenv activate venv` / `pyenv deactivate`

  5. 删除：`pyenv uninstall venv`





## pipenv

```shell
$ pip install pipenv
```



## 11. Anaconda3

> 安装 anaconda3 （在 pyenv 里）([ref](https://blog.csdn.net/l1216766050/article/details/77526455))

为了避免各种虚拟环境不兼容的问题（ref：[Mac 下实现 pyenv/virtualenv 与 Anaconda 的兼容](https://blog.csdn.net/vencent7/article/details/76849849)），所有的安装包，包括 anaconda 都将安装到 pyenv 的环境里。



## 12. MXNet-cu92

> 在新建的工作目录 `py4GW` 中，局部本地化 anaconda3 环境，并在此环境中安装 `mxnet-cu92`：
>

```shell
~/py4GW$ pyenv local anconda3-5.3.0
(anaconda3-5.3.0) ~/py4GW$ pip install -U --pre mxnet-cu92
(anaconda3-5.3.0) ~/py4GW$ sudo ldconfig /usr/local/cuda-9.2/lib64
# 启动 anaconda3
(anaconda3-5.3.0) ~/py4GW$ anaconda-navigator
```

> 注意：
>
> **这样的安装步骤，mxnet 将只能在 `anaconda3-5.3.0` 环境下才能调用成功！**



## 13. Visual Studio Code

> 安装 Visual Studio Code  （官网 [rpm](https://code.visualstudio.com/docs/setup/linux#_rhel-fedora-and-centos-based-distributions)）（在 `anaconda3-5.3.0` 环境下安装）

1. ```shell
   # Install the key and repository
   $ sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
   $ sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/yum.repos.d/vscode.repo'
   # Update the package cache and install the package
   $ yum check-update
   $ sudo yum install code
   ```


## 14. 坚果云

- 通过源码安装。（[ref](https://www.jianguoyun.com/s/downloads/linux)）

1. 解决软件包依赖关系

坚果云Linux客户端依赖于这些包: `glib2.0-dev`, `gtk2.0-dev`, `libnautilus-extension-dev`, `gvfs-bin`, `JRE`。 

```shell
$ sudo yum install glib2-devel gtk2-devel nautilus-devel gvfs java-1.7.0-openjdk-headless
```

2. 下载Nautilus插件源代码包: [nutstore_linux_src_installer.tar.gz](https://www.jianguoyun.com/static/exe/installer/nutstore_linux_src_installer.tar.gz)

```shell
$ wget http://www.jianguoyun.com/static/exe/installer/nutstore_linux_src_installer.tar.gz
```

3. 解压缩，编译和安装Nautilus插件

```shell
$ tar zxf nutstore_linux_src_installer.tar.gz
$ cd nutstore_linux_src_installer && ./configure && make
$ sudo make install
```

4. 重启Nautilus

```shell
$ nautilus -q
```

5. 运行以下命令，自动下载和安装坚果云其他二进制组件

```shell
$ ./runtime_bootstrap
```



## 15. Synergy

 自己有 Synergy1 pro 的账号 ($19)，访问官网下载客户端并安装：https://symless.com/download

1. 先安装依赖（[ref](https://github.com/symless/synergy-core/wiki/Compiling#centos-7)）

   ```shell
   $ sudo yum groupinstall "Development Tools"
   $ sudo yum -y install epel-release cmake3 boost-static git libXtst-devel qt5-qtbase-devel qt5-qtdeclarative-devel libcurl-devel openssl-devel
   ```

2. 下载 rpm 包（如 `synergy_1.10.1-91.stable.8941241e.centos.el7.x86_64.rpm` ）后，在相应目录中执行下方代码：

   ```shell
   $ sudo rpm -ivh synergy_1.10.1-91.stable.8941241e.centos.el7.x86_64.rpm
   ```

3. 第一次打开 Synergy 会需要输出 Synergy 1 的 serial key。（在自己的账户(*hewang@mail.bnu.edu.cn*)中可查询）

4. Centos 作为客户端的配置注意事项：

   - 服务器端和客户端都要关闭 `TLS Encrption`。

   - 服务器端先要“设置服务端”，添加新客户端到网格中，并且设置其"名称"；在客户端中设定该“屏幕名称”要与“名称”一致。

   - 客户端的设置中，关闭 `Auto Config`，并且填写的服务器端 ip 地址要和服务器上的 Synergy 的 IP 地址一致（或要与服务器端上的 `$ ifconfig` 查询的 ip 地址一致）

   - 先开始运行服务器端的 Synergy，然后再运行客户端的 Synergy。




## 14. Floydhub





## 15. Screen

```bash
$ yum install screen
```



（持续更新中。。。。）





---

[返回到首页](../index.html) | [返回到顶部](./Centos7_CUDA_9.2_cuDNN_7.3_mxnet_cu92_Floydhub_MyInstallitionNotes.html)


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



