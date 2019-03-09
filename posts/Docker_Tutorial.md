---
title: Docker Tutorial
date: 2019-1-14
---

[返回到首页](../index.html)

---

# Docker 简易入门教程

[TOC]

> 此文是为自己学习 Docker 为未来的自己看的 Note，所以内容是很基础的，并且内容有相当一部分是转载和 copy 自其它牛人的博文等其他技术资料。所以写作此文的目的，就是为了降低个人搜索和查阅的麻烦，每学一手，就记一点。
>
> Ref：
>
> - [Docker 入门教程](http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html) - [阮一峰](http://www.ruanyifeng.com/) （我的最初级入门资料）
> - [Docker — 从入门到实践](https://legacy.gitbook.com/book/yeasy/docker_practice/details) or [在线阅读国内镜像](https://docker_practice.gitee.io)（非常全的中文教程电子书|[repo](https://github.com/yeasy/docker_practice)）
> - [10张图带你深入理解Docker容器和镜像](http://dockone.io/article/783)（英文原文：[Visualizing Docker Containers and Images](http://merrigrove.blogspot.sg/2015/10/visualizing-docker-containers-and-images.html)）
> - [Docker系列之一：入门介绍](https://tech.meituan.com/docker_introduction.html)（美团技术资料）
> - [Docker系列之二：基于容器的自动构建](https://tech.meituan.com/auto_build.html)（美团技术资料）
> - [如何使用docker部署c/c++程序](https://blog.csdn.net/len_yue_mo_fu/article/details/80189035)
> - [*Docker Get Started Tutorial*](https://docs.docker.com/get-started/)



（以下为正文）

---



2013年发布至今， [Docker](https://www.docker.com/) 一直广受瞩目，被认为可能会改变软件行业。

![](https://i.loli.net/2018/12/07/5c09e3e4dc7c0.png)

## 1. 环境配置的难题

软件开发最大的麻烦事之一，就是环境配置。用户计算机的环境都不相同，你怎么知道自家的软件，能在那些机器跑起来？

用户必须保证两件事：操作系统的设置，各种库和组件的安装。只有它们都正确，软件才能运行。举例来说，安装一个 Python 应用，计算机必须有 Python 引擎，还必须有各种依赖，可能还要配置环境变量。

如果某些老旧的模块与当前环境不兼容，那就麻烦了。开发者常常会说："它在我的机器可以跑了"（It works on my machine），言下之意就是，其他机器很可能跑不了。

环境配置如此麻烦，换一台机器，就要重来一次，旷日费时。很多人想到，能不能从根本上解决问题，软件可以带环境安装？也就是说，安装的时候，把原始环境一模一样地复制过来。



## 2. 虚拟机

虚拟机（virtual machine）就是带环境安装的一种解决方案。它可以在一种操作系统里面运行另一种操作系统，比如在 Windows 系统里面运行 Linux 系统。应用程序对此毫无感知，因为虚拟机看上去跟真实系统一模一样，而对于底层系统来说，虚拟机就是一个普通文件，不需要了就删掉，对其他部分毫无影响。

虽然用户可以通过虚拟机还原软件的原始环境。但是，这个方案有几个缺点。

**（1）资源占用多**

虚拟机会独占一部分内存和硬盘空间。它运行的时候，其他程序就不能使用这些资源了。哪怕虚拟机里面的应用程序，真正使用的内存只有 1MB，虚拟机依然需要几百 MB 的内存才能运行。

**（2）冗余步骤多**

虚拟机是完整的操作系统，一些系统级别的操作步骤，往往无法跳过，比如用户登录。

**（3）启动慢**

启动操作系统需要多久，启动虚拟机就需要多久。可能要等几分钟，应用程序才能真正运行。



## 3. Linux 容器

由于虚拟机存在这些缺点，Linux 发展出了另一种虚拟化技术：Linux 容器（Linux Containers，缩写为 LXC）。

**Linux 容器不是模拟一个完整的操作系统，而是对进程进行隔离。**或者说，在正常进程的外面套了一个[保护层](https://opensource.com/article/18/1/history-low-level-container-runtimes)。对于容器里面的进程来说，它接触到的各种资源都是虚拟的，从而实现与底层系统的隔离。

由于容器是进程级别的，相比虚拟机有很多优势。

**（1）启动快**

容器里面的应用，直接就是底层系统的一个进程，而不是虚拟机内部的进程。所以，启动容器相当于启动本机的一个进程，而不是启动一个操作系统，速度就快很多。

**（2）资源占用少**

容器只占用需要的资源，不占用那些没有用到的资源；虚拟机由于是完整的操作系统，不可避免要占用所有资源。另外，多个容器可以共享资源，虚拟机都是独享资源。

**（3）体积小**

容器只要包含用到的组件即可，而虚拟机是整个操作系统的打包，所以容器文件比虚拟机文件要小很多。

总之，容器有点像轻量级的虚拟机，能够提供虚拟化的环境，但是成本开销小得多。



## 4. 为什么要使用 Docker

作为一种新兴的虚拟化方式，Docker 跟传统的虚拟化方式相比具有众多的优势。

- 更高效的利用系统资源

由于容器不需要进行硬件虚拟以及运行完整操作系统等额外开销，Docker 对系统资源的利用率更高。无论是应用执行速度、内存损耗或者文件存储速度，都要比传统虚拟机技术更高效。因此，相比虚拟机技术，一个相同配置的主机，往往可以运行更多数量的应用。

- 更快速的启动时间

传统的虚拟机技术启动应用服务往往需要数分钟，而 Docker 容器应用，由于直接运行于宿主内核，无需启动完整的操作系统，因此可以做到秒级、甚至毫秒级的启动时间。大大的节约了开发、测试、部署的时间。

- 一致的运行环境

开发过程中一个常见的问题是环境一致性问题。由于开发环境、测试环境、生产环境不一致，导致有些 bug 并未在开发过程中被发现。而 Docker 的镜像提供了除内核外完整的运行时环境，确保了应用运行环境一致性，从而不会再出现 *「这段代码在我机器上没问题啊」* 这类问题。

- 持续交付和部署

对开发和运维（[DevOps](https://zh.wikipedia.org/wiki/DevOps)）人员来说，最希望的就是一次创建或配置，可以在任意地方正常运行。

使用 Docker 可以通过定制应用镜像来实现持续集成、持续交付、部署。开发人员可以通过 [Dockerfile](https://docker_practice.gitee.io/image/dockerfile) 来进行镜像构建，并结合 [持续集成(Continuous Integration)](https://en.wikipedia.org/wiki/Continuous_integration) 系统进行集成测试，而运维人员则可以直接在生产环境中快速部署该镜像，甚至结合 [持续部署(Continuous Delivery/Deployment)](https://en.wikipedia.org/wiki/Continuous_delivery) 系统进行自动部署。

而且使用 `Dockerfile` 使镜像构建透明化，不仅仅开发团队可以理解应用运行环境，也方便运维团队理解应用运行所需条件，帮助更好的生产环境中部署该镜像。

- 更轻松的迁移

由于 Docker 确保了执行环境的一致性，使得应用的迁移更加容易。Docker 可以在很多平台上运行，无论是物理机、虚拟机、公有云、私有云，甚至是笔记本，其运行结果是一致的。因此用户可以很轻易的将在一个平台上运行的应用，迁移到另一个平台上，而不用担心运行环境的变化导致应用无法正常运行的情况。

- 更轻松的维护和扩展

Docker 使用的分层存储以及镜像的技术，使得应用重复部分的复用更为容易，也使得应用的维护更新更加简单，基于基础镜像进一步扩展镜像也变得非常简单。此外，Docker 团队同各个开源项目团队一起维护了一大批高质量的 [官方镜像](https://store.docker.com/search?q=&source=verified&type=image)，既可以直接在生产环境使用，又可以作为基础进一步定制，大大的降低了应用服务的镜像制作成本。

- 对比传统虚拟机总结

| 特性       | 容器               | 虚拟机      |
| ---------- | ------------------ | ----------- |
| 启动       | 秒级               | 分钟级      |
| 硬盘使用   | 一般为 `MB`        | 一般为 `GB` |
| 性能       | 接近原生           | 弱于        |
| 系统支持量 | 单机支持上千个容器 | 一般几十个  |

![](https://i.loli.net/2018/12/07/5c09ea37056ae.png)



## 5. Docker 是什么？

**Docker 属于 Linux 容器的一种封装，提供简单易用的容器使用接口。**它是目前最流行的 Linux 容器解决方案。

Docker 将应用程序与该程序的依赖，打包在一个文件里面。运行这个文件，就会生成一个虚拟容器。程序在这个虚拟容器里运行，就好像在真实的物理机上运行一样。有了 Docker，就不用担心环境问题。

总体来说，Docker 的接口相当简单，用户可以方便地创建和使用容器，把自己的应用放入容器。容器还可以进行版本管理、复制、分享、修改，就像管理普通的代码一样。

详细的 Docker 介绍可见：[XX](https://docker_practice.gitee.io/introduction/what.html)



## 6. Docker 的用途

Docker 的主要用途，目前有三大类。

**（1）提供一次性的环境。**比如，本地测试他人的软件、持续集成的时候提供单元测试和构建的环境。

**（2）提供弹性的云服务。**因为 Docker 容器可以随开随关，很适合动态扩容和缩容。

**（3）组建微服务架构。**通过多个容器，一台机器可以跑多个服务，因此在本机就可以模拟出微服务架构。



## 7. Docker 的安装

Docker 是一个开源的商业产品，有两个版本：社区版（Community Edition，缩写为 CE）和企业版（Enterprise Edition，缩写为 EE）。企业版包含了一些收费服务，个人开发者一般用不到。下面的介绍都针对社区版。

Docker CE 的安装请参考[官方文档](https://docs.docker.com/install/)，也可参考这本[中文线上电子书教程](https://docker_practice.gitee.io/install/)。

=========经过一段安装过程==========

安装完成后，运行下面的命令，验证是否安装成功。

```shell
$ docker version
$ docker info
```

Docker 需要用户具有 sudo 权限，为了避免每次命令都输入`sudo`，可以把用户加入 Docker 用户组（[官方文档](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user)）。也可以每次执行 Docker 的命令时都带上 `sudo`。

```shell
$ sudo usermod -aG docker $USER
```

Docker 是服务器----客户端架构。命令行运行`docker`命令的时候，需要本机有 Docker 服务。如果这项服务没有启动，可以用下面的命令启动（[官方文档](https://docs.docker.com/config/daemon/systemd/)）。

```bash
# 以下命令仅适用于 Linux 系统
# service 命令的用法
$ sudo service docker start

# systemctl 命令的用法 (RHEL7/Centos7)
$ sudo systemctl start docker
```

> **接下来很重要的事，是修改 Docker 的官方仓库到国内的镜像网站。**
>
> - 参考[镜像加速器](https://docker_practice.gitee.io/install/mirror.html)！



## 8. 实例：Hello-world

由于 Docker 官方提供的 image 文件，都放在[`library`](https://hub.docker.com/r/library/)组里面，所以它的是默认组，可以省略。

```shell
# 抓取官方的 hello-world 镜像：
$ docker image pull hello-world

# 查看
$ docker image ls

# 运行这个 image 文件。
$ docker container run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.

... ...
# (运行成功！)
```

注意，`docker container run`命令具有自动抓取 image 文件的功能。如果发现本地没有指定的 image 文件，就会从仓库自动抓取。因此，前面的`docker image pull`命令并不是必需的步骤。

有些容器不会自动终止，因为提供的是服务。对于那些不会自动终止的容器，必须使用[`docker container kill`](https://docs.docker.com/engine/reference/commandline/container_kill/) 命令手动终止。

```shell
$ docker container kill [containID]
```





## 9. 容器文件

**image 文件生成的容器实例，本身也是一个文件，称为容器文件。**也就是说，一旦容器生成，就会同时存在两个文件： image 文件和容器文件。而且关闭容器并不会删除容器文件，只是容器停止运行而已。

```shell
# 列出本机正在运行的容器
$ docker container ls

# 列出本机所有容器，包括终止运行的容器
$ docker container ls --all
```

上面命令的输出结果之中，包括容器的 ID。很多地方都需要提供这个 ID，比如上一节终止容器运行的`docker container kill`命令。

终止运行的容器文件，依然会占据硬盘空间，可以使用[`docker container rm`](https://docs.docker.com/engine/reference/commandline/container_rm/)命令删除。

```shell
$ docker container rm [containerID]
```

运行上面的命令之后，再使用`docker container ls --all`命令，就会发现被删除的容器文件已经消失了。





## 10. Dockerfile 文件

Dockerfile 文件，它是一个文本文件，用来配置 image。Docker 根据 该文件生成二进制的 image 文件。

> 关于 Dockerfile 的写作规则和可用的指令，请查阅：
>
> - [Docker — 从入门到实践](https://legacy.gitbook.com/book/yeasy/docker_practice/details) or [在线阅读国内镜像](https://docker_practice.gitee.io)（非常全的中文教程电子书|[repo](https://github.com/yeasy/docker_practice)）

下面是记录我制作某次 Dockerfile 镜像时用的代码：

```dockerfile
# 该 image 文件继承我自己的 gwave image，冒号表示标签，这里标签是2.0.0，即2.0.0版本的 gwave。
FROM iphysreserch/gwave:2.0.0

# 将当前目录下的所有文件(除了.dockerignore排除的路径),都拷贝进入 image 文件里微系统的/waveform目录
COPY . /waveform

# 指定接下来的工作路径为/waveform (也就是微系统的 pwd)
WORKDIR /waveform

# 定义一个微系统里的环境变量
ENV VERSION=2.0.0	# optional

# 将容器 3000 端口暴露出来， 允许外部连接这个端口
EXPOSE 3000			# optional

# 在/waveform目录下，运行以下命令更新系统程序包。注意，安装后所有的依赖都将打包进入 image 文件
RUN apt-get update && apt-get upgrade	# optional

# 将我这个 image 做成一个 app 可执行程序，容器启动后自动执行下面指令
ENTRYPOINT ["bash", "setup.sh"]
```

可以在项目的根目录下创建一个 `.dockerignore` 文件夹，表示可排除的文件，类似 `.gitignore`。

也可将 `ENTRYPOINT` 换做 `CMD` ，都是容器启动后自动执行指令，简单区别就是 `ENTRYPOINT` 可以在本地启动容器时加额外的shell参数。另外，一个 Dockerfile 可以包含多个`RUN`命令，但是只能有一个`CMD` 或者 `ENTRYPOINT` 命令。

```dockerfile
CMD bash setup.sh
```



## 11. 创建 image

有了 Dockerfile 文件以后，就可以使用`docker image build`命令创建 image 文件了

```bash
$ docker image build -t my-demo .
# 或者
$ docker image build -t my-demo:0.0.1 .
```

上面代码中，`-t`参数用来指定 image 文件的名字，后面还可以用冒号指定标签。如果不指定，默认的标签就是`latest`。最后的那个点表示 Dockerfile 文件所在的路径，上例是当前路径，所以是一个点。

如果运行成功，就可以看到新生成的 image 文件`my-demo`了。

```bash
$ docker image ls -a
```

> 注：
>
> - 当母 image 是有 `ENTRYPOINT` 时，在其基础上创建的子 image 会继承其 `ENTRYPOINT`，并且不会被子 image 的容器在 docker run 时提供的参数覆盖。只有在子 image 的 `Dockerfile` 中指定 `ENTRYPOINT`再 build 后才可以覆盖。（子 image 的`ENTRYPOINT` 为空也可）[REF](https://segmentfault.com/q/1010000004861105/a-1020000005367169) [REF](https://www.cnblogs.com/lienhua34/p/5170335.html)
>
> - **基于容器来创建 image**：[REF](https://blog.csdn.net/leo15561050003/article/details/71274718)
>
>   先运行一个容器，并在运行容器的基础上进行修改（不要使用 `docker run --rm` 参数会自动删除容器，应使用`-it` 参数来可交互 ），如：
>
>   ```bash
>   $ sudo docker container run -it <image_name> /bin/bash
>   ```
>
>   然后将正在运行的容器导出为 image。
>
>   ```bash
>   $ docker commit -m “Description” -a “users <users@email.com>” <ID> <your_repo:tags>
>   ```
>
>   其中：
>
>   - `-m` 指定提交的说明信息
>   - `-a` 指定更新的作者和邮箱 
>   - `<ID>` 想要保存为 image 的容器 ID
>   - `<your_repo:tags>` 欲新建镜像的 repository:tags

## 12. 删除 image

```shell
$ docker rmi [image ID]
```



> 若生成 image 有误等情况，导致出现难以正常删除的 image，可执行下面的代码即可！
>
> ```shell
> $ docker rmi $(docker images -f "dangling=true" -q)
> ```
>



## 13. 生成容器

`docker container run`命令会从 image 文件生成容器。

比如下面的例子：

```bash
$ docker container run -p 8000:3000 -it my-demo /bin/bash
# 或者
$ docker container run -p 8000:3000 -it my-demo:0.0.1 /bin/bash
```

上面命令的各个参数含义如下：

- `-p`参数：容器的 3000 端口映射到本机的 8000 端口。
- `-it`参数：容器的 Shell 映射到当前的 Shell，然后你在本机窗口输入的命令，就会传入容器。
- `my-demo:0.0.1`：image 文件的名字（如果有标签，还需要提供标签，默认是 latest 标签）。
- `/bin/bash`：容器启动以后，内部第一个执行的命令。这里是启动 Bash，保证用户可以使用 Shell。

如果一切正常，运行上面例子的命令以后，就会返回一个命令行提示符，进入到“微系统”里啦！

```bash
root@66d80f4aaf1e:/app#
```





## 14. 终止容器

若在容器的命令行中，按下 Ctrl + c 停止进程，然后按下 Ctrl + d （或者输入 exit）退出容器。

此外，不管是容器中，还是本机的终端里，也都可以用`docker container kill`终止容器运行。

```bash
# 在本机的另一个终端窗口，查出容器的 ID
$ docker container ls

# 停止指定的容器运行
$ docker container kill [containerID]
```

**容器停止运行之后，并不会消失**，用下面的命令删除容器文件。

```bash
# 查出容器的 ID
$ docker container ls --all

# 删除指定的容器文件
$ docker container rm [containerID] # containerID 可以输入前几个关键词即可
```

或者对于指定过 `name` 的容器进程，可如下例子来关闭 `stop` 并删除 `rm` 容器：

```shell
$ docker run -p 6379:6379 --name gredis -d redis
# ... runing ...
$ docker stop gredis
$ docker rm gredis
```

也可以使用`docker container run`命令的`--rm`参数，在容器终止运行后自动删除容器文件，如下面的例子：

```shell
$ docker container run --rm -p 8000:3000 -it koa-demo /bin/bash
```



## 15. 发布 image

容器运行成功后，就确认了 image 文件的有效性。这时，我们就可以考虑把 image 文件分享到网上，让其他人使用。

首先，去 [hub.docker.com](https://hub.docker.com/) 或 [cloud.docker.com](https://cloud.docker.com/) 注册一个账户。然后，用下面的命令登录。

```bash
$ docker login
```

接着，为本地的 image 标注用户名和版本。

```bash
$ docker image tag [imageName] [username]/[repository]:[tag]
# 实例
$ docker image tag my-demos:0.0.1 iphysresearch/my-demos:0.0.1
```

也可以不标注用户名，重新构建一下 image 文件。

```bash
$ docker image build -t [username]/[repository]:[tag] .
```

最后，发布 image 文件。

```bash
$ docker image push [username]/[repository]:[tag]
```

发布成功以后，登录 hub.docker.com，就可以看到已经发布的 image 文件。







## Appendix. 其他有用的命令

docker 的主要用法就是上面这些，此外还有几个命令，也非常有用。

**（1）docker container start**

前面的`docker container run`命令是新建容器，每运行一次，就会新建一个容器。同样的命令运行两次，就会生成两个一模一样的容器文件。如果希望重复使用容器，就要使用`docker container start`命令，它用来启动已经生成、已经停止运行的容器文件。

```bash
$ docker container start [containerID]
```

**（2）docker container stop**

前面的`docker container kill`命令终止容器运行，相当于向容器里面的主进程发出 SIGKILL 信号。而`docker container stop`命令也是用来终止容器运行，相当于向容器里面的主进程发出 SIGTERM 信号，然后过一段时间再发出 SIGKILL 信号。

```bash
$ bash container stop [containerID]
```

这两个信号的差别是，应用程序收到 SIGTERM 信号以后，可以自行进行收尾清理工作，但也可以不理会这个信号。如果收到 SIGKILL 信号，就会强行立即终止，那些正在进行中的操作会全部丢失。

**（3）docker container logs**

`docker container logs`命令用来查看 docker 容器的输出，即容器里面 Shell 的标准输出。如果`docker run`命令运行容器的时候，没有使用`-it`参数，就要用这个命令查看输出。

```bash
$ docker container logs [containerID]
```

**（4）docker container exec**

`docker container exec`命令用于进入一个正在运行的 docker 容器。如果`docker run`命令运行容器的时候，没有使用`-it`参数，就要用这个命令进入容器。一旦进入了容器，就可以在容器的 Shell 执行命令了。

```bash
$ docker container exec -it [containerID] /bin/bash
```

**（5）docker container cp**

`docker container cp`命令用于从正在运行的 Docker 容器里面，将文件拷贝到本机。下面是拷贝到当前目录的写法。

```bash
$ docker container cp [containID]:[/path/to/file] .
```

（待续）



---

[返回到首页](../index.html) | [返回到顶部](./Docker_Tutorial.html)


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

