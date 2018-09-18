

# Linux 就该这么学 | 小抄

Website: https://www.linuxprobe.com/

![](https://www.linuxprobe.com/imgs/cover.png)


[TOC]


## Linux 系统

### RPM（红帽软件包管理器）

```shell
rpm -ivh filename.rmp	# 安装软件
rpm -Uvh filename.rmp	# 升级软件
rpm -e filename.rpm		# 卸载软件
rpm -qpi filename.rpm	# 查询软件描述信息
rpm -qpl filename.rpm	# 列出软件文件信息
rpm -qf filename		# 查询文件属于哪个 RPM 
```



### Yum 软件仓库

```shell
yum repolist all			# 列出所有仓库
yum list all				# 列出仓库中所有软件包
yum info "软件包名称"		# 查看软件包信息
yum install "软件包名称"		# 安装软件包
yum reinstall "软件包名称"	# 重新安装软件包
yum update "软件包名称"		# 升级软件包
yum remove "软件包名称"		# 移除软件包
yum clean all				# 清楚所有仓库缓存
yum check-update			# 检查可更新的软件包
yum grouplist				# 查看系统中已经安装的软件包组
yum groupinstall "软件包名称"	# 安装指定的软件包组
yum groupremove "软件包名称"		# 移除指定的软件包组
yum groupinfo "软件包名称"		# 查询指定的软件包组信息
```



### systemd 初始化进程

```shell
systemctl start foo.service		# 启动服务
systemctl restart foo.service	# 重启服务
systemctl stop foo.service		# 停止服务
systemctl reload foo.service	# 重新加载配置文件（不终止服务）
systemctl status foo.service	# 查看服务状态

systemctl enable foo.service	# 开机自动启动
systemctl disable foo.service	# 开机不自动启动
systemctl is-enabled foo.service			# 查看特定服务是否为开机自启动
systemctl list-unit-files --type=service	# 查看各个级别下服务的启动与禁用情况
```





## 新手必须掌握的 Linux 命令

### `man` 执行查看帮助命令

- `man`命令中常用按键以及用途

| 按键      | 用处                               |
| --------- | ---------------------------------- |
| 空格键    | 向下翻一页                         |
| PaGe down | 向下翻一页                         |
| PaGe up   | 向上翻一页                         |
| home      | 直接前往首页                       |
| end       | 直接前往尾页                       |
| /         | 从上至下搜索某个关键词，如“/linux” |
| ?         | 从下至上搜索某个关键词，如“?linux” |
| n         | 定位到下一个搜索到的关键词         |
| N         | 定位到上一个搜索到的关键词         |
| q         | 退出帮助文档                       |

- `man`命令帮助信息的结构以及意义

| 结构名称    | 代表意义                 |
| ----------- | ------------------------ |
| NAME        | 命令的名称               |
| SYNOPSIS    | 参数的大致使用方法       |
| DESCRIPTION | 介绍说明                 |
| EXAMPLES    | 演示（附带简单说明）     |
| OVERVIEW    | 概述                     |
| DEFAULTS    | 默认的功能               |
| OPTIONS     | 具体的可用选项（带介绍） |
| ENVIRONMENT | 环境变量                 |
| FILES       | 用到的文件               |
| SEE ALSO    | 相关的资料               |
| HISTORY     | 维护历史与联系方式       |





### 常用系统工作命令

#### `echo`

在终端输出字符串或变量提取后的值。Eg: `echo $SHELL`

#### `date `

Eg: `date "+%Y-%m-%d %H:%M:%S"`

| 参数 | 作用           |
| ---- | -------------- |
| %t   | 跳格[Tab键]    |
| %H   | 小时（00～23） |
| %I   | 小时（00～12） |
| %M   | 分钟（00～59） |
| %S   | 秒（00～59）   |
| %j   | 今年中的第几天 |

#### `reboot`

默认只能 root 管理员来重启

#### `poweroff`

默认只能 root 管理员来关闭

#### `wget`

Eg: `wget -r -p https://www.linuxprobe.com`

| 参数 | 作用                                 |
| ---- | ------------------------------------ |
| -b   | 后台下载模式                         |
| -P   | 下载到指定目录                       |
| -t   | 最大尝试次数                         |
| -c   | 断点续传                             |
| -p   | 下载页面内所有资源，包括图片、视频等 |
| -r   | 递归下载                             |

#### `ps`

静态的系统进程状态。

| 参数 | 作用                               |
| ---- | ---------------------------------- |
| -a   | 显示所有进程（包括其他用户的进程） |
| -u   | 用户以及其他详细信息               |
| -x   | 显示没有控制终端的进程             |

**R（运行）：**进程正在运行或在运行队列中等待。

**S（中断）：**进程处于休眠中，当某个条件形成后或者接收到信号时，则脱离该状态。

**D（不可中断）：**进程不响应系统异步信号，即便用kill命令也不能将其中断。

**Z（僵死）：**进程已经终止，但进程描述符依然存在, 直到父进程调用wait4()系统函数后将进程释放。

**T（停止）：**进程收到停止信号后停止运行。

#### `top`

动态的系统进程状态。

![](https://i.loli.net/2018/09/17/5b9fbcf51ea40.jpeg)

#### `pidof`

 查询某服务的 PID

#### `kill`

终止某 PID 的服务进程

#### `killall`

终止某进程所对应的全部进程



### 系统状态监测命令

#### `ifconfig`

网卡配置与网络状态

![](https://i.loli.net/2018/09/17/5b9fbd193d49e.jpeg)



#### `uname`

系统内核与系统版本  `uname   -a`

#### `uptime`

系统的负载信息

#### `free`

内存的使用量信息  `free   -h`

#### `who`

登入主机的用户终端信息

#### `last`

所有系统的登录记录

#### `history`

历史执行过的命令

Eg: `history  -c` 清空所有的命令历史记录；`!"编码数字"` 重复执行某一次的命令

#### `sosreport`

收集系统配置以及架构信息并输出诊断文档



### 工作目录切换命令

#### `pwd`

当前工作目录

#### `cd`

切换工作目录

Eg：`cd ~` ；`cd ..` ； `cd  -`

#### `ls`

显示目录中文件信息

Eg: `ls -a` 可看到隐藏文件；`ls -l` 可看到文件属性；`ls -h` 人可读的。。。



### 文本文件编辑命令

#### `cat`

查看小文件

Eg: `cat   -n` 显示行号

#### `more`

查看大文件

#### `head`

查看前几行   `head  -n`

#### `tail`

查看后几行 `tail  -n`

Eg: `tail -f` 实时持续查看！

#### `tr`

替换文本文件中的字符（正则表达式）

Eg: `cat  anaconda-ks.cfg  |  tr  [a-z]  [A-Z]` 

#### `wc`

统计文本的行数、字数、字符数

| 参数 | 作用         |
| ---- | ------------ |
| -l   | 只显示行数   |
| -w   | 只显示单词数 |
| -c   | 只显示字节数 |

#### `stat`

查看文件的具体存储信息和时间等信息

![](https://i.loli.net/2018/09/17/5b9fbd36884ef.jpeg)

#### `cut`

按“列”提取文本字符

Eg: `cut   -d:   -f1   /etc/passwd` 以冒号(:)为间隔符号提取第一列内容

#### `diff`

比较多个文本文件的差异

Eg: `diff   --brief   diff_A.txt   diff_B.txt` 显示比较后的结果，判断文件是否相同

Eg: `diff   -c   diff_A.txt   diff_B.txt`  描述文件内容具体的不同



### 文件目录管理命令

#### `touch`

创建空白文件或设置文件的时间

Eg: `touch   -d   "2017-05-04 15:44"   anaconda-ks.cfg `

| 参数 | 作用                      |
| ---- | ------------------------- |
| -a   | 仅修改“读取时间”（atime） |
| -m   | 仅修改“修改时间”（mtime） |
| -d   | 同时修改 atime 与 mtime   |

#### `mkdir`

创建空白的目录

#### `cp`

复制文件或目录

| 参数 | 作用                                         |
| ---- | -------------------------------------------- |
| -p   | 保留原始文件的属性                           |
| -d   | 若对象为“链接文件”，则保留该“链接文件”的属性 |
| -r   | 递归持续复制（用于目录）                     |
| -i   | 若目标文件存在则询问是否覆盖                 |
| -a   | 相当于 `-pdr`（p、d、r为上述参数）           |

#### `mv`

剪切文件或将文件重命名

#### `rm`

删除文件或目录

Eg: `rm   -f` 强制删除

#### `dd`

按照指定大小和个数的数据块来复制文件或转换文件

Eg: `dd   if=/dev/zero   of=560_file   count=1   bs=560M`从/dev/zero中取出一个大小为560MB的数据块，保存名为560_file的文件

| 参数  | 作用                 |
| ----- | -------------------- |
| if    | 输入的文件名称       |
| of    | 输出的文件名称       |
| bs    | 设置每个“块”的大小   |
| count | 设置要复制“块”的个数 |

#### `file`

查看“任何”文件的类型



### 打包压缩与搜索命令

#### `tar`

对文件进行打包压缩或解压

Eg: `tar   czvf  etc.tar.gz   /etc` 压缩；`tar   xzvf   etc.tar.gz   -C   /root/etc` 解压

| 参数 | 作用                   |
| ---- | ---------------------- |
| -c   | 创建压缩文件           |
| -x   | 解开压缩文件           |
| -t   | 查看压缩包内有哪些文件 |
| -z   | 用Gzip压缩或解压       |
| -j   | 用bzip2压缩或解压      |
| -v   | 显示压缩或解压的过程   |
| -f   | 目标文件名             |
| -p   | 保留原始的权限与属性   |
| -P   | 使用绝对路径来压缩     |
| -C   | 指定解压到的目录       |

#### `grep`

在文本中执行关键词搜索，并显示匹配的结果

Eg: `grep   /sbin/nologin   /etc/passwd`

| 参数 | 作用                                           |
| ---- | ---------------------------------------------- |
| -b   | 将可执行文件(binary)当作文本文件（text）来搜索 |
| -c   | 仅显示找到的行数                               |
| -i   | 忽略大小写                                     |
| -n   | 显示行号                                       |
| -v   | 反向选择——仅列出没有“关键词”的行。             |

#### `find`

按照指定条件来查找文件

Eg: `find   /etc   -name   "host*"   -print`  获取到 /etc 目录中所有以 host 开头的文件列表

Eg: `find  /  -perm   -4000   -print`  在整个系统中搜索权限中包括SUID权限的所有文件

| 参数               | 作用                                                         |
| ------------------ | ------------------------------------------------------------ |
| -name              | 匹配名称                                                     |
| -perm              | 匹配权限（mode为完全匹配，-mode为包含即可）                  |
| -user              | 匹配所有者                                                   |
| -group             | 匹配所有组                                                   |
| -mtime -n +n       | 匹配修改内容的时间（-n指n天以内，+n指n天以前）               |
| -atime -n +n       | 匹配访问文件的时间（-n指n天以内，+n指n天以前）               |
| -ctime -n +n       | 匹配修改文件权限的时间（-n指n天以内，+n指n天以前）           |
| -nouser            | 匹配无所有者的文件                                           |
| -nogroup           | 匹配无所有组的文件                                           |
| -newer f1 !f2      | 匹配比文件f1新但比f2旧的文件                                 |
| --type b/d/c/p/l/f | 匹配文件类型（后面的字幕字母依次表示块设备、目录、字符设备、管道、链接文件、文本文件） |
| -size              | 匹配文件的大小（+50KB为查找超过50KB的文件，而-50KB为查找小于50KB的文件） |
| -prune             | 忽略某个目录                                                 |
| -exec …… {}\;      | 后面可跟用于进一步处理搜索结果的命令（下文会有演示）         |









