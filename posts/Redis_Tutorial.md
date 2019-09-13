---
title: Redis Tutorial
date: 2019-1-14
---

[返回到首页](../index.html)

---

![](https://i.loli.net/2019/01/14/5c3c0bed0425b.png)

***Redis* is an open source (BSD licensed), in-memory data structure store, used as a database, cache and message broker.**

---

# Redis 超浅显入门教程

## Redis 是什么？

究竟是什么是 Redis 呢？官方的介绍是这样子滴：

> *Redis* is an *open source* (*BSD licensed*), in-*memory data structure store*, *used* as a *database*, *cache*and *message broker*.
>
> —— 来自 [Redis 官网](https://redis.io)


Redis是一个开源的使用 ANSI C 语言编写、遵守BSD协议、支持网络、可基于**内存**亦可持久化的日志型、Key-Value 数据库，并提供多种语言的 API。它通常被称为数据结构服务器，因为值（value）可以是 字符串(String)，哈希(Map)，列表(list)，集合(sets) 和有序集合(sorted sets)等类型。从2010年3月15日起，Redis的开发工作由VMware主持。从2013年5月开始，Redis的开发由Pivotal赞助。

![](https://i.loli.net/2019/01/14/5c3c0c0ca2e8d.png)



> 此文是什么？
>
> 答：针对 Redis 超级新手的一个快速入门小教程，力争实用主义者的福音。
>
> 此文不是什么？
>
> 答：不是 Redis 的详尽教程教材，不会对内部原理和技术过度解释，但会尽可能得链接相关资料资源。

## Redis 的优缺点

- 优点
  - 性能极高 – Redis能读的速度是110000次/s，写的速度是81000次/s 。
  - 丰富的数据类型 – Redis支持二进制案例的 Strings, Lists, Hashes, Sets 及 Ordered Sets 数据类型操作。
  - 原子 – Redis的所有操作都是原子性的，意思就是要么成功执行要么失败完全不执行。单个操作是原子性的。多个操作也支持事务，即原子性，通过MULTI和EXEC指令包起来。
  - 丰富的特性 – Redis还支持 publish/subscribe, 通知, key 过期等等特性。
- Redis与其他key-value存储有什么不同？
  - Redis有着更为复杂的数据结构并且提供对他们的原子性操作，这是一个不同于其他数据库的进化路径。Redis的数据类型都是基于基本数据结构的同时对程序员透明，无需进行额外的抽象。
  - Redis运行在**内存**中但是可以**持久化**到磁盘，所以在对不同数据集进行高速读写时需要权衡内存，因为数据量不能大于硬件内存。在内存数据库方面的另一个优点是，相比在磁盘上相同的复杂的数据结构，在内存中操作起来非常简单，这样Redis可以做很多内部复杂性很强的事情。同时，在磁盘格式方面他们是紧凑的以追加的方式产生的，因为他们并不需要进行随机访问。
  - 常见数据库对比：　(来源于：[ref](https://www.cnblogs.com/jing99/p/6112055.html))![](https://i.loli.net/2019/02/01/5c53ad282b63c.png)
  - **使用 Redis 而不是 memcached 来解决问题，不仅可以让代码变得更简短、更易懂、更易维护，而且还可以使代码的运行速度更快（因为用户不需要通过读取数据库来更新数据）。除此之外，在其他许多情况下，Redis的效率和易用性也比关系数据库要好得多。**
  - 使用 Redis 而不是关系数据库或者其他硬盘存储数据库，可以**避免写入不必要的临时数据，也免去了对临时数据进行扫描或者删除的麻烦**，并最终改善程序的性能。

---

## Redis 参考资料

- **Redis** official website: http://redis.io

- **Redis** 中文教程：http://www.runoob.com/redis/redis-tutorial.html
- **Redis** 中文命令大全：http://redisdoc.com/index.html
- **Redis** 的官方详尽参考学习资料 [Redis Command Reference](http://redis.io/commands) 和 [Redis Documentation](http://redis.io/documentation)。

- [如何在 Python 语言下调用 docker 环境安装的 Redis](http://gree2.github.io/python/2016/05/14/python-with-docker-redis)
- [Redis【入门】就这一篇！](https://zhuanlan.zhihu.com/p/37982685)（知乎）
- **Redis** 在线学习小程序：http://try.redis.io



[TOC]

![](https://i.loli.net/2019/01/14/5c3c0c34e6d93.jpeg)



---

> A  [TUTORIAL](http://try.redis.io/)  for Redis.



## 基本指令 SET/GET/INCR/DEL

Redis 里的数据都是所谓的 key-value 储存体系，通常也称为是 NoSQL (Not Only SQL) 数据库，意思是“不仅仅是SQL”。对于“键-值”对形式的数据储存结构，关键是对于某个 key，肯定存在数据与之对应，叫做 value。以后只要告诉 Redis 准确的某 key，就会得到相应的 value。我们可以用 [SET](http://redisdoc.com/string/set.html) 指令来将一个为 `fido` 的 value 储存在 key `server:name` 中：

```
SET server:name "fido"
```

这个指令一打出去，Redis 就会永久的储存了这一对数据。于是乎，当我们想要用 [GET](http://redisdoc.com/string/get.html) 指令问 Redis “一个 key 为 `server:name` 的 value 是啥来着？”，那么 Redis 就会这样回答你：

```bash
GET server:name # => "fido"
```

（注：上面代码里的符号 "=>" 表示会得到的系统返回结果）

介绍了“增”和“查”了以后，那就是要介绍一下“删”。[DEL](http://redisdoc.com/database/del.html) 这个指令就可以删除一个给定的 key 及其与之相对应的 value。此外，SET-if-not-exists （也就是 [SETNX](http://redisdoc.com/string/setnx.html) 指令）是说增加一个 key 在当前仅当 Redis 中还不存在这个 key 的时候。对于已经存在的 key 来说，要用 [INCR](http://redisdoc.com/string/incr.html) 指令来实现在相应的 key 上对 value 数值加一。看下面的例子体会一二：

```bash
SET connections # 10
INCR connections # => 11
INCR connections # => 12
DEL connections
INCR connections # => 1
```

可以看到 INCR 指令很适合作为一个简单计数器来使用。但或许你会疑惑，计数不就是数学运算“+1”这么简单呗，把对应 key 的 value 用 GET 指令拿出来，运算一下后，再 SET 回到 Redis 效率是一样的嘛？其实对于一个端（client）来说确实如此，但对于多个端同时进行计数的话就明显不是了，比如下面这个小反例：

1. A 端通过 GET 得到名为 `count` 的 KEY，其 value 是 10.
2. 同理，B 端也 GET `count`的 value 是 10.
3. A 端计数运算后要 SET `count` 为 11.
4. B 端计数运算后也要 SET `count` 为 11.

上面就是同时多端同时计数的效果，但我们期待的是最后计数为 12，不是 11 哦！这里的关键要素是这个例子的操作不是 atomic operation（原子操作）。而 INCR 指令就可以避免上面例子的情况，因为它就是 atomic 的数据操作。就算是你再很多的复杂端同时给出要“计数”的数据操作请求，都不会乱，在 Redis 中有很多类似 atomic 的数据操作用在各种各样类型的数据上。



## 基本指令 EXPIRE/TTL

在 Redis 中，可以让某指定的 key 存在在内存中一段有限的时间。就好比是给某数据设定一个“自毁倒计时”似的，这是通过 [EXPIRE](http://redisdoc.com/expire/expire.html) 和  [TTL](http://redisdoc.com/expire/ttl.html) 指令来实现：

```bash
SET resource:lock "Redis Demo"
EXPIRE resource:lock 120
```

上面例子的第二行就是在要求 key 为 `resource:lock` 的数据将会在 120 秒后自动删除。你可以在这段 120 秒的等待时间里，用 TTL 指令来测试某 key 还能"活"多久？TTL 指令会返回被删除前的剩余时间。

```bash
TTL resource:lock # => 113
# after 113s
TTL resource:lock # => -2
```

TTL 指令返回 -2 的意思就是该 key 已经不存在了，返回 -1 的意思就是该 key 还没有被 EXPIRE 操作过。如果用 SET 指令操作过该 key，"自毁倒计时"会解除，TTL 指令将会被重置：

```bash
SET resource:lock "Redis Demo 1"
EXPIRE resource:lock 120
TTL resource:lock # => 119
SET resource:lock "Redis Demo 2"
TTL resource:lock # => -1
```



## 数据结构：list

Redis 也可以使用更复杂些的数据结构，也不是数值那么简单，首先来介绍一下 `list`。`list` 就是一串有序的数值。对于操作这种数据结构的常见 Redis 指令有 [RPUSH](http://redisdoc.com/list/rpush.html)，[LPUSH](http://redisdoc.com/list/lpush.html)，[LLEN](http://redisdoc.com/list/llen.html)，[LRANGE](http://redisdoc.com/list/lrange.html)，[LPOP](http://redisdoc.com/list/lpop.html) 和 [RPOP](http://redisdoc.com/list/rpop.html) 等。我们可以直接将一个 key 的 value 数值视作 list 的一个元素开始搞事情。

[RPUSH](http://redisdoc.com/list/rpush.html) 可以添加一个新的 value 到 list 的右末端：

```bash
RPUSH friends "Alice"
RPUSH friends "Bob"
```

[LPUSH](http://redisdoc.com/list/lpush.html) 可以添加一个新的 value 到 list 的左前端：

```bash
LPUSH friends "Sam"
```

（此时，我们的 key `friends` 的 list 应该是："Sam" "Alice" "Bob"）

[LRANGE](http://redisdoc.com/list/lrange.html) 可以返回一个 list 的子集，这也就是所谓的切片操作，指令后面跟上的两个数字分别代表的是被切片 list 两端的索引值。其中，-1 代表的是直接到底，这些规则与 Python 等编程语言是一致的。

```bash
LRANGE friends 0 -1 # => 1) "Sam", 2) "Alice", 3) "Bob"
LRANGE friends 0 1 # => 1) "Sam", 2) "Alice"
LRANGE friends 1 2 # => 1) "Alice", 2) "Bob"
```

[LLEN](http://redisdoc.com/list/llen.html) 返回某 key 中 list 的长度。

```shell
LLEN friends # => 3
```

[LPOP](http://redisdoc.com/list/lpop.html) 去掉 list 中的第一个元素，并且返回该元素。

```shell
LPOP friends # => "Sam"
```

[RPOP](http://redisdoc.com/list/rpop.html) 去掉 list 中的最后一个元素，并且返回该元素。

```shell
RPOP friends # => "Bob"
```

经过上面一串操作后，我们的 `friends` 中就是只有一个元素的 list 了。

```shell
LLEN friends # => 1
LRANGE friends 0 -1 # => 1) "Alice"
```



## 数据结构：set

接下来，我们来看下 set 这个数据结构，它和 list 非常像，但是没有有序性，并且其中每个元素都只能出现一次，不可以重复出现。关于 set 常用的数据操作有 [SADD](http://redisdoc.com/set/sadd.html)，[SREM](http://redisdoc.com/set/srem.html)，[SISMEMBER](http://redisdoc.com/set/sismember.html)，[SMEMBERS](http://redisdoc.com/set/sismember.html) 和 [SUNION](http://redisdoc.com/set/sunion.html)。

[SADD](http://redisdoc.com/set/sadd.html) 指令是给某 key 中的 set 增加元素用的。

```shell
SADD superpowers "flight"
SADD superpowers "x-ray vision"
SADD superpowers "reflexes"
```

[SREM](http://redisdoc.com/set/srem.html) 指令是删除某 key 中指定的元素。

```shell
SREM superpowers "reflexes"
```

[SISMEMBER](http://redisdoc.com/set/sismember.html) 指令可以返回是否某给定元素在某 key 中。1 就是 True，0 就是 False。

```shell
SISMEMBER superpowers "flight" # => 1
SISMEMBER superpowers "reflexes" # => 0
```

[SMEMBERS](http://redisdoc.com/set/sismember.html) 指令可以返回某 key 中所有的元素。

```shell
SMEMBERS superpowers # => 1) "flight", 2) "x-ray vision"
```

[SUNION](http://redisdoc.com/set/sunion.html) 指令可以合并两个以上 key，并且返回合并后的所有元素。（合并=并集）

```shell
SADD birdpowers "pecking"
SADD birdpowers "flight"
SUNION superpowers birdpowers # => 1) "pecking", 2) "x-ray vision", 3) "flight"
```







## 数据结构：sorted set

虽然 Set 用起来已经很带感了，但是完全的无序性还是有些不是那么的用途广泛。于是 Redis 1.2 开始引入了新的 sorted set 数据结构。

sorted set 和 set 其实还是非常像的，仅仅是某 key 中的每个元素中附带了一个数值，用以对元素进行排序。

```shell
ZADD hackers 1940 "Alan Kay"
ZADD hackers 1906 "Grace Hopper"
ZADD hackers 1953 "Richard Stallman"
ZADD hackers 1965 "Yukihiro Matsumoto"
ZADD hackers 1916 "Claude Shannon"
ZADD hackers 1969 "Linus Torvalds"
ZADD hackers 1957 "Sophie Wilson"
ZADD hackers 1912 "Alan Turing"
```

在上面这个例子中，我们使用 [ZADD](http://redisdoc.com/sorted_set/zadd.html) 指令在 key 为 `hackers` 中添加了每一个指明黑客的出生年月和名字作为元素（value）。同样的，我们可以使用 [ZRANGE](http://redisdoc.com/sorted_set/zrange.html) 来对 sorted set 进行切片：

```shell
ZRANGE hackers 2 4 # => 1) "Claude Shannon", 2) "Alan Kay", 3) "Richard Stallman"
```









## 数据结构：hashes

之前介绍的数据结构都算是很简单，Redis 其实还可以处理 Hashes（哈希）数据结构。

Hashes 可以实现一种嵌套式的映射（比方说，一个用户左右一个 key 的话，其 value 可以进一步赋予很多 key （叫做 field）如姓名，电子邮件，密码等等）：

```shell
HSET user:1000 name "John Smith"
HSET user:1000 email "john.smith@example.com"
HSET user:1000 password "s3cret"
```

上面例子中使用了 [HSET](http://redisdoc.com/hash/hset.html) 指令来创建 hash。获取所有值的指令是 [HGETALL](http://redisdoc.com/hash/hgetall.html)：

```shell
HGETALL user:1000
```

其实你也可以一次性创建，使用 [HMSET](http://redisdoc.com/hash/hmset.html) 指令就可以：

```shell
HMSET user:1001 name "Mary Jones" password "hidden" email "mjones@example.com"
```

如果你想获取某 key 下的 field 信息，可以使用 [HGET](http://redisdoc.com/hash/hget.html) 指令：

```shell
HGET user:1001 name # => "Mary Jones"
```

其实你可以感受到对 hash 中的 field 进行数据操作是非常熟悉的，比方说你若想对某 field 进行计数操作的话，可以使用 [HINCRBY](http://redisdoc.com/hash/hincrby.html) 指令即可，这也是 atomic 的数据操作哦！

```shell
HSET user:1000 visits 10
HINCRBY user:1000 visits 1 # => 11
HINCRBY user:1000 visits 10 # => 21
HDEL user:1000 visits
HINCRBY user:1000 visits 1 # => 1
```

更多关于 Hash 的指令可以查看[这里](https://redis.io/commands#hash)。

---



如果你读到这里感到轻松，操作起来已经深谙其道的话，建议阅读上面的 **Redis 参考资料** 所罗列的资料，也强烈建议读一读这一篇官方文章（[Introduction to Redis Data Types](http://redis.io/topics/data-types-intro)），文中会深入浅出的介绍所有 Redis 数据类型的差异和特点，也会对各数据类型的用途用法有详细介绍。




（END）

---

[返回到首页](../index.html) | [返回到顶部](./Redis_Tutorial.html)


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


