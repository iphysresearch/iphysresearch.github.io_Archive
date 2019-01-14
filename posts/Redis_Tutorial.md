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


Redis是一个开源的使用ANSI C语言编写、遵守BSD协议、支持网络、可基于**内存**亦可持久化的日志型、Key-Value 数据库，并提供多种语言的API。它通常被称为数据结构服务器，因为值（value）可以是 字符串(String), 哈希(Map), 列表(list), 集合(sets) 和 有序集合(sorted sets)等类型。从2010年3月15日起，Redis的开发工作由VMware主持。从2013年5月开始，Redis的开发由Pivotal赞助。

![](https://i.loli.net/2019/01/14/5c3c0c0ca2e8d.png)

- 优点
  - 性能极高 – Redis能读的速度是110000次/s,写的速度是81000次/s 。
  - 丰富的数据类型 – Redis支持二进制案例的 Strings, Lists, Hashes, Sets 及 Ordered Sets 数据类型操作。
  - 原子 – Redis的所有操作都是原子性的，意思就是要么成功执行要么失败完全不执行。单个操作是原子性的。多个操作也支持事务，即原子性，通过MULTI和EXEC指令包起来。
  - 丰富的特性 – Redis还支持 publish/subscribe, 通知, key 过期等等特性。
- Redis与其他key-value存储有什么不同？
  - Redis有着更为复杂的数据结构并且提供对他们的原子性操作，这是一个不同于其他数据库的进化路径。Redis的数据类型都是基于基本数据结构的同时对程序员透明，无需进行额外的抽象。
  - Redis运行在内存中但是可以持久化到磁盘，所以在对不同数据集进行高速读写时需要权衡内存，因为数据量不能大于硬件内存。在内存数据库方面的另一个优点是，相比在磁盘上相同的复杂的数据结构，在内存中操作起来非常简单，这样Redis可以做很多内部复杂性很强的事情。同时，在磁盘格式方面他们是紧凑的以追加的方式产生的，因为他们并不需要进行随机访问。

---

**Redis** official website: http://redis.io

**Redis** 中文教程资源：http://www.runoob.com/redis/redis-tutorial.html

> **Redis** 命令参考：http://redisdoc.com/index.html
>
> 本文档是 [Redis Command Reference](http://redis.io/commands) 和 [Redis Documentation](http://redis.io/documentation) 的中文翻译版。

[python with docker redis](http://gree2.github.io/python/2016/05/14/python-with-docker-redis)



> A list of supported commands, or any valid Redis command to play with the database.
>
> [DECR](http://try.redis.io/#help), [DECRBY](http://try.redis.io/#help), [DEL](http://try.redis.io/#help), [EXISTS](http://try.redis.io/#help), [EXPIRE](http://try.redis.io/#help), [GET](http://try.redis.io/#help), [GETSET](http://try.redis.io/#help), [HDEL](http://try.redis.io/#help), [HEXISTS](http://try.redis.io/#help), [HGET](http://try.redis.io/#help), [HGETALL](http://try.redis.io/#help), [HINCRBY](http://try.redis.io/#help), [HKEYS](http://try.redis.io/#help), [HLEN](http://try.redis.io/#help), [HMGET](http://try.redis.io/#help), [HMSET](http://try.redis.io/#help), [HSET](http://try.redis.io/#help), [HVALS](http://try.redis.io/#help), [INCR](http://try.redis.io/#help), [INCRBY](http://try.redis.io/#help), [KEYS](http://try.redis.io/#help), [LINDEX](http://try.redis.io/#help), [LLEN](http://try.redis.io/#help), [LPOP](http://try.redis.io/#help), [LPUSH](http://try.redis.io/#help), [LRANGE](http://try.redis.io/#help), [LREM](http://try.redis.io/#help), [LSET](http://try.redis.io/#help), [LTRIM](http://try.redis.io/#help), [MGET](http://try.redis.io/#help), [MSET](http://try.redis.io/#help), [MSETNX](http://try.redis.io/#help), [MULTI](http://try.redis.io/#help), [PEXPIRE](http://try.redis.io/#help), [RENAME](http://try.redis.io/#help), [RENAMENX](http://try.redis.io/#help), [RPOP](http://try.redis.io/#help), [RPOPLPUSH](http://try.redis.io/#help), [RPUSH](http://try.redis.io/#help), [SADD](http://try.redis.io/#help), [SCARD](http://try.redis.io/#help), [SDIFF](http://try.redis.io/#help), [SDIFFSTORE](http://try.redis.io/#help), [SET](http://try.redis.io/#help), [SETEX](http://try.redis.io/#help), [SETNX](http://try.redis.io/#help), [SINTER](http://try.redis.io/#help), [SINTERSTORE](http://try.redis.io/#help), [SISMEMBER](http://try.redis.io/#help), [SMEMBERS](http://try.redis.io/#help), [SMOVE](http://try.redis.io/#help), [SORT](http://try.redis.io/#help), [SPOP](http://try.redis.io/#help), [SRANDMEMBER](http://try.redis.io/#help), [SREM](http://try.redis.io/#help), [SUNION](http://try.redis.io/#help), [SUNIONSTORE](http://try.redis.io/#help), [TTL](http://try.redis.io/#help), [TYPE](http://try.redis.io/#help), [ZADD](http://try.redis.io/#help), [ZCARD](http://try.redis.io/#help), [ZCOUNT](http://try.redis.io/#help), [ZINCRBY](http://try.redis.io/#help), [ZRANGE](http://try.redis.io/#help), [ZRANGEBYSCORE](http://try.redis.io/#help), [ZRANK](http://try.redis.io/#help), [ZREM](http://try.redis.io/#help), [ZREMRANGEBYSCORE](http://try.redis.io/#help), [ZREVRANGE](http://try.redis.io/#help), [ZSCORE](http://try.redis.io/#help)



[TOC]

![](https://i.loli.net/2019/01/14/5c3c0c34e6d93.jpeg)



---

> A  [TUTORIAL](http://try.redis.io/)  for Redis.



## SET/GET/INCR/DEL

Redis is what is called a key-value store, often referred to as a NoSQL database. The essence of a key-value store is the ability to store some data, called a value, inside a key. This data can later be retrieved only if we know the exact key used to store it. We can use the command [SET](http://try.redis.io/#help) to store the value "fido" at key "server:name":

```shell
SET server:name "fido"
```

Redis will store our data permanently, so we can later ask "What is the value stored at key server:name?" and Redis will reply with "fido":

```bash
GET server:name # => "fido"
```

Tip: The text after the arrow (=>) shows the expected output.

Other common operations provided by key-value stores are [DEL](http://try.redis.io/#help) to delete a given key and associated value, SET-if-not-exists (called [SETNX](http://try.redis.io/#help) on Redis) that sets a key only if it does not already exist, and [INCR](http://try.redis.io/#help) to atomically increment a number stored at a given key:

```shell
SET connections # 10
INCR connections # => 11
INCR connections # => 12
DEL connections
INCR connections # => 1
```

There is something special about [INCR](http://try.redis.io/#help). Why do we provide such an operation if we can do it ourself with a bit of code? After all it is as simple as:

```shell
x = GET count
x = x + 1
SET count x
```

The problem is that doing the increment in this way will only work as long as there is a single client using the key. See what happens if two clients are accessing this key at the same time:

1. Client A reads *count* as 10.
2. Client B reads *count* as 10.
3. Client A increments 10 and sets *count* to 11.
4. Client B increments 10 and sets *count* to 11.

We wanted the value to be 12, but instead it is 11! This is because incrementing the value in this way is not an atomic operation. Calling the [INCR](http://try.redis.io/#help) command in Redis will prevent this from happening, because it *is* an atomic operation. Redis provides many of these atomic operations on different types of data.



## EXPIRE/TTL

Redis can be told that a key should only exist for a certain length of time. This is accomplished with the [EXPIRE](http://try.redis.io/#help) and [TTL](http://try.redis.io/#help) commands.

```shell
SET resource:lock "Redis Demo"
EXPIRE resource:lock 120
```

This causes the key *resource:lock* to be deleted in 120 seconds. You can test how long a key will exist with the [TTL](http://try.redis.io/#help) command. It returns the number of seconds until it will be deleted.

```shell
TTL resource:lock # => 113
# after 113s
TTL resource:lock # => -2
```

The *-2* for the [TTL](http://try.redis.io/#help) of the key means that the key does not exist (anymore). A *-1* for the [TTL](http://try.redis.io/#help) of the key means that it will never expire. Note that if you [SET](http://try.redis.io/#help) a key, its [TTL](http://try.redis.io/#help) will be reset.

```shell
SET resource:lock "Redis Demo 1"
EXPIRE resource:lock 120
TTL resource:lock # => 119
SET resource:lock "Redis Demo 2"
TTL resource:lock # => -1
```



## Data Structures: list

Redis also supports several more complex data structures. The first one we'll look at is a list. A list is a series of ordered values. Some of the important commands for interacting with lists are [RPUSH](http://try.redis.io/#help), [LPUSH](http://try.redis.io/#help), [LLEN](http://try.redis.io/#help), [LRANGE](http://try.redis.io/#help), [LPOP](http://try.redis.io/#help), and [RPOP](http://try.redis.io/#help). You can immediately begin working with a key as a list, as long as it doesn't already exist as a different type.

[RPUSH](http://try.redis.io/#help) puts the new value at the end of the list.

```shell
RPUSH friends "Alice"
RPUSH friends "Bob"
```

[LPUSH](http://try.redis.io/#help) puts the new value at the start of the list.

```shell
LPUSH friends "Sam"
```

[LRANGE](http://try.redis.io/#help) gives a subset of the list. It takes the index of the first element you want to retrieve as its first parameter and the index of the last element you want to retrieve as its second parameter. A value of -1 for the second parameter means to retrieve elements until the end of the list.

```shell
LRANGE friends 0 -1 # => 1) "Sam", 2) "Alice", 3) "Bob"
LRANGE friends 0 1 # => 1) "Sam", 2) "Alice"
LRANGE friends 1 2 # => 1) "Alice", 2) "Bob"
```

[LLEN](http://try.redis.io/#help) returns the current length of the list.

```shell
LLEN friends # => 3
```

[LPOP](http://try.redis.io/#help) removes the first element from the list and returns it.

```shell
LPOP friends # => "Sam"
```

[RPOP](http://try.redis.io/#help) removes the last element from the list and returns it.

```shell
RPOP friends # => "Bob"
```

Note that the list now only has one element:

```shell
LLEN friends # => 1
LRANGE friends 0 -1 # => 1) "Alice"
```





## Data Structures: set

The next data structure that we'll look at is a set. A set is similar to a list, except it does not have a specific order and each element may only appear once. Some of the important commands in working with sets are [SADD](http://try.redis.io/#help), [SREM](http://try.redis.io/#help), [SISMEMBER](http://try.redis.io/#help), [SMEMBERS](http://try.redis.io/#help) and [SUNION](http://try.redis.io/#help).

[SADD](http://try.redis.io/#help) adds the given value to the set.

```shell
SADD superpowers "flight"
SADD superpowers "x-ray vision"
SADD superpowers "reflexes"
```

[SREM](http://try.redis.io/#help) removes the given value from the set.

```shell
SREM superpowers "reflexes"
```

[SISMEMBER](http://try.redis.io/#help) tests if the given value is in the set. It returns *1* if the value is there and *0* if it is not.

```shell
SISMEMBER superpowers "flight" # => 1
SISMEMBER superpowers "reflexes" # => 0
```

[SMEMBERS](http://try.redis.io/#help) returns a list of all the members of this set.

```shell
SMEMBERS superpowers # => 1) "flight", 2) "x-ray vision"
```

[SUNION](http://try.redis.io/#help) combines two or more sets and returns the list of all elements.

```shell
SADD birdpowers "pecking"
SADD birdpowers "flight"
SUNION superpowers birdpowers # => 1) "pecking", 2) "x-ray vision", 3) "flight"
```



## Data Structures: sorted set

Sets are a very handy data type, but as they are unsorted they don't work well for a number of problems. This is why Redis 1.2 introduced Sorted Sets.

A sorted set is similar to a regular set, but now each value has an associated score. This score is used to sort the elements in the set.

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

In these examples, the scores are years of birth and the values are the names of famous hackers.

```shell
ZRANGE hackers 2 4 # => 1) "Claude Shannon", 2) "Alan Kay", 3) "Richard Stallman"
```



## Data Structures: hashes

Simple strings, sets and sorted sets already get a lot done but there is one more data type Redis can handle: Hashes.

Hashes are maps between string fields and string values, so they are the perfect data type to represent objects (eg: A User with a number of fields like name, surname, age, and so forth):

```shell
HSET user:1000 name "John Smith"
HSET user:1000 email "john.smith@example.com"
HSET user:1000 password "s3cret"
```

To get back the saved data use [HGETALL](http://try.redis.io/#help):

```shell
HGETALL user:1000
```

You can also set multiple fields at once:

```shell
HMSET user:1001 name "Mary Jones" password "hidden" email "mjones@example.com"
```

If you only need a single field value that is possible as well:

```shell
HGET user:1001 name # => "Mary Jones"
```

Numerical values in hash fields are handled exactly the same as in simple strings and there are operations to increment this value in an atomic way.

```shell
HSET user:1000 visits 10
HINCRBY user:1000 visits 1 # => 11
HINCRBY user:1000 visits 10 # => 21
HDEL user:1000 visits
HINCRBY user:1000 visits 1 # => 1
```

Check the [full list of Hash commands](http://redis.io/commands#hash) for more information.

---

That wraps up the *Try Redis* tutorial. Please feel free to goof around with this console as much as you'd like.

Check out the following links to continue learning about Redis.

- [Redis Documentation](http://redis.io/documentation)
- [Command Reference](http://redis.io/commands)
- [Implement a Twitter Clone in Redis](http://redis.io/topics/twitter-clone)
- [Introduction to Redis Data Types](http://redis.io/topics/data-types-intro)




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





