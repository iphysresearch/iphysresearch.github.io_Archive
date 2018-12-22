---
title: Ray Tutorial
date: 2018-12-13
---

[返回到首页](../index.html)

---

![](https://i.loli.net/2018/12/13/5c11e943c8355.png)

**Ray is a flexible, high-performance distributed execution framework.**

---

[TOC]

# Ray 

> **Some important references:** 
>
> - Ray论文：[Real-Time Machine Learning: The Missing Pieces](https://arxiv.org/abs/1703.03924)
>
> - Ray开发手册：<http://ray.readthedocs.io/en/latest/index.html>
>
> - Ray源代码：<https://github.com/ray-project/ray>
> - [高性能分布式执行框架——Ray](https://www.cnblogs.com/fanzhidongyzby/p/7901139.html) （blog）
>
> - [Ray - 面向增强学习场景的分布式计算框架](https://www.jianshu.com/p/a5f8665d84ff) （blog）
> - [Ray——为AI而生的高性能分布式执行框架](http://gaiding.com/index.html)（book）
>
>
>
> **Core concepts**:
>
> Ray 是使用什么样的架构对分布式计算做出如上抽象的呢，一下给出了 Ray 的系统架构（来自 Ray [论文](https://arxiv.org/abs/1703.03924)）。
>
> ![](https://i.loli.net/2018/12/13/5c11e73c9a133.png)
>
> 作为分布式计算系统，Ray 仍旧遵循了典型的 Master-Slave 的设计：Master 负责全局协调和状态维护，Slave 执行分布式计算任务。不过和传统的分布式计算系统不同的是，Ray使用了**混合任务调度**的思路。在集群部署模式下，Ray 启动了以下关键组件：
>
> 1. **GlobalScheduler**：Master上启动了一个全局调度器，用于接收本地调度器提交的任务，并将任务分发给合适的本地任务调度器执行。
> 2. **RedisServer**：Master上启动了一到多个RedisServer用于保存分布式任务的状态信息（ControlState），包括对象机器的映射、任务描述、任务debug信息等。
> 3. **LocalScheduler**：每个Slave上启动了一个本地调度器，用于提交任务到全局调度器，以及分配任务给当前机器的Worker进程。
> 4. **Worker**：每个Slave上可以启动多个Worker进程执行分布式任务，并将计算结果存储到ObjectStore。
> 5. **ObjectStore**：每个Slave上启动了一个ObjectStore存储只读数据对象，Worker可以通过共享内存的方式访问这些对象数据，这样可以有效地减少内存拷贝和对象序列化成本。ObjectStore底层由Apache Arrow实现。
> 6. **Plasma**：每个Slave上的ObjectStore都由一个名为Plasma的对象管理器进行管理，它可以在Worker访问本地ObjectStore上不存在的远程数据对象时，主动拉取其它Slave上的对象数据到当前机器。
>
> 需要说明的是，Ray的论文中提及，全局调度器可以启动一到多个，而目前Ray的实现文档里讨论的内容都是基于一个全局调度器的情况。我猜测可能是Ray尚在建设中，一些机制还未完善，后续读者可以留意此处的细节变化。
>
> Ray的任务也是通过类似Spark中Driver的概念的方式进行提交的，有所不同的是：
>
> 1. Spark的Driver提交的是任务DAG，一旦提交则不可更改。
> 2. 而Ray提交的是更细粒度的remote function，任务DAG依赖关系由函数依赖关系自由定制。
>
> 论文给出的架构图里并未画出Driver的概念，因此这位 [牛人](http://www.cnblogs.com/fanzhidongyzby/p/7901139.html) 在其基础上做了一些修改和扩充。
>
> ![](https://i.loli.net/2018/12/13/5c11e7a40f05a.png)
>
> Ray的Driver节点和和Slave节点启动的组件几乎相同，不过却有以下区别：
>
> 1. Driver上的工作进程DriverProcess一般只有一个，即用户启动的PythonShell。Slave可以根据需要创建多个WorkerProcess。
> 2. Driver只能提交任务，却不能接收来自全局调度器分配的任务。Slave可以提交任务，也可以接收全局调度器分配的任务。
> 3. Driver可以主动绕过全局调度器给Slave发送Actor调用任务（此处设计是否合理尚不讨论）。Slave只能接收全局调度器分配的计算任务。



---

# Ray Tutorial (with solutions)

**Github**：https://github.com/ray-project/tutorial



---

## Try Ray on Binder (Experimental)

Try the Ray tutorials online on [Binder](https://mybinder.org/v2/gh/ray-project/tutorial/master).



> 为了方便学习和查找，将 12 个 Exercise 都整理如下，并附有自己写好的答案（仅供参考）。

---

## Exercise 1 - Simple Data Parallel Example

**GOAL:** The goal of this exercise is to show how to run simple tasks in parallel.

This script is too slow, and the computation is embarrassingly parallel. In this exercise, you will use Ray to execute the functions in parallel to speed it up.

### Concept for this Exercise - Remote Functions

> The standard way to turn a Python function into a remote function is to add the `@ray.remote` decorator. Here is an example.
>
> ```python
> # A regular Python function.
> def regular_function():
>     return 1
> 
> # A Ray remote function.
> @ray.remote
> def remote_function():
>     return 1
> ```
>
> The differences are the following:
>
> 1. **Invocation:** The regular version is called with `regular_function()`, whereas the remote version is called with `remote_function.remote()`.
> 2. **Return values:** `regular_function` immediately executes and returns `1`, whereas `remote_function` immediately returns an object ID (a future) and then creates a task that will be executed on a worker process. The result can be obtained with `ray.get`.
>     ```python
>     >>> regular_function()
>     1
>     
>     >>> remote_function.remote()
>     ObjectID(1c80d6937802cd7786ad25e50caf2f023c95e350)
>     
>     >>> ray.get(remote_function.remote())
>     1
>     ```
> 3. **Parallelism:** Invocations of `regular_function` happen **serially**, for example
>     ```python
>     # These happen serially.
>     for _ in range(4):
>         regular_function()
>     ```
>     whereas invocations of `remote_function` happen in **parallel**, for example
>     ```python
>     # These happen in parallel.
>     for _ in range(4):
>         remote_function.remote()
>     ```

---

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import time
```

Start Ray. By default, Ray does not schedule more tasks concurrently than there are CPUs. This example requires four tasks to run concurrently, so we tell Ray that there are four CPUs. Usually this is not done and Ray computes the number of CPUs using `psutil.cpu_count()`. The argument `ignore_reinit_error=True` just ignores errors if the cell is run multiple times.

The call to `ray.init` starts a number of processes.

```python
ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)
```

**EXERCISE:** The function below is slow. Turn it into a remote function using the `@ray.remote` decorator.

```python
# This function is a proxy for a more interesting and computationally
# intensive function.
@ray.remote
def slow_function(i):
    time.sleep(1)
    return i
```

**EXERCISE:** The loop below takes too long. The four function calls could be executed in parallel. Instead of four seconds, it should only take one second. Once `slow_function` has been made a remote function, execute these four tasks in parallel by calling `slow_function.remote()`. Then obtain the results by calling `ray.get` on a list of the resulting object IDs.

```python
# Sleep a little to improve the accuracy of the timing measurements below.
# We do this because workers may still be starting up in the background.
time.sleep(2.0)
start_time = time.time()

results = ray.get([slow_function.remote(i) for i in range(4)])

end_time = time.time()
duration = end_time - start_time

print('The results are {}. This took {} seconds. Run the next cell to see '
      'if the exercise was done correctly.'.format(results, duration))
# The results are [0, 1, 2, 3]. This took 1.0055913925170898 seconds. Run the next cell to see if the exercise was done correctly.
```

**VERIFY:** Run some checks to verify that the changes you made to the code were correct. Some of the checks should fail when you initially run the cells. After completing the exercises, the checks should pass.

```python
assert results == [0, 1, 2, 3], 'Did you remember to call ray.get?'
assert duration < 1.1, ('The loop took {} seconds. This is too slow.'
                        .format(duration))
assert duration > 1, ('The loop took {} seconds. This is too fast.'
                      .format(duration))

print('Success! The example took {} seconds.'.format(duration))
# Success! The example took 1.0055913925170898 seconds.
```

**EXERCISE:** Use the UI to view the task timeline and to verify that the four tasks were executed in parallel. After running the cell below, you'll need to click on **View task timeline**".
- Using the **second** button, you can click and drag to **move** the timeline.
- Using the **third** button, you can click and drag to **zoom**. You can also zoom by holding "alt" and scrolling.

**NOTE:** Normally our UI is used as a separate Jupyter notebook. However, for simplicity we embedded the relevant feature here in this notebook.

**NOTE:** The first time you click **View task timeline** it may take **several minutes** to start up. This will change.

**NOTE:** If you run more tasks and want to regenerate the UI, you need to move the slider bar a little bit and then click **View task timeline** again.

**NOTE:** The timeline visualization may only work in **Chrome**.

```python
import ray.experimental.ui as ui
ui.task_timeline()
```

![](https://i.loli.net/2018/12/13/5c11e25231d2a.png)

---

## Exercise 2 - Parallel Data Processing with Task Dependencies

**GOAL:** The goal of this exercise is to show how to pass object IDs into remote functions to encode dependencies between tasks.

In this exercise, we construct a sequence of tasks each of which depends on the previous mimicking a data parallel application. Within each sequence, tasks are executed serially, but multiple sequences can be executed in parallel.

In this exercise, you will use Ray to parallelize the computation below and speed it up.

### Concept for this Exercise - Task Dependencies

> Suppose we have two remote functions defined as follows.
>
> ```python
> @ray.remote
> def f(x):
>     return x
> ```
>
> Arguments can be passed into remote functions as usual.
>
> ```python
> >>> x1_id = f.remote(1)
> >>> ray.get(x1_id)
> 1
> 
> >>> x2_id = f.remote([1, 2, 3])
> >>> ray.get(x2_id)
> [1, 2, 3]
> ```
>
> **Object IDs** can also be passed into remote functions. When the function actually gets executed, **the argument will be a retrieved as a regular Python object**.
>
> ```python
> >>> y1_id = f.remote(x1_id)
> >>> ray.get(y1_id)
> 1
> 
> >>> y2_id = f.remote(x2_id)
> >>> ray.get(y2_id)
> [1, 2, 3]
> ```
>
> So when implementing a remote function, the function should expect a regular Python object regardless of whether the caller passes in a regular Python object or an object ID.
>
> **Task dependencies affect scheduling.** In the example above, the task that creates `y1_id` depends on the task that creates `x1_id`. This has the following implications.
>
> - The second task will not be executed until the first task has finished executing.
> - If the two tasks are scheduled on different machines, the output of the first task (the value corresponding to `x1_id`) will be copied over the network to the machine where the second task is scheduled.

---

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time
```

```python
ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)
```

These are some helper functions that mimic an example pattern of a data parallel application.

**EXERCISE:** You will need to turn all of these functions into remote functions. When you turn these functions into remote function, you do not have to worry about whether the caller passes in an object ID or a regular object. In both cases, the arguments will be regular objects when the function executes. This means that even if you pass in an object ID, you **do not need to call `ray.get`** inside of these remote functions.

```python
@ray.remote
def load_data(filename):
    time.sleep(0.1)
    return np.ones((1000, 100))

@ray.remote
def normalize_data(data):
    time.sleep(0.1)
    return data - np.mean(data, axis=0)

@ray.remote
def extract_features(normalized_data):
    time.sleep(0.1)
    return np.hstack([normalized_data, normalized_data ** 2])

@ray.remote
def compute_loss(features):
    num_data, dim = features.shape
    time.sleep(0.1)
    return np.sum((np.dot(features, np.ones(dim)) - np.ones(num_data)) ** 2)

assert hasattr(load_data, 'remote'), 'load_data must be a remote function'
assert hasattr(normalize_data, 'remote'), 'normalize_data must be a remote function'
assert hasattr(extract_features, 'remote'), 'extract_features must be a remote function'
assert hasattr(compute_loss, 'remote'), 'compute_loss must be a remote function'
```

**EXERCISE:** The loop below takes too long. Parallelize the four passes through the loop by turning `load_data`, `normalize_data`, `extract_features`, and `compute_loss` into remote functions and then retrieving the losses with `ray.get`.

**NOTE:** You should only use **ONE** call to `ray.get`. For example, the object ID returned by `load_data` should be passed directly into `normalize_data` without needing to be retrieved by the driver.

```python
# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

losses = []
for filename in ['file1', 'file2', 'file3', 'file4']:
    inner_start = time.time()

    data = load_data.remote(filename)
    normalized_data = normalize_data.remote(data)
    features = extract_features.remote(normalized_data)
    loss = compute_loss.remote(features)
    losses.append(loss)
    
    inner_end = time.time()
    
    if inner_end - inner_start >= 0.1:
        raise Exception('You may be calling ray.get inside of the for loop! '
                        'Doing this will prevent parallelism from being exposed. '
                        'Make sure to only call ray.get once outside of the for loop.')

print('The losses are {}.'.format(losses) + '\n')
loss = sum(ray.get(losses))

end_time = time.time()
duration = end_time - start_time

print('The loss is {}. This took {} seconds. Run the next cell to see '
      'if the exercise was done correctly.'.format(loss, duration))

# The losses are [ObjectID(c93d08295a9c442613ed4b4eca48f94ec6814f5b), ObjectID(b2826a902ef0845f30bc2ee0dd1ea4f78629bd8c), ObjectID(7dff67fd2906233ff53a5ea8d13932bb33f0031a), ObjectID(01d0071b7d8705f17673f5e660bd3d9c8a2c8ba1)].

# The loss is 4000.0. This took 0.6542365550994873 seconds. Run the next cell to see if the exercise was done correctly.
```

**VERIFY:** Run some checks to verify that the changes you made to the code were correct. Some of the checks should fail when you initially run the cells. After completing the exercises, the checks should pass.

```python
assert loss == 4000
assert duration < 0.8, ('The loop took {} seconds. This is too slow.'
                        .format(duration))
assert duration > 0.4, ('The loop took {} seconds. This is too fast.'
                        .format(duration))

print('Success! The example took {} seconds.'.format(duration))
# Success! The example took 0.6542365550994873 seconds.
```

**EXERCISE:** Use the UI to view the task timeline and to verify that the relevant tasks were executed in parallel. After running the cell below, you'll need to click on **View task timeline**".
- Using the **second** button, you can click and drag to **move** the timeline.
- Using the **third** button, you can click and drag to **zoom**. You can also zoom by holding "alt" and scrolling.

In the timeline, click on **View Options** and select **Flow Events** to visualize tasks dependencies.

```python
import ray.experimental.ui as ui
ui.task_timeline()
```

![](https://i.loli.net/2018/12/13/5c11e28c5c1a2.png)

---

## Exercise 3 - Nested Parallelism

**GOAL:** The goal of this exercise is to show how to create nested tasks by calling a remote function inside of another remote function.

In this exercise, you will implement the structure of a parallel hyperparameter sweep which trains a number of models in parallel. Each model will be trained using parallel gradient computations.

### Concepts for this Exercise - Nested Remote Functions

> Remote functions can call other functions. For example, consider the following.
>
> ```python
> @ray.remote
> def f():
>     return 1
> 
> @ray.remote
> def g():
>     # Call f 4 times and return the resulting object IDs.
>     return [f.remote() for _ in range(4)]
> 
> @ray.remote
> def h():
>     # Call f 4 times, block until those 4 tasks finish,
>     # retrieve the results, and return the values.
>     return ray.get([f.remote() for _ in range(4)])
> ```
>
> Then calling `g` and `h` produces the following behavior.
>
> ```python
> >>> ray.get(g.remote())
> [ObjectID(b1457ba0911ae84989aae86f89409e953dd9a80e),
>  ObjectID(7c14a1d13a56d8dc01e800761a66f09201104275),
>  ObjectID(99763728ffc1a2c0766a2000ebabded52514e9a6),
>  ObjectID(9c2f372e1933b04b2936bb6f58161285829b9914)]
> 
> >>> ray.get(h.remote())
> [1, 1, 1, 1]
> ```
>
> **One limitation** is that the definition of `f` must come before the definitions of `g` and `h` because as soon as `g` is defined, it will be pickled and shipped to the workers, and so if `f` hasn't been defined yet, the definition will be incomplete.

---

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time
```

```python
ray.init(num_cpus=9, include_webui=False, ignore_reinit_error=True)
```

This example represents a hyperparameter sweep in which multiple models are trained in parallel. Each model training task also performs data parallel gradient computations.

**EXERCISE:** Turn `compute_gradient` and `train_model` into remote functions so that they can be executed in parallel. Inside of `train_model`, do the calls to `compute_gradient` in parallel and fetch the results using `ray.get`.

```python
@ray.remote
def compute_gradient(data, current_model):
    time.sleep(0.03)
    return 1

@ray.remote
def train_model(hyperparameters):
    current_model = 0
    # Iteratively improve the current model. This outer loop cannot be parallelized.
    for _ in range(10):
        # EXERCISE: Parallelize the list comprehension in the line below. After you
        # turn "compute_gradient" into a remote function, you will need to call it
        # with ".remote". The results must be retrieved with "ray.get" before "sum"
        # is called.
        total_gradient = sum(ray.get([compute_gradient.remote(j, current_model) for j in range(2)]))
        current_model += total_gradient

    return current_model

assert hasattr(compute_gradient, 'remote'), 'compute_gradient must be a remote function'
assert hasattr(train_model, 'remote'), 'train_model must be a remote function'
```

**EXERCISE:** The code below runs 3 hyperparameter experiments. Change this to run the experiments in parallel.

```python
# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

# Run some hyperparaameter experiments.
results = []
for hyperparameters in [{'learning_rate': 1e-1, 'batch_size': 100},
                        {'learning_rate': 1e-2, 'batch_size': 100},
                        {'learning_rate': 1e-3, 'batch_size': 100}]:
    results.append(train_model.remote(hyperparameters))

# EXERCISE: Once you've turned "results" into a list of Ray ObjectIDs
# by calling train_model.remote, you will need to turn "results" back
# into a list of integers, e.g., by doing "results = ray.get(results)".
results = ray.get(results)

end_time = time.time()
duration = end_time - start_time

assert all([isinstance(x, int) for x in results]), 'Looks like "results" is {}. You may have forgotten to call ray.get.'.format(results)
```

**VERIFY:** Run some checks to verify that the changes you made to the code were correct. Some of the checks should fail when you initially run the cells. After completing the exercises, the checks should pass.

```python
assert results == [20, 20, 20]
assert duration < 0.5, ('The experiments ran in {} seconds. This is too '
                         'slow.'.format(duration))
assert duration > 0.3, ('The experiments ran in {} seconds. This is too '
                        'fast.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))
# Success! The example took 0.32144594192504883 seconds.
```

**EXERCISE:** Use the UI to view the task timeline and to verify that the pattern makes sense.

```python
import ray.experimental.ui as ui
ui.task_timeline()
```

![](https://i.loli.net/2018/12/13/5c11e2e06ae8c.png)

---

## Exercise 4 - Introducing Actors

**Goal:** The goal of this exercise is to show how to create an actor and how to call actor methods.

See the documentation on actors at http://ray.readthedocs.io/en/latest/actors.html.

Sometimes you need a "worker" process to have "state". For example, that state might be a neural network, a simulator environment, a counter, or something else entirely. However, remote functions are side-effect free. That is, they operate on inputs and produce outputs, but they don't change the state of the worker they execute on.

Actors are different. When we instantiate an actor, a brand new worker is created, and all methods that are called on that actor are executed on the newly created worker.

This means that with a single actor, no parallelism can be achieved because calls to the actor's methods will be executed one at a time. However, multiple actors can be created and methods can be executed on them in parallel.

### Concepts for this Exercise - Actors

> To create an actor, decorate Python class with the `@ray.remote` decorator.
>
> ```python
> @ray.remote
> class Example(object):
>     def __init__(self, x):
>         self.x = x
>     
>     def set(self, x):
>         self.x = x
>     
>     def get(self):
>         return self.x
> ```
>
> Like regular Python classes, **actors encapsulate state that is shared across actor method invocations**.
>
> Actor classes differ from regular Python classes in the following ways.
> 1. **Instantiation:** A regular class would be instantiated via `e = Example(1)`. Actors are instantiated via
>     ```python
>     e = Example.remote(1)
>     ```
>     When an actor is instantiated, a **new worker process** is created by a local scheduler somewhere in the cluster.
> 2. **Method Invocation:** Methods of a regular class would be invoked via `e.set(2)` or `e.get()`. Actor methods are invoked differently.
>     ```python
>     >>> e.set.remote(2)
>     ObjectID(d966aa9b6486331dc2257522734a69ff603e5a1c)
>     
>     >>> e.get.remote()
>     ObjectID(7c432c085864ed4c7c18cf112377a608676afbc3)
>     ```
> 3. **Return Values:** Actor methods are non-blocking. They immediately return an object ID and **they create a task which is scheduled on the actor worker**. The result can be retrieved with `ray.get`.
>     ```python
>     >>> ray.get(e.set.remote(2))
>     None
>     
>     >>> ray.get(e.get.remote())
>     2
>     ```

---

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time
```

```python
ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)
```

**EXERCISE:** Change the `Foo` class to be an actor class by using the `@ray.remote` decorator.

```python
@ray.remote
class Foo(object):
    def __init__(self):
        self.counter = 0

    def reset(self):
        self.counter = 0

    def increment(self):
        time.sleep(0.5)
        self.counter += 1
        return self.counter

assert hasattr(Foo, 'remote'), 'You need to turn "Foo" into an actor with @ray.remote.'
```

**EXERCISE:** Change the intantiations below to create two actors by calling `Foo.remote()`.

```python
# Create two Foo objects.
f1 = Foo.remote()
f2 = Foo.remote()
```

**EXERCISE:** Parallelize the code below. The two actors can execute methods in parallel (though each actor can only execute one method at a time).

```python
# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

# Reset the actor state so that we can run this cell multiple times without
# changing the results.
f1.reset.remote()
f2.reset.remote()

# We want to parallelize this code. However, it is not straightforward to
# make "increment" a remote function, because state is shared (the value of
# "self.counter") between subsequent calls to "increment". In this case, it
# makes sense to use actors.
results = []
for _ in range(5):
    results.append(f1.increment.remote())
    results.append(f2.increment.remote())

results = ray.get(results)

end_time = time.time()
duration = end_time - start_time

assert not any([isinstance(result, ray.ObjectID) for result in results]), 'Looks like "results" is {}. You may have forgotten to call ray.get.'.format(results)
```

**VERIFY:** Run some checks to verify that the changes you made to the code were correct. Some of the checks should fail when you initially run the cells. After completing the exercises, the checks should pass.

```python
assert results == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

assert duration < 3, ('The experiments ran in {} seconds. This is too '
                      'slow.'.format(duration))
assert duration > 2.5, ('The experiments ran in {} seconds. This is too '
                        'fast.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))
# Success! The example took 2.5102529525756836 seconds.
```



---

## Exercise 5 - Actor Handles

**GOAL:** The goal of this exercise is to show how to pass around actor handles.

Suppose we wish to have multiple tasks invoke methods on the same actor. For example, we may have a single actor that records logging information from a number of tasks. We can achieve this by passing a handle to the actor as an argument into the relevant tasks.

### Concepts for this Exercise - Actor  Handles

> First of all, suppose we've created an actor as follows.
>
> ```python
> @ray.remote
> class Actor(object):
>     def method(self):
>         pass
> 
> # Create the actor
> actor = Actor.remote()
> ```
>
> Then we can define a remote function (or another actor) that takes an actor handle as an argument.
>
> ```python
> @ray.remote
> def f(actor):
>     # We can invoke methods on the actor.
>     x_id = actor.method.remote()
>     # We can block and get the results.
>     return ray.get(x_id)
> ```
>
> Then we can invoke the remote function a few times and pass in the actor handle.
>
> ```python
> # Each of the three tasks created below will invoke methods on the same actor.
> f.remote(actor)
> f.remote(actor)
> f.remote(actor)
> ```

---

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import ray
import time
```

```python
ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)
```

In this exercise, we're going to write some code that runs several "experiments" in parallel and has each experiment log its results to an actor. The driver script can then periodically pull the results from the logging actor.

**EXERCISE:** Turn this `LoggingActor` class into an actor class.

```python
@ray.remote
class LoggingActor(object):
    def __init__(self):
        self.logs = defaultdict(lambda: [])
    
    def log(self, index, message):
        self.logs[index].append(message)
    
    def get_logs(self):
        return dict(self.logs)


assert hasattr(LoggingActor, 'remote'), ('You need to turn LoggingActor into an '
                                         'actor (by using the ray.remote keyword).')
```

**EXERCISE:** Instantiate the actor.

```python
logging_actor = LoggingActor.remote()

# Some checks to make sure this was done correctly.
assert hasattr(logging_actor, 'get_logs')
```

Now we define a remote function that runs and pushes its logs to the `LoggingActor`.

**EXERCISE:** Modify this function so that it invokes methods correctly on `logging_actor` (you need to change the way you call the `log` method).

```python
@ray.remote
def run_experiment(experiment_index, logging_actor):
    for i in range(60):
        time.sleep(1)
        # Push a logging message to the actor.
        logging_actor.log.remote(experiment_index, 'On iteration {}'.format(i))
```

Now we create several tasks that use the logging actor.

```python
experiment_ids = [run_experiment.remote(i, logging_actor) for i in range(3)]
```

While the experiments are running in the background, the driver process (that is, this Jupyter notebook) can query the actor to read the logs.

**EXERCISE:** Modify the code below to dispatch methods to the `LoggingActor`.

```python
logs = ray.get(logging_actor.get_logs.remote())
print(logs)

assert isinstance(logs, dict), ("Make sure that you dispatch tasks to the "
                                "actor using the .remote keyword and get the results using ray.get.")
#{0: ['On iteration 0'], 
# 1: ['On iteration 0'], 
# 2: ['On iteration 0']}
```

**EXERCISE:** Try running the above box multiple times and see how the results change (while the experiments are still running in the background). You can also try running more of the experiment tasks and see what happens.

```python
#{2: ['On iteration 0', 'On iteration 1'], 
# 0: ['On iteration 0', 'On iteration 1'], 
# 1: ['On iteration 0', 'On iteration 1']}
#{0: ['On iteration 0', 'On iteration 1', 'On iteration 2'], 
# 2: ['On iteration 0', 'On iteration 1', 'On iteration 2'], 
# 1: ['On iteration 0', 'On iteration 1', 'On iteration 2']}
```



---

## Exercise 6 - Handling Slow Tasks

**GOAL:** The goal of this exercise is to show how to use `ray.wait` to avoid waiting for slow tasks.

See the documentation for ray.wait at https://ray.readthedocs.io/en/latest/api.html#ray.wait.

This script starts 6 tasks, each of which takes a random amount of time to complete. We'd like to process the results in two batches (each of size 3). Change the code so that instead of waiting for a fixed set of 3 tasks to finish, we make the first batch consist of the first 3 tasks that complete. The second batch should consist of the 3 remaining tasks. Do this exercise by using `ray.wait`.

### Concepts for this Exercise - ray.wait

> After launching a number of tasks, you may want to know which ones have finished executing. This can be done with `ray.wait`. The function works as follows.
>
> ```python
> ready_ids, remaining_ids = ray.wait(object_ids, num_returns=1, timeout=None)
> ```
>
> **Arguments:**
>
> - `object_ids`: This is a list of object IDs.
> - `num_returns`: This is maximum number of object IDs to wait for. The default value is `1`.
> - `timeout`: This is the maximum amount of time in milliseconds to wait for. So `ray.wait` will block until either `num_returns` objects are ready or until `timeout` milliseconds have passed.
>
> **Return values:**
> - `ready_ids`: This is a list of object IDs that are available in the object store.
> - `remaining_ids`: This is a list of the IDs that were in `object_ids` but are not in `ready_ids`, so the IDs in `ready_ids` and `remaining_ids` together make up all the IDs in `object_ids`.

---

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time
```

```python
ray.init(num_cpus=6, include_webui=False, ignore_reinit_error=True)
```

Define a remote function that takes a variable amount of time to run.

```python
@ray.remote
def f(i):
    np.random.seed(5 + i)
    x = np.random.uniform(0, 4)
    time.sleep(x)
    return i, time.time()
```

**EXERCISE:** Using `ray.wait`, change the code below so that `initial_results` consists of the outputs of the first three tasks to complete instead of the first three tasks that were submitted.

```python
# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

# This launches 6 tasks, each of which takes a random amount of time to
# complete.
result_ids = ray.wait([f.remote(i) for i in range(6)], num_returns=3, timeout=None)
# Get one batch of tasks. Instead of waiting for a fixed subset of tasks, we
# should instead use the first 3 tasks that finish.
# initial_results = ray.get(result_ids[:3])
initial_results = ray.get(result_ids[0])

end_time = time.time()
duration = end_time - start_time
```

**EXERCISE:** Change the code below so that `remaining_results` consists of the outputs of the last three tasks to complete.

```python
# Wait for the remaining tasks to complete.
# remaining_results = ray.get(result_ids[3:])
remaining_results = ray.get(result_ids[1])
```

**VERIFY:** Run some checks to verify that the changes you made to the code were correct. Some of the checks should fail when you initially run the cells. After completing the exercises, the checks should pass.

```python
assert len(initial_results) == 3
assert len(remaining_results) == 3

initial_indices = [result[0] for result in initial_results]
initial_times = [result[1] for result in initial_results]
remaining_indices = [result[0] for result in remaining_results]
remaining_times = [result[1] for result in remaining_results]

assert set(initial_indices + remaining_indices) == set(range(6))

assert duration < 1.5, ('The initial batch of ten tasks was retrieved in '
                        '{} seconds. This is too slow.'.format(duration))

assert duration > 0.8, ('The initial batch of ten tasks was retrieved in '
                        '{} seconds. This is too slow.'.format(duration))

# Make sure the initial results actually completed first.
assert max(initial_times) < min(remaining_times)

print('Success! The example took {} seconds.'.format(duration))
# Success! The example took 0.893179178237915 seconds.
```



---

## Exercise 7 - Process Tasks in Order of Completion

**GOAL:** The goal of this exercise is to show how to use `ray.wait` to process tasks in the order that they finish.

See the documentation for ray.wait at https://ray.readthedocs.io/en/latest/api.html#ray.wait.

The code below runs 10 tasks and retrieves the results in the order that the tasks were launched. However, since each task takes a random amount of time to finish, we could instead process the tasks in the order that they finish.

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time
```

```python
ray.init(num_cpus=5, include_webui=False, ignore_reinit_error=True)
```

```python
@ray.remote
def f():
    time.sleep(np.random.uniform(0, 5))
    return time.time()
```

**EXERCISE:** Change the code below to use `ray.wait` to get the results of the tasks in the order that they complete.

**NOTE:** It would be a simple modification to maintain a pool of 10 experiments and to start a new experiment whenever one finishes.

```python
# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

# result_ids = [f.remote() for _ in range(10)]
temp = ray.wait( [f.remote() for _ in range(10)] , num_returns=1, timeout=None)
result_ids = temp[0]
while temp[1]:
    temp = ray.wait( temp[1] , num_returns=1, timeout=None)
    result_ids.extend(temp[0])

# Get the results.
results = []
for result_id in result_ids:
    result = ray.get(result_id)
    results.append(result)
    print('Processing result which finished after {} seconds.'
          .format(result - start_time))

end_time = time.time()
duration = end_time - start_time
# Processing result which finished after 1.5440089702606201 seconds.
# Processing result which finished after 1.8363125324249268 seconds.
# Processing result which finished after 2.719313144683838 seconds.
# Processing result which finished after 3.2043678760528564 seconds.
# Processing result which finished after 3.8053157329559326 seconds.
# Processing result which finished after 3.9189162254333496 seconds.
# Processing result which finished after 4.422319412231445 seconds.
# Processing result which finished after 5.62132453918457 seconds.
# Processing result which finished after 6.22131085395813 seconds.
# Processing result which finished after 6.867010593414307 seconds.
```

**VERIFY:** Run some checks to verify that the changes you made to the code were correct. Some of the checks should fail when you initially run the cells. After completing the exercises, the checks should pass.

```python
assert results == sorted(results), ('The results were not processed in the '
                                    'order that they finished.')

print('Success! The example took {} seconds.'.format(duration))
# Success! The example took 6.874698162078857 seconds.
```



---

## Exercise 8 - Speed up Serialization

**GOAL:** The goal of this exercise is to illustrate how to speed up serialization by using `ray.put`.

### Concepts for this Exercise - ray.put

> Object IDs can be created in multiple ways.
> - They are returned by remote function calls.
> - They are returned by actor method calls.
> - They are returned by `ray.put`.
>
> When an object is passed to `ray.put`, the object is serialized using the Apache Arrow format (see https://arrow.apache.org/ for more information about Arrow) and copied into a shared memory object store. This object will then be available to other workers on the same machine via shared memory. If it is needed by workers on another machine, it will be shipped under the hood.
>
> **When objects are passed into a remote function, Ray puts them in the object store under the hood.** That is, if `f` is a remote function, the code
>
> ```python
> x = np.zeros(1000)
> f.remote(x)
> ```
>
> is essentially transformed under the hood to
>
> ```python
> x = np.zeros(1000)
> x_id = ray.put(x)
> f.remote(x_id)
> ```
>
> The call to `ray.put` copies the numpy array into the shared-memory object store, from where it can be read by all of the worker processes (without additional copying). However, if you do something like
>
> ```python
> for i in range(10):
>     f.remote(x)
> ```
>
> then 10 copies of the array will be placed into the object store. This takes up more memory in the object store than is necessary, and it also takes time to copy the array into the object store over and over. This can be made more efficient by placing the array in the object store only once as follows.
>
> ```python
> x_id = ray.put(x)
> for i in range(10):
>     f.remote(x_id)
> ```
>
> In this exercise, you will speed up the code below and reduce the memory footprint by calling `ray.put` on the neural net weights before passing them into the remote functions.
>
> **WARNING:** This exercise requires a lot of memory to run. If this notebook is running within a Docker container, then the docker container must be started with a large shared-memory file system. This can be done by starting the docker container with the `--shm-size` flag.

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import ray
import time
```

```python
ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)
```

Define some neural net weights which will be passed into a number of tasks.

```python
# neural_net_weights = {'variable{}'.format(i): np.random.normal(size=1000000)
#                       for i in range(50)}  # 这个好像有误
neural_net_weights = np.random.normal(size=1000000)
```

**EXERCISE:** Compare the time required to serialize the neural net weights and copy them into the object store using Ray versus the time required to pickle and unpickle the weights. The big win should be with the time required for *deserialization*.

Note that when you call `ray.put`, in addition to serializing the object, we are copying it into shared memory where it can be efficiently accessed by other workers on the same machine.

**NOTE:** You don't actually have to do anything here other than run the cell below and read the output.

**NOTE:** Sometimes `ray.put` can be faster than `pickle.dumps`. This is because `ray.put` leverages multiple threads when serializing large objects. Note that this is not possible with `pickle`.

```python
print('Ray - serializing')
%time x_id = ray.put(neural_net_weights)
print('\nRay - deserializing')
%time x_val = ray.get(x_id)

print('\npickle - serializing')
%time serialized = pickle.dumps(neural_net_weights)
print('\npickle - deserializing')
%time deserialized = pickle.loads(serialized)

# Ray - serializing
# CPU times: user 35.9 ms, sys: 47.9 ms, total: 83.7 ms
# Wall time: 61.6 ms

# Ray - deserializing
# CPU times: user 1.07 ms, sys: 0 ns, total: 1.07 ms
# Wall time: 1.04 ms

# pickle - serializing
# CPU times: user 85.8 ms, sys: 103 ms, total: 189 ms
# Wall time: 193 ms

# pickle - deserializing
# CPU times: user 2.25 ms, sys: 0 ns, total: 2.25 ms
# Wall time: 2.28 ms
```

Define a remote function which uses the neural net weights.

```python
@ray.remote
def use_weights(weights, i):
    return i
```

**EXERCISE:** In the code below, use `ray.put` to avoid copying the neural net weights to the object store multiple times.

```python
# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)

start_time = time.time()

temp = ray.put(neural_net_weights)
results = ray.get([use_weights.remote(temp, i) for i in range(20)])

end_time = time.time()
duration = end_time - start_time
```

**VERIFY:** Run some checks to verify that the changes you made to the code were correct. Some of the checks should fail when you initially run the cells. After completing the exercises, the checks should pass.

```python
assert results == list(range(20))
assert duration < 1, ('The experiments ran in {} seconds. This is too '
                      'slow.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))
# Success! The example took 0.10664176940917969 seconds.
```



---

## Exercise 9 - Using the GPU API

**GOAL:** The goal of this exercise is to show how to use GPUs with remote functions and actors.

**NOTE:** These exercises are designed to run on a machine without GPUs.

See the documentation on using Ray with GPUs http://ray.readthedocs.io/en/latest/using-ray-with-gpus.html.

### Concepts for this Exercise - Using Ray with GPUs

> We can indicate that a remote function or an actor requires some GPUs using the `num_gpus` keyword.
>
> ```python
> @ray.remote(num_gpus=1)
> def f():
>     # The command ray.get_gpu_ids() returns a list of the indices
>     # of the GPUs that this task can use (e.g., [0] or [1]).
>     ray.get_gpu_ids()
> 
> @ray.remote(num_gpus=2)
> class Foo(object):
>     def __init__(self):
>         # The command ray.get_gpu_ids() returns a list of the
>         # indices of the GPUs that this actor can use
>         # (e.g., [0, 1] or [3, 5]).
>         ray.get_gpu_ids()
> ```
>
> Then inside of the actor constructor and methods, we can get the IDs of the GPUs allocated for that actor with `ray.get_gpu_ids()`.

---

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time
```

Start Ray, note that we pass in `num_gpus=4`. Ray will assume this machine has 4 GPUs (even if it does not). When a task or actor requests a GPU, it will be assigned a GPU ID from the set `[0, 1, 2, 3]`. It is then the responsibility of the task or actor to make sure that it only uses that specific GPU (e.g., by setting the `CUDA_VISIBLE_DEVICES` environment variable).

```python
ray.init(num_cpus=4, num_gpus=2, include_webui=False, ignore_reinit_error=True)
```

**EXERCISE:** Change the remote function below to require one GPU.

**NOTE:** This change does not make the remote function actually **use** the GPU, it simply **reserves** the GPU for use by the remote function. To actually use the GPU, the remote function would use a neural net library like TensorFlow or PyTorch after setting the `CUDA_VISIBLE_DEVICES` environment variable properly. This can be done as follows.

> ```python
> import os
> os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in ray.get_gpu_ids()])
> ```

```python
@ray.remote(num_gpus = 1)
def f():
    time.sleep(0.5)
    return ray.get_gpu_ids()
```

**VERIFY:** This code checks that each task was assigned one GPU and that not more than two tasks are run at the same time (because we told Ray there are only two GPUs).

```python
start_time = time.time()

gpu_ids = ray.get([f.remote() for _ in range(3)])  # [[1], [0], [0]]

end_time = time.time()

for i in range(len(gpu_ids)):
    assert len(gpu_ids[i]) == 1

assert end_time - start_time > 1

print('Sucess! The test passed.')
# Sucess! The test passed.
```

**EXERCISE:** The code below defines an actor. Make it require one GPU.

```python
@ray.remote(num_gpus = 1)
class Actor(object):
    def __init__(self):
        pass

    def get_gpu_ids(self):
        return ray.get_gpu_ids()
```

**VERIFY:** This code checks that the actor was assigned a GPU.

```python
actor = Actor.remote()

gpu_ids = ray.get(actor.get_gpu_ids.remote()) # [0]

assert len(gpu_ids) == 1

print('Sucess! The test passed.')
# Sucess! The test passed.
```



---

## Exercise 10 - Custom Resources

**GOAL:** The goal of this exercise is to show how to use custom resources

See the documentation on using Ray with custom resources http://ray.readthedocs.io/en/latest/resources.html#custom-resources.

### Concepts for this Exercise - Using Custom Resources

> We've discussed how to specify a task's CPU and GPU requirements, but there are many other kinds of resources. For example, a task may require a dataset, which only lives on a few machines, or it may need to be scheduled on a machine with extra memory. These kinds of requirements can be expressed through the use of custom resources.
>
> Custom resources are most useful in the multi-machine setting. However, this exercise illustrates their usage in the single-machine setting.
>
> Ray can be started with a dictionary of custom resources (mapping resource name to resource quantity) as follows.
>
> ```python
> ray.init(resources={'CustomResource1': 1, 'CustomResource2': 4})
> ```
>
> The resource requirements of a remote function or actor can be specified in a similar way.
>
> ```python
> @ray.remote(resources={'CustomResource2': 1})
> def f():
>     return 1
> ```
>
> Even if there are many CPUs on the machine, only 4 copies of `f` can be executed concurrently.
>
> Custom resources give applications a great deal of flexibility. For example, if you wish to control precisely which machine a task gets scheduled on, you can simply start each machine with a different custom resource (e.g., start machine `n` with resource `Custom_n` and then tasks that should be scheduled on machine `n` can require resource `Custom_n`. However, this usage has drawbacks because it makes the code less portable and less resilient to machine failures.

---

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import time
```

In this exercise, we will start Ray using custom resources.

```python
ray.init(num_cpus=8, resources={'Custom1': 4}, include_webui=False, ignore_reinit_error=True)
```

**EXERCISE:** Modify the resource requirements of the remote functions below so that the following hold.
- The number of concurrently executing tasks is at most 8 (note that there are 8 CPUs).
- No more than 4 copies of `g` can execute concurrently.
- If 4 `g` tasks are executing, then an additional 4 `f` tasks can execute.

You should only need to use the `Custom1` resource.

```python
@ray.remote
def f():
    time.sleep(0.1)

@ray.remote(resources={'Custom1': 1})
def g():
    time.sleep(0.1)
```

If you did the above exercise correctly, the next cell should execute without raising an exception.

```python
start = time.time()
ray.get([f.remote() for _ in range(8)])
duration = time.time() - start 
assert duration >= 0.1 and duration < 0.19, '8 f tasks should be able to execute concurrently.'

start = time.time()
ray.get([f.remote() for _ in range(9)])
duration = time.time() - start 
assert duration >= 0.2 and duration < 0.29, 'f tasks should not be able to execute concurrently.'

start = time.time()
ray.get([g.remote() for _ in range(4)])
duration = time.time() - start 
assert duration >= 0.1 and duration < 0.19, '4 g tasks should be able to execute concurrently.'

start = time.time()
ray.get([g.remote() for _ in range(5)])
duration = time.time() - start 
assert duration >= 0.2 and duration < 0.29, '5 g tasks should not be able to execute concurrently.'

start = time.time()
ray.get([f.remote() for _ in range(4)] + [g.remote() for _ in range(4)])
duration = time.time() - start 
assert duration >= 0.1 and duration < 0.19, '4 f and 4 g tasks should be able to execute concurrently.'

start = time.time()
ray.get([f.remote() for _ in range(5)] + [g.remote() for _ in range(4)])
duration = time.time() - start 
assert duration >= 0.2 and duration < 0.29, '5 f and 4 g tasks should not be able to execute concurrently.'

print('Success!')
# Success!
```



---

## Exercise 11 - Pass Neural Net Weights Between Processes

**GOAL:** The goal of this exercise is to show how to send neural network weights between workers and the driver.

For more details on using Ray with TensorFlow, see the documentation at http://ray.readthedocs.io/en/latest/using-ray-with-tensorflow.html.

### Concepts for this Exercise - Getting and Setting Neural Net Weights

> Since pickling and unpickling a TensorFlow graph can be inefficient or may not work at all, it is most efficient to ship the weights between processes as a dictionary of numpy arrays (or as a flattened numpy array).
>
> We provide the helper class `ray.experimental.TensorFlowVariables` to help with getting and setting weights. Similar techniques should work other neural net libraries.
>
> Consider the following neural net definition.
>
> ```python
> import tensorflow as tf
> 
> x_data = tf.placeholder(tf.float32, shape=[100])
> y_data = tf.placeholder(tf.float32, shape=[100])
> 
> w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
> b = tf.Variable(tf.zeros([1]))
> y = w * x_data + b
> 
> loss = tf.reduce_mean(tf.square(y - y_data))
> optimizer = tf.train.GradientDescentOptimizer(0.5)
> grads = optimizer.compute_gradients(loss)
> train = optimizer.apply_gradients(grads)
> 
> init = tf.global_variables_initializer()
> sess = tf.Session()
> sess.run(init)
> ```
>
> Then we can use the helper class as follows.
>
> ```python
> variables = ray.experimental.TensorFlowVariables(loss, sess)
> # Here 'weights' is a dictionary mapping variable names to the associated
> # weights as a numpy array.
> weights = variables.get_weights()
> variables.set_weights(weights)
> ```
>
> Note that there are analogous methods `variables.get_flat` and `variables.set_flat`, which concatenate the weights as a single array instead of a dictionary.
>
> ```python
> # Here 'weights' is a numpy array of all of the neural net weights
> # concatenated together.
> weights = variables.get_flat()
> variables.set_flat(weights)
> ```
>
> In this exercise, we will use an actor containing a neural network and implement methods to extract and set the neural net weights.
>
> **WARNING:** This exercise is more complex than previous exercises.

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import tensorflow as tf
import time
```

```python
ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)
```

The code below defines a class containing a simple neural network.

**EXERCISE:** Implement the `set_weights` and `get_weights` methods. This should be done using the `ray.experimental.TensorFlowVariables` helper class.

```python
@ray.remote
class SimpleModel(object):
    def __init__(self):
        x_data = tf.placeholder(tf.float32, shape=[100])
        y_data = tf.placeholder(tf.float32, shape=[100])

        w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        b = tf.Variable(tf.zeros([1]))
        y = w * x_data + b

        self.loss = tf.reduce_mean(tf.square(y - y_data))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        grads = optimizer.compute_gradients(self.loss)
        self.train = optimizer.apply_gradients(grads)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()

        # Here we create the TensorFlowVariables object to assist with getting
        # and setting weights.
        self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)

        self.sess.run(init)

    def set_weights(self, weights):
        """Set the neural net weights.
        
        This method should assign the given weights to the neural net.
        
        Args:
            weights: Either a dict mapping strings (the variable names) to numpy
                arrays or a single flattened numpy array containing all of the
                concatenated weights.
        """
        # EXERCISE: You will want to use self.variables here.
        self.variables.set_weights(weights)
#         raise NotImplementedError

    def get_weights(self):
        """Get the neural net weights.
        
        This method should return the current neural net weights.
        
        Returns:
            Either a dict mapping strings (the variable names) to numpy arrays or
                a single flattened numpy array containing all of the concatenated
                weights.
        """
        # EXERCISE: You will want to use self.variables here.
        return self.variables.get_weights()
#         raise NotImplementedError
```

Create a few actors.

```python
actors = [SimpleModel.remote() for _ in range(4)]
```

**EXERCISE:** Get the neural net weights from all of the actors.

```python
# raise Exception('Implement this.')
ray.get([actor.get_weights.remote() for actor in actors])

# [{'Variable': array([-0.6429639], dtype=float32),
#   'Variable_1': array([0.], dtype=float32)},
#  {'Variable': array([-0.789418], dtype=float32),
#   'Variable_1': array([0.], dtype=float32)},
#  {'Variable': array([-0.49405766], dtype=float32),
#   'Variable_1': array([0.], dtype=float32)},
#  {'Variable': array([0.7267263], dtype=float32),
#   'Variable_1': array([0.], dtype=float32)}]
```

**EXERCISE:** Average all of the neural net weights.

**NOTE:** This will be easier to do if you chose to use `get_flat`/`set_flat` instead of `get_weights`/`set_weights` in the implementation of `SimpleModel.set_weights` and `SimpleModel.get_weights` above..

```python
# raise Exception('Implement this.')
Variable_mean = np.mean( [ actor_dict['Variable'] for actor_dict in ray.get([actor.get_weights.remote() for actor in actors]) ] )
Variable_1_mean = np.mean( [ actor_dict['Variable_1'] for actor_dict in ray.get([actor.get_weights.remote() for actor in actors]) ] )
average_weights = [{ 'Variable': np.array([Variable_mean])
                    ,'Variable_1': np.array([Variable_1_mean]) } for _ in range(4) ]

average_weights
# [{'Variable': array([-0.2999283], dtype=float32),
#   'Variable_1': array([0.], dtype=float32)},
#  {'Variable': array([-0.2999283], dtype=float32),
#   'Variable_1': array([0.], dtype=float32)},
#  {'Variable': array([-0.2999283], dtype=float32),
#   'Variable_1': array([0.], dtype=float32)},
#  {'Variable': array([-0.2999283], dtype=float32),
#   'Variable_1': array([0.], dtype=float32)}]
```

**EXERCISE:** Set the average weights on the actors.

```python
# raise Exception('Implement this.')
[actor.set_weights.remote(average_weight) for actor, average_weight in zip(actors, average_weights) ];
```

**VERIFY:** Check that all of the actors have the same weights.

```python
weights = ray.get([actor.get_weights.remote() for actor in actors])

for i in range(len(weights)):
    np.testing.assert_equal(weights[i], weights[0])

print('Success! The test passed.')
# Success! The test passed.
```



---

## Exercise 12 - Tree Reduce

**GOAL:** The goal of this exercise is to show how to implement a tree reduce in Ray by passing object IDs into remote functions to encode dependencies between tasks.

In this exercise, you will use Ray to implement parallel data generation and a parallel tree reduction.

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time
```

```python
ray.init(num_cpus=8, include_webui=False, ignore_reinit_error=True)
```

**EXERCISE:** These functions will need to be turned into remote functions so that the tree of tasks can be executed in parallel.

```python
# This is a proxy for a function which generates some data.
@ray.remote
def create_data(i):
    time.sleep(0.3)
    return i * np.ones(10000)

# This is a proxy for an expensive aggregation step (which is also
# commutative and associative so it can be used in a tree-reduce).
@ray.remote
def aggregate_data(x, y):
    time.sleep(0.3)
    return x * y
```

**EXERCISE:** Make the data creation tasks run in parallel. Also aggregate the vectors in parallel. Note that the `aggregate_data` function must be called 7 times. They cannot all run in parallel because some depend on the outputs of others. However, it is possible to first run 4 in parallel, then 2 in parallel, and then 1.

```python
# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(1.0)
start_time = time.time()

# EXERCISE: Here we generate some data. Do this part in parallel.
vectors = [create_data.remote(i + 1) for i in range(8)]

# Here we aggregate all of the data repeatedly calling aggregate_data. This
# can be sped up using Ray.
#
# NOTE: A direct translation of the code below to use Ray will not result in
# a speedup because each function call uses the output of the previous function
# call so the function calls must be executed serially.
#
# EXERCISE: Speed up the aggregation below by using Ray. Note that this will
# require restructuring the code to expose more parallelism. First run 4 tasks
# aggregating the 8 values in pairs. Then run 2 tasks aggregating the resulting
# 4 intermediate values in pairs. then run 1 task aggregating the two resulting
# values. Lastly, you will need to call ray.get to retrieve the final result.
#
# Exposing more parallelism means aggregating the vectors in a DIFFERENT ORDER.
# This can be done because we are simply summing the data and the order in
# which the values are summed doesn't matter (it's commutative and associative).
# result = aggregate_data(vectors[0], vectors[1])
# result = aggregate_data(result, vectors[2])
# result = aggregate_data(result, vectors[3])
# result = aggregate_data(result, vectors[4])
# result = aggregate_data(result, vectors[5])
# result = aggregate_data(result, vectors[6])
# result = aggregate_data(result, vectors[7])

while len(vectors) > 1:
    vectors.append(aggregate_data.remote(vectors.pop(0), vectors.pop(0)) )# + vectors[2:]
result = ray.get(vectors[0])

# NOTE: For clarity, the aggregation above is written out as 7 separate function
# calls, but this can be done more easily in a while loop via
#
#     while len(vectors) > 1:
#         vectors = aggregate_data(vectors[0], vectors[1]) + vectors[2:]
#     result = vectors[0]
#
# When expressed this way, the change from serial aggregation to tree-structured
# aggregation can be made simply by appending the result of aggregate_data to the
# end of the vectors list as opposed to the beginning.
#
# EXERCISE: Think about why this is true.

end_time = time.time()
duration = end_time - start_time
```

**EXERCISE:** Use the UI to view the task timeline and to verify that the vectors were aggregated with a tree of tasks.

You should be able to see the 8 `create_data` tasks running in parallel followed by 4 `aggregate_data` tasks running in parallel followed by 2 more `aggregate_data` tasks followed by 1 more `aggregate_data` task.

In the timeline, click on **View Options** and select **Flow Events** to visualize tasks dependencies.

```python
import ray.experimental.ui as ui
ui.task_timeline()
```

![](https://i.loli.net/2018/12/22/5c1e536115fb2.png)

VERIFY:** Run some checks to verify that the changes you made to the code were correct. Some of the checks should fail when you initially run the cells. After completing the exercises, the checks should pass.

```python
assert np.all(result == 40320 * np.ones(10000)), ('Did you remember to '
                                                  'call ray.get?')
assert duration < 0.3 + 0.9 + 0.3, ('FAILURE: The data generation and '
                                    'aggregation took {} seconds. This is '
                                    'too slow'.format(duration))
assert duration > 0.3 + 0.9, ('FAILURE: The data generation and '
                              'aggregation took {} seconds. This is '
                              'too fast'.format(duration))

print('Success! The example took {} seconds.'.format(duration))
# Success! The example took 1.2151989936828613 seconds.
```





（END）

---

[返回到首页](../index.html) | [返回到顶部](./Ray_Tutorial.html)


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





