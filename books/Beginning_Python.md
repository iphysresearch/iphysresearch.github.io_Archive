[返回到首页](../index.html)

---

[TOC]



# Python 基础教程（第3版）

> - 该笔记是个人在学习和阅读《**Beginning Python** From Novice to Professional <u>Third Edition</u>》时候，所作的私人笔记。不保证覆盖书籍所有知识点和内容，仅摘录个人觉得重要和有趣的信息，以及相关资料的搜集和讨论。相当于是原书上做的批注笔记。
> - 以下代码都可以在 `Python 3.6` 环境下顺利运行。





---

## 第6章 抽象



- 抽象就是要节省人力，追求的是一种“懒惰”的美德。
- 抽象的关键作用是是要程序能够被人理解。也就是说，掩盖操作细节，只需要组织必要的逻辑。



### 自定义函数

> 函数是结构化编程的核心

```python
def test(num):
    '斐波那契数列' # 函数文档
    result = [0, 1]
    for i in range(num-2):
        result.append(result[-2] + result[-1])
	return result
	print('Finished!')	# 不会执行
```

- 函数的内部变量 (`num`, `result`) 可以使用任何名字。
- 内置函数 `callable ` （判断一个对象是否可调用）
- 查看函数的文档可以用 `test.__doc__` 或者 `help(test)`
- `return` 只是为了结束函数，可以不跟任何变量，至少会返回 `None`
- `return` 之后的代码行将不会执行。



### 函数的参数怎么修改

- 函数内的局部名称（包括参数）不会与函数外的名称（全局名称）冲突。

- 函数的参数（'形参'）对于**不可变的数据结构**，如**字符串、数和元组**等，函数内部重新关联参数（赋值），函数外部的变量不受影响；但是**可变的数据结构**，如**列表**等，就非常不同了！

  ```python
  def try_to_change(s,n,t,l1,l2):
      s = 'IPhysResearch'
      n = 2018
      t = ('changed', 'changed')
      l1 = ['changed','changed']
      l2[0] = 'changed'
      
  name = 'BLOG'
  year = 2017
  tuple_ = ('unchanged', 'unchanged')
  list1_ = ['unchanged', 'unchanged']
  list2_ = ['unchanged', 'unchanged']
  
  try_to_change(name, year, tuple_, list1_, list2_)
  print(name)
  print(year)
  print(tuple_)
  print(list1_)
  print(list2_)
  
  # 打印结果
  # BLOG
  # 2017
  # ('unchanged', 'unchanged')
  # ['unchanged', 'unchanged']
  # ['changed', 'unchanged']    # 只有这列表的第一个元素改变了值！
  ```

  函数内对可变化参数的关联赋值，其实等价于同一个列表赋值给了两个变量。

  ```python
  def try_to_change(l2):
      l2[0] = 'changed'
  list2_ = ['unchanged', 'unchanged']
  try_to_change(list2_)
  
  # 上述操作等价于：
  
  l2 = list2_
  l2[0] = 'changed'
  ```

  若想不影响原变量的值，需要创建列表的**副本**，从而使得给出两个**相等**但**不同**的列表：

  ```python
  list2_ = ['unchanged', 'unchanged']
  l2 = list2_[:]		# 也可等价的写 l2 = list2_.copy()
  print(l2 is list2_)
  print(l2 == list2_)
  l2[0] = 'changed'
  print(l2)
  print(list2_)
  
  # 打印结果
  # False
  # True
  # ['changed', 'unchanged']
  # ['unchanged', 'unchanged']
  ```

- 对于函数内参数是不可变的数据结构，如**字符串、数和元组**等，就需要用 `return` 返回所需要的值了，如下：

  ```python
  def inc(x): x+1
  foo = 10
  foo = inc(foo)	# 将输出赋值到原输入参数变量，不然全局的 foo 不会有任何变化。
  foo		# 11
  ```

  

### 关键字参数 + 默认值

- 可以结合使用位置参数和关键字参数，但**必须先指定**所有的**位置参数**。

  ```python
  def Hello_4(name, greeting='Hello', punctuation='!')
  	print('{},{}{}'.format(greeting, name, punctuation))
  # 函数内的 name 参数是必须指定的，其他都是可选的。
  ```

- 一般而言，除非必不可少的参数很少，而带默认值的可选参数很多，否则不应结合使用关键字参数和位置参数。



### 收集参数

- 带一个星号的参数变量会收集参数为一个元组。（不同于序列接包是，带星号的变量收集的是一个列表）

- 无可供收集的带星号参数时，其将返回一个空元组。

    ```python
    def print_params_2(title, *params):
        print(title, params)
    
    print_params_2('Params:', 1, 2 ,3)
    # 打印结果
    # Params: (1, 2, 3)
    
    print_params_2('Nothing:')
    # 打印结果
    # Nothing: ()
    ```

- 带两个星号的参数收集关键字参数，并且其得到的是一个字典。无可提供，将返回空字典。

    ```python
    def print_params_2(x, y, z = 3, *pospar, **keypar):
        print(x, y, z, pospar, keypar)
        
    print_params_2(3, 2, 1, 5, 6, 7, foo=1, bar=2)
    # 打印结果
    # 3 2 1 (5, 6, 7) {'foo': 1, 'bar': 2}
    ```



### 分配参数

- 简单说，意思就是在调用函数时，使用带星号的参数在调用函数中。

- 使用拆分运算符来传递参数（又如调用超类的构造函数时）很有用，因为这样就无需操心参数个数之类的问题，如下所示：

  ```python
  def foo(x, y, z, m=0, n=0):
      print(x, y, z, m, n)
  def call_foo(*args, **kwds):
      print("Calling foo!")
      foo(*args, **kwds)
      
  call_foo(4,3,2,m=1)
  # 打印结果：
  # Calling foo!
  # 4 3 2 1 0
  ```



### 作用域

- 变量究竟是什么？其实就是一个指向“值”的名称。相当于字典中“键值对”里的“键”，这个“字典”可以用函数 `vars` 来调用。（注：不要修改 `vars` 返回的字典！）

- 除全局作用域外，每个函数调用都会创建一个新作用域（或叫命名空间）。

- 函数中可以随性读取全局变量的值，**只要不是重新关联它（赋值）**，但要务必慎用全局变量！

- 函数中的局部变量会**遮盖**住同名的全局变量。如需要，可以用函数 `globals` 来读取全局变量的值。

- 函数中若想对某全局变量**重新关联（使其指向新值）**，就需要 `global` 声明出来。

  ```python
  def combine(parameter):
      print(parameter + globals()['parameter'])
  def change_global():
      global x
      x += 1
  
  parameter = 'berry'
  combine('Shrub')
  x = 1
  change_global()
  # 打印结果
  # Shrubbery
  # 2
  ```

- 作用域嵌套（略）



### 递归（略）







## 第7章 再谈抽象



### 对象魔法

> **对象**意味着一系列数据（属性）以及一套访问和操作这些数据的**方法**。

面向对象编程的好处多多：

- **多态**(polymorphism)：可对不同类型的对象执行相同的操作。

  - 与属性相关联的**函数**称为**方法**。
  - 说白了，多态体现在各种类型的变量(对象)或“值”上可以直接调用的相同函数。（也称为**鸭子类型**）
  - 函数 `repo` 是多态的集大成者之一，可用于任何对象。其返回指定值的字符串表示。
  - 应尽量避免使用函数显示地“类型检查”来破坏多态。若考虑**抽象基类**和模块 `abs` 后，可另当别论。

- **封装**(encapsulation)：对外部隐藏有关工作原理的细节。

  - 说白了，封装主要聊的是**属性**的封装，和方法一样，其归属于对象的变量，对象是通过调用类创建的。

  - 每个对象都有自己的**状态**，即使都由相同的类中创建而来。对象的状态由其属性（如名称）描述。对象的方法可能修改这些属性，因此对象将一系列函数（方法）组合起来，并赋予它们访问一些变量（属性）的群贤，而属性可用于在两次函数调用之间存储值。

    ```python
    >>> c = ClosedObject()  # 创建了对象 c (实例 c)
    >>> c.set_name('Sir Lancelot')   # 调用对象 c 的方法 set_name
    >>> c.get_name()		# 调用对象 c 的方法 get_name
    'Sir Lancelot'			# 输出了属性(对象的状态之一)
    >>> r = ColsedObject()
    >>> r.set_name('Sir Robin')
    >>> r.get_name()
    'Sir Robin'
    >>> c.get_name()		# 对象 c 和对象 r 各自有各自的状态(属性)
    'Sir Lancelot'
    ```

- **继承**：可基于通用类创造出专用类。

  - 说白了，在创建新类（子类）时可以继承（调用）其他类（超类）中的方法等。



### 类

- 每个对象都**属于**特定的类，并被称为该类的**实例**。
- 类的所有实例都有该类的所有方法，因此**子类**的所有实例都有**超类**的所有方法。
- 在 Python 中，类名称约定使用单数并将首字母大写，如 Bird 和 Lark。
- 在类里，方法中的参数 `self` 总是指向对象本身，像一个“**通配的对象**”一样，关联到它所属的实例。

```python
class Person:
    def set_name(self, name):
        self.name = name  			# 属性 self.name
	def get_name(self):
        return self.name
def function():
    print('I don"t have a "self"!')

>>> foo = Person()					# 创建对象 foo
>>> bar = Person()					# 创建对象 bar
>>> foo.set_name('Luke Skywalker')	# 关联属性 foo.name 的值为 'Luke Skywalker'
>>> bar.set_name('Anakin Skywalker')# 关联属性 bar.name 的值为 'Anakin Skywalker'
>>> foo.get_name()
'Luke Skywalker'
>>> bar.get_name()
'Anakin Skywalker'
>>> bar.get_name = function			# 可以将属性关联到普通的函数上，此时没有特殊的 self 参数
>>> bar.get_name()
"I don't have a 'self'!"
```

- 要让**方法或属性**成为**私有的**（不能从外部访问），只需让其名称以两个下划线打头即可。但仍可从类外访问私有方法，即幕后其实是修改了其名称。

- 如果想名称不被修改，也想告诫别人不要从类外发出访问请求，让其名称以一个下划线打头即可。虽是约定，但还是有作用的，如：`from module import *` 就不会导入以一个下划线打头的名称。

- 类的命名空间范围内所定义的变量，所有的实例都可像方法一样访问它！

  ```python
  class MemberCounter():
      members = 0
      member_s = 0
      def init(self):
          MemberCounter.members += 1
          self.member_s += 1
  
  m1 = MemberCounter()
  print(MemberCounter.members, MemberCounter.member_s, m1.members, m1.member_s)
  m1.init()
  print(MemberCounter.members, MemberCounter.member_s, m1.members, m1.member_s)
  print(MemberCounter.members is m1.members)	# 相同的两个变量
  
  m2 = MemberCounter()
  print(MemberCounter.members, MemberCounter.member_s, m2.members, m2.member_s)
  m2.init()
  print(MemberCounter.members, MemberCounter.member_s, m2.members, m2.member_s)
  print(MemberCounter.members is m2.members)	# 相同的两个变量
  
  # 打印结果
  # 0 0 0 0
  # 1 0 1 1
  # True
  # 1 0 1 0
  # 2 0 2 1
  # True
  ```

  虽然 `members` 和 `members_s` 都是在类作用域内定义的变量，但是经过不同初始化操作后的表现大有不同。对于 `members` 来说，初始化都是针对其相对于类 `MemberCounter` 的属性操作，所以根据第一列和第三列的表现看来，不论是 `MemberCounter.members` 还是在实例上对此属性的访问表现都是一致的，而且是**相同**且**相等**的；对于 `members_s` 来说，初始化时就关联到了新的相对于**实例**的属性变量 `self.member_s`，所以根据第二列和第四列的表现看来，初始化并没有影响类的 `MemberCounter.member_s` 变量，而`self.member_s` 是依赖对象实例的。

  在类外给属性（不论是类的，还是实例的）赋值的时候，要留意：

  ```python
  # 对类的属性赋值
  MemberCounter.members = 'Two'
  print(m1.members, m2.members)	# 将会把定义在类上的属性(所有实例里)都会被赋值
  MemberCounter.member_s = 'One'
  print(m1.member_s, m2.member_s)	# 关联在实例上的属性不会被影响
  
  # 打印结果
  # Two Two
  # 1 1
  ```

  ```python
  # 对实例的属性赋值
  m1.members = 'Two'				# 在实例上对类的属性赋值后，
  print(m1.members, m2.members)	# 将会遮住类级变量，新值被写入到相应实例上。
  m2.member_s = 'One'
  print(m1.member_s, m2.member_s)	# 新值被写入到相应实例上。
  
  # 打印结果
  # Two 2
  # 1 One
  ```

- 关于超类的继承，很有趣的一点就是属性可以在连接起来的类定义内不受限的访问。

  ```python
  class Filter:
      def init(self):
          self.blocked = []
      def filter(self, sequence):
          self.filtered = [x for x in sequence if x not in self.blocked]
          print(self.filtered)
      
  class SPAMFilter(Filter):	# SPAMFilter 是 Filter 的子类
      def init(self):			# 重写超类 Filter 的方法 init
          self.blocked = ['SPAM']
      def count_num(self):
          print(len(self.filtered))	# 访问了类 Filter 内的属性
          
  class SPAMFilter_count(SPAMFilter,Filter):	
      # 多重继承：反转超类的排序会出错，因为它们中相同的函数 init 是有子类继承顺序的。
      pass
  
  s = SPAMFilter_count()
  s.init()
  s.filter(['SPAM', 'SPAM', 'eggs', 'SPAM', 'bacon'])
  s.count_num()
  
  # 打印结果
  # ['eggs', 'bacon']
  # 2
  ```

- 用 Python 内置的方法 `issubclass` 考察一个类是否是另一个类的子类；

- 访问类内的特殊属性 `__bases__` 可得知其基类；

- 用函数 `isinstance` 来确定对象是否是特定类的实例（通常不用，会影响多态）；也可以通过访问对象的属性 `__class__` 得知对象属于哪个类。

- 在 Python 中，不显式地指定对象必须包含哪些方法才能用作参数。所以检查某对象所需的方法是否存在，可用函数 `hasattr`，`getattr` 与 `callable` 。也可检查其 `__dict__` 属性，得到对象中储存的所有值。更多详情已略。



### 抽象基类（略）

- 抽象基类用于指定子类必须提供哪些功能，却不实现这些功能。



### 关于面向对象设计的一些思考

- Tips：

  - 将相关的东西放在一起。如果一个函数操作一个**全局变**量，最好将它们**作为一个类的属性和方法**。
  - 不要让对象之间过于亲密。**方法应只关心其所属实例的属性**，对于其他实例的状态，让它们自己去管理就好了。
  - **慎用继承**，尤其是多重继承。继承有时很有用，但在有些情况下可能带来不必要的复杂性。要正确地使用多重继承很难，要排除其中的 bug 更难。
  - 保持简单。**让方法短小紧凑**。一般而言，应确保大多数方法都能在30秒内读完并理解。对于其余的方法，尽可能将其篇幅控制在一页或一屏内。

- 确定需要哪些类和相应的方法时：

  1. 将有关问题的描述（程序需要做什么）记录下来，并给所有的名词、动词和形容词加上标记。
  2. 在名词中找出可能的类。
  3. 在动词中找出可能的方法。
  4. 在形容词中找出可能的属性。
  5. 将找出的方法和属性分配给各个类。

- 有了**面向对象模型**的草图后，还需考虑类和对象之间的关系（如继承或协作）以及他们的职责。

  进一步改进模型的办法：

  1. 记录（或设想）一系列**用例**，即使用程序的场景，并尽力确保这些用例涵盖了所有的功能。
  2. 透彻而仔细地考虑每个场景，确保模型包含了所需的一切。如果有遗漏，就加上；如果有不太对的地方，就修改。不断地重复这个过程，直到对模型满意为止。

  有了你认为行之有效的模型后，就可以着手编写程序了。







## 第8章 异常



### 让事情沿着你指定的轨道出错！

- 使用 `raise` 语句来引发异常，并将一个类（必须是`Exception` 的子类或实例作为参数）。

  如：`raise Exception('hyperdrive overload')`

- 最重要的几个内置的异常类

|                类名 | 描述                                                         |
| ------------------: | :----------------------------------------------------------- |
|         `Exception` | 几乎所有的异常类都是从它派生而来的                           |
|    `AttributeError` | 引用属性或给它赋值失败时引发                                 |
|           `OSError` | 操作系统不能执行指定的任务（如打开文件）时引发，有多个子类   |
|        `IndexError` | 使用序列中不存在的索引时引发，为 `LookupError` 的子类        |
|          `KeyError` | 使用映射中不存在的键时引发，为 `LookupError` 的子类          |
|         `NameError` | 找不到名称（变量）时引发                                     |
|       `SyntaxError` | 代码不正确时引发                                             |
|         `TypeError` | 将内置操作或函数用于类型不正确的对象时引发                   |
|        `ValueError` | 将内置操作或函数用于这样的对象时引发：其类型正确但包含的值不合适 |
| `ZeroDivisionError` | 在出发或求模运算的第二个参数为零时引发                       |

- 也可以自定义异常类，但务必直接或间接地继承 `Exception`。

  如：

  ```python
  calss SomeCustomException(Exception): pass
  ```



### 捕获异常

- 异常从函数向外传播到调用函数的地方，异常会向程序的最顶层传播。

- 如果想“抑制”异常的传播的话，`try/except` 语句在异常时是打印错误消息即可。

- 若想异常后继续向上传播，就可以在异常时执行不带参数的 `raise` 即可。

- 带其他异常类参数的 `raise` 是将进入 `except` 子句的异常作为**上下文**显示出来。如果想禁用上下文，使用 `raise ... from …` 语句和用 `None` 即可。

  ```python
  try:
      x = int(input('Enter the first number: '))
      y = int(input('Enter the second number: '))
      print(x/y)
  except (TypeError, NameError) as e:		# 同时捕获两种异常，不向上传播，但会打印异常 e
      print('Your numbers were bogus...')
      print(e)
  except ZeroDivisionError: 				# 捕获 ZeroDivisionError 异常
      print("Division by zero is illegal!")
      raise ValueError from None			# 仅显示 ValueError 异常
      print("You can't see me!")			# 被忽略了
  ```

- 一网打尽的捕获所有异常（`except` 语句不带任何参数）是非常危险的。与之相比，更好的选择是：`except Exception as e` 并对异常(e)进行检查。

- 搭配 `else` 语句会很帅，类似于条件语句和循环一样。

  ```python
  while True:
      try:
          x = int(input('Enter the first number: '))
          y = int(input('Enter the second number: '))
  		print('x/y is', x/y)
  	except Exception as e:
          print('Invalid input:', e)
          print('Please try again!')
  	else:							# 当 try 的子句里没有异常才执行下面
          break
  ```

- 另有，`finally` 语句，暂略。。。

- 在函数内使用异常处理很好用，可以避免异常被不断的向上传播到主程序（全局作用域）上被处理。

- 很多时候，相比于使用 `if/else`，使用 `try/except` 语句更自然，也更符合 Python 的习惯。

- 警告类似于异常，但（通常）只打印一条错误信息。它们都在模块 `warnings` 的子类中。





## 第9章 魔法方法、特性和迭代器



### 构造函数

- 所谓构造函数（constructor）就是在对象创建后会自动调用的初始化方法，命名为 `__init__`。

- 超类和子类在继承和重写时，如何让各自的初始化方法不会发生冲突和重写呢？有两种办法：

  1. 在子类中调用未关联实例的超类构造函数。（旧式）

  ```python
  class SongBird(Bird):
  	def __init__(self):
          Bird.__init__(self)	# 用超类 Bird 关联并无实例关联，self 仍是子类的当前实例
          self.sound = 'Squawk!'
  	def sing(self):
          print(self.sound)
  ```

  2. 使用函数 `super`。（新式）

  ```python
  class SongBird(Bird):
  	def __init__(self):
  		super().__init__()	# 调用超类(Bird)的方法
          self.sound = 'Squawk!'
  	def sing(self):
          print(self.sound)
  ```

  - ​	即使有多个超类，也只需调用函数 `super` 一次（条件是所有的超类的构造函数也使用函数 `super`，如下所示：）。函数 `super` 就像是打了个“书签标记”，像是表示“可被寻址解析”。如果子类的超级类里排序（1, 2, 3）中，假设（1，2）打了标签，则 3 也会被寻址；假设只有（1）打了标签，则 2 会被寻址，3则不会；假设都未打标签，则只有 1 会被寻址。<u>没事干的话，还是统统都打上标签为好。</u>

    ```python
    class Bird:
        def __init__(self):
            super().__init__()
            self.hungry = True
        def eat(self):
            if self.hungry:  print('Aaaah ...')
    
    class SongBird:
        def __init__(self):
            super().__init__()
            self.sound = 'Squawk!'
        def sing(self):
            print(self.sound)
            
    class JumpBird:
        def __init__(self):
            super().__init__()
            self.jump = 'Sooooo!'
        def act(self):
            print(self.jump)        
    
    class Play(SongBird,JumpBird,Bird):
        def __init__(self):
            super().__init__()
            pass
    ```

    - 函数 `super` 返回的是一个 `super` 对象，该对象将负责为你执行方法解析。当访问它的属性时，它将在所有的超类（以及超类的超类，等等），直到找到指定的属性或引发 `AttributeError` 异常。

  

  

### 元素访问

- `__len__(self)`
- `__getitem__(self, key)`
- `__setitem__(self, key, value)`
- `__delitem__(self, key)`

（暂略）

  

### 其他魔法方法

- 特性 `property` 是通过存取方法定义的属性。

- 特性函数 `property(fget, fset, fdel, doc)`

  ```python
  class Rectangle:
      def __init__(self):
          self.width = 0
          self.height = 0
  	def set_size(self, size):
          self.width, self.height = size
  	def get_size(self):
          return self.width, self.height
      size = property(get_size, set_size)
  ```
  > 书中居然没有好好讲装饰器和特性装饰器。。。
  >
  > 参考文献：[廖雪峰-装饰器](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014318435599930270c0381a3b44db991cd6d858064ac0000)，[廖雪峰-@property](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143186781871161bc8d6497004764b398401a401d4cce000)

- 在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator），本质是一个返回函数的高阶函数。多说无益，其实不太好懂，下面是一个自己定义的日志装饰器，可以作用在任何函数上，并打印该函数的耗费时间和执行时间：

  ```python
  import functools, time
  from inspect import isfunction
  
  def log(*args):
      def decorator(func):
          @functools.wraps(func)
          def wrapper(*func_args, **kwargs):
              start = time.time() #执行前计时
              fn = func(*func_args, **kwargs) #执行函数
              end = time.time() #执行后计时
              print('{} executed in {:.2f}ms, finished {}'.format(func.__name__, 
                                                                  1000*(end-start),
                                                                  time.asctime()))
              return fn
          return wrapper
  
      if isfunction(args[0]):		# 判断是否有参数，默认无参数
          return decorator(args[0])
      else:						# 若有参数，可以进一步丰富这个装饰器的功能
          return decorator
  
  >>> @log
  >>> def now(value = 30):
  >>> 	print('YESSSSSS', value)
  >>> 	return value+3
  >>> now()
  ```

  - 指定多个装饰器时，应用的顺序与列出的顺序相反。

- 特性装饰器 `@property`。习惯了高级语言的严谨，总想对属性加以访问控制，相对安全些，比如直接在`__init__`中定义公用属性，从封装性来说，它是不好的写法。 属性之访问，它亦有机制，其一便是`@propery`关键字。用此关键字，其获取、设置函数，须与属性名一致。`@property`可以把一个实例方法变成其同名属性，以支持.号访问，它亦可标记设置限制，加以规范，如下代码：

  ```python
  class Animal(object):
      def __init__(self, name, age):
          self._name = name
          self._age = age
          self._color = 'Black'
  
      @property
      def name(self):
          return self._name
  
      @name.setter
      def name(self, value):
          if isinstance(value, basestring):
              self._name = value
          else:
              self._name = 'No name'
  
      @property
      def age(self):
          return self._age
  
      @age.setter
      def age(self, value):
          if value > 0 and value < 100:
              self._age = value
          else:
              self._age = 0
              # print 'invalid age value.'
  
      @property
      def color(self):
          return self._color
  
      @color.setter
      def color(self, value):
          self._color = value;
  ```

  关于 `@property` 的用法始终总有种知其然不知其所以然的感觉。从各类教程看过来，基本可以知晓 `@prperty` 下面固定紧跟属性的方法，定义的方式也是固定的形式，函数的名称就是类属性名称，实例关联的对应属性名称前要加上一个 `_` 下划线以示隐藏，该函数就表示对该属性的提取（读取），最后别忘了 `return ` 出来。除了"可读"外，还可以可选的定义"可写"该属性的方法，那就是在 `@property` 对应下面任何位置处定义相应的 `@[属性名].setter` 装饰器和该属性的写入方法函数，名称依旧是类属性的名字。



### 迭代器（暂略）



`__next__`

`__iter__`



### 生成器（暂略）

- 包含 `yield` 语句的函数都被称为生成器。





## 第10章 开箱即用











---

[返回到首页](../index.html) | [返回到顶部](./Beginning_Python.html)

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





