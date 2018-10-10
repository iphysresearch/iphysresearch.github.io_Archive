C++与并行计算



# Basic Concepts and Tools

## Introduction to C++

> Any programming language can be broken down into two high level concepts:
>
> - Data, and 
> - Operations on data.

- Top-down manner：从上层的编程概念以及技巧，到下层逐步的学习方式。

- 主函数：In C++, the starting point of execution is the “main” function, which is called **main**. 

- **main** 会告诉你的电脑从哪里开始执行你的程序。其用最简单的 C++ 代码来表示：

  ```c++
  int main(int argc, char ** argv){
  }
  ```

  该代码会编译(compile)并且执行(execute)，不过是毛事不干的！

- Hello wolrd 代码！

  ```c++
  #include <iostream>
  
  int main(int argc, char ** argv){
  	std::cout << "Hello World" << std::endl;
  }
  
  >>> g++ -o hello_world hello_world.cpp
  >>> ./hello_world
  ```

- 在 C++ 编程语言中，有两个基本概念贯彻其中：**Function**, and of **Class**. 从 C 语言开始，模块化（modularity） 就是很重要特点。此外，在 C 语言里，一切都是 function. 通常会有两个基本组成部分：一个核心语言，描述了基本数据类型和逻辑关系，以及一堆预定义好的库（libraries）。在 C++ 里，会在此基础上，增加一个新的叫做“类”（class） 的基本结构。

- Function 和 Class 的区别：函数是算法的进一步抽象（abstractions of algorithms），而类是数据及数据操作的进一步抽象（abstraction of data and operations on that data）。

- Function: 

  - 一个函数的三大要素：input, output, and contract (or algorithm).

  ![](https://i.loli.net/2018/10/10/5bbce4b964da0.png)

  - 每个函数分两步走：function declaration, and function definition.

    - 函数声明（function declaration）的格式：

      ```c++
      float f(float x);
      ```

      ![](https://i.loli.net/2018/10/10/5bbce5fb7d429.png)

    - 函数声明的位置，要么会加到头文件中（the header files with a `.h` extension），要么就会在主函数（main） 之前。

      > *A function must be declared before it can be used.*

    - Within the **argument list** (the list of inputs and outputs given as arguments to the function), the names specified are <u>irrelevant</u>. 在函数的声明里，输入输出的变量名字并不重要，只需要确保数据类型被声明出来即可：

      ```c++
      float f(float);
      ```

      那为啥有时候人们会多此一举呢？因为懒人爱用之“复制-粘贴”大法啊！在函数定义（function definition）的时候，其格式和函数的声明很像，但变量名字是必要的：

      ```c++
      float f(float x){
          float y;
          y = x*x*x - x*x + 2;
          return y;
      }
      ```

    - 函数定义好后，就可以很容易调用了：

      ```c++
      float w;
      w = f(2);
      ```

  -  这里对函数（function）做一个小结：

    > - Every function must have a function **declaration** and **definition**.
    > - Function **declarations** specify the <u>name of the function</u> and the <u>data types of the inputs and outputs</u>.
    > - Function **definitions** specify the implementation of the algorithm used to carry out the contract of the function.
    > - Variable names in function **declarations** do not matter.
    > - Variable names in function **definitions** do matter because they specify how the data is to be referred to in the implementation of the function.
    > - Variables passed to C++ functions are passed by value unless otherwise specified.

  - 接下来，综上给出一个完整的代码例子：

    ```c++
    #include <iostream>		// inclusion of library header file
    						// for use of cout
    
    float f(float x);		// function declaration
    
    int main(int argc, char ** argv){
        float w;
        w = f(2);
        std::cout << "The value of w is: " << w << std::endl;
    }
    
    float f(float x){		// function definition
        float y;
        y = x*x*x - x*x +2;
        return y;
    }
    ```

- Class:

  - 



