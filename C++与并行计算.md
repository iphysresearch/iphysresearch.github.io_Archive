C++与并行计算



# Basic Concepts and Tools

## Introduction to C++

> Any programming language can be broken down into two high level concepts:
>
> - Data, and 
> - Operations on data.

- Top-down manner：从上层的编程概念以及技巧，到下层逐步的学习方式。

- In C++, the starting point of execution is the “main” function, which is called **main**. 

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
  ```




