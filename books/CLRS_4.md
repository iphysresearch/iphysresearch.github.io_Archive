







# 第4章 分治策略



最大子数组问题

```python
def FIND_MAX_CROSSING_SUBARRAY(A, low, mid, high):
    left_sum = -float('inf')
    sum_, max_left = 0, low
    for i in range(low, mid+1)[::-1]:
        sum_ += A[i]
        if sum_ > left_sum:
            left_sum = sum_
            max_left = i
    right_sum = -float('inf')
    sum_, max_right = 0, high
    for j in range(mid, high+1):
        sum_ += A[j]
        if sum_ > right_sum:
            right_sum = sum_
            max_right = j
    return (max_left, max_right, left_sum+right_sum)


def FIND_MAXIMUM_SUBARRAY(A, low, high):
    if low == high:
        return (low, high, A[low])
    else:
        mid = (low + high)//2
        (left_low, left_high, left_sum) = FIND_MAXIMUM_SUBARRAY(A, low, mid)
        (right_low, right_high, right_sum) = FIND_MAXIMUM_SUBARRAY(A, mid+1, high)
        (cross_low, cross_high, cross_sum) = FIND_MAX_CROSSING_SUBARRAY(A, low, mid, high)
        if left_sum >= right_sum and left_sum >= cross_sum:
            return (left_low, left_high, left_sum)
        elif right_sum >= left_sum and right_sum >= cross_sum:
            return (right_low, right_high, right_sum)
        else:
            return (cross_low, cross_high, cross_sum)
```







矩阵乘法

```python
def SQUARE_MATRIX_MULTIPLY(A, B):
    n = A.shape[0]
    C = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            C[i,j] = 0
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]
    return C

>>> A = np.array([[1,2,3], [2,3,4], [5,6,7]])
>>> B = np.array([[9,8,7], [8,7,6], [7,6,5]])
>>> SQUARE_MATRIX_MULTIPLY(A, B)
```



```python
def SQUARE_MATRIX_MULTIPLY_RECURSIVE(A, B):
    n = A.shape[0]
    C = np.empty((n,n))
    if n == 1:
        C[0,0] = A[0,0] * B[0,0] # 其实可以不用写 [0,0]
    else:
        C[:n//2, :n//2] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:n//2, :n//2], B[:n//2, :n//2]) + SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:n//2, n//2:], B[n//2:, :n//2])
        C[:n//2, n//2:] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:n//2, :n//2], B[:n//2, n//2:]) + SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:n//2, n//2:], B[n//2:, n//2:])
        C[n//2:, :n//2] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[n//2:, :n//2], B[:n//2, :n//2]) + SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[n//2:, n//2:], B[n//2:, :n//2])
        C[n//2:, n//2:] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[n//2:, :n//2], B[:n//2, n//2:]) + SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[n//2:, n//2:], B[n//2:, n//2:])
    return C

>>> A = np.array([[1,2,3,4], [2,3,4,5], [5,6,7,8], [6,7,8,9]])
>>> B = np.array([[9,8,7,6], [8,7,6,5], [7,6,5,4], [6,5,4,3]])
>>> SQUARE_MATRIX_MULTIPLY_RECURSIVE(A, B)
```



















