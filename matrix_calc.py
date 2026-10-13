"""矩阵乘法。改下面 A、B 两个矩阵，运行就出结果。"""

import numpy as np
from math import sin,cos
# ---------------- 改这里 ----------------
pi = np.pi
x = pi/2
L = 1
tx = 1
ty = 1
A = [[cos(x), -sin(x),tx],
     [sin(x), cos(x),ty],
     [0,0,1]]

B = [[1],
     [1],
     [1]]
# ---------------------------------------

np.set_printoptions(linewidth=200, suppress=True)  # suppress: 把 1e-16 这类浮点噪声显示成 0
a, b = np.asarray(A), np.asarray(B)

if a.ndim != 2 or b.ndim != 2:
    raise ValueError(f"两个矩阵都要是二维，现在是 {a.shape} 和 {b.shape}")
if a.shape[1] != b.shape[0]:
    raise ValueError(f"A 的列数({a.shape[1]}) 要等于 B 的行数({b.shape[0]})，现在不相等")

print(a @ b)
