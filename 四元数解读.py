"""输入一个四元数，直接打印它代表什么姿态。

跑法：
    D:\\MuJoCo_Projects\\mujoco_env\\python.exe D:\\MuJoCo_Projects\\test\\四元数解读.py
    D:\\MuJoCo_Projects\\mujoco_env\\python.exe D:\\MuJoCo_Projects\\test\\四元数解读.py 0.7071 0 0 0.7071
    （不带参数 = 看几个例子；带参数 = 解读你自己那个，顺序是 w x y z）
"""

import sys
import math
import numpy as np
import mink


def 解读(q, 名字=""):
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    w = q[0]
    theta = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, w))))
    s = math.sqrt(max(0.0, 1.0 - w * w))
    axis = q[1:] / s if s > 1e-9 else np.zeros(3)
    R = mink.SO3(q)
    rpy = np.degrees(R.as_rpy_radians())

    print("输入 [w,x,y,z] =", np.array2string(q, precision=4, suppress_small=True))
    if 名字:
        print("   （%s）" % 名字)
    print("   旋转角 theta  = %.2f 度" % theta)
    print("   旋转轴(单位)  =", np.array2string(axis, precision=4, suppress_small=True))
    print("   等价 RPY(度)  =", np.array2string(rpy, precision=3, suppress_small=True), "  [roll, pitch, yaw]")
    print("   等价旋转矩阵  =")
    for row in R.as_matrix():
        print("        ", np.array2string(row, precision=4, suppress_small=True))
    print()


例子 = [
    ([1.0, 0.0, 0.0, 0.0], "完全不转"),
    ([0.9914, 0.1305, 0.0, 0.0], "绕 x 轴转 15 度"),
    ([0.7071, 0.0, 0.0, 0.7071], "绕 z 轴转 90 度"),
    ([0.0, 0.7071, 0.7071, 0.0], "Panda home 姿态"),
    ([0.5, 0.5, 0.5, 0.5], "绕 (1,1,1) 方向转 120 度"),
]

if len(sys.argv) > 1:
    数字 = " ".join(sys.argv[1:]).replace(",", " ").split()
    解读([float(v) for v in 数字])
else:
    for 四元数, 说明 in 例子:
        解读(四元数, 说明)
