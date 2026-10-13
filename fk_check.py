"""拿手写齐次矩阵和 MuJoCo 的 FK 对一遍。

左边：你写的 R(q1)@Tr(L1,0) @ R(q2)@Tr(L2,0) @ R(q3)@Tr(L3,0)
右边：MuJoCo 读 data.body(...).xpos / data.site('ee').xpos
两边应该一模一样。
"""

import numpy as np
import mujoco

XML = r"D:\MuJoCo_Projects\test\three_link_2d.xml"

L = [0.5, 0.4, 0.3]          # 三个连杆的长度
Q = [30.0, 45.0, -60.0]      # 三个关节角（度）：j1 相对世界，j2/j3 相对上一节


def Tr(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1.0]])


def R(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1.0]])


T = np.eye(3)
rows = []
for i in range(3):
    T = T @ R(Q[i]) @ Tr(L[i], 0)
    rows.append((i, T[:2, 2].copy()))

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
data.qpos[:] = np.radians(Q)
mujoco.mj_forward(model, data)

sim = [data.body("link2").xpos[:2].copy(),
       data.body("link3").xpos[:2].copy(),
       data.site("ee").xpos[:2].copy()]

print(f"{'':8}{'手写矩阵':>22}{'MuJoCo':>22}")
labels = ["关节2", "关节3", "末端"]
for (i, p), s, lab in zip(rows, sim, labels):
    print(f"{lab:8}({p[0]:9.6f},{p[1]:9.6f})  ({s[0]:9.6f},{s[1]:9.6f})")

err = max(np.abs(rows[k][1] - sim[k]).max() for k in range(3))
print(f"\n最大差值 = {err:.3e}")
