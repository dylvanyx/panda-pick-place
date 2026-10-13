import numpy as np

# ====================== 改这里 ======================
L     = 1     # 连杆长度
yaw   = 90.0    # 绕 z 轴转（度）—— 水平面内转，决定朝哪个方向
pitch = -90.0   # 绕 y 轴转（度）—— 负值向上抬，正值向下压
roll  = 0.0     # 绕 x 轴转（度）—— 绕杆自己转，末端位置不变
# ===================================================

np.set_printoptions(linewidth=200, suppress=True, precision=4)


def Tr(x, y, z):
    """平移矩阵：把 (x, y, z) 放进第 4 列。"""
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1.0]])


def Rx(d):
    """绕 x 轴转 d 度。"""
    a = np.radians(d)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1.0]])


def Ry(d):
    """绕 y 轴转 d 度。"""
    a = np.radians(d)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1.0]])


def Rz(d):
    """绕 z 轴转 d 度。"""
    a = np.radians(d)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1.0]])


T = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ Tr(L, 0, 0)

e = T[:3, 3]

print("T =\n", T, "\n")
print(f"末端 = ({e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f})")
