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


def get_T(roll,pitch,yaw):
    T = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ Tr(L, 0, 0)
    return T

T1 = get_T(0,-90,0)
T2 = T1 @ get_T(0,180,0)
e1 = T1[:3, 3]
e2 = T2[:3,3]
print("T1 =\n", T1, "\n")
print("T2 =\n", T2, "\n")

print(f"末端1 = ({e1[0]:.4f}, {e1[1]:.4f}, {e1[2]:.4f})")
print(f"末端2 = ({e2[0]:.4f}, {e2[1]:.4f}, {e2[2]:.4f})")
