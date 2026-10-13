import numpy as np

# ====================== 改这里 ======================
L1  = 1      # 连杆 1 长度
L2  = 1      # 连杆 2 长度
th1 = 90.0     # 关节 1 角度（度，相对世界）
th2 = 90.0     # 关节 2 角度（度，相对【连杆 1】不是世界）
L3  = 1      # 连杆 3 长度
th3 = 90.0     # 关节 3 角度（度，相对【连杆 2】不是世界）
# ===================================================

np.set_printoptions(linewidth=200, suppress=True, precision=4)


def link(theta_deg, length):
    """一个连杆的齐次变换矩阵：先转 theta，再沿自己的 x 轴走 length。"""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, length * c],
                     [s,  c, length * s],
                     [0,  0, 1.0]])


T1 = link(th1, L1)          # 底座 -> 连杆1 末端（=关节2）
T2 = link(th2, L2)          # 关节2 -> 连杆2 末端（=末端执行器）
T3 = link(th3,L3)
T  = T1 @ T2 @ T3
print("T1 =\n", T1, "\n")
print("T2 =\n", T2, "\n")
print("T3 =\n", T3, "\n")
print("T1 @ T2 @ T3 =\n", T, "\n")

j1   = np.array([0.0, 0.0])
j2   = T1[:2, 2]            # T1 的平移列
j3   = T2[:2, 2]            # T2 的平移列
end  = T[:2, 2]             # 连乘后的平移列

print(f"关节1 = ({j1[0]:.4f}, {j1[1]:.4f})")
print(f"关节2 = ({j2[0]:.4f}, {j2[1]:.4f})")
print(f"关节3 = ({j3[0]:.4f}, {j3[1]:.4f})")
print(f"末端  = ({end[0]:.4f}, {end[1]:.4f})")
