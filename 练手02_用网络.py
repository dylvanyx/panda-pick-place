"""练手 02：载入练手01训练好的网络，输入一个坐标，让它自己走过去。

先跑一次练手01生成模型文件（模型_学IK.pt），然后：

    # 演示 3 个点（会开窗，看着机械臂动）
    python 练手02_用网络.py

    # 指定一个坐标
    python 练手02_用网络.py 0.5 0 0.6

    # 指定坐标 + 朝向（四元数 w x y z，w 在前）
    python 练手02_用网络.py 0.5 0 0.6 --quat 0 0.7071 0.7071 0

    # 不开窗，只打印
    python 练手02_用网络.py 0.5 0 0.6 --无窗

在自己代码里调用：

    from 练手02_用网络 import 网络IK
    q = 网络IK([0.5, 0, 0.6])          # 返回 7 个关节角(rad)
    data.ctrl[:7] = q                  # 直接喂给仿真

注意三件事：
  1. 它只认训练覆盖的那块区域（x 0.25~0.68, y ±0.36, z 0.29~0.97），出去就乱指
  2. 它不认识关节限位，输出必须自己 np.clip 到 model.jnt_range
  3. 它不会纠错：一次算完就交给伺服，误差里含伺服死区和重力下垂
"""

import os
import sys
import numpy as np
import mujoco
import torch

sys.path.insert(0, r"D:\MuJoCo_Projects\test")
from 封装函数.package import make_ik

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"
模型路径 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "模型_学IK.pt")

model, data, cfg, task, limits, dt = make_ik(SCENE)
LO, HI = model.jnt_range[:7, 0], model.jnt_range[:7, 1]

# home 姿态下 hand 的朝向，作为"不指定朝向"时的默认值
_fk = mujoco.MjData(model)
_fk.qpos[:7] = model.key_qpos[0][:7]
mujoco.mj_kinematics(model, _fk)
HOME_QUAT = _fk.body("hand").xquat.copy()
if HOME_QUAT[0] < 0:
    HOME_QUAT = -HOME_QUAT

_包 = None


def 载入网络():
    """读 模型_学IK.pt。里面是网络本体 + 四个标准化参数，缺一不可。"""
    global _包
    if _包 is None:
        if not os.path.exists(模型路径):
            raise FileNotFoundError("找不到 %s，先跑一遍：python 练手01_学IK.py" % 模型路径)
        # weights_only=False：这个文件存的是整个网络对象，不是纯权重字典
        _包 = torch.load(模型路径, weights_only=False)
    return _包


def 网络IK(位置, 四元数=None):
    """核心就这一个函数：给末端位置(3)，返回 7 个关节角(rad)。

    四元数不给就用 home 的朝向。内部做的三件事：
      1. 拼成 7 维输入 [x, y, z, qw, qx, qy, qz]
      2. 用训练时的 mu/sd 标准化（**必须用训练集的那一套**，换一套结果全错）
      3. 前向一次，再把输出反标准化回 rad
    """
    包 = 载入网络()
    位置 = np.asarray(位置, dtype=float).reshape(3)
    四元数 = HOME_QUAT if 四元数 is None else np.asarray(四元数, dtype=float).reshape(4)
    x = np.r_[位置, 四元数]
    with torch.no_grad():
        y = 包["net"](torch.tensor((x - 包["mu"]) / 包["sd"], dtype=torch.float32)).numpy()
    q = y * 包["qs"] + 包["qm"]
    return np.clip(q, LO, HI)          # 网络不认识限位，这一行不能省


def 回到home():
    mujoco.mj_resetData(model, data)
    data.qpos[:9] = model.key_qpos[0][:9]
    data.ctrl[:] = 0.0
    data.ctrl[7] = 255.0               # 夹爪张开，方便看
    mujoco.mj_forward(model, data)


def 走过去(目标位置, 目标四元数=None, viewer=None, 步数=600):
    """网络给关节角 → 写 ctrl → 让伺服把手臂带过去。过程不纠错。"""
    q = 网络IK(目标位置, 目标四元数)
    回到home()
    data.ctrl[:7] = q
    误差 = np.zeros(步数)
    for i in range(步数):
        mujoco.mj_step(model, data)
        误差[i] = np.linalg.norm(data.body("hand").xpos - 目标位置) * 1000
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                break
    return q, float(误差[-1]), 误差


def 演示(目标列表, 四元数=None, 开窗=True):
    载入网络()
    viewer = None
    ctx = None
    if 开窗:
        import mujoco.viewer
        ctx = mujoco.viewer.launch_passive(model, data)
        viewer = ctx.__enter__()
    try:
        for 目标 in 目标列表:
            目标 = np.asarray(目标, dtype=float)
            q, 末误差, 历史 = 走过去(目标, 四元数, viewer=viewer)
            实际 = data.body("hand").xpos.copy()
            print("目标 (%5.2f, %+5.2f, %5.2f)" % tuple(目标))
            print("   网络给的关节角(度) : %s" % np.round(np.degrees(q), 1))
            print("   实际到达           : (%5.3f, %+5.3f, %5.3f)" % tuple(实际))
            print("   误差 %6.2f mm   （过程中最小 %.2f mm）\n"
                  % (末误差, 历史.min()))
        if viewer is not None:
            print("窗口保持打开，看完关掉即可。")
            while viewer.is_running():
                viewer.sync()
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)


def main():
    参数 = sys.argv[1:]
    开窗 = True
    if "--无窗" in 参数:
        参数.remove("--无窗")
        开窗 = False

    四元数 = None
    if "--quat" in 参数:
        i = 参数.index("--quat")
        四元数 = [float(v) for v in 参数[i + 1: i + 5]]
        参数 = 参数[:i] + 参数[i + 5:]

    def 是数(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    数值 = [float(v) for v in 参数 if 是数(v)]
    if len(数值) == 3:
        目标列表 = [tuple(数值)]
    else:
        目标列表 = [(0.50, 0.00, 0.60), (0.45, 0.25, 0.52), (0.55, -0.20, 0.70)]

    包 = 载入网络()
    print("载入网络：%s" % 模型路径)
    print("训练时的采样幅度 ±%.2f rad，超出覆盖范围的坐标不可信" % 包["采样幅度"])
    print("训练覆盖的末端范围 x 0.25~0.68  y -0.36~+0.36  z 0.29~0.97 (m)\n")

    演示(目标列表, 四元数=四元数, 开窗=开窗)


if __name__ == "__main__":
    main()
