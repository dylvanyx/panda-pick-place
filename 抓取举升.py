"""
抓取 -> 举升 -> (可选)转腕

这个脚本解决一个具体问题：夹住方块之后举起来，方块被甩掉。

实测结论（Panda + 这个 scene，夹爪 ctrl=0 全闭合）：

  只平移、不转姿态：   250 mm/s 漂移 1.4mm | 1000 mm/s 4.2mm | 1515 mm/s 9.3mm | 2500 mm/s 16.1mm 掉了
  夹着方块转腕（绕水平轴，方块会以 44mm 半径画圆）：
      转  15 度 /  8 度每秒 -> 漂移 4.7mm   正常
      转  45 度 / 22 度每秒 -> 漂移 5.1mm   正常
      转  90 度 / 45 度每秒 -> 漂移 6.8mm   正常
      转 180 度 / 90 度每秒 -> 漂移 17.5mm  被甩掉
      转 180 度 / 45 度每秒 -> 漂移 12.4mm  被甩掉
      转 180 度 / 22 度每秒 -> 漂移 9.5mm   勉强

所以规矩就两条：
  1. 举升的时候别改朝向。这是 99% 的情况，直接解决。
  2. 真要转，转角越大允许的角速度越低，180 度要 8 秒以上才稳。

转腕快慢怎么控（实测，空手从 home 转 90 度）：
  FrameTask(gain=...)  1.0 -> 0.47秒(189度/秒)  0.5 -> 0.81秒  0.2 -> 1.87秒(48度/秒)  0.1 -> 3.49秒(25度/秒)
                       但 gain 一小，位置跟着漂：gain=1.0 漂 59mm、gain=0.2 漂 35mm、gain=0.05 漂 136mm
  自己插值姿态        转 90 度分 500 步 -> 78 度/秒，位置只漂 7mm      <-- 用这个
  改 dt               完全没用。solve_ik 给 Δq/dt，integrate_inplace 又乘回 dt，约掉了。
                      实测 dt 从 0.004 改到 0.0005，步数一步不差都是 237 步。
"""

import os
import sys
import numpy as np
import mujoco
import mink

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 封装函数.package import make_ik

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"

# ---- 这几个数是本脚本唯一需要你调的地方 ----
下降速度 = 0.25        # m/s，下探接近方块
举升速度 = 0.25        # m/s，抓起后往上抬
夹紧等待 = 0.4         # 秒，夹爪闭合后先停一会儿再动，等夹持力建立
举升高度 = 0.25        # m
转腕角度 = 0.0         # 度。0 = 不转（推荐）。要转就填 90 以下
转腕速度 = 45.0        # 度/秒。就是你要的那个参数
转腕轴 = (1, 0, 0)     # 世界坐标下的转轴
# --------------------------------------------

model, data, cfg, task, limits, dt = make_ik(SCENE)

R抓取 = mink.SO3(data.body("hand").xquat.copy())   # 抓取时的朝向，整段都别改它
手柄中心 = lambda: data.body("hand").xpos.copy()
方块中心 = lambda: data.body("cube").xpos.copy()


def 姿态夹角(Ra, Rb):
    """两个旋转之间差多少度"""
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def slerp(q0, q1, t):
    """四元数球面插值。t=0 是 q0，t=1 是 q1"""
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:                       # q 和 -q 是同一个姿态，取同一半球
        q1, d = -q1, -d
    if d > 0.9995:                  # 两者几乎重合，线性插值就够
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(np.clip(d, -1, 1))
    q = (np.sin(th0 - th0 * t) * q0 + np.sin(th0 * t) * q1) / np.sin(th0)
    return q / np.linalg.norm(q)


def 安全转速(转角):
    """实测出来的安全角速度上限（度/秒）。180 度时是 22"""
    if 转角 <= 90:
        return 45.0
    return max(8.0, 45.0 * 90.0 / 转角)


def 漂移():
    """方块相对手指的位置跑了多远（毫米）。<12 说明还稳稳夹着"""
    Rh = data.body("hand").xmat.reshape(3, 3)
    rel = Rh.T @ (方块中心() - 手柄中心())
    return np.linalg.norm(rel - 抓取时相对位置) * 1000


def 移动(目标位置, 目标姿态=None, 速度=0.25, 转速=None, viewer=None):
    """
    末端走直线到目标位置。姿态从当前朝向 slerp 插值到目标姿态。

    速度          位置速度，m/s
    转速          姿态角速度，度/秒。不填就按实测安全线自动取
                  这是「转腕快慢」唯一靠谱的旋钮，别去调 gain，也别调 dt
    """
    起点 = 手柄中心()
    距离 = float(np.linalg.norm(np.asarray(目标位置) - 起点))
    时长 = max(dt * 20, 距离 / 速度)

    Rh_now = data.body("hand").xmat.reshape(3, 3)
    if 目标姿态 is None:
        目标姿态 = mink.SO3.from_matrix(Rh_now.copy())
    转角 = 姿态夹角(Rh_now, 目标姿态.as_matrix())

    if 转角 > 0.5:
        允许 = 转速 if 转速 else 安全转速(转角)
        if 转速 and 转速 > 安全转速(转角) * 1.01:
            print("    [警告] %.0f 度/秒 超过了实测安全线 %.0f 度/秒，夹着的方块可能会滑"
                  % (转速, 安全转速(转角)))
        该用多慢 = 转角 / 时长
        if 该用多慢 > 允许:
            新时长 = 转角 / 允许
            print("    [限速] 要转 %.0f 度，原时长会让腕子以 %.0f 度/秒转（上限 %.0f），已延长到 %.1f 秒"
                  % (转角, 该用多慢, 允许, 新时长))
            时长 = 新时长

    N = max(1, int(round(时长 / dt)))
    q0 = mink.SO3.from_matrix(Rh_now.copy()).wxyz

    for i in range(N):
        t = (i + 1) / N
        R = mink.SO3(slerp(q0, 目标姿态.wxyz, t))
        p = 起点 + (np.asarray(目标位置, float) - 起点) * t
        task.set_target(mink.SE3.from_rotation_and_translation(R, p))

        vel = mink.solve_ik(tasks=[task], dt=dt, solver="quadprog",
                            configuration=cfg, limits=limits, damping=1e-3)
        cfg.integrate_inplace(vel, dt)

        data.ctrl[:7] = cfg.q[:7]        # 七个臂关节
        mujoco.mj_step(model, data)      # 走一步
        cfg.update(data.qpos)            # 把真实姿势读回 cfg

        if viewer is not None and not viewer.is_running():
            return False
        if viewer is not None:
            viewer.sync()
    return True


def 转腕(角度, 转速=45.0, 轴=(1, 0, 0), viewer=None):
    """
    原地转腕：手柄中心一点不动，只转朝向。

    角度   度，正负都行
    转速   度/秒。这是你要的那个参数 —— 45 就是 90 度转 2 秒
    轴     世界坐标下的转轴。默认 (1,0,0) 绕世界 x 轴；手指朝下时绕 (0,0,1)
           就是「拧」手指（方块不画圆，最不容易掉）
    """
    h0 = 手柄中心()
    R0 = mink.SO3.from_matrix(data.body("hand").xmat.reshape(3, 3).copy())
    th = np.radians(角度)
    ax = np.asarray(轴, float)
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Rm = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    R目标 = mink.SO3.from_matrix(Rm @ R0.as_matrix())
    print("转腕 %.0f 度，%.0f 度/秒（约 %.1f 秒）..." % (角度, 转速, abs(角度) / 转速))
    return 移动(h0, R目标, 速度=0.05, 转速=转速, viewer=viewer)


def main():
    global 抓取时相对位置

    目标 = 方块中心() + np.array([0, 0, 0.1])     # hand 在方块上方 10cm
    print("方块在", np.round(方块中心(), 4), " hand 在", np.round(手柄中心(), 4))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # ---- 1. 张开夹爪，下探到方块 ----
        data.ctrl[7] = 255            # 255 = 张到 80mm
        print("下探...")
        移动(目标, R抓取, 速度=下降速度, viewer=viewer)

        # ---- 2. 闭合夹爪，原地等一下 ----
        data.ctrl[7] = 0              # 0 = 全闭合
        print("夹紧，等 %.1f 秒..." % 夹紧等待)
        移动(手柄中心(), R抓取, 速度=0.05, viewer=viewer)
        for _ in range(int(夹紧等待 / dt)):
            data.ctrl[:7] = cfg.q[:7]
            mujoco.mj_step(model, data)
            cfg.update(data.qpos)
            if not viewer.is_running():
                return
            viewer.sync()

        Rh = data.body("hand").xmat.reshape(3, 3)
        抓取时相对位置 = Rh.T @ (方块中心() - 手柄中心())
        print("  夹住时方块相对手指 =", np.round(抓取时相对位置, 4), " 漂移 %.1f mm" % 漂移())

        # ---- 3. 举升。注意：姿态不传，默认保持当前朝向，一个字都不改 ----
        print("举升 %.0f cm（只动位置，不转腕）..." % (举升高度 * 100))
        移动(手柄中心() + np.array([0, 0, 举升高度]), 速度=举升速度, viewer=viewer)
        print("  方块现在在", np.round(方块中心(), 4), " 漂移 %.1f mm" % 漂移())

        # ---- 4. 可选：转腕 ----
        if abs(转腕角度) > 0.5:
            转腕(转腕角度, 转速=转腕速度, 轴=转腕轴, viewer=viewer)
            print("  漂移 %.1f mm" % 漂移())

        print("松手...")
        data.ctrl[7] = 255
        for _ in range(int(0.5 / dt)):
            mujoco.mj_step(model, data)
            cfg.update(data.qpos)
            if not viewer.is_running():
                return
            viewer.sync()

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
