"""
位置直线 + 姿态球面，两条曲线同步插值。

用法：
    python 插值转身.py            开窗口看
    python 插值转身.py --headless  不开窗口，只打印结果

要改的地方只有 main() 里那几行：目标位置、绕哪个世界轴转多少度、两个速度。
"""

import sys
import numpy as np
import mujoco
import mink

sys.path.insert(0, r"D:\MuJoCo_Projects\test")
from 封装函数.package import make_ik

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"
看窗口 = "--headless" not in sys.argv

model, data, cfg, task, limits, dt = make_ik(SCENE)

# 冗余自由度正则：7 个关节只干 6 个自由度（位置 3 + 姿态 3），多出来的那 1 个没人管，
# IK 就会随便挑分支，转腕时容易一头撞进关节限位。这个任务让它"想回 home 位"。
posture = mink.PostureTask(model, cost=1e-2)
posture.set_target(cfg.q.copy())


def 姿态角(Ra, Rb):
    """两个旋转矩阵之间差多少度"""
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def 走(目标位置, 目标姿态=None, 位置速度=0.25, 角速度=45.0, 平滑=True, viewer=None):
    """
    末端从当前位置/姿态走到目标。
    位置走直线，姿态走球面短弧，两者共用同一个 t —— 所以是「边走边转」，
    不会出现「先原地转完再平移」或者反过来。

    姿态插值只用一行：起姿态.interpolate(目标姿态, t)，
    等价于四元数 slerp（实测两者差 5e-16），180 度也不奇异。

    位置速度  m/s
    角速度    度/秒。想转得慢就调它，不要调 gain，也不要调 dt
    平滑      True  = S 曲线，起步和停止都是慢慢加速/减速（峰值会比匀速高 1.5 倍）
              False = 匀速，第一帧就要全速，最后一帧急停
    """
    起点 = data.body("hand").xpos.copy()
    起姿态 = mink.SO3.from_matrix(data.body("hand").xmat.reshape(3, 3).copy())
    目标位置 = np.asarray(目标位置, float)
    if 目标姿态 is None:
        目标姿态 = 起姿态

    距离 = float(np.linalg.norm(目标位置 - 起点))
    转角 = 姿态角(起姿态.as_matrix(), 目标姿态.as_matrix())

    # 谁需要的步数多，就按谁算 —— 保证两个速度都不超
    步数位置 = 距离 / 位置速度
    步数姿态 = np.radians(转角) / np.radians(角速度) if 转角 > 1e-6 else 0.0
    步数 = max(1, int(round(max(步数位置, 步数姿态) / dt)))

    for i in range(1, 步数 + 1):
        t = i / 步数
        u = t * t * (3 - 2 * t) if 平滑 else t     # S 曲线 = 时间参数化，只改这一个映射
        p = 起点 + (目标位置 - 起点) * u
        R = 起姿态.interpolate(目标姿态, u)
        task.set_target(mink.SE3.from_rotation_and_translation(R, p))

        vel = mink.solve_ik(tasks=[task, posture], dt=dt, solver="quadprog",
                            configuration=cfg, limits=limits, damping=1e-3)
        cfg.integrate_inplace(vel, dt)
        data.ctrl[:7] = cfg.q[:7]        # 只有前 7 个是臂关节，第 8 个 ctrl[7] 是夹爪
        mujoco.mj_step(model, data)
        cfg.update(data.qpos)            # 必须在 mj_step 之后，把真实关节角读回 cfg

        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                return 步数 * dt
    return 步数 * dt


def main():
    R0 = mink.SO3.from_matrix(data.body("hand").xmat.reshape(3, 3).copy())

    # 「在起始姿态基础上，绕世界 x 轴转 90 度」的正确写法：
    #   先造出这个转动本身的绝对旋转 Rx90，再右乘到起始姿态上
    # 绕世界 y 轴转 90 度在这个 home 姿态下关节限位不够（实测漂 67mm），绕 x / z 才行
    Rx90 = mink.SO3(np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]))
    R目标 = mink.SO3.from_matrix(Rx90.as_matrix() @ R0.as_matrix())

    print("起始 wxyz :", np.round(R0.wxyz, 4))
    print("目标 wxyz :", np.round(R目标.wxyz, 4), "  <- 绕着世界 x 轴转了 90 度")
    print("姿态夹角  : %.1f 度" % 姿态角(R0.as_matrix(), R目标.as_matrix()))

    起点 = data.body("hand").xpos.copy()
    目标 = 起点.copy()          # 位置一点不动，就是原地转腕
    用时 = 走(目标, R目标, 位置速度=0.25, 角速度=45.0, viewer=viewer)

    终点 = data.body("hand").xpos.copy()
    末姿态 = data.body("hand").xmat.reshape(3, 3).copy()
    print("步数 %d 步，用时 %.2f 秒" % (int(round(用时 / dt)), 用时))
    print("位置漂移 %.1f mm" % (np.linalg.norm(终点 - 起点) * 1000))
    print("姿态残差 %.2f 度" % 姿态角(末姿态, R目标.as_matrix()))
    return 末姿态


if __name__ == "__main__":
    if 看窗口:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            main()
            while viewer.is_running():
                mujoco.mj_step(model, data)
                data.ctrl[:7] = cfg.q[:7]
                viewer.sync()
    else:
        viewer = None
        main()
