"""
闭环 IK 控制：把「窗口开着就一直转」换成「目标到了就停」。

两个要点
--------
1. 到位判据 = 位置误差够小 **且** 几乎不动了。
   - 只看误差会在「路过」时误判 —— 手经过目标附近但还在飞。
   - 位置伺服抗不住重力，稳态残差约 5~6 mm，**永远不会归零**。
     所以容差必须设得比它大（这里 10 mm），设成 1 mm 会跑到超时也停不下来。

2. 每个动作封装成 move_to()，内部闭环、到位就返回。
   → 外面看起来是「一步到位」，多阶段任务就是连续调用几次。
   难点不在循环，在于**明确「什么时候算走完」**。

实测（scene_pick_table.xml）
----------------------------
    第1段 -> [0.5  0.   0.45]   到位   249 步 (0.50 s)  残差 4.9 mm
    第3段 -> [0.5  0.   0.62]   到位   263 步 (0.53 s)  残差 6.9 mm
    只解一次 IK 然后死转 2000 步 -> 残差 33.9 mm（所以闭环不能省）

注意：hand 中心低于 0.3688 m 时手指会插进桌面（桌面顶面 z=0.25，
手指最下端比 hand 中心低 0.1188 m）。别把目标设那么低。
"""

import numpy as np
import mujoco
import mujoco.viewer
import mink

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"

# 想让它去几个地方就加几行，按顺序走
任务表 = [
    np.array([0.50, 0.00, 0.45]),
    np.array([0.60, 0.15, 0.50]),
    np.array([0.50, 0.00, 0.62]),
    np.array([0.55, 0.00, 0.6245]),
]

位置容差 = 0.010     # m，必须大于伺服稳态残差（5~6 mm），否则永远停不下来
速度阈值 = 0.05      # rad/s，关节几乎不动了
稳定次数 = 25        # 连续满足多少次才算「稳」，防止路过时误判
最多步数 = 4000      # 每段动作的安全上限


model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)


def 摆到home():
    """机器人摆成 home，方块留在 XML 里给的位置。

    不能直接用 mj_resetDataKeyframe：panda.xml 里的 <key> 只写了机器人那 9 个数，
    方块占的 7 维（freejoint）会被补成「原点 + 单位姿态」，
    一 reset 方块就瞬移到世界原点，正好卡在机械臂底座里。
    """
    mujoco.mj_resetData(model, data)          # 方块回到 XML 的 (0.5, 0, 0.27)
    data.qpos[:9] = model.key_qpos[0][:9]     # 机器人摆成 home
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)            # 算一遍各零件在哪


摆到home()

cfg = mink.Configuration(model)
cfg.update(data.qpos)
task = mink.FrameTask("hand", "body", position_cost=1.0, orientation_cost=0.0)
limits = [mink.ConfigurationLimit(model)]
dt = float(model.opt.timestep)
R = mink.SO3.from_matrix(data.body("hand").xmat.reshape(3, 3).copy())


def 到位了吗(goal):
    """误差够小 且 几乎不动了。两个都要满足。"""
    误差 = np.linalg.norm(data.body("hand").xpos - goal)
    速度 = np.linalg.norm(data.qvel[:7])
    return 误差 < 位置容差 and 速度 < 速度阈值


def move_to(goal, viewer=None):
    """闭环走到 goal。返回 (是否到位, 走了几步, 残余误差 mm)。"""
    task.set_target(mink.SE3.from_rotation_and_translation(R, np.asarray(goal, float)))
    连续满足 = 0
    for k in range(最多步数):
        if viewer is not None and not viewer.is_running():
            return False, k, np.linalg.norm(data.body("hand").xpos - goal) * 1000

        vel = mink.solve_ik(tasks=[task], dt=dt, solver="quadprog",
                            configuration=cfg, limits=limits, damping=1e-3)
        cfg.integrate_inplace(vel, dt)
        data.ctrl[:7] = cfg.q[:7]
        mujoco.mj_step(model, data)
        cfg.update(data.qpos)
        if viewer is not None:
            viewer.sync()

        if 到位了吗(goal):
            连续满足 += 1
            if 连续满足 >= 稳定次数:
                return True, k + 1, np.linalg.norm(data.body("hand").xpos - goal) * 1000
        else:
            连续满足 = 0

    return False, 最多步数, np.linalg.norm(data.body("hand").xpos - goal) * 1000


with mujoco.viewer.launch_passive(model, data) as viewer:
    for i, 目标 in enumerate(任务表):
        ok, 步, 误差 = move_to(目标, viewer=viewer)
        print("第 %d 段 -> %-24s %-12s %4d 步 (%.2f 秒)  残余误差 %5.1f mm" % (
            i + 1, np.round(目标, 3),
            "到位" if ok else "超时/窗口关闭", 步, 步 * dt, 误差))
        if not ok:
            break
    print("结束，hand 停在", np.round(data.body("hand").xpos, 4))
