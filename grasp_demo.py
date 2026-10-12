"""
抓取流程演示：位点 IK + 分步移动 + 到位判定 + 夹爪时序。

为什么需要这个：ik_demo.py 只做"给个坐标走过去"，走完就停在那儿。
真实的抓取是一串**有序动作**，每步都得知道"上一步完成了没有"。

用法:
    python grasp_demo.py           全流程（开窗口看）
    python grasp_demo.py --dry     无窗口，只打印每阶段结果

三个关键概念:
  1. IK 只回答"手该摆成什么姿势"，不回答"怎么过去、什么时候到"。
  2. "到位"的判据不是误差 = 0（永远到不了），而是【误差 < 容差 且 速度 < 阈值】。
  3. 想让末端走直线，得把直线拆成小段、每段解一次 IK —— 这就是轨迹的雏形。
"""

import sys
import contextlib
import numpy as np
import mujoco
import mink

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"

# ---- 实测标定出来的常量 ----
TIP_OFFSET = 0.0584                   # hand 中心 -> 两指夹取中心，沿世界 -Z（实测标定）
FINGER_BOTTOM = 0.1188                # hand 中心 -> 手指最低点，沿世界 -Z（实测标定）
# 上面两个数差 0.0604，意思是：手指比夹取中心还往下伸 6cm。
# 所以夹取中心对准球心时，手指会插到球下方 6cm —— 抓桌面上的小物体必然撞桌子。
# 对策：夹取中心比球心高一点，且球要够大（半径 > 0.031 才可能从正上方抓着）。
# 抓取高度：让手指下缘比物体中心低一点点，这样手指能覆盖物体重心所在的区域。
# 为什么不能更高：手指只夹住物体上半截时，重心落在夹持区外面，一抬就翻转滑脱
#   （实测：手指下缘 = 物体中心时，抬起 5mm 物体就掉了）。
# 为什么不能更低：手指下端低于桌面就会被桌面挡住。
GRASP_RAISE = 0.045                    # 手指下缘比物体中心低约 1.5cm
GRIP_OPEN, GRIP_CLOSE = 255.0, 0.0    # 实测：255 -> 两指间距 80mm；0 -> 完全闭拢
TOL_POS = 0.015                       # 到位判据：位置容差 15 mm
TOL_VEL = 0.05                        # 到位判据：关节速度阈值 rad/s
STAGE_TIMEOUT = 6.0                   # 单个阶段超时（秒）

dry = "--dry" in sys.argv

# ================= 初始化 =================
model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)

data.qpos[:7] = model.key_qpos[0][:7]     # 从 home 开始（不用 reset，球就留在 XML 写的 pos）
data.ctrl[:] = model.key_ctrl[0]
mujoco.mj_forward(model, data)

R_home = mink.SO3.from_matrix(data.body("hand").xmat.reshape(3, 3).copy())
cube = data.body("cube").xpos.copy()
print(f"物体位置 {np.round(cube, 4)}")
print(f"末端起点 {np.round(data.body('hand').xpos, 4)}")

# 下面所有目标点都用【指尖坐标】表示。IK 内部再换算成 hand 中心的位置。
# ⚠️ 这一步换算很容易漏：IK 约束的是 body "hand" 的位置，而我们要的是指尖，
#    两者沿世界 Z 差 0.105 m。漏掉它就会出现"永远差 10cm 到位不了"。
PRE_GRASP = cube + np.array([0, 0, 0.30])          # 指尖停在球正上方 30cm
GRASP = cube + np.array([0, 0, GRASP_RAISE])       # 指尖降到球心略上方

# ================= IK =================
cfg = mink.Configuration(model)
task = mink.FrameTask("hand", "body", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
limits = [mink.ConfigurationLimit(model)]


def solve_ik(tip_target):
    """给【指尖】目标点 -> 返回 7 个关节角。每次从 home 起解，同一个点必得同一个姿势。"""
    hand_target = np.asarray(tip_target, dtype=float) + np.array([0, 0, TIP_OFFSET])
    cfg.update_from_keyframe("home")
    goal = mink.SE3.from_rotation_and_translation(R_home, hand_target)
    for _ in range(500):
        task.set_target(goal)
        vel = mink.solve_ik(cfg, [task], dt=1e-2, solver="quadprog",
                            limits=limits, damping=1e-3)
        cfg.integrate_inplace(vel, 1e-2)
        if np.linalg.norm(
                cfg.get_transform_frame_to_world("hand", "body").translation()
                - hand_target) < 1e-5:
            break
    return cfg.q[:7].copy()


def tip_pos():
    """指尖（≈夹爪中心）的世界坐标。"""
    return data.body("hand").xpos.copy() + np.array([0, 0, -TIP_OFFSET])


def report(tag):
    """阶段体检：物体有没有被动过、手指开多大。"""
    p = data.body("cube").xpos
    gap = np.linalg.norm(data.body("left_finger").xpos
                         - data.body("right_finger").xpos) * 1000
    drift = np.linalg.norm(p[:2] - cube[:2]) * 1000
    print(f"      [{tag}] 物块 {np.round(p, 4)}  偏移 {drift:4.1f}mm  "
          f"指间距 {gap:5.1f}mm  夹持力 {data.actuator_force[7]:+.2f}")


# ================= 窗口（dry 模式下用替身） =================
@contextlib.contextmanager
def open_viewer():
    if dry:
        class _Fake:
            def is_running(self):
                return True

            def sync(self):
                pass
        yield _Fake()
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as v:
            yield v


def run_steps(viewer, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        viewer.sync()


def wait_reached(viewer, target_tip, label):
    """到位判据 = 位置够近 且 速度够小。缺任何一个都会误判（'路过'也算到）。"""
    t0, err, spd = data.time, np.inf, np.inf
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        err = np.linalg.norm(tip_pos() - target_tip)
        spd = np.linalg.norm(data.qvel[:7])
        if err < TOL_POS and spd < TOL_VEL:
            print(f"  [{label}] 到位  用时 {data.time - t0:5.2f}s  "
                  f"残差 {err * 1000:5.1f}mm  速度 {spd:.3f} rad/s")
            return True
        if data.time - t0 > STAGE_TIMEOUT:
            print(f"  [{label}] 超时  残差 {err * 1000:5.1f}mm  速度 {spd:.3f} rad/s")
            return False
    return False


def move_linear(viewer, p_from, p_to, n=10, label=""):
    """笛卡尔直线：拆成 n 小段，每段解一次 IK 并等它走完。
    不拆的话，伺服走的是关节空间直线，末端会划一道弧线。"""
    p = p_to
    for i in range(1, n + 1):
        p = p_from + (p_to - p_from) * (i / n)
        data.ctrl[:7] = solve_ik(p)
        t0 = data.time
        while viewer.is_running() and data.time - t0 < 1.0:
            mujoco.mj_step(model, data)
            viewer.sync()
            if (np.linalg.norm(tip_pos() - p) < 0.01
                    and np.linalg.norm(data.qvel[:7]) < 0.3):
                break
    print(f"  [{label}] {n} 段走完  终点残差 {np.linalg.norm(tip_pos() - p_to) * 1000:.1f}mm")


# ================= 主流程 =================
with open_viewer() as viewer:
    print("\n阶段 1  张开夹爪")
    data.ctrl[7] = GRIP_OPEN
    run_steps(viewer, 0.8)

    print("阶段 2  平移到物体正上方 30cm")
    data.ctrl[:7] = solve_ik(PRE_GRASP)
    wait_reached(viewer, PRE_GRASP, "预抓取点")
    report("阶段2 后")

    print("阶段 3  垂直下降，手指下端落到物体中心高度")
    move_linear(viewer, PRE_GRASP, GRASP, n=10, label="下降")
    report("阶段3 后")

    print("阶段 4  闭合夹爪（分 30 步慢速，否则手指会高速撞飞物体）")
    for v in np.linspace(GRIP_OPEN, GRIP_CLOSE, 30):
        data.ctrl[7] = v
        run_steps(viewer, 0.02)
    run_steps(viewer, 0.5)
    report("阶段4 后")
    gap = np.linalg.norm(data.body("left_finger").xpos
                         - data.body("right_finger").xpos) * 1000
    print(f"  指间距 {gap:.1f} mm   (方块宽 50 mm，夹住的话会停在 50 附近)")

    print("阶段 5  抬起")
    move_linear(viewer, GRASP, PRE_GRASP, n=10, label="上升")
    run_steps(viewer, 0.5)

    print("\n结果")
    now = data.body("cube").xpos.copy()
    print(f"  物体位置 {np.round(now, 4)}   初始 {np.round(cube, 4)}")
    lift = (now[2] - cube[2]) * 1000
    print(f"  抬升 {lift:6.1f} mm  ->  {'抓起来了' if lift > 20 else '没抓住'}")
    print("  （抓不住通常是摩擦/夹紧力问题，不是 IK 问题）")

    if not dry:
        print("\n窗口保持打开，关闭窗口即结束。")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
