"""
mink IK 黑箱演示 —— 输入末端目标坐标，机械臂自己走过去。

用法:
    python ik_demo.py                          默认目标 (0.45, 0.25, 0.55)
    python ik_demo.py 0.5 0.2 0.4              指定末端目标 xyz
    python ik_demo.py --scene 路径 0.5 0.2 0.4  指定场景
    python ik_demo.py --dry 0.5 0.2 0.4        只解 IK + 跑 3 秒，不开窗口

要点:
    - 约束的是 body 帧 "hand"，不需要在 XML 里加 <site>
    - 只约束位置 (position_cost=1)，不管姿态 (orientation_cost=0)
    - 关节限位由 mink.ConfigurationLimit 自动处理，解出的角一定在限位内
    - 全程不调 mj_resetDataKeyframe，所以场景里球的位置就是 XML 里写的那个
"""

import sys
import numpy as np
import mujoco
import mink

DEFAULT_SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick.xml"
DEFAULT_TARGET = [0.45, 0.25, 0.55]

# ---------------- 命令行 ----------------
argv = sys.argv[1:]
dry = "--dry" in argv
if dry:
    argv.remove("--dry")

scene = DEFAULT_SCENE
if "--scene" in argv:
    i = argv.index("--scene")
    scene = argv[i + 1]
    del argv[i:i + 2]

target = np.array([float(v) for v in argv[:3]]) if len(argv) >= 3 else np.array(DEFAULT_TARGET)

# ---------------- 1. 解 IK ----------------
model = mujoco.MjModel.from_xml_path(scene)
data = mujoco.MjData(model)

cfg = mink.Configuration(model)
cfg.update_from_keyframe("home")              # 拿 home 当迭代起点

task = mink.FrameTask("hand", "body",
                      position_cost=1.0,      # 位置权重大 -> 使劲往目标点凑
                      orientation_cost=0.0,   # 姿态权重 0 -> 不管朝向
                      lm_damping=1.0)
goal = mink.SE3.from_rotation_and_translation(mink.SO3.identity(), target)
limits = [mink.ConfigurationLimit(model)]

err = np.inf
for it in range(300):
    task.set_target(goal)
    vel = mink.solve_ik(cfg, [task], dt=1e-2, solver="quadprog",
                        limits=limits, damping=1e-3)
    cfg.integrate_inplace(vel, 1e-2)
    err = np.linalg.norm(
        cfg.get_transform_frame_to_world("hand", "body").translation() - target)
    if err < 1e-6:
        break

q_ik = cfg.q[:7].copy()
print(f"[IK]   目标 {np.round(target, 3)}   迭代 {it + 1} 次   残差 {err:.2e} m")
print(f"[IK]   解出关节角 {np.round(q_ik, 4)}")

# ---------------- 2. 喂给位置伺服 ----------------
data.qpos[:7] = model.key_qpos[0][:7]         # 从 home 姿势出发
data.ctrl[:] = model.key_ctrl[0]              # 夹爪等沿用 home
data.ctrl[:7] = q_ik                          # 目标换成 IK 解
mujoco.mj_forward(model, data)

p0 = data.body("hand").xpos.copy()
print(f"[伺服] 起点末端 {np.round(p0, 4)}   离目标 {np.linalg.norm(p0 - target) * 1000:.1f} mm")

# ---------------- 3. 开窗口 / 无头跑 ----------------
if dry:
    for _ in range(1500):                     # 3 s
        mujoco.mj_step(model, data)
    p1 = data.body("hand").xpos.copy()
    print(f"[伺服] 3s 后末端 {np.round(p1, 4)}   离目标 {np.linalg.norm(p1 - target) * 1000:.1f} mm")
    print("       残差不为 0 = 重力下垂 + 位置伺服不是刚性锁定")
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        n = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            n += 1
            if n % 500 == 0:
                p = data.body("hand").xpos
                print(f"t={data.time:5.1f}s  末端 {np.round(p, 4)}  "
                      f"离目标 {np.linalg.norm(p - target) * 1000:5.1f} mm")
