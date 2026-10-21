"""数据流追踪：把 IK 控制的每一站数据打出来看。不打开窗口，纯打印。

跑法（在任意目录都行）：
    D:\\MuJoCo_Projects\\mujoco_env\\python.exe D:\\MuJoCo_Projects\\test\\dataflow_trace.py
"""
import numpy as np, mujoco, mink

np.set_printoptions(precision=4, suppress=True)

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"
GOAL  = np.array([0.55, 0.0, 0.35])          # 想让 hand 去的地方（世界系，米）


def line():
    print("-" * 74)


# ── 站 1：磁盘上的 XML → 内存里的 model ──────────────────────────
line(); print("[1] XML → model    （一次性编译，之后 XML 再改它也不变）")
model = mujoco.MjModel.from_xml_path(SCENE)
print("    nq=%d  nv=%d  nu=%d" % (model.nq, model.nv, model.nu))

# ── 站 2/3：model → 两个互不相干的 data ──────────────────────────
line(); print("[2][3] model → 两个独立的 data 容器")
data = mujoco.MjData(model)                  # 你的：跑物理
cfg  = mink.Configuration(model)             # mink 的：跑求解
cfg.update_from_keyframe("home")             # 只改 cfg 那份，不动 data

mujoco.mj_forward(model, data)
print("    data.qpos[:7] =", np.round(data.qpos[:7], 4))
print("       data 的 hand @", np.round(data.body("hand").xpos, 4))
print("    cfg.q[:7]     =", np.round(cfg.q[:7], 4))
print("       cfg  的 hand @", np.round(cfg.data.body("hand").xpos, 4))
print("    → 两条轨此刻对不上：一个是 XML 默认姿态，一个是 home")

# 可选：把物理轨也拨到 home，让两边起点一致（注意只是对齐起点，不是绑定）
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
print("    把 data 也 reset 到 home 之后：")
print("       data 的 hand @", np.round(data.body("hand").xpos, 4),
      "   cfg 的 hand @", np.round(cfg.data.body("hand").xpos, 4), " ← 对齐了")

# ── 站 4：目标写进 task（世界系绝对位姿） ────────────────────────
line(); print("[4] task.set_target(SE3)   目标存进 task 对象里")
task   = mink.FrameTask("hand", "body", position_cost=1.0, orientation_cost=0.0)
limits = [mink.ConfigurationLimit(model)]
R_home = mink.SO3.from_matrix(cfg.data.body("hand").xmat.reshape(3, 3).copy())
task.set_target(mink.SE3.from_rotation_and_translation(R_home, GOAL))
print("    目标 hand 位置 =", GOAL)

# ── 站 5：求解环 —— 全在 cfg 内部跑，一个数都没往外漏 ────────────
line(); print("[5] 求解环（cfg 内部）：q → 运动学 → 雅可比 → QP → vel → 积分 → 新 q")
print("     %4s  %10s   %s   %s" % ("迭代", "max|vel[:7]|", "cfg.q[:4]".ljust(26), "cfg 的 hand"))
for i in range(60):
    vel = mink.solve_ik(cfg, [task], 0.01, "quadprog", limits=limits, damping=1e-3)
    cfg.integrate_inplace(vel, 0.01)
    p = cfg.data.body("hand").xpos
    if i < 3 or i % 20 == 0:
        print("     %4d  %10.2f   %s   %s" % (
            i + 1, np.abs(vel[:7]).max(), np.round(cfg.q[:4], 4), np.round(p, 4)))
    if np.linalg.norm(p - GOAL) < 1e-6:
        break
print("     收敛：迭代 %d 次，残差 %.1e m" % (
    i + 1, np.linalg.norm(cfg.data.body("hand").xpos - GOAL)))

# ── 站 6：跨轨 —— 唯一的接口就是 data.ctrl ──────────────────────
line(); print("[6] 跨轨搬运：cfg.q → data.ctrl      （整个流程唯一的接口）")
target_q = cfg.q[:7].copy()
data.ctrl[:7] = target_q
print("    data.ctrl[:7] =", np.round(data.ctrl[:7], 4))

# ── 站 7：物理轨自己跑 —— ctrl 是目标，qpos 慢慢追 ──────────────
line(); print("[7] mj_step：ctrl（目标）→ 伺服出力 → qpos（结果）")
print("     %5s  %10s   %s   %s" % ("步数", "qpos[3]", "data 的 hand".ljust(26), "离目标"))
for k in range(1, 1501):
    mujoco.mj_step(model, data)
    if k in (1, 100, 300, 600, 1000, 1500):
        print("     %5d  %10.4f   %s   %.4f m" % (
            k, data.qpos[3], np.round(data.body("hand").xpos, 4),
            np.linalg.norm(data.body("hand").xpos - GOAL)))
print("     目标 ctrl[3] = %.4f  实际 qpos[3] = %.4f  （永远差一点，伺服有滞后）"
      % (target_q[3], data.qpos[3]))

# ── 站 8：回到求解轨 —— 同步 ────────────────────────────────────
line(); print("[8] cfg.update(data.qpos)：把求解轨拉回真实状态")
print("    同步前  cfg 的 hand @", np.round(cfg.data.body("hand").xpos, 4))
cfg.update(data.qpos)
print("    同步后  cfg 的 hand @", np.round(cfg.data.body("hand").xpos, 4))
print("            data 的 hand @", np.round(data.body("hand").xpos, 4))
line()
