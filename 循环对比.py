"""同一个目标，两种循环写法，自己看差别。

跑法：
    D:\\MuJoCo_Projects\\mujoco_env\\python.exe D:\\MuJoCo_Projects\\test\\循环对比.py
"""

import numpy as np
import mujoco
import mink

np.set_printoptions(linewidth=200, suppress=True)

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"
GOAL  = np.array([0.5, 0.0, 0.4])
DT    = 0.01
N     = 800

model = mujoco.MjModel.from_xml_path(SCENE)
LIMIT = [mink.ConfigurationLimit(model)]


def 建():
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    c = mink.Configuration(model)
    c.update(d.qpos)
    t = mink.FrameTask("hand", "body", position_cost=1.0, orientation_cost=0.0)
    R = mink.SO3.from_matrix(d.body("hand").xmat.reshape(3, 3).copy())
    t.set_target(mink.SE3.from_rotation_and_translation(R, GOAL))
    return d, c, t


def 报(标签, d):
    p = d.body("hand").xpos
    print("  %-18s hand = %s   离目标 %7.1f mm"
          % (标签, np.array2string(p, precision=4), np.linalg.norm(p - GOAL) * 1000))


d, c, t = 建()
print("起点 hand =", np.array2string(d.body("hand").xpos, precision=4))
print("目标      =", np.array2string(GOAL, precision=4))
print()

print("--- 一、solve_ik 返回的到底是什么 ---")
vel = mink.solve_ik(c, [t], DT, "quadprog", limits=LIMIT, damping=1e-3)
print("  一共 %d 个数（= model.nv）" % len(vel))
print("  [0:7]   手臂 7 个关节的速度 =", np.array2string(vel[:7], precision=3))
print("  [7:9]   夹爪 2 根手指的速度 =", np.array2string(vel[7:9], precision=3))
print("  [9:15]  方块自由关节的速度  =", np.array2string(vel[9:15], precision=3))
print("  solve_ik 之后 cfg.q 变了吗？", not np.allclose(c.q, d.qpos))
c.integrate_inplace(vel, DT)
print("  integrate_inplace 之后 cfg.q[:7] =", np.array2string(c.q[:7], precision=4))
print()

print("--- 二、写法 A：ctrl = cfg.q（没有 integrate_inplace）---")
d1, c1, t1 = 建()
for k in range(N):
    d1.ctrl[:7] = c1.q[:7]
    c1.update(d1.qpos)
    mujoco.mj_step(model, d1)
报("ctrl=cfg.q", d1)
print("      ctrl[:7] =", np.array2string(d1.ctrl[:7], precision=4))
print()

print("--- 三、写法 B：solve -> integrate -> ctrl（正确闭环）---")
d2, c2, t2 = 建()
for k in range(N):
    v = mink.solve_ik(c2, [t2], DT, "quadprog", limits=LIMIT, damping=1e-3)
    c2.integrate_inplace(v, DT)
    d2.ctrl[:7] = c2.q[:7]
    mujoco.mj_step(model, d2)
    c2.update(d2.qpos)
报("正确闭环", d2)
print("      ctrl[:7] =", np.array2string(d2.ctrl[:7], precision=4))
