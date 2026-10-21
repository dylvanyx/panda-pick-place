"""让 Franka Panda 的手臂把头移动到指定位置 —— 完整、能跑的那一份。

跑法（改 GOAL 那三个数就能去别的地方）：
    D:\\MuJoCo_Projects\\mujoco_env\\python.exe D:\\MuJoCo_Projects\\test\\完整示例.py

=========================================================================
整条线只有六个动作，从上往下读：

    1. 装机器人         model / data
    2. 摆好初始姿势      reset + forward
    3. 告诉它目标        task.set_target
    4. 算关节速度        solve_ik
    5. 积成关节角度      integrate_inplace
    6. 喂给它 + 让它走   ctrl，然后 mj_step
       └ 走完把真实姿势读回来，回到第 4 步重算（因为它一次到不了位）

底下"求解器"和"机器人"是两套各自独立的家伙，各自记着一份关节角度。
所以每一轮都要用 cfg.update 把机器人的真实状态抄回给求解器。
=========================================================================
"""

import numpy as np
import mujoco
import mujoco.viewer
import mink

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"
GOAL  = np.array([0.55, 0.0, 0.35])       # ← 想让它去哪就改这里（米，世界坐标 x y z）

# ← 目标姿态的四元数 [w, x, y, z]，w 在最前面。填 None = 保持起点时的朝向
# ⚠ 姿态别乱填：填一个胳膊够不到的姿态，位置也会被拖着一起到不了
QUAT  = None      # 例：[0.0, 0.70710678, 0.70710678, 0.0] 就是起点这个朝下姿态的写法


# ══════════════════════════════════════════════════════════════════
# 一、准备：把机器人和求解器都装好
# ══════════════════════════════════════════════════════════════════

model = mujoco.MjModel.from_xml_path(SCENE)     # 读 XML，造出机器人（结构、关节、限位都在里面）
data  = mujoco.MjData(model)                    # 机器人的运行数据：现在在哪、速度多少

mujoco.mj_resetDataKeyframe(model, data, 0)     # 把机器人摆成 home 那个存档姿势
mujoco.mj_forward(model, data)                  # 算一遍：它现在每个零件都在哪（不调的话读出来全是 0）

cfg = mink.Configuration(model)                 # 求解器：负责算"关节该转多少"
cfg.update(data.qpos)                           # 让求解器和机器人站在同一个姿势上（这一步必须做）

task = mink.FrameTask("hand", "body",
                      position_cost=1.0,        # 盯住模型里叫 hand 的那个 body
                      orientation_cost=0.0)     # 位置要准；朝向不管它，随缘
# ⚠ orientation_cost 是 0.0 的时候，下面 QUAT 填什么都不会有任何效果（它压根不管朝向）。
#   想让 QUAT 真的起作用，把上面那个 0.0 改成 1.0。

limit = [mink.ConfigurationLimit(model)]        # 关节限位：别让关节转出它能转的范围
DT    = 0.01                                    # 每一轮打算用多久走完（秒）


# ══════════════════════════════════════════════════════════════════
# 二、告诉它目标
# ══════════════════════════════════════════════════════════════════

# 目标姿态：QUAT 填了就用它，没填就沿用起点时的朝向
if QUAT is None:
    R_goal = mink.SO3.from_matrix(cfg.data.body("hand").xmat.reshape(3, 3).copy())
else:
    q = np.array(QUAT, dtype=float)
    R_goal = mink.SO3(q / np.linalg.norm(q))   # 四元数必须是单位长度，先除一下模长最保险

# set_target 只收一个参数：把「姿态 + 位置」打包成的 SE3
task.set_target(mink.SE3.from_rotation_and_translation(R_goal, GOAL))

# 等价的偷懒写法：一口气给 7 个数，前 4 个是四元数，后 3 个是位置
# task.set_target(mink.SE3(np.array([0.70710678, 0.0, 0.0, 0.70710678, 0.55, 0.0, 0.35])))

print("起点     =", np.round(cfg.data.body("hand").xpos, 4))
print("目标     =", np.round(GOAL, 4))
print("（关掉窗口就退出）")


# ══════════════════════════════════════════════════════════════════
# 三、主循环：反复"看现在在哪 → 算要转多少 → 让它转"
# ══════════════════════════════════════════════════════════════════

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # ① 算：按当前姿势，各关节的速度应该是多少（注意给的是"速度"，不是角度）
        vel = mink.solve_ik(cfg, [task], DT, "quadprog", limits=limit, damping=1e-3)

        # ② 把速度积成"新的关节角度"（少了这步，拿到的就是半成品）
        cfg.integrate_inplace(vel, DT)

        # ③ 把新的关节角度交给机器人（它的输入口只收关节角度，不收位置也不收坐标）
        data.ctrl[:7] = cfg.q[:7]

        # ④ 让它按这个命令往前走一小步
        mujoco.mj_step(model, data)

        # ⑤ 它真正走到了哪？读回来交给求解器（不写这步，下一轮算的时候用的还是旧姿势）
        cfg.update(data.qpos)

        # ⑥ 刷新窗口画面
        viewer.sync()

    print()
    print("最终位置 =", np.round(data.body("hand").xpos, 4))
    print("目标位置 =", np.round(GOAL, 4))
    print("差       = %.1f mm" % (np.linalg.norm(data.body("hand").xpos - GOAL) * 1000))
