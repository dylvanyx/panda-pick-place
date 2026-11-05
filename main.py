# main.py —— 主文件（程序入口）
# 放在 D:\MuJoCo_Projects\test\ 下，和 robot.py 同一个目录。
#
# 运行方式（在 D:\MuJoCo_Projects 目录下）：
#   .\mujoco_env\python.exe test\main.py
#
# 这个文件只负责「先做什么、后做什么」，具体怎么动全部交给 robot.py。
# 以后加相机、加随机落点、加成功率统计，都往这里加，robot.py 基本不用动。

import mujoco.viewer

from robot import Robot

# 场景文件。改场景只改这一行。
SCENE_PATH = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"

# 创建机器人对象：这一步就把 model / data / IK 全部准备好了
r = Robot(SCENE_PATH)

# 记下 home 的位置和姿态，作为「回到原位」的目标
HOME = r.hand_pos()
R_HOME = r.hand_rot()

# ----------------------------------------------------------------------
# 开窗口。launch_passive 只负责「显示」和「响应鼠标拖拽视角」，
# 它不会自己推进仿真——推进一步是 r._step() 干的，也就是 r.move() 里每步做的。
# ----------------------------------------------------------------------
with mujoco.viewer.launch_passive(r.model, r.data) as viewer:

    r.attach_viewer(viewer)     # 把窗口交给机器人，之后每步自动刷新画面

    # 每个阶段写成一行：(去哪, 什么朝向, 夹爪, 走几步)
    #   去哪     —— 一个函数，运行时才求值。写成函数是因为「方块跟着手走」，
    #               必须到那一刻再读方块坐标，现在读的是旧值。
    #   夹爪     —— 0 全闭 / 255 全开
    #   走几步   —— 越大越慢，500 步约 1 秒
    phase = [
        (lambda: r.obj_pos("cube", dz=0.10), R_HOME, 0,   500),   # 1. 去方块上方并闭合
        (lambda: HOME,                      R_HOME, 255, 500),   # 2. 回原位，张开手（放下）
        (lambda: HOME,                      R_HOME, 255, 1000),  # 3. 原地等，让方块落稳
        (lambda: r.obj_pos("cube", dz=0.10), R_HOME, 0,   500),   # 4. 再下去抓
        (lambda: HOME,                      R_HOME, 0,   100),   # 5. 抬起
    ]

    grip_last = r.data.ctrl[7]

    for target, R, grip, steps in phase:

        # 拿到这一阶段的目标位置（lambda 在这里才被调用）
        p = target()

        # 走过去
        r.move(p, R, steps=steps)

        # 走完再原地补一下，消除 10~30 毫米的残余误差
        r.refine(p, R)

        # 夹爪值变了才动手，避免每阶段都重新发一遍
        if grip != grip_last:
            r.grip(grip, hold=0.2)
            grip_last = grip

    # 流程走完别急着关窗，否则你根本看不到最后停在哪
    while viewer.is_running():
        r._step()
