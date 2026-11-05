# robot.py —— 机器人控制模块
# 这个文件只负责一件事：让机械臂「怎么动」。它不知道你要抓什么、先做什么。
#
# 缩写对照（第一次出现都展开一次，往后你就认得它们了）：
#   MjModel / MjData   MuJoCo 的「模型」（常量，从 XML 编译出来，从头到尾不改）
#                      和「数据」（会变的状态：关节角、速度、位置）
#   ctrl               control，发给驱动器的目标量。data.ctrl[0:7] 是 7 个关节角，
#                      data.ctrl[7] 是夹爪（0 = 全闭，255 = 全开）
#   qpos               joint position，7 个关节的「实际」角度（不是目标）
#   dt                 delta time，一次迭代对应的时间长度，单位秒
#   IK                 Inverse Kinematics，逆运动学：给定手的位姿，反推关节角
#   SE3                Special Euclidean group 3，三维的「旋转 + 平移」，4x4 矩阵
#   SO3                Special Orthogonal group 3，三维的「只有旋转」，3x3 矩阵
#   cfg                configuration，mink 用来存「当前关节角」的容器
#   vel                velocity，mink 解出来的关节速度（7 个数，单位 rad/s）

import mujoco
import mink
import numpy as np


class Robot:
    """把「加载模型 / 解 IK / 步进仿真 / 驱动夹爪」包成一个对象。"""

    def __init__(self, scene_path):
        # 1) 加载场景。整个程序只会有一个 model，永远。
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # 2) 回到存档点（XML 里的 <key name="home">）。
        #    不做这一步，qpos 全 0，机械臂是躺平的，后面全部坐标系都对不上。
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # 3) 算一遍正运动学，把 xpos / xquat 这些「算出来的位置」填上。
        #    不做这一步，读到的 body 位置是上一帧的残值（刚 reset 完是 0）。
        mujoco.mj_forward(self.model, self.data)

        self.dt = 0.002                 # 每次 IK 迭代对应 2 毫秒
        self.viewer = None              # 还没开窗口，attach_viewer() 时才填

        # 4) mink 的三件套。cfg 是「影子机器人」，专门给 IK 用。
        self.cfg = mink.Configuration(self.model)
        self.cfg.update_from_keyframe("home")

        # task：告诉 IK「我要让哪个 body 追哪个位姿」
        #   position_cost=1.0      位置误差的权重
        #   orientation_cost=1.0   姿态误差的权重
        # 两个都是 1.0 就是「位置和姿态一样重要」。
        self.task = mink.FrameTask("hand", "body",
                                   position_cost=1.0, orientation_cost=1.0)

        # limits：关节限位。去掉它，IK 会给出 joint4 转 +200 度这种答案，
        # 伺服会一路顶到限位卡死。
        self.limits = [mink.ConfigurationLimit(self.model)]

    # ------------------------------------------------------------------
    # 下面这几个是「读状态」的小工具，避免你在主文件里到处写 data.body(...)
    # ------------------------------------------------------------------

    def attach_viewer(self, viewer):
        """把已经开好的窗口交给机器人，之后每步自动刷新画面。"""
        self.viewer = viewer

    def _step(self):
        """推进一步物理仿真，顺手刷一下窗口（如果开了的话）。"""
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def hand_pos(self):
        """返回手（hand 这个 body）当前在世界坐标系里的位置，3 个数，单位米。"""
        return self.data.body("hand").xpos.copy()

    def hand_rot(self):
        """返回手当前的姿态，SO3 对象（内部是 3x3 旋转矩阵）。

        .copy() 不能省：xquat 是 MuJoCo 内部数组的视图，
        仿真一跑它就变，你手里那个就跟着变了。
        """
        return mink.SO3(self.data.body("hand").xquat.copy())

    def obj_pos(self, name, dz=0.0):
        """返回某个物体（比如 'cube'）的位置，可以顺手抬高 dz 米。"""
        return self.data.body(name).xpos.copy() + np.array([0.0, 0.0, dz])

    # ------------------------------------------------------------------
    # 下面两个是「动」的核心
    # ------------------------------------------------------------------

    def move(self, p, R=None, steps=500):
        """让手沿直线走到 p，同时把姿态从当前转到 R。

        p      目标位置，3 个数
        R      目标姿态（SO3）。给 None 就保持当前朝向不变
        steps  走多少步。步数越多越慢，这就是控制速度的唯一旋钮
        """
        p0 = self.hand_pos()            # 起点位置
        R0 = self.hand_rot()            # 起点姿态
        if R is None:
            R = R0
        p = np.asarray(p, dtype=float)

        N = max(1, steps - 1)

        # np.linspace 把起点到终点切成 steps 个点，每个点都解一次 IK。
        # 不切段、只解一次的话，手会沿一条弧线甩过去，不是直线。
        for k, mid in enumerate(np.linspace(p0, p, steps)):

            # 当前这一步对应的时间进度，0 到 1
            t = k / N

            # 影子机器人先跟上真机器人的实际关节角。
            # 漏了这句，cfg 永远停在初始姿态，IK 算出来的增量全是错的。
            self.cfg.update(self.data.qpos)

            # 姿态也按同样的 t 插值。
            # 为什么必须插：如果 R 和 R0 差 90 度，第 1 步就让 IK 追 90 度，
            # 它会要求 400 度/秒的瞬转，夹着的方块当场被甩飞。
            # 这里的 interpolate 就是球面插值，等价于常说的 slerp。
            pose = mink.SE3.from_rotation_and_translation(R0.interpolate(R, t), mid)
            self.task.set_target(pose)

            # 解一次逆运动学，拿到的不是角度，是「关节速度」
            vel = mink.solve_ik(tasks=[self.task], solver="quadprog",
                                configuration=self.cfg, limits=self.limits,
                                dt=self.dt)

            # 速度乘时间，累加进影子机器人的关节角里
            self.cfg.integrate_inplace(vel, self.dt)

            # 把影子机器人的关节角抄给真正的驱动器（只抄 7 个臂关节，
            # 第 8 个是夹爪，由 grip() 管，别在这里覆盖）
            self.data.ctrl[:7] = self.cfg.q[:7]

            # 推进一步仿真
            self._step()

    def refine(self, p, R=None, tol=0.003, max_steps=800):
        """到位精修：原地把同一个目标反复追，直到误差小于 tol。

        move() 走完其实还差 10~30 毫米——最后一步只是「设了个新目标 + 走一步」，
        伺服有惯性，追不上。想抓准就必须补这一步。
        """
        p = np.asarray(p, dtype=float)
        if R is None:
            R = self.hand_rot()
        self.task.set_target(mink.SE3.from_rotation_and_translation(R, p))

        for _ in range(max_steps):
            if np.linalg.norm(self.hand_pos() - p) < tol:
                break
            self.cfg.update(self.data.qpos)
            vel = mink.solve_ik(tasks=[self.task], solver="quadprog",
                                configuration=self.cfg, limits=self.limits,
                                dt=self.dt)
            self.cfg.integrate_inplace(vel, self.dt)
            self.data.ctrl[:7] = self.cfg.q[:7]
            self._step()

    def grip(self, value, hold=0.2):
        """驱动夹爪。0 = 全闭（0mm），255 = 全开（80mm）。

        hold 是夹紧后原地等多久（秒）。夹爪自己也要时间走到位，
        设完就立刻抬手，等于没夹上。
        """
        self.data.ctrl[7] = value
        for _ in range(int(hold / self.dt)):
            self._step()

    def hold(self, seconds, n=1):
        """什么都不动，原地等一会儿（让画面/伺服稳定下来）。"""
        for _ in range(int(seconds / self.dt)):
            self._step()
