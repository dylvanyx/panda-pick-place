# 导入两个核心库：
#   mujoco —— 物理仿真引擎，负责模型加载与动力学
#   mink   —— 基于 MuJoCo 的逆运动学(IK)求解库
import mujoco, mink


def make_ik(scene_path, keyframe="home"):
    # 作用：一次性把做 IK 需要的所有对象都准备好并返回
    # 参数：
    #   scene_path —— MJCF/XML 场景文件的路径
    #   keyframe   —— 初始姿态的名字，默认 "home"（对应 XML 里 <key> 定义的姿态）

    # 从 XML 文件加载模型（描述机器人的结构、关节、几何等静态信息）
    model  = mujoco.MjModel.from_xml_path(scene_path)
    # 创建 data（运行时的动态数据：当前关节角、速度、传感器读数等）
    data   = mujoco.MjData(model)
    # 用 model 构造一个 mink 配置对象，用于求解 IK
    cfg    = mink.Configuration(model)
    # 把配置的初始状态设置为指定的 keyframe（即机器人的起始姿态）
    cfg.update_from_keyframe(keyframe)
    # 定义一个任务：让 "hand" 这个刚体去对齐 "body" 坐标系
    #   position_cost=1.0     —— 位置误差权重（越大越优先满足位置）
    #   orientation_cost=0.0  —— 姿态误差权重（0 表示不关心朝向）
    task   = mink.FrameTask("hand", "body", position_cost=1.0, orientation_cost=1.0)
    mujoco.mj_resetData(model, data)            # 方块回到 XML 写的 (0.5, 0, 0.27)
    data.qpos[:9] = model.key_qpos[0][:9]       # 机器人摆成 home
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)              # 读坐标/做 IK 之前必须算这一遍
    # 约束：把关节限制在 model 定义的运动范围内（防止越界）
    limits = [mink.ConfigurationLimit(model)]

    # 仿真步长(单位:秒)，取自模型设置，后面用于时间积分
    dt     = float(model.opt.timestep)

    # 返回所有需要的对象，供调用方使用
    return model, data, cfg, task, limits, dt
