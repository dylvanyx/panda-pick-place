from operator import pos
import mujoco, mujoco.viewer,math
import numpy as np,mink
import sys
from mink import solve_ik
from 封装函数.package import make_ik
from 练手02_用网络 import 网络IK

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"
model, data, cfg, task, limits, dt = make_ik(SCENE)

cube = data.body("cube").xpos.copy()
R = mink.SO3(data.body('hand').xquat.copy())    # xquat 也是 wxyz，可以直接喂
pi = np.pi
ang_start = model.key_qpos[0][:9]

pos_start = data.body("hand").xpos.copy()
goal = cube + np.array([0,0,0.1])

def move_liner(start,end,steplong,R):
    for mid in np.linspace(start,end,steplong):
        task.set_target(mink.SE3.from_rotation_and_translation(R,mid))
        vel = solve_ik(tasks=[task],dt=dt,solver="quadprog",configuration=cfg,limits=limits)   
        cfg.integrate_inplace(vel, dt)
        data.ctrl[:7] = cfg.q[:7]
        mujoco.mj_step(model, data)  # 推进物理仿真
        cfg.update(data.qpos)
        viewer.sync() 

def rotate_global(R0,x,y,z,t):
    R1 = mink.SO3.from_z_radians(np.radians(z)) @ mink.SO3.from_y_radians(np.radians(y)) @ mink.SO3.from_x_radians(np.radians(x)) @ R0
    N = int(t / dt)
    p = data.body("hand").xpos.copy()
    for i in range(1,N+1):
        R = R0.interpolate(R1, i / N)
        task.set_target(mink.SE3.from_rotation_and_translation(R, p))
        vel = solve_ik(tasks=[task],dt=dt,solver="quadprog",configuration=cfg,limits=limits)   
        cfg.integrate_inplace(vel, dt)
        data.ctrl[:7] = cfg.q[:7]
        mujoco.mj_step(model, data)  # 推进物理仿真
        cfg.update(data.qpos)
        viewer.sync() 

def move_liner2(目标位置, 秒=1.5, viewer=None):
    q = 网络IK(目标位置)              # 一次前向，算出 7 个关节角
    for _ in range(int(秒 / dt)):
        data.ctrl[:7] = q
        mujoco.mj_step(model, data)
        viewer.sync()
    cfg.update(data.qpos)              # ← 这句不能漏，见下

print(model)
print(cfg)
print(task)
print(limits)
print(dt)

print(f"物体位置 {np.round(cube, 4)}")
print(f"末端起点 {np.round(data.body('hand').xpos, 4)}")

with mujoco.viewer.launch_passive(model, data) as viewer: 
    # 打开夹爪
    data.ctrl[7] = np.array([255])
    # 移动到目标位置
    move_liner2(goal, viewer=viewer)
    # 关闭夹爪
    data.ctrl[7] = np.array([0])
    rotate_global(R,0,0,90,2)
    #move_liner(pos_start,pos_start,500,R)
    
    viewer.sync()                # 同步刷新画面       
    while viewer.is_running():                  # 走完别急着关窗
        mujoco.mj_step(model, data)
        viewer.sync()