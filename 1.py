import mujoco, mujoco.viewer,math
import numpy as np,mink
import sys
from mink import solve_ik
from 封装函数.package import make_ik

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"
model, data, cfg, task, limits, dt = make_ik(SCENE)

cube = data.body("cube").xpos.copy()
goal = cube + np.array([0,0,0.2])
R = mink.SO3(np.array([1, 1, 1, 1]))  
pi = np.pi



print(model)
print(cfg)
print(task)
print(limits)
print(dt)

print(f"物体位置 {np.round(cube, 4)}")
print(f"末端起点 {np.round(data.body('hand').xpos, 4)}")

with mujoco.viewer.launch_passive(model, data) as viewer: 
    while viewer.is_running():
        mujoco.mj_step(model, data)  # 推进物理仿真
        viewer.sync()                # 同步刷新画面
        task.set_target(mink.SE3.from_rotation_and_translation(R,goal))
        vel = solve_ik(tasks=[task],dt=dt,solver="quadprog",configuration=cfg,limits=limits)   
        cfg.integrate_inplace(vel, dt) 
        data.ctrl[:6] = cfg.q[:6]
        data.ctrl[6] = np.array([255])
        cfg.update(data.qpos)
    mujoco.mj_step(model, data)                           
    viewer.sync()                                         