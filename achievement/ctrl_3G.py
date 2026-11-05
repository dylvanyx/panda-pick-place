import mujoco,mujoco.viewer,mink,numpy as np,time

SENCE_PATH = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"
# 加载模型
model = mujoco.MjModel.from_xml_path(SENCE_PATH)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,data,0)
mujoco.mj_forward(model,data)

def reach(p1,p2):
    
    return True

def get_xpos(x,dz=0.1):
    return data.body(x).xpos.copy() + np.array([0,0,dz])

def cube_realtime():
    return get_xpos("cube")
#常量
dt = 0.002
x_home = data.body("hand").xpos.copy()
R_home = mink.SO3(data.body('hand').xquat.copy())

#阶段表（xpos,(0~255),duration）
phase = [(cube_realtime,R_home,0,500),(x_home,R_home,255,500),(x_home,R_home,255,1000),(cube_realtime,R_home,0,500),(x_home,R_home,0,100)]

def make_ik(model,data):    #准备
    cfg = mink.Configuration(model)
    cfg.update_from_keyframe("home")

    task = mink.FrameTask("hand","body",position_cost=1.0,orientation_cost=1.0)     #pos位置权重，ori姿态权重
    limits = [mink.ConfigurationLimit(model)]   #限制   
    return cfg,task,limits
# 接住变量
cfg,task,limits = make_ik(model,data)


def move(p,R,step_size): 
    for i in np.linspace(data.body("hand").xpos,p,step_size):
        cfg.update(data.qpos)
        task.set_target(mink.SE3.from_rotation_and_translation(R, i))
        vel = mink.solve_ik(tasks=[task],solver="quadprog",configuration=cfg,limits=limits,dt=dt)
        cfg.integrate_inplace(vel,dt)
        data.ctrl[:7] = cfg.q[:7]
        mujoco.mj_step(model, data)
        viewer.sync()

with mujoco.viewer.launch_passive(model,data) as viewer:


    grip_last = data.ctrl[7].copy()
   

    for p,R,grip,step_size in phase:
        p = p() if callable(p) else p
        
        wait = 0.0
        move(p,R,step_size)
        viewer.sync()
        if grip != grip_last:
            data.ctrl[7] = grip
            while wait < 0.2:
                mujoco.mj_step(model, data)
                viewer.sync()
                wait += dt
        grip_last = grip
        
    while viewer.is_running():              # 走完别急着关窗
        mujoco.mj_step(model, data)
        viewer.sync()
