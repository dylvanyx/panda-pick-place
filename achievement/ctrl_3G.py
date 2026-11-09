from typing import NamedTuple
import mujoco,mujoco.viewer,mink,numpy as np,time

SENCE_PATH = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"
# 加载模型
model = mujoco.MjModel.from_xml_path(SENCE_PATH)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,data,0)
mujoco.mj_forward(model,data)

class Phase(NamedTuple): #位置target 夹爪grip 等待wait 姿态R
    target: object
    grip: int
    wait: float = 0.0
    R:object = None

def hold(sec, grip=None):
    if grip is not None:
        data.ctrl[7] = grip
    for _ in range(max(1, int(sec / dt))):
        mujoco.mj_step(model, data)
        viewer.sync()

def reach(p1,p2,tol=0.1):
    return False if np.linalg.norm(p1-p2) < tol else True

# 获取手和方块的 body ID (或者 geom ID)
def is_grasped(hand_id = model.body("hand").id,cube_id = model.body("cube").id):
    # 遍历当前所有的接触
    for i in range(data.ncon):
        c = data.contact[i]
        # 获取发生接触的两个物体的 body ID
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        
        # 只要检测到 hand 和 cube 碰在一起了，直接返回 True
        if (b1 == hand_id and b2 == cube_id) or (b1 == cube_id and b2 == hand_id):
            return True
    return False
    

def get_xpos(x,dz=0.1):
    return data.body(x).xpos.copy() + np.array([0,0,dz])

def cube_realtime():
    return get_xpos("cube")
#常量
dt = 0.002 # s
v = 0.2 # m/s
x_home = data.body("hand").xpos.copy()
R_home = mink.SO3(data.body('hand').xquat.copy())
GRIPTIME = 0.2
FALLDOWN = 1.5
#阶段表（xpos,(0~255),wait,R）
phase = [Phase(cube_realtime,0,GRIPTIME),Phase(x_home,255,FALLDOWN),Phase(cube_realtime,0,GRIPTIME),Phase(x_home,0),Phase(x_home,255,FALLDOWN),Phase(x_home,0,GRIPTIME)]

def make_ik(model,data):    #准备
    cfg = mink.Configuration(model)
    cfg.update_from_keyframe("home")

    task = mink.FrameTask("hand","body",position_cost=1.0,orientation_cost=1.0)     #pos位置权重，ori姿态权重
    limits = [mink.ConfigurationLimit(model)]   #限制   
    return cfg,task,limits
# 接住变量
cfg,task,limits = make_ik(model,data)


def move(p,R,step_size=500): 
    for i in np.linspace(data.body("hand").xpos,p,step_size):
        cfg.update(data.qpos)
        task.set_target(mink.SE3.from_rotation_and_translation(R, i))
        vel = mink.solve_ik(tasks=[task],solver="quadprog",configuration=cfg,limits=limits,dt=dt)
        cfg.integrate_inplace(vel,dt)
        data.ctrl[:7] = cfg.q[:7]
        mujoco.mj_step(model, data)
        viewer.sync()

with mujoco.viewer.launch_passive(model,data) as viewer:




    for p,grip,wait,R in phase:
        p = p() if callable(p) else p
        R = R if R is not None else R_home  
        smart_size = int(np.linalg.norm(data.body("hand").xpos-p)/(v*dt)) 
        move(p,R,smart_size)
        mujoco.mj_step(model, data)
        hold(wait,grip) #开到闭
        mujoco.mj_step(model, data)
        viewer.sync()
        
        
    while viewer.is_running():              # 走完别急着关窗
        mujoco.mj_step(model, data)
        viewer.sync()
