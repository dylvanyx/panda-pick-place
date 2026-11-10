from typing import NamedTuple
import os,mujoco,mujoco.viewer,mink,numpy as np,time
from perc_1G import make_cube_locator

SENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..","franka_emika_panda","scene_pick_table_cam.xml")
# 加载模型
model = mujoco.MjModel.from_xml_path(SENCE_PATH)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,data,0)
mujoco.mj_forward(model,data)

np.random.seed(67)

locate = make_cube_locator(model)   # 感知层：locate(data) -> 方块中心世界坐标


def place_cube(pos):
    mujoco.mj_resetDataKeyframe(model,data,0)
    jid = model.joint("cube_free").id
    q = model.jnt_qposadr[jid]
    data.qpos[q:q+3] = pos
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
    


#常量
dt = 0.002 # s
v = 0.2 # m/s
x_home = data.body("hand").xpos.copy()
R_home = mink.SO3(data.body('hand').xquat.copy())
x_obs = np.array([0.25, 0, 0.45])
GRIPTIME = 0.2
FALLDOWN = 1.5

def prepare(model,data):    #准备
    cfg = mink.Configuration(model)
    cfg.update_from_keyframe("home")

    task = mink.FrameTask("hand","body",position_cost=1.0,orientation_cost=1.0)     #pos位置权重，ori姿态权重
    limits = [mink.ConfigurationLimit(model)]   #限制   
    return cfg,task,limits
# 接住变量
cfg,task,limits = prepare(model,data)


def move(p,R,step_size=500): 
    for i in np.linspace(data.body("hand").xpos,p,step_size):
        cfg.update(data.qpos)
        task.set_target(mink.SE3.from_rotation_and_translation(R, i))
        vel = mink.solve_ik(tasks=[task],solver="quadprog",configuration=cfg,limits=limits,dt=dt)
        cfg.integrate_inplace(vel,dt)
        data.ctrl[:7] = cfg.q[:7]
        mujoco.mj_step(model, data)
        viewer.sync()


if __name__ == "__main__":
    
    with mujoco.viewer.launch_passive(model,data) as viewer:
        for _ in range(3):
            x_rand = np.random.uniform(0.35,0.65)
            y_rand = np.random.uniform(-0.15, 0.15)
            z_rand = 0.27
            pos = np.array([x_rand,y_rand,z_rand])
            place_cube(pos)
            
            move(x_obs, R_home, 300)
            
            x_cube = locate(data) + np.array([0,0,0.1])  # 感知定位；+0.1 保持悬停语义
            x_high = x_cube + np.array([0,0,0.15])       # 抓取点正上方，垂直下降的入口
            print("真值", pos[:2].round(3), "估计", x_cube[:2].round(3))

            #阶段表每回合重建：target 是本回合拍到的方块位置（xpos,(0~255),wait,R）
            #先高位悬停再垂直下降：防止张开的手指横扫时把方块推开
            phase = [Phase(x_high,255),Phase(x_cube,0,GRIPTIME),Phase(x_home,255,FALLDOWN)]
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
