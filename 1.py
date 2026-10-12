import mujoco, mujoco.viewer,math
import numpy as np
import sys

from grasp_demo import 

SCENE = sys.argv[1]
model = mujoco.MjModel.from_xml_path(SCENE)                    # 1 图纸 → 结构
data  = mujoco.MjData(model)                                   # 2 分配状态
                  # 3 摆到 home 姿态
print(model)
with mujoco.viewer.launch_passive(model, data) as viewer:      # 4 开窗口
    print(model.key_ctrl[0])
    while viewer.is_running():
        data.ctrl[:] = np.array([0.3, -0.2, 0.1, -1.5, 0.4, 1.2, -0.6, 200.0])                                # 先打底
        data.ctrl[0] = model.key_ctrl[0][0] + 0.5 * math.sin(2 * math.pi * 0.3 * data.time)    
        data.ctrl[1] = model.key_ctrl[0][0] + 0.5 * math.sin(2 * math.pi * 0.3 * data.time)  # 5 写控制 ← 见下面坑 2
        mujoco.mj_step(model, data)                            # 6 推进物理
        viewer.sync()                                          # 7 刷新画面
