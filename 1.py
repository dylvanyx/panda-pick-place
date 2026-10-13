import mujoco, mujoco.viewer,math
import numpy as np
import sys

#SCENE = sys.argv[1]
SCENE = "franka_emika_panda\scene_pick_table.xml"
model = mujoco.MjModel.from_xml_path(SCENE)                    # 1 图纸 → 结构
data  = mujoco.MjData(model)                                   # 2 分配状态
                  # 3 摆到 home 姿态
print(model)
with mujoco.viewer.launch_passive(model, data) as viewer:      # 4 开窗口
    print(model.key_ctrl[0])
    
     # 5 写控制 ← 见下面坑 2
    mujoco.mj_step(model, data)                            # 6 推进物理
    viewer.sync()                                          # 7 刷新画面
