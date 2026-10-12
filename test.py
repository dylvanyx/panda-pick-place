import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path(r"D:\MuJoCo_Projects\test\franka_emika_panda\scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 运行 5 秒的仿真
    while viewer.is_running():
        mujoco.mj_step(model, data)  # 推进物理仿真
        viewer.sync()                # 同步刷新画面

print("测试完成！您的 MuJoCo 环境已完全就绪。")