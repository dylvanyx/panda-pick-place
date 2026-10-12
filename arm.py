import mujoco, mujoco.viewer, time

ARM = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="3 3 0.1" rgba="0.3 0.3 0.35 1"/>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.08 0.08" pos="0 0 0.08" rgba="0.55 0.55 0.6 1"/>
      <body name="link1" pos="0 0 0.16">
        <joint name="joint1" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.45 0 0" size="0.035" rgba="0.2 0.7 0.9 1"/>
        <body name="link2" pos="0.45 0 0">
          <joint name="joint2" type="hinge" axis="0 1 0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03" rgba="0.9 0.6 0.2 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="20"/>
    <motor name="motor2" joint="joint2" gear="20"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(ARM)
data = mujoco.MjData(model)
print("关节数 =", model.nq, "执行器 =", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)   # ← 少了这行，机械臂会以几千步/秒疯掉
