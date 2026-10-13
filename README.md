# MuJoCo 机械臂仿真 — Franka Panda 抓取

基于 [MuJoCo](https://mujoco.org/) 与 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 的
Franka Emika Panda 模型，练习**正运动学推导 → 逆运动学求解 → 抓取闭环**的完整链路。

## 环境

| 组件 | 版本 |
|---|---|
| Python | 3.10.21（便携 conda，位于 `../mujoco_env/`） |
| mujoco | 3.13.0 |
| mink | 1.2.0（IK 求解，quadprog 后端） |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| gymnasium | 1.3.0 |

解释器用绝对路径调用（未做 `conda init`）：

```
D:\MuJoCo_Projects\mujoco_env\python.exe
```

## 目录

| 路径 | 说明 |
|---|---|
| `franka_emika_panda/` | Panda 模型本体 + 网格资源（来自 MuJoCo Menagerie） |
| `franka_emika_panda/scene_pick_table.xml` | 自建抓取场景：桌面 + 50 mm 方块 |
| `franka_emika_panda/scene_pick.xml` | 官方场景：含小球 |
| `scene.xml` | 官方场景副本 |
| `run.bat` | 启动器，把场景路径透传给 `1.py` |
| `1.py` | 模型加载 + 可视化窗口实验 |
| `ik_demo.py` | 逆运动学最小示例（`--dry` 可无窗口运行） |
| `grasp_demo.py` | 分步抓取状态机：IK → 到位判定 → 闭合 → 抬升 |
| `two_link.py` | 平面二连杆手写 FK |
| `three_link_2d.xml` | 平面三连杆模型（配合对拍脚本） |
| `fk_check.py` | 手写矩阵 FK ↔ MuJoCo 结果对拍 |
| `three_d_link.py` | 3D 单连杆手写 FK（RPY + 4×4 齐次矩阵） |
| `two_link_3D.py` | 3D 二连杆手写 FK |
| `matrix_calc.py` | 矩阵乘法小工具 |

## 运行

启动可视化窗口：

```
run.bat
```

指定场景：

```
run.bat franka_emika_panda\scene_pick_table.xml
```

命令行运行：

```
D:\MuJoCo_Projects\mujoco_env\python.exe ik_demo.py
D:\MuJoCo_Projects\mujoco_env\python.exe grasp_demo.py
D:\MuJoCo_Projects\mujoco_env\python.exe fk_check.py
```

## 已完成

- **手写正运动学**：4×4 齐次变换矩阵连乘，从平面二/三连杆推广到 3D（RPY）。
  与 MuJoCo 的 `body.xpos` 逐点对拍，最大差值 **1.1e-16**（机器精度）。
- **逆运动学**：`mink` 的 `FrameTask` 约束 `hand` 帧，quadprog 后端 4~5 次迭代收敛，
  耗时约 6 ms，残差约 1e-8 m，关节限位检查无越界。
- **抓取闭环**：桌面 50 mm 方块，IK → 到位判定 → 夹爪闭合 → 抬升，实测抬升 **223.4 mm**。
- **夹爪几何标定**：真实内开口 59 mm（非 80 mm），两指夹取中心相对 `hand` 偏移 58.4 mm。

## 进行中

- Panda 七关节手写 FK 与 MuJoCo 逐段对拍
- gymnasium 风格环境封装（`reset()` / `step()`）
- 为场景添加 `<camera>`，做视觉输入

## 许可

`franka_emika_panda/` 目录取自 MuJoCo Menagerie，采用 Apache-2.0 许可，
版权归原项目所有。本项目对该模型的使用、修改与再分发遵循同一许可。
