"""感知层：俯视相机 + 颜色分割 + 反投影，输出方块中心世界坐标。

原理：颜色负责"在哪里"（mask 质心），先验负责"有多远"（顶面高度恒定）。
所有标定常数在 make_cube_locator 里一次算好，locate() 每次调用只做最少计算。
"""
import mujoco
import numpy as np

# 场景先验（来自 scene_pick_table_cam.xml）
CUBE_CENTER_Z = 0.27   # 方块中心高度（抓取目标 z）
CUBE_TOP_Z = 0.29      # 顶面高度 = 桌面 0.25 + 半高 0.02 * 2


def make_cube_locator(model, cam_name="top_cam", size=256):
    """返回 locate(data) -> np.array([x, y, z])，与控制层 cube_realtime 同接口。"""
    renderer = mujoco.Renderer(model, height=size, width=size)  # 只建一次（GL 上下文贵）
    cam_id = model.camera(cam_name).id

    # 内参：fovy → 像素焦距 f；光心 = 图像中心
    f = (size / 2) / np.tan(np.radians(model.cam_fovy[cam_id]) / 2)
    c = size / 2
    # 外参：固定相机，位置恒定，直接从 model 拿
    cam_x, cam_y, cam_z = model.cam_pos[cam_id]
    d = cam_z - CUBE_TOP_Z   # 相机到顶面的垂直距离（俯视 + 高度恒定 → 常数）

    def locate(data) -> np.ndarray:
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render().astype(int)

        # 色度分割：R 显著大于 G、B 即红。抗光照（亮面粉红、暗面深红都满足）
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = (R - np.maximum(G, B) > 40) & (R > 80)

        v, u = np.nonzero(mask)
        if len(u) < 100:     # 完整顶面约 1600px，过少说明被遮挡或出视野
            raise RuntimeError(f"方块 mask 仅 {len(u)} 像素，检查遮挡/视野/阈值")

        # 反投影（俯视简化版）：像素偏移 * d / f = 物理偏移；v 轴朝下，y 取负
        x = cam_x + (u.mean() - c) * d / f
        y = cam_y - (v.mean() - c) * d / f
        return np.array([x, y, CUBE_CENTER_Z])

    return locate
