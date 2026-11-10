"""离屏录制演示 GIF。

不修改 ctrl_3G_random.py：它在 `with mujoco.viewer.launch_passive(...)` 里跑主流程，
这里把 launch_passive 替换成一个"假 viewer"，它的 sync() 就是截一帧。

用法：python achievement/record_demo.py
输出：docs/demo.gif
"""
import os, sys
import mujoco, mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "demo.gif")

FPS = 12            # 输出帧率
MAX_FRAMES = 120    # 最多采集帧数（约 10 秒），录满后仿真继续但不采集
W, H = 400, 300     # 输出分辨率；想更清晰可改 640x480，但 GIF 会变大
COLORS = 96         # 调色板颜色数，越小文件越小

_recorder = None


class Recorder:
    """冒充 passive viewer：提供 sync() / is_running()，并顺手渲染存帧。"""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.renderer = mujoco.Renderer(model, height=H, width=W)
        self.dt_frame = 1.0 / FPS
        self.next_t = 0.0
        self.frames = []

        # 第三人称视角：同时框住机械臂、桌面和方块
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.42, 0.0, 0.32]
        self.cam.distance = 1.45
        self.cam.azimuth = 135.0
        self.cam.elevation = -20.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def is_running(self):
        return False     # 让主循环末尾的 while 立即退出，脚本自然结束

    def sync(self):
        if len(self.frames) >= MAX_FRAMES or self.data.time < self.next_t:
            return
        self.next_t += self.dt_frame
        self.renderer.update_scene(self.data, camera=self.cam)
        self.frames.append(self.renderer.render().copy())


def fake_launch_passive(model, data, **kwargs):
    global _recorder
    _recorder = Recorder(model, data)
    return _recorder


if __name__ == "__main__":
    mujoco.viewer.launch_passive = fake_launch_passive

    sys.path.insert(0, HERE)
    import runpy
    runpy.run_path(os.path.join(HERE, "ctrl_3G_random.py"), run_name="__main__")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    from PIL import Image
    imgs = [Image.fromarray(f).quantize(colors=COLORS) for f in _recorder.frames]
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / FPS), loop=0, optimize=True, disposal=2)
    mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"\n[录屏] {len(imgs)} 帧 -> {os.path.abspath(OUT)}  ({mb:.1f} MB)")
    if mb > 3:
        print("[提示] 超过 3MB，可调小 W,H 或 COLORS 后重跑")
