"""
抓取成功率统计：方块落到不同位置，每次都从 home 重来，统计多少次抓起来了。

这就是招聘 JD 里"成功率 >80%"那个数字的来源。没有这个数，你的项目在简历上
只是一句"我实现了抓取"，有了这个数才是"在 XX 条件下成功率 XX%"。

用法：
    python 抓取成功率.py            # 跑 20 次，范围 ±5cm
    python 抓取成功率.py 50 0.08    # 跑 50 次，范围 ±8cm

判成功的标准（实测最可靠的）：
    抓起来之后，方块在 hand 局部坐标里的位置，和刚夹住时相比漂移 < 12mm。
    不要用"抬多高"判断，也不要用"接触数"判断——前者翻转后会误判，后者一直在抖。
"""
import sys
import time

import numpy as np
import mujoco
import mink

import 抓取举升 as G      # 复用它的 移动() / 转腕() / R抓取

model, data, cfg, dt = G.model, G.data, G.cfg, G.dt
task, limits = G.task, G.limits

HOME9 = model.key_qpos[0][:9].copy()
方块初始z = 0.27
夹紧等待 = 0.4
举升高度 = 0.25
成功阈值 = 0.012          # 12 mm
精修容差 = 0.003          # 3 mm


def 精修(位置, 姿态, 容差=精修容差, 最多=800):
    """
    停在目标点，让伺服把最后那点滞后补上。

    为什么必须有这一步：移动() 的最后一步只是"把目标设成终点、解一次 IK、走一步"，
    但伺服是有惯性的，目标停下时手还在后面追。实测追不上的量是 10~28mm，
    而夹爪的有效余量只有 9mm（开口 59mm - 方块 50mm），偏心超过 12mm 就夹在角上。
    所以"走完"不等于"到了"，得再给它时间把误差吃掉。

    返回用掉的步数。跑到 最多 还没进容差 = 这个点机械臂够不到。
    """
    task.set_target(mink.SE3.from_rotation_and_translation(姿态, 位置))
    for n in range(1, 最多 + 1):
        vel = mink.solve_ik(tasks=[task], dt=dt, solver="quadprog",
                            configuration=cfg, limits=limits, damping=1e-3)
        cfg.integrate_inplace(vel, dt)
        data.ctrl[:7] = cfg.q[:7]
        mujoco.mj_step(model, data)
        cfg.update(data.qpos)
        if np.linalg.norm(data.body("hand").xpos - 位置) < 容差:
            return n
    return 最多


def 复位(rng, xy范围):
    """把机器人摆回 home，方块随机挪到新位置。返回方块新位置。"""
    mujoco.mj_resetData(model, data)
    data.qpos[:9] = HOME9

    # 方块 free joint 在 qpos[9:16]：xyz + 四元数(wxyz)
    data.qpos[9] = 0.5 + rng.uniform(-xy范围, xy范围)
    data.qpos[10] = rng.uniform(-xy范围, xy范围)
    data.qpos[11] = 方块初始z
    data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]        # 不转，保持轴对齐
    mujoco.mj_forward(model, data)

    # 关键：手臂的 ctrl 必须等于当前关节角，不能写 0
    # 写 0 等于命令七个关节全部转到零位，一 reset 手臂就往零位暴冲
    data.ctrl[:7] = data.qpos[:7]
    data.ctrl[7] = 255.0                            # 夹爪张开 80mm

    cfg.update(data.qpos)                           # IK 的起点必须和仿真一致
    return data.body("cube").xpos.copy()


def 抓一次(rng, xy范围):
    """跑一整套抓取流程，返回 (漂移mm, 抬起mm, 是否成功)"""
    位置 = 复位(rng, xy范围)

    # 1. 张开夹爪，走到方块上方 10cm（姿态全程锁定为 home 朝向）
    目标 = 位置 + np.array([0, 0, 0.1])
    G.移动(目标, G.R抓取, 速度=0.25)
    精修(目标, G.R抓取)          # ← 关键：走完不等于到了

    # 2. 闭合夹爪，原地停一会儿等夹持力建立
    data.ctrl[7] = 0.0
    for _ in range(int(夹紧等待 / dt)):
        data.ctrl[:7] = cfg.q[:7]
        mujoco.mj_step(model, data)
        cfg.update(data.qpos)

    # 3. 记下"刚夹住时"方块相对手指的位置 —— 后面拿它当基准
    Rh = data.body("hand").xmat.reshape(3, 3).copy()
    相对0 = Rh.T @ (data.body("cube").xpos - data.body("hand").xpos)
    # 夹爪中心与方块中心的水平偏离。理想 0，偏了就是"夹在边角上"，夹不住
    偏心 = float(np.linalg.norm((data.body("hand").xpos - data.body("cube").xpos)[:2])) * 1000

    # 4. 举升。只动位置，朝向一个字不改
    G.移动(data.body("hand").xpos + np.array([0, 0, 举升高度]), 速度=0.25)

    # 5. 量漂移
    相对1 = Rh.T @ (data.body("cube").xpos - data.body("hand").xpos)
    漂移 = float(np.linalg.norm(相对1 - 相对0)) * 1000
    抬起 = float(data.body("cube").xpos[2] - 方块初始z) * 1000
    return 漂移, 抬起, 偏心, 漂移 < 成功阈值 * 1000


def main():
    参数 = [a for a in sys.argv[1:] if not a.startswith("-")]
    次数 = int(参数[0]) if len(参数) > 0 else 20
    范围 = float(参数[1]) if len(参数) > 1 else 0.05

    rng = np.random.default_rng(0)
    print("方块随机范围 ±%.0f mm，共 %d 次" % (范围 * 1000, 次数))
    print("%-4s %-22s %8s %8s %8s %6s" % ("次", "方块位置", "漂移mm", "抬起mm", "偏心mm", "结果"))
    print("-" * 66)

    成 = 0
    漂移们, 偏心们 = [], []
    t0 = time.time()
    for i in range(1, 次数 + 1):
        try:
            漂移, 抬起, 偏心, ok = 抓一次(rng, 范围)
        except Exception as e:
            print("%-4d 崩溃：%s" % (i, e))
            continue
        漂移们.append(漂移)
        偏心们.append(偏心)
        成 += int(ok)
        print("%-4d (%.3f,%+.3f,%.3f) %8.1f %8.1f %8.1f %6s"
              % (i, *data.body("cube").xpos[:3], 漂移, 抬起, 偏心, "成功" if ok else "失败"))

    用时 = time.time() - t0
    print("-" * 66)
    print("成功率  %d / %d = %.0f%%" % (成, 次数, 100.0 * 成 / 次数))
    if 漂移们:
        print("漂移    中位 %.1f mm   最差 %.1f mm   阈值 %.0f mm"
              % (np.median(漂移们), np.max(漂移们), 成功阈值 * 1000))
        print("偏心    最大 %.1f mm" % np.max(偏心们))
    print("总耗时  %.1f 秒（平均每次 %.1f 秒）" % (用时, 用时 / 次数))


if __name__ == "__main__":
    main()
