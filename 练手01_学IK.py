"""练手 01：用一个神经网络替掉 mink 的 IK。

这一步是你后面「强化学习抓取」的同一件事：
    策略网络干的就是「看末端位姿 → 吐 7 个关节角」，跟这里一模一样，
    只是训练信号从"标准答案"换成"奖励"。

整条链只有四步，跟训练回路图一一对应：
    ① 造数据   用 MuJoCo 正运动学(q → 末端位姿)造配对样本
    ② 定模型   7 个数进 → 64 → 64 → 7 个数出
    ③ 算损失   MSE，一个标量，越小越好
    ④ 反传更新 PyTorch 的 loss.backward() 自动算梯度，optimizer 负责走

为什么用正运动学造数据而不是 IK：正运动学是一次矩阵乘法，0.00003 秒；
IK 要迭代几百步。4000 组数据，造得快 1000 倍，而且答案天然正确（没有不收敛的问题）。

跑：
    D:\\MuJoCo_Projects\\mujoco_env\\python.exe 练手01_学IK.py
"""

import sys
import os
import time
import numpy as np
import mujoco
import torch
import torch.nn as nn

sys.path.insert(0, r"D:\MuJoCo_Projects\test")
from 封装函数.package import make_ik

# ─── 可调参数：改这几个数，观察曲线怎么变 ──────────────────────────
N        = 4000      # 样本数。改成 200 看"过拟合"；改成 20000 看"数据够不够"
噪声幅度  = 0.35      # 关节角在 home 附近采样的范围(rad)。越大样本越分散，越难学
学习率    = 1e-3      # 改成 0.5 会炸成 nan（发散）；改成 1e-5 会像蜗牛
轮数      = 800       # 一次"轮"= 把训练集完整看一遍
验证比例  = 0.2       # 留 20% 数据不参与训练，只看它判好坏
模型文件  = "模型_学IK.pt"   # 训练完存这里，练手02_用网络.py 会来读
# ─────────────────────────────────────────────────────────────────

SCENE = r"D:\MuJoCo_Projects\test\franka_emika_panda\scene_pick_table.xml"


def 正运动学(model, data, q):
    """给 7 个关节角，返回 hand 的位置(3) 和姿态四元数 wxyz(4)。"""
    data.qpos[:7] = q
    mujoco.mj_kinematics(model, data)
    p = data.body("hand").xpos.copy()
    quat = data.body("hand").xquat.copy()
    if quat[0] < 0:            # q 与 -q 是同一姿态，统一成 w≥0，否则网络会学到两个答案
        quat = -quat
    return p, quat


def 造数据(model, data, n, 幅度, rng):
    q_home = model.key_qpos[0][:7].copy()
    lo, hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
    Q = np.zeros((n, 7))       # 答案：关节角
    P = np.zeros((n, 7))       # 输入：末端位姿（位置 3 + 四元数 4）
    for i in range(n):
        q = np.clip(q_home + rng.uniform(-幅度, 幅度, 7), lo + 1e-3, hi - 1e-3)
        p, quat = 正运动学(model, data, q)
        Q[i], P[i] = q, np.r_[p, quat]
    return Q, P


def 评测(model, data, q_pred, P_true):
    """把预测的关节角喂回 MuJoCo，看末端到底偏了多少毫米。"""
    误差位置, 误差姿态 = [], []
    for q, 真 in zip(q_pred, P_true):
        p, quat = 正运动学(model, data, q)
        误差位置.append(np.linalg.norm(p - 真[:3]) * 1000)
        d = abs(float(np.dot(quat, 真[3:])))
        误差姿态.append(np.degrees(2 * np.arccos(np.clip(d, -1, 1))))
    return float(np.mean(误差位置)), float(np.mean(误差姿态))


def main():
    model, data, cfg, task, limits, dt = make_ik(SCENE)
    rng = np.random.default_rng(0)

    # ① 造数据
    t0 = time.time()
    Q, P = 造数据(model, data, N, 噪声幅度, rng)
    print("① 造 %d 组数据：%.2f 秒" % (N, time.time() - t0))
    print("   末端能到的范围 x %.2f~%.2f  y %+.2f~%+.2f  z %.2f~%.2f (m)"
          % (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(),
             P[:, 2].min(), P[:, 2].max()))

    # 划分：训练集用来改参数，验证集只用来"打分"，绝不参与训练
    打乱 = rng.permutation(N)
    训练集 = 打乱[: int(N * (1 - 验证比例))]
    验证集 = 打乱[int(N * (1 - 验证比例)):]

    # 标准化：把输入压到均值 0、方差 1。不做这一步，学习率会变得极其难调
    mu, sd = P[训练集].mean(0), P[训练集].std(0)
    qm, qs = Q[训练集].mean(0), Q[训练集].std(0)
    Xn = (P - mu) / sd
    Yn = (Q - qm) / qs

    X = torch.tensor(Xn[训练集], dtype=torch.float32)
    Y = torch.tensor(Yn[训练集], dtype=torch.float32)
    Xv = torch.tensor(Xn[验证集], dtype=torch.float32)
    Yv = torch.tensor(Yn[验证集], dtype=torch.float32)

    # 先立两个"不学习"的对照，才知道网络到底有没有用
    print("\n② 对照基线（这两个是尺子）")
    q_mean = Q[训练集].mean(0)
    m, a = 评测(model, data, np.tile(q_mean, (len(验证集), 1)), P[验证集])
    print("   常数基线（永远输出平均关节角） ：位置 %6.2f mm  姿态 %5.2f 度" % (m, a))

    W, *_ = np.linalg.lstsq(Xn[训练集], Yn[训练集], rcond=None)
    q_lin = (Xn[验证集] @ W) * qs + qm
    m, a = 评测(model, data, q_lin, P[验证集])
    print("   线性模型（只有一层、无激活）   ：位置 %6.2f mm  姿态 %5.2f 度" % (m, a))

    # ② 模型：7 → 64 → 64 → 7。两个 Tanh 是"非线性"，没有它上面那行就是天花板
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(7, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 7),
    )
    opt = torch.optim.Adam(net.parameters(), lr=学习率)
    lossf = nn.MSELoss()          # 就是 mean((预测-答案)^2)，一个标量

    print("\n③ 训练（loss 是训练集上的，val 是验证集上的；两个差很多 = 过拟合）")
    t0 = time.time()
    for ep in range(1, 轮数 + 1):
        opt.zero_grad()                   # 不清零，梯度会累加
        loss = lossf(net(X), Y)           # 前向 + 算损失
        loss.backward()                   # 反向：一次算完所有参数该往哪改
        opt.step()                        # 按梯度和学习率走一步
        if ep == 1 or ep % max(1, 轮数 // 6) == 0:
            with torch.no_grad():
                print("   第 %4d 轮   训练 loss %.5f   验证 loss %.5f"
                      % (ep, loss.item(), lossf(net(Xv), Yv).item()))
    print("   训练耗时 %.1f 秒" % (time.time() - t0))

    with torch.no_grad():
        q_net = net(Xv).numpy() * qs + qm
    m, a = 评测(model, data, q_net, P[验证集])
    print("\n④ 小网络 (7-64-64-7)              ：位置 %6.2f mm  姿态 %5.2f 度" % (m, a))

    print("\n注意：这张表要横着读。常数列告诉你「不学能到哪」，线性列告诉你「这关系是不是直的」，")
    print("      网络列告诉你「弯曲的关系值多少毫米」。三列一起看，才知道自己改的参数有没有用。")

    # 存盘：网络本体 + 四个标准化参数（换台机器、换个脚本也得配套用，不能只存网络）
    存 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 模型文件)
    torch.save({"net": net, "mu": mu, "sd": sd, "qm": qm, "qs": qs,
                "采样幅度": 噪声幅度}, 存)
    print("\n已保存：%s（%.1f KB）" % (存, os.path.getsize(存) / 1024))
    print("体验：python 练手02_用网络.py 0.5 0 0.6")


if __name__ == "__main__":
    main()
