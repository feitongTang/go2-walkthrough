import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

from pynput import keyboard

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """
    target_q: 目标关节角度，也就是期望达到的位置
    q: 当前实际关节角度
    kp: 位置控制增益，决定位置误差的相应强度，过大导致震荡，过小导致响应慢
    target_dq: 目标关节速度，通常为0
    dq: 当前实际关节速度
    kd: 速度控制增益，提供阻尼，抑制震荡

    kp确保准确性，kd提供稳定性
    """
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def key_callback(key):
    try:
        if key.char == '6':
            cmd[0] += 0.5
        elif key.char == '7':
            cmd[0] -= 0.5
        elif key.char == '8':
            cmd[1] += 0.5
        elif key.char == '9':
            cmd[1] -= 0.5
        elif key.char == '-':
            cmd[2] += 0.5
        elif key.char == '=':
            cmd[2] -= 0.5
        elif key.char == '0':
            cmd[0] = 0
            cmd[1] = 0
            cmd[2] = 0
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=key_callback)
listener.start()


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_cmds = config["num_cmds"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            # 计算PD控制力矩
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            
            # 应用控制并推进物理仿真
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # 控制降频处理（每control_decimation步执行一次策略）
            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]     # 关节角度
                dqj = d.qvel[6:]    # 关节角速度
                quat = d.qpos[3:7]  # 姿态四元数
                omega = d.qvel[3:6] # 角速度

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                # 添加步态相位信息
                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                # 观测包含以下信息：
                # 机体角速度（3维）
                # 机体重力方向向量（3维）
                # 速度命令（3维：前进、侧向、转向）
                # 关节位置和速度（各12维）
                # 上一周期动作（12维）（提供动作连续性，避免抖动）
                # 步态相位（sin/cos）
                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:6 + num_cmds] = cmd * cmd_scale
                obs[6 + num_cmds : 6 + num_cmds + num_actions] = qj
                obs[6 + num_cmds + num_actions : 6 + num_cmds + 2 * num_actions] = dqj
                obs[6 + num_cmds + 2 * num_actions : 6 + num_cmds + 3 * num_actions] = action
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
