from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
import torch
import numpy as np

class Go2Robot(LeggedRobot):
    def post_physics_step(self):
        super().post_physics_step()
        self.actions_history = torch.cat([self.actions_history[:, self.num_actions:], self.actions], dim=-1)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        
        if self.cfg.commands.num_commands > 3:
            self.commands[env_ids, 3:6] = torch.tensor([0, 0.5, 0], device=self.device)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[: 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3: 6] = noise_scales.gravity * noise_level
        noise_vec[6: 6 + self.cfg.commands.num_commands] = 0
        noise_vec[6 + self.cfg.commands.num_commands: 6 + self.cfg.commands.num_commands + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[6 + self.cfg.commands.num_commands + self.num_actions: 6 + self.cfg.commands.num_commands + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[6 + self.cfg.commands.num_commands + 6 + self.cfg.commands.num_commands * self.num_actions: 12 + 3 * self.num_actions] = 0. # previous actions

        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel,
                                            self.obs_scales.lin_vel,
                                            self.obs_scales.ang_vel,
                                            self.obs_scales.gait_phase,
                                            self.obs_scales.gait_offset,
                                            self.obs_scales.gait_bound], device=self.device, requires_grad=False)[:self.cfg.commands.num_commands]
        
        self.feet_num = len(self.feet_indices)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.actions_history = torch.zeros(self.num_envs, self.cfg.env.num_actions * self.cfg.env.num_history, dtype=torch.float, device=self.device, requires_grad=False)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        if self.cfg.env.observe_gaits:
            self.update_feet_state()
            period = 0.8
            phases = self.commands[:, 3]
            offsets = self.commands[:, 4]
            bounds = self.commands[:, 5]
            self.phase = (self.episode_length_buf * self.dt) % period / period
            self.phase_fl = torch.sin(2 * np.pi * ((self.phase + phases + offsets + bounds) % 1))
            self.phase_fr = torch.sin(2 * np.pi * ((self.phase + bounds) % 1))
            self.phase_rl = torch.sin(2 * np.pi * ((self.phase + offsets) % 1))
            self.phase_rr = torch.sin(2 * np.pi * ((self.phase + phases) % 1))
            self.leg_phase = torch.cat([self.phase_fl.unsqueeze(1), self.phase_fr.unsqueeze(1), self.phase_rl.unsqueeze(1), self.phase_rr.unsqueeze(1)], dim=-1)
        
        super()._post_physics_step_callback()

    def compute_observations(self):
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.actions_history
                                    ),dim=-1)
        if self.cfg.env.observe_gaits:
            self.obs_buf = torch.cat((self.obs_buf, self.leg_phase), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.privileged_obs_buf = torch.cat((   self.base_lin_vel * self.obs_scales.lin_vel,
                                                self.base_ang_vel * self.obs_scales.ang_vel,
                                                self.projected_gravity,
                                                self.commands * self.commands_scale,
                                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                                self.dof_vel * self.obs_scales.dof_vel,
                                                self.actions,
                                                self.actions_history), dim=-1)
        if self.cfg.env.observe_gaits:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.leg_phase), dim=-1)

    def _reward_base_height(self):
        base_height = self.root_states[:, 2]
        if self.cfg.env.observe_body_height:
            height_diff = base_height - (self.commands[:, 3] + self.cfg.rewards.base_height_target)
        else:
            height_diff = base_height - self.cfg.rewards.base_height_target
        return torch.square(height_diff)