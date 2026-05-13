import gymnasium as gym
import gymnasium_robotics
import numpy as np

# Ensure base robotics environments are registered
gym.register_envs(gymnasium_robotics)

class AdroitHandHammerPEMPWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        renderer = getattr(env.unwrapped, "mujoco_renderer", None)
        viewer = renderer._get_viewer(render_mode="human")
        viewer._hide_menu = True
        
    def reset(self, **kwargs):
        # 1. Call the ordinary super reset
        obs, info = super().reset(**kwargs)
        
        # 2. Apply the specific configuration
        qp = np.array([0.05362812, -0.15676938,  0.1124559,  -0.19211253, -0.00677962,  0.94733362,
            0.68165027,  0.95151664, -0.04922544,  1.11051917,  0.84514704,  0.93555745,
            -0.07568273,  0.93582394,  0.93026161,  1.2665799,   0.14275029, -0.38195376,
            1.22411986,  1.04177562,  1.09699174,  0.10221559,  1.12782028,  0.14026537,
            -0.1667768,  -0.43379269,  0.,          0.02543423,  0.10667372, -0.01684074,
            -0.0044481,  -0.29020183,  0.25038974])
        qv = np.zeros(33)
        bp = np.array([0.05, 0., 0.18356179])
        
        init_state = dict(qpos=qp, qvel=qv, board_pos=bp)
        
        # Pass the state down to the core mujoco environment
        base_env = self.unwrapped
        if hasattr(base_env, "set_env_state"):
            base_env.set_env_state(init_state)
            
        # 3. Settle simulator for 10 steps using 0 action
        for _ in range(10):
            _, _, _, _, step_info = self.env.step(np.zeros(self.action_space.shape))
            info.update(step_info)
            
        # 4. Return the new observation after settling
        obs = base_env._get_obs()
        return obs, info

# Factory function required for gym registration
def make_adroit_hammer_pemp(**kwargs):
    env = gym.make("AdroitHandHammer-v1", **kwargs)
    return AdroitHandHammerPEMPWrapper(env)

# Register the new environment
gym.envs.registration.register(
    id="AdroitHandHammer-vPEMP",
    entry_point="adroit_hand_hammer_updated:make_adroit_hammer_pemp",
)
