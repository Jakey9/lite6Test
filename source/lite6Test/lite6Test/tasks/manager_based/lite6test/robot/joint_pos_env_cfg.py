import math
from isaaclab.utils import configclass

import lite6Test.tasks.manager_based.lite6test.mdp as mdp
from lite6Test.tasks.manager_based.lite6test.lite6test_env_cfg import Lite6testEnvCfg

from .lite6 import LITE6_CFG


@configclass
class Lite6ReachEnvCfg(Lite6testEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # switch robot to Lite6
        self.scene.robot = LITE6_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ✅ end-effector link (CHANGE if different)
        ee_link = "link6"   # or "link6" depending on your USD

        # override rewards (fix MISSING)
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [ee_link]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [ee_link]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [ee_link]

        # override actions (fix MISSING)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            scale=0.5,
            use_default_offset=True,
        )

        # override command generator body
        self.commands.ee_pose.body_name = ee_link
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class Lite6ReachEnvCfg_PLAY(Lite6ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
