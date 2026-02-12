import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os

# Path
_THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LITE6_PATH = os.path.join(_THIS_SCRIPT_DIR, "asset", "lite6_v1.usd")

LITE6_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LITE6_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        },
    ),
    actuators={
        "joint1": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            velocity_limit=100.0,
            effort_limit=50.0,
            stiffness=120.0,
            damping=60.0,
        ),
        "joint2": ImplicitActuatorCfg(
            joint_names_expr=["joint2"],
            velocity_limit=100.0,
            effort_limit=50.0,
            stiffness=67.0,
            damping=0.0268,
        ),
        "joint3": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=62.4,
            damping=0.0249,
        ),
        "joint4": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            velocity_limit=100.0,
            effort_limit=50.0,
            stiffness=188.8,
            damping=0.0755,
        ),
        "joint5": ImplicitActuatorCfg(
            joint_names_expr=["joint5"],
            velocity_limit=100.0,
            effort_limit=50.0,
            stiffness=4.02,
            damping=0.00161,
        ),
        "joint6": ImplicitActuatorCfg(
            joint_names_expr=["joint6"],
            velocity_limit=100.0,
            effort_limit=50.0,
            stiffness=3.552,
            damping=0.00152,
        ),
    },
)