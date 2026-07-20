from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BoyingRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        # head_joint dont_collapse removed: head merges into base, 17 rigid bodies (same as go1)
        # privileged_obs = obs(45) + lin_vel(3) + contact_forces(17*3=51) + heights(17*11=187) = 286
        num_privileged_obs = 286

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.40] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint':  0.1,   # [rad]
            'RL_hip_joint':  0.1,   # [rad]
            'FR_hip_joint': -0.1,   # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 60.}  # [N*m/rad]
        damping = {'joint': 4.5}    # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/boying_description/urdf/boying_description_withouthm.urdf'
        name = "boying"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35

        class scales( LeggedRobotCfg.rewards.scales ):
            # --- stability: boying has higher CoM, penalize pitch/roll more ---
            orientation  = -0.3       # go1: -0.2
            ang_vel_xy   = -0.08      # go1: -0.05

            # --- energy: boying is 19kg vs go1's 11kg, natural power is higher ---
            joint_power        = -1e-5   # go1: -2e-5
            power_distribution = -5e-6   # go1: -10e-6

            # --- smoothness: high damping (4.5) already damps motion naturally ---
            smoothness   = -0.005     # go1: -0.01

            # --- acceleration: heavier thigh (1.5kg vs 0.9kg) has larger natural acc ---
            dof_acc      = -1.5e-7    # go1: -2.5e-7

            # --- gait: longer legs (0.49m vs 0.43m), encourage larger steps ---
            feet_air_time = 0.15      # go1: 0.1

class BoyingRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_boying'
