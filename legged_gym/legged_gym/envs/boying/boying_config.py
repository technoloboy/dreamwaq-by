from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BoyingRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        # head_joint dont_collapse removed: head merges into base, 17 rigid bodies (same as go1)
        # privileged_obs = obs(45) + lin_vel(3) + contact_forces(17*3=51) + heights(17*11=187) = 286
        num_privileged_obs = 286

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
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
        control_type = 'P'
        # [Method A] Motor spec ratio (empirical, based on GO1 stall torque ratio):
        # k: 28 * (60/33.5) = 50,  d: GO1 uses 58.5% of d_max=T_stall/v_max=60/15.6=3.846 -> 2.25
        stiffness = {'joint': 50.}
        damping   = {'joint': 2.25}

        # [Method B] 2nd-order system + equivalent inertia (matched to GO1, currently active):
        # target: same ωn=28.38 rad/s and ζ=0.355 as GO1 thigh joint
        # GO1:    J=0.03477 kg·m²  =>  k=28,  d=0.70
        # Boying: J=0.03187 kg·m²  =>  k=ωn²·J=25.7≈26,  d=2·ζ·ωn·J=0.64
        # stiffness = {'joint': 26.}   # [N*m/rad]
        # damping   = {'joint': 0.64}  # [N*m*s/rad]
        
        action_scale = 0.25
        decimation   = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/boying_description/urdf/boying_description_withouthm.urdf'
        name = "boying"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.3

        class scales( LeggedRobotCfg.rewards.scales ):
            # --- stability: boying has higher CoM ---
            orientation  = -0.3       # go1: -0.2
            ang_vel_xy   = -0.05      # go1: -0.05

            # --- energy: boying 19kg, natural power higher, keep loose ---
            joint_power        = -1e-5   # go1: -2e-5
            power_distribution = -5e-7   # go1: -10e-6 (reduced from -2e-6, structural asymmetry cannot be eliminated)

            # --- anti-oscillation: 2x go1, not 4x; heavy penalty blocks all gradient ---
            action_rate  = -0.015     # go1: -0.01  (relaxed from -0.02, k=50 causes natural impact on complex terrain)
            smoothness   = -0.008     # go1: -0.01

            # --- acceleration: heavier thigh (1.5kg vs 0.9kg) ---
            dof_acc      = -1.5e-7    # go1: -2.5e-7

            # --- gait: longer legs (0.49m vs 0.43m) ---
            feet_air_time = 0.15      # go1: 0.1

        # [GO1 same weights - commented out]
        # class scales( LeggedRobotCfg.rewards.scales ):
        #     orientation  = -0.2
        #     ang_vel_xy   = -0.05
        #     joint_power        = -2e-5
        #     power_distribution = -10e-6
        #     action_rate  = -0.01
        #     smoothness   = -0.01
        #     dof_acc      = -2.5e-7
        #     feet_air_time = 0.1

class BoyingRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_boying'
        max_iterations = 30000
