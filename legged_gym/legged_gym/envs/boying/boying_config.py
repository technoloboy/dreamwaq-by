from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BoyingRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        # head_joint dont_collapse removed: head merges into base, 17 rigid bodies (same as go1)
        # privileged_obs = obs(45) + lin_vel(3) + contact_forces(17*3=51) + heights(17*11=187) = 286
        num_privileged_obs = 286

    class terrain( LeggedRobotCfg.terrain ):
        # cap max step height at ~0.117m (difficulty=0.9): 0.05 + 0.074*0.9 = 0.117m
        # go1 default scale=0.18 gives 0.212m at difficulty=0.9, beyond boying's leg reach
        step_height_scale = 0.074

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.40] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint':  0.1,   # [rad]
            'RL_hip_joint':  0.1,   # [rad]
            'FR_hip_joint': -0.1,   # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.7,  # [rad]
            'FR_thigh_joint': 0.7,  # [rad]
            'RL_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 0.8,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        # [Method A] Motor spec ratio (empirical, based on GO1 stall torque ratio):
        # k: 28 * (60/33.5) = 50,  d: GO1 uses 58.5% of d_max=T_stall/v_max=60/15.6=3.846 -> 2.25
        # Training result: action_rate and smoothness ~50% worse than GO1 at same steps,
        # raw joint_power ~7x GO1. High stiffness causes structural oscillation that
        # reward tuning alone cannot compensate. Deprecated after v1~v4 experiments.
        # stiffness = {'joint': 50.}
        # damping   = {'joint': 2.25}

        # [Method B] 2nd-order system + equivalent inertia (matched to GO1 dynamics):
        # target: same ωn=28.38 rad/s and ζ=0.355 as GO1 thigh joint
        # GO1:    J=0.03477 kg·m²  =>  k=28,  d=0.70
        # Boying: J=0.03187 kg·m²  =>  k=ωn²·J=25.7≈26,  d=2·ζ·ωn·J=0.64
        # Method B (k=26/d=0.64) failed: underdamped, terrain degraded after step 3200
        # Method C: k=40/d=1.5 — compromise: lower k reduces power, higher d suppresses oscillation
        stiffness = {'joint': 40.}   # [N*m/rad]
        damping   = {'joint': 1.5}   # [N*m*s/rad]
        
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
        only_positive_rewards = False
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32
        # v18: v17 stayed negative for the full 30k run at 0.5 s and converged
        # to a low-clearance shuffling gait. Lowering only this threshold makes
        # attainable swing phases positively reinforced while preserving the
        # proven v17 reward scales and PPO settings.
        feet_air_time_threshold = 0.3
        # Next single-variable experiment: reward a 10 cm swing peak relative
        # to local terrain at touchdown. The bounded kernel avoids incentivizing
        # arbitrarily high or long swing trajectories.
        feet_clearance_target = 0.10
        feet_clearance_sigma = 0.03

        class scales( LeggedRobotCfg.rewards.scales ):
            # --- stability: boying has higher CoM; orientation still ~50% worse than GO1 at equal steps ---
            orientation  = -0.3       # go1: -0.2; -0.5 caused catastrophic collapse (Jul23 v7), -0.3 is upper limit for Boying
            ang_vel_xy   = -0.05      # go1: -0.05

            # --- height: go1 default -1.0 is too strong for Boying; stairs require natural trunk height adjustment ---
            # v14 analysis: base_height deviation was 0.118m (vs v9/GO1: 0.042m) under -1.0, 2.8x worse
            base_height  = -0.5       # go1: -1.0 (halved: Boying at 0.32m target on stairs needs more flexibility)

            # --- energy: restored to GO1 value; raw joint_power was ~7x GO1 under k=50 ---
            joint_power        = -5e-6   # go1: -2e-5 (loosened: k=50 causes ~7x raw power vs GO1, penalty can't fix physics)
            power_distribution = -5e-7   # go1: -10e-6 (kept loose: structural asymmetry from k=50)

            # --- anti-oscillation: equal to GO1; raw oscillation still ~50% higher due to k=50 ---
            action_rate  = -0.01      # go1: -0.01 (was -0.015, relaxed Jul22)
            smoothness   = -0.01      # go1: -0.01 (was -0.008, aligned Jul22)

            # --- acceleration: heavier thigh (1.5kg vs 0.9kg) ---
            dof_acc      = -1.5e-7    # go1: -2.5e-7

            # --- gait: longer legs (0.49m vs 0.43m) ---
            feet_air_time = 0.15      # go1: 0.1
            feet_clearance_target = 0.05

        # [GO1 baseline weights for reference]
        # class scales( LeggedRobotCfg.rewards.scales ):
        #     orientation        = -0.2
        #     ang_vel_xy         = -0.05
        #     joint_power        = -2e-5
        #     power_distribution = -10e-6
        #     action_rate        = -0.01
        #     smoothness         = -0.01
        #     dof_acc            = -2.5e-7
        #     feet_air_time      = 0.1

class BoyingRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # v13: entropy_coef=0.01 + max_lr=5e-4 caused noise_std to diverge (0.99→2.36 over 20k steps)
        # low LR (~1.5e-4 mean) couldn't counteract the entropy bonus, so policy kept expanding variance
        # v15: entropy_coef=0.001 caused terrain to drop to 0 — policy over-converged to flat-ground
        # strategy before learning stairs (noise_std=0.35 by step 200, too deterministic to explore)
        # v16: entropy_coef=0.01 diverged to noise_std=5.23 by step 30k (same root cause as v13)
        # v17: entropy_coef=0.005 — midpoint between 0.001 (too low) and 0.01 (too high)
        entropy_coef = 0.005
        schedule = 'adaptive'
        learning_rate = 1e-3
        max_learning_rate = 8e-4
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_boying'
        max_iterations = 30000
