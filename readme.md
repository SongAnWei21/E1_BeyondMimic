python scripts/rsl_rl/train.py --task=Tracking-Flat-x1-25dof-Wo-v0 --motion_file /home/rytech/agibot_x1/Mini-Pi-Plus_BeyondMimic/source/motion/x1/npz/x1_25dof_walk1_subject1.npz --log_project_name x1_25dof_beyondminic --logger wandb


python scripts/bvh_to_robot.py --bvh_file /home/rytech/agibot_x1/GMR/lafan1/walk1_subject1.bvh --robot x1_25dof --save_path /home/rytech/agibot_x1/GMR/lafan1/x1_25dof_walk1_subject1.pkl --rate_limit --format lafan1 --record_video


python -m mujoco.viewer --mjcf /home/rytech/agibot_x1/Mini-Pi-Plus_BeyondMimic/source/whole_body_tracking/whole_body_tracking/assets/hightorque/hi/mjcf/hi_25dof.xml


python scripts/bvh_to_robot.py --bvh_file /home/rytech/agibot_x1/GMR/lafan1/walk1_subject1.bvh --robot unitree_g1_23dof --save_path /home/rytech/agibot_x1/GMR/lafan1_g1/walk.pkl --rate_limit --format lafan1 --record_video


python ./scripts/batch_gmr_pkl_to_csv.py --folder /home/rytech/agibot_x1/GMR/lafan1_x1_25dof



python scripts/csv_cut.py --input_csv /home/rytech/agibot_x1/whole_body_tracking/source/motion/g1_29dof/walk.csv --output_csv /home/rytech/agibot_x1/whole_body_tracking/source/motion/g1_29dof/walk_cut.csv --start_frame 200 --end_frame 800 --remove_frame_column --z_offset 0.0 --decimal_places 6


python scripts/csv_to_npz.py --robot g1 --input_file source/motion/g1_29dof/walk_cut.csv --input_fps 30 --output_name source/motion/g1_29dof/walk_cut.npz --no_wandb


python scripts/replay_npz.py --robot g1 --motion_file /home/rytech/agibot_x1/whole_body_tracking/source/motion/g1_29dof/walk_cut.npz.npz


python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 --motion_file source/motion/g1_29dof/walk_cut.npz --headless --log_project_name g1_beyondmimic


python scripts/rsl_rl/play.py --task=Tracking-Flat-x1-Wo-v0  --checkpoint /home/rytech/agibot_x1/Mini-Pi-Plus_BeyondMimic/logs/rsl_rl/x1_23dof_flat/2026-01-28_12-23-27/model_2000.pt --num_envs=1 --motion_file /home/rytech/agibot_x1/Mini-Pi-Plus_BeyondMimic/source/motion/x1/npz/dance1.npz


python ./scripts/sim2sim.py --robot x1_23dof --motion_file source/motion/x1/npz/dance1.npz --xml_path source/whole_body_tracking/whole_body_tracking/assets/x1/x1_23dof/mjcf/x1_23dof.xml --policy_path /home/rytech/agibot_x1/Mini-Pi-Plus_BeyondMimic/logs/rsl_rl/x1_23dof_flat/2026-01-28_12-23-27/exported/model_2000.onnx --save_json

python ./scripts/sim2sim.py --robot n1 --motion_file source/motion/n1/npz/dance1.npz --xml_path /home/saw/RL/humanoid_robot/BeyondMimic/source/whole_body_tracking/whole_body_tracking/assets/n1/mjcf/n1_mocap.xml --policy_path /home/saw/RL/humanoid_robot/BeyondMimic/logs/rsl_rl/n1_flat/2026-01-25_19-14-15/exported/model_43500.onnx --save_json

python scripts/rsl_rl/train.py --task=Tracking-Flat-x1-Wo-v0 --motion_file /home/saw/RL/humanoid_robot/BeyondMimic/source/motion/x1/npz/dance1.npz --headless --log_project_name x1_beyondmimic

python scripts/rsl_rl/play.py --task=Tracking-Flat-x1-Wo-v0  --checkpoint /home/saw/RL/humanoid_robot/BeyondMimic/logs/rsl_rl/x1_23dof_flat/2026-02-09_11-12-37/model_11000.pt --num_envs=1 --motion_file /home/saw/RL/humanoid_robot/BeyondMimic/source/motion/x1/npz/dance1.npz


export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890

python scripts/rsl_rl/train.py --task=Tracking-Flat-e1-Wo-v0 --motion_file source/motion/E1/npz/dance.npz --headless --log_project_name e1_beyondmimic

python ./scripts/sim2sim.py --robot e1 --motion_file source/motion/E1/npz/dance.npz --xml_path /home/saw/RL/humanoid_robot/BeyondMimic/source/whole_body_tracking/whole_body_tracking/assets/e1/mjcf/test.xml --policy_path /home/saw/RL/humanoid_robot/BeyondMimic/logs/rsl_rl/e1_flat/2026-02-17_22-57-31/exported/model_16500.onnx --save_json

python scripts/rsl_rl/play.py --task=Tracking-Flat-e1-Wo-v0  --checkpoint /home/saw/RL/humanoid_robot/BeyondMimic/logs/rsl_rl/e1_flat/2026-02-17_22-57-31/model_16500.pt --num_envs=1 --motion_file /home/saw/RL/humanoid_robot/BeyondMimic/source/motion/E1/npz/dance.npz

python scripts/rsl_rl/play.py --task=Tracking-Flat-e1-Wo-v0  --checkpoint /home/saw/RL/humanoid_robot/BeyondMimic/logs/rsl_rl/e1_flat/2026-02-18_23-58-46/model_14500.pt --num_envs=1 --motion_file /home/saw/RL/humanoid_robot/BeyondMimic/source/motion/E1/npz/dance.npz


36Nm   50.27   10.47    30.52
60Nm   20.42   18.85
14Nm   32.99   27.23


python ./scripts/sim2sim.py --motion_file /home/saw/droidup/E1_BeyondMimic/motion/e1_21dof/MJ_dance.npz --xml_path /home/saw/droidup/E1_BeyondMimic/e1_lab/assets/e1_21dof/mjcf/E1_21dof.xml --policy_path /home/saw/droidup/E1_BeyondMimic/logs/rsl_rl/e1_flat/2026-03-20_10-15-20/exported/model_2000.onnx

python ./scripts/sim2sim_his.py --motion_file /home/saw/droidup/E1_BeyondMimic/motion/e1_21dof/MJ_dance.npz --xml_path /home/saw/droidup/E1_BeyondMimic/e1_lab/assets/e1_21dof/mjcf/E1_21dof.xml --policy_path /home/saw/droidup/E1_BeyondMimic/logs/rsl_rl/e1_flat/2026-03-20_11-10-07/exported/model_10000.onnx
