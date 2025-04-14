SHARED_ARGS="--replay_buffer_capacity 10_000 --batch_size 32 --update_horizon 1 --gamma 0.99 --learning_rate 3e-4 \
    --horizon 1_000 --architecture_type fc --n_epochs 50 --n_training_steps_per_epoch 10_000 --update_to_data 1 \
    --n_initial_samples 1_000 --epsilon_end 0.01 --epsilon_duration 1_000"

FEATURE=100
N_NETWORS=3
TARGET_UPDATE_FREQ=200
TARGET_SYNC_FREQ=10

SHARED_ARGS="$SHARED_ARGS --features $FEATURE $FEATURE --target_update_frequency $TARGET_UPDATE_FREQ \
    --experiment_name randreward_f${FEATURE}_target_freq${TARGET_UPDATE_FREQ}_n_net${N_NETWORS}_sync_freq${TARGET_SYNC_FREQ}"

# ----- L2 Loss -----
# launch_job/lunar_lander/cluster_dqn.sh --first_seed 1 --last_seed 10 --n_parallel_seeds 1 $SHARED_ARGS
# sleep 20

# iDQN_ARGS="--n_networks $N_NETWORS --target_sync_frequency $TARGET_SYNC_FREQ"
# launch_job/lunar_lander/cluster_idqn.sh --first_seed 1 --last_seed 10 --n_parallel_seeds 1 $SHARED_ARGS $iDQN_ARGS
# sleep 20

# aGIDQN_ARGS="--n_networks $N_NETWORS"
# launch_job/lunar_lander/cluster_agidqn.sh --first_seed 1 --last_seed 10 --n_parallel_seeds 1 $SHARED_ARGS $aGIDQN_ARGS
# sleep 20

# GIDQN_ARGS="--n_networks $N_NETWORS"
# launch_job/lunar_lander/cluster_gidqn.sh --first_seed 1 --last_seed 10 --n_parallel_seeds 1 $SHARED_ARGS $GIDQN_ARGS
# sleep 20

# ----- KL Loss -----
# HLDQN_ARGS="--n_bins 50 --min_value -100 --max_value 100 --sigma 3"
# launch_job/lunar_lander/cluster_hldqn.sh --first_seed 1 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $HLDQN_ARGS
# sleep 20

# iHLDQN_ARGS="--n_networks $N_NETWORS --target_sync_frequency $TARGET_SYNC_FREQ --n_bins 50 --min_value -100 --max_value 100 --sigma 3"
# launch_job/lunar_lander/cluster_ihldqn.sh --first_seed 1 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $iHLDQN_ARGS
# sleep 20

# aGIHLDQN_ARGS="--n_networks $N_NETWORS --n_bins 50 --min_value -100 --max_value 100 --sigma 3"
# launch_job/lunar_lander/cluster_agihldqn.sh --first_seed 1 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $aGIHLDQN_ARGS
# sleep 20

# GIHLDQN_ARGS="--n_networks $N_NETWORS --n_bins 50 --min_value -100 --max_value 100 --sigma 3"
# launch_job/lunar_lander/cluster_gihldqn.sh --first_seed 1 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $GIHLDQN_ARGS
# sleep 20