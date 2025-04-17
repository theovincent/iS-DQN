SHARED_ARGS="--replay_buffer_capacity 500_000 --batch_size 32 --update_horizon 1 --gamma 0.99 --learning_rate 1e-3 \
    --horizon 1_000 --architecture_type fc --n_epochs 50 --n_training_steps_per_epoch 10_000 --data_to_update 1 \
    --n_initial_samples 1_000 --epsilon_end 0.01 --epsilon_duration 1_000"

FEATURE=100
N_NETWORS=5
TARGET_UPDATE_FREQ=800
TARGET_SYNC_FREQ=5

PLATFORM="normal/cluster"  # normal/local

if [ $PLATFORM == "normal/local" ]
then
    SHARED_ARGS="$SHARED_ARGS --tmux_name slimdqn"
fi

SHARED_ARGS="$SHARED_ARGS --features $FEATURE $FEATURE --target_update_frequency $TARGET_UPDATE_FREQ"

# ----- L2 Loss -----
L2_ARGS="--experiment_name L2_n_net${N_NETWORS}_target_freq${TARGET_UPDATE_FREQ}_sync_freq${TARGET_SYNC_FREQ}"

launch_job/lunar_lander/${PLATFORM}_dqn.sh --first_seed 6 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS
sleep 40

iDQN_ARGS="--n_networks $N_NETWORS --target_sync_frequency $TARGET_SYNC_FREQ"
launch_job/lunar_lander/${PLATFORM}_idqn.sh --first_seed 6 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $iDQN_ARGS
sleep 40

aGIDQN_ARGS="--n_networks $N_NETWORS"
launch_job/lunar_lander/${PLATFORM}_agidqn.sh --first_seed 6 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $aGIDQN_ARGS
sleep 40

GIDQN_ARGS="--n_networks $N_NETWORS"
launch_job/lunar_lander/${PLATFORM}_gidqn.sh --first_seed 6 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $GIDQN_ARGS
sleep 600

# ----- KL Loss -----
KL_ARGS="--experiment_name KL_n_net${N_NETWORS}_target_freq${TARGET_UPDATE_FREQ}_sync_freq${TARGET_SYNC_FREQ} \
    --n_bins 51 --min_value -100 --max_value 100 --sigma 2.94117647"

launch_job/lunar_lander/${PLATFORM}_hldqn.sh --first_seed 6 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS
sleep 40

iHLDQN_ARGS="--n_networks $N_NETWORS --target_sync_frequency $TARGET_SYNC_FREQ"
launch_job/lunar_lander/${PLATFORM}_ihldqn.sh --first_seed 6 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS $iHLDQN_ARGS
sleep 40

aGIHLDQN_ARGS="--n_networks $N_NETWORS"
launch_job/lunar_lander/${PLATFORM}_agihldqn.sh --first_seed 6 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS $aGIHLDQN_ARGS
sleep 40

GIHLDQN_ARGS="--n_networks $N_NETWORS"
launch_job/lunar_lander/${PLATFORM}_gihldqn.sh --first_seed 6 --last_seed 20 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS $GIHLDQN_ARGS