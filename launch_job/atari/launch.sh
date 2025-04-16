SHARED_ARGS="--features 32 64 64 512 --replay_buffer_capacity 1_000_000 --batch_size 32 --update_horizon 1 --gamma 0.99 \
    --horizon 27_000 --architecture_type cnn --n_epochs 50 --n_training_steps_per_epoch 10_000 --data_to_update 4 \
    --n_initial_samples 1_000 --epsilon_end 0.01 --epsilon_duration 250_000"

N_NETWORS=5
TARGET_UPDATE_FREQ=8000
TARGET_SYNC_FREQ=30

PLATFORM="normal/cluster"  # nhrfau/cluster normal/local

SHARED_ARGS="$SHARED_ARGS --target_update_frequency $TARGET_UPDATE_FREQ"

# ----- L2 Loss -----
L2_ARGS="--experiment_name L2_n_net${N_NETWORS}_target_freq${TARGET_UPDATE_FREQ}_sync_freq${TARGET_SYNC_FREQ} \
    --learning_rate 6.25e-5"

launch_job/atari/${PLATFORM}_dqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS
launch_job/atari/${PLATFORM}_dqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS
sleep 20

iDQN_ARGS="--n_networks $N_NETWORS --target_sync_frequency $TARGET_SYNC_FREQ"
launch_job/atari/${PLATFORM}_idqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS $iDQN_ARGS
launch_job/atari/${PLATFORM}_idqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $iDQN_ARGS
sleep 20

aGIDQN_ARGS="--n_networks $N_NETWORS"
launch_job/atari/${PLATFORM}_agidqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS $aGIDQN_ARGS
launch_job/atari/${PLATFORM}_agidqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $aGIDQN_ARGS
sleep 20

GIDQN_ARGS="--n_networks $N_NETWORS"
launch_job/atari/${PLATFORM}_gidqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS $GIDQN_ARGS
launch_job/atari/${PLATFORM}_gidqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $GIDQN_ARGS
sleep 40

# ----- KL Loss -----
KL_ARGS="--experiment_name KL_n_net${N_NETWORS}_target_freq${TARGET_UPDATE_FREQ}_sync_freq${TARGET_SYNC_FREQ} \
    --learning_rate 2.5e-4 --n_bins 51 --min_value -10 --max_value 10 --sigma 0.294117647"

launch_job/atari/${PLATFORM}_hldqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $KL_ARGS
launch_job/atari/${PLATFORM}_hldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS
sleep 20

iHLDQN_ARGS="--n_networks $N_NETWORS --target_sync_frequency $TARGET_SYNC_FREQ"
launch_job/atari/${PLATFORM}_ihldqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $KL_ARGS $iHLDQN_ARGS
launch_job/atari/${PLATFORM}_ihldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS $iHLDQN_ARGS
sleep 20

aGIHLDQN_ARGS="--n_networks $N_NETWORS"
launch_job/atari/${PLATFORM}_agihldqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $KL_ARGS $aGIHLDQN_ARGS
launch_job/atari/${PLATFORM}_agihldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS $aGIHLDQN_ARGS
sleep 20

GIHLDQN_ARGS="--n_networks $N_NETWORS"
launch_job/atari/${PLATFORM}_gihldqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $KL_ARGS $GIHLDQN_ARGS
launch_job/atari/${PLATFORM}_gihldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS $GIHLDQN_ARGS