SHARED_ARGS="--features 32 64 64 512 --replay_buffer_capacity 1_000_000 --batch_size 32 --update_horizon 1 --gamma 0.99 \
    --horizon 27_000 --architecture_type cnn --n_epochs 50 --n_training_steps_per_epoch 250_000 --data_to_update 4 \
    --n_initial_samples 20_000 --epsilon_end 0.01 --epsilon_duration 250_000"

N_NETWORS=3
TARGET_UPDATE_FREQ=8000
TARGET_SYNC_FREQ=30
GAME="Assault"

PLATFORM="normal/cluster"  # nhrfau/cluster normal/local

if [ $PLATFORM == "normal/local" ]
then
    SHARED_ARGS="$SHARED_ARGS --tmux_name slimdqn"
fi

SHARED_ARGS="$SHARED_ARGS --target_update_frequency $TARGET_UPDATE_FREQ"

# ----- L2 Loss -----
L2_ARGS="--experiment_name L2_K${N_NETWORS}_T${TARGET_UPDATE_FREQ}_${GAME} \
    --learning_rate 6.25e-5"

launch_job/atari/${PLATFORM}_dqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS
launch_job/atari/${PLATFORM}_dqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS
sleep 20

iDQN_ARGS="--n_networks $N_NETWORS --target_sync_frequency $TARGET_SYNC_FREQ"
launch_job/atari/${PLATFORM}_idqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS $iDQN_ARGS
launch_job/atari/${PLATFORM}_idqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $iDQN_ARGS
sleep 20

GIDQN_ARGS="--n_networks $N_NETWORS"
launch_job/atari/${PLATFORM}_gidqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS $GIDQN_ARGS
launch_job/atari/${PLATFORM}_gidqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $GIDQN_ARGS
sleep 20

SHAREDGIDQN_ARGS="--n_networks $N_NETWORS"
launch_job/atari/${PLATFORM}_sharedgidqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS $SHAREDGIDQN_ARGS
launch_job/atari/${PLATFORM}_sharedgidqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $SHAREDGIDQN_ARGS
sleep 20

CROWNGIDQN_ARGS="--n_networks $N_NETWORS"
launch_job/atari/${PLATFORM}_crowngidqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS $CROWNGIDQN_ARGS
launch_job/atari/${PLATFORM}_crowngidqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $CROWNGIDQN_ARGS
sleep 40

# ----- KL Loss -----
KL_ARGS="--experiment_name KL_K${N_NETWORS}_T${TARGET_UPDATE_FREQ}_${GAME} \
    --learning_rate 2.5e-4 --n_bins 51 --min_value -10 --max_value 10 --sigma 0.294117647"

launch_job/atari/${PLATFORM}_hldqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $KL_ARGS
launch_job/atari/${PLATFORM}_hldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS
sleep 20

iHLDQN_ARGS="--n_networks $N_NETWORS --target_sync_frequency $TARGET_SYNC_FREQ"
launch_job/atari/${PLATFORM}_ihldqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $KL_ARGS $iHLDQN_ARGS
launch_job/atari/${PLATFORM}_ihldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS $iHLDQN_ARGS
sleep 20

GIHLDQN_ARGS="--n_networks $N_NETWORS"
launch_job/atari/${PLATFORM}_gihldqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $KL_ARGS $GIHLDQN_ARGS
launch_job/atari/${PLATFORM}_gihldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $KL_ARGS $GIHLDQN_ARGS