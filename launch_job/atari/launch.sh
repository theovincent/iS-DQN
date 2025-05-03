SHARED_ARGS="--features 32 64 64 512 --replay_buffer_capacity 1_000_000 --batch_size 32 --update_horizon 1 --gamma 0.99 \
    --horizon 27_000 --n_epochs 100 --n_training_steps_per_epoch 250_000 --data_to_update 4 --n_initial_samples 20_000 \
    --epsilon_end 0.01 --epsilon_duration 250_000"

GAME="Assault"
N_BELLMAN_ITERATIONS=1  # 1 3 5 10
LAYER_NORM=1  # 0 1
ARCHITECTURE_TYPE="cnn"  # cnn impala
TARGET_UPDATE_FREQ=8000
TARGET_SYNC_FREQ=30

PLATFORM="normal/cluster"  # nhrfau/cluster normal/cluster normal/local

if [ $PLATFORM == "normal/local" ]
then
    SHARED_ARGS="$SHARED_ARGS --tmux_name slimdqn"
fi
if [ $LAYER_NORM == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --layer_norm"
fi

SHARED_ARGS="$SHARED_ARGS --target_update_frequency $TARGET_UPDATE_FREQ --architecture_type $ARCHITECTURE_TYPE"

# ----- L2 Loss -----
L2_ARGS="--learning_rate 6.25e-5"

DQN_ARGS="--experiment_name L2_LN${LAYER_NORM}_${ARCHITECTURE_TYPE}_T${TARGET_UPDATE_FREQ}_${GAME}"
# launch_job/atari/${PLATFORM}_dqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS $DQN_ARGS
# launch_job/atari/${PLATFORM}_dqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $DQN_ARGS

ISDQN_ARGS="--experiment_name L2_K${N_BELLMAN_ITERATIONS}_LN${LAYER_NORM}_${ARCHITECTURE_TYPE}_T${TARGET_UPDATE_FREQ}_D${TARGET_SYNC_FREQ}_${GAME} \
    --n_bellman_iterations $N_BELLMAN_ITERATIONS --target_sync_frequency $TARGET_SYNC_FREQ"
# launch_job/atari/${PLATFORM}_isdqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $L2_ARGS $ISDQN_ARGS
# launch_job/atari/${PLATFORM}_isdqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $L2_ARGS $ISDQN_ARGS

# ----- HL Loss -----
HL_ARGS="--learning_rate 2.5e-4 --n_bins 51 --min_value -10 --max_value 10 --sigma 0.294117647"

HLDQN_ARGS="--experiment_name HL_LN${LAYER_NORM}_${ARCHITECTURE_TYPE}_T${TARGET_UPDATE_FREQ}_${GAME}"
# launch_job/atari/${PLATFORM}_hldqn.sh --first_seed 1 --last_seed 3 --n_parallel_seeds 1 $SHARED_ARGS $HL_ARGS $HLDQN_ARGS
# launch_job/atari/${PLATFORM}_hldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $HL_ARGS $HLDQN_ARGS 

ISHLDQN_ARGS="--experiment_name HL_K${N_BELLMAN_ITERATIONS}_LN${LAYER_NORM}_${ARCHITECTURE_TYPE}_T${TARGET_UPDATE_FREQ}_D${TARGET_SYNC_FREQ}_${GAME} \
    --n_bellman_iterations $N_BELLMAN_ITERATIONS --target_sync_frequency $TARGET_SYNC_FREQ"
# launch_job/atari/${PLATFORM}_ishldqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $HL_ARGS $ISHLDQN_ARGS
# launch_job/atari/${PLATFORM}_ishldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $HL_ARGS $ISHLDQN_ARGS