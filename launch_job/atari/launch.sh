SHARED_ARGS="--features 32 64 64 512 --replay_buffer_capacity 1_000_000 --batch_size 32 --update_horizon 1 --gamma 0.99 \
    --horizon 27_000 --n_epochs 40 --n_training_steps_per_epoch 250_000 --data_to_update 4 --n_initial_samples 20_000 \
    --epsilon_end 0.01 --epsilon_duration 250_000 --learning_rate 6.25e-5"

GAME="Breakout"
N_BELLMAN_ITERATIONS=1  # 1 3 5 10
LAYER_NORM=1  # 0 1
BATCH_NORM=0  # 0 1
ARCHITECTURE_TYPE="cnn"  # cnn impala
TARGET_UPDATE_FREQ=8000

PLATFORM="normal/local"  # nhrfau/cluster normal/cluster normal/local

if [ $PLATFORM == "normal/local" ]
then
    SHARED_ARGS="$SHARED_ARGS --tmux_name slimdqn"
fi
if [ $LAYER_NORM == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --layer_norm"
fi
if [ $BATCH_NORM == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --batch_norm"
fi


SHARED_ARGS="$SHARED_ARGS --target_update_frequency $TARGET_UPDATE_FREQ --architecture_type $ARCHITECTURE_TYPE"
SHARED_NAME="FP16_LN${LAYER_NORM}_BN${BATCH_NORM}_${ARCHITECTURE_TYPE}_T${TARGET_UPDATE_FREQ}"
# ----- L2 Loss -----

DQN_ARGS="--experiment_name L2_${SHARED_NAME}_${GAME}"
# launch_job/atari/${PLATFORM}_dqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $DQN_ARGS
# launch_job/atari/${PLATFORM}_dqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $DQN_ARGS
# launch_job/atari/${PLATFORM}_tfdqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $DQN_ARGS
# launch_job/atari/${PLATFORM}_tfdqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $DQN_ARGS

ISDQN_ARGS="--experiment_name L2_K${N_BELLMAN_ITERATIONS}_${SHARED_NAME}_${GAME} --n_bellman_iterations $N_BELLMAN_ITERATIONS"
# launch_job/atari/${PLATFORM}_isdqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SHARED_ARGS $ISDQN_ARGS
# launch_job/atari/${PLATFORM}_isdqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $ISDQN_ARGS
