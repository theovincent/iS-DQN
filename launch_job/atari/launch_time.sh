SHARED_ARGS="--features 32 64 64 512 --replay_buffer_capacity 1_000_000 --batch_size 32 --update_horizon 1 --gamma 0.99 \
    --horizon 27_000 --n_epochs 1 --n_training_steps_per_epoch 250_000 --data_to_update 4 --n_initial_samples 100 \
    --epsilon_end 0.01 --epsilon_duration 1 --learning_rate 6.25e-5 --disable_wandb"

GAME="Asterix"
LAYER_NORM=1  # 0 1
BATCH_NORM=0  # 0 1
TARGET_UPDATE_FREQ=8000
ANALYSIS=0 # 0 1

PLATFORM="normal/local"

for ARCHITECTURE_TYPE in cnn impala
do 
    SHARED_ARGS="--tmux_name slimdqn $SHARED_ARGS --target_update_frequency $TARGET_UPDATE_FREQ --architecture_type $ARCHITECTURE_TYPE"
    SHARED_NAME="LN${LAYER_NORM}_BN${BATCH_NORM}_${ARCHITECTURE_TYPE}_T${TARGET_UPDATE_FREQ}_A${ANALYSIS}"
    
    DQN_ARGS="--experiment_name L2_${SHARED_NAME}_${GAME}"
    launch_job/atari/${PLATFORM}_dqn.sh --first_seed 2 --last_seed 2 --n_parallel_seeds 1 $SHARED_ARGS $DQN_ARGS
    launch_job/atari/${PLATFORM}_tfdqn.sh --first_seed 2 --last_seed 2 --n_parallel_seeds 1 $SHARED_ARGS $DQN_ARGS

    for N_BELLMAN_ITERATIONS in 1 4 9 49
    do
        ISDQN_ARGS="--experiment_name L2_K${N_BELLMAN_ITERATIONS}_${SHARED_NAME}_${GAME} --n_bellman_iterations $N_BELLMAN_ITERATIONS"
        launch_job/atari/${PLATFORM}_isdqn.sh --first_seed 2 --last_seed 2 --n_parallel_seeds 1 $SHARED_ARGS $ISDQN_ARGS
    done
done