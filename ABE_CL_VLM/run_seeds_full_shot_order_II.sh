#!/bin/bash

GPU_ID=0
EXP_BASE_NAME="./Output_Order2"  
SEEDS=(0)

TASKS=(
    "stanford_cars"
    "aircraft"
    "oxford_pets"
    "food101"
    "sun397"
    "mnist"
    "flowers"
    "dtd"
    "caltech101"
    "eurosat"
)

set -e 

echo "============================================================"
echo "Starting automated training script..."
echo "Scheduled seeds to run: ${SEEDS[*]}"
echo "Using GPU: $GPU_ID"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ">>> Starting full sequence training with random seed SEED: $SEED"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    for TASK in "${TASKS[@]}"; do
        echo "--> [SEED $SEED] Training task: $TASK"
        
        case "$TASK" in
            "stanford_cars") EPOCH_ARGS="10 30 0" ;;
            "aircraft")      EPOCH_ARGS="5 40 0" ;;
            "oxford_pets")   EPOCH_ARGS="5 0 0" ;;
            "food101")       EPOCH_ARGS="5 5 0" ;;
            "sun397")        EPOCH_ARGS="2 4 0" ;;
            "mnist")         EPOCH_ARGS="5 20 0" ;;
            "flowers")       EPOCH_ARGS="10 30 0" ;;
            "dtd")           EPOCH_ARGS="20 0 0" ;;
            "caltech101")    EPOCH_ARGS="5 20 0" ;;
            "eurosat")       EPOCH_ARGS="10 0 0" ;;
            *)               EPOCH_ARGS="5 0 0" ;; 
        esac
        
        python -m xtail_ABE \
            --dataset "$TASK" \
            --gpu $GPU_ID \
            --seed "$SEED" \
            --exp_base "$EXP_BASE_NAME" \
            --epochs $EPOCH_ARGS \
            --task_order "${TASKS[@]}"
            
        echo "--> [SEED $SEED] Task $TASK training complete!"
    done
    
    echo ">>> All tasks for seed SEED: $SEED are finished! Models saved in ${EXP_BASE_NAME}_seed_${SEED}/"

    EXP_ROOT="${EXP_BASE_NAME}_seed_${SEED}"

    echo ">>> [SEED $SEED] Starting evaluation for Order II ..."
    python -m xtail_eval \
        --exp_root "$EXP_ROOT" \
        --gpu "$GPU_ID" \
        --target_step 0 \
        --task_order "${TASKS[@]}"

    echo ">>> [SEED $SEED] Evaluation complete!"

done

echo "============================================================"
echo "All experiments for all seeds have been completed!"
echo "============================================================"