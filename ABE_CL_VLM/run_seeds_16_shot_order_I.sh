#!/bin/bash

GPU_ID=0
EXP_BASE_NAME="./Output_16shot_Order1"  
SEEDS=(0)                

TASKS=(
    "aircraft"
    "caltech101"
    "dtd"
    "eurosat"
    "flowers"
    "food101"
    "mnist"
    "oxford_pets"
    "stanford_cars"
    "sun397"
)

set -e 

echo "============================================================"
echo "Starting 16-shot automated training script (Order 1)..."
echo "Scheduled seeds to run: ${SEEDS[*]}"
echo "Using GPU: $GPU_ID"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ">>> [Order 1] Starting full sequence training with random seed SEED: $SEED"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    for TASK in "${TASKS[@]}"; do
        echo "--> [SEED $SEED] Training task: $TASK"
        
        case "$TASK" in
            "aircraft")      EPOCH_ARGS="5 40 0" ;;
            "caltech101")    EPOCH_ARGS="10 0 0" ;;
            "dtd")           EPOCH_ARGS="20 0 0" ;;
            "eurosat")       EPOCH_ARGS="1 25 0" ;;
            "flowers")       EPOCH_ARGS="25 0 0" ;;
            "food101")       EPOCH_ARGS="5 0 0" ;;
            "mnist")         EPOCH_ARGS="40 0 0" ;;
            "oxford_pets")   EPOCH_ARGS="10 0 0" ;;
            "stanford_cars") EPOCH_ARGS="30 0 0" ;;
            "sun397")        EPOCH_ARGS="4 0 0" ;;
            *)               EPOCH_ARGS="5 0 0" ;; 
        esac
        

        python -m xtail_ABE_16_shot \
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

    echo ">>> [SEED $SEED] Starting 16-shot evaluation for Order I ..."
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