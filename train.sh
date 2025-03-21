#!/bin/bash
# filepath: run_experiment.sh

# 默认参数
SEED=156862
LR=1e-4
GAMMA=0.95
BATCH_SIZE=128
EXPECTILE=0.5

SAVE_MODEL_DIR="models/${EXPECTILE}${BATCH_SIZE}"
LOG_DIR="logs/expectile${EXPECTILE}${BATCH_SIZE}"
NON_IID_LEVEL=1
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED=$2
            shift 2
            ;;
        --lr)
            LR=$2
            shift 2
            ;;
        --save_model_dir)
            SAVE_MODEL_DIR=$2
            shift 2
            ;;
        --log_dir)
            LOG_DIR=$2
            shift 2
            ;;
        --non_iid_level)
            NON_IID_LEVEL=$2
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE=$2
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 设置实验名称，用于日志目录
EXPERIMENT_NAME="pareto_exp_expectile${EXPECTILE}_batch${BATCH_SIZE}_$(date +'%Y%m%d%H%M%S')"

echo "开始实验: ${EXPERIMENT_NAME}"
echo "Expectile: $EXPECTILE, 非独立同分布等级: $NON_IID_LEVEL,BatchSize: $BATCH_SIZE,种子: $SEED, 学习率: $LR, 折扣因子: $GAMMA,"

# 运行程序
python run.py \
    --seed $SEED \
    --log_dir $LOG_DIR \
    --save_model_dir $SAVE_MODEL_DIR \
    --lr $LR \
    --gamma $GAMMA \
    --expectile $EXPECTILE \
    --non_iid_level $NON_IID_LEVEL \
    --batch_size $BATCH_SIZE


echo "实验 ${EXPERIMENT_NAME} 已完成"