#!/bin/bash
# filepath: run_experiment.sh

# 默认参数
SEED=156862
LR=1e-4
GAMMA=0.95
BATCH_SIZE=64
EXPECTILE=0.2
IS_TEST=1
EXCEL_DIR="results/expectile2"

SAVE_MODEL_DIR="models/${EXPECTILE}-${BATCH_SIZE}"
NON_IID_LEVEL=1
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --excel_dir)
            EXCEL_DIR=$2
            shift 2
            ;;
        --expectile)
            EXPECTILE=$2
            shift 2
            ;;
        --is_test)
            IS_TEST=$2
            shift 2
            ;;
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
EXPERIMENT_NAME="pareto_exp_expectile${EXPECTILE}_test_$(date +'%Y%m%d%H%M%S')"

echo "开始实验: ${EXPERIMENT_NAME}"
echo "Expectile: $EXPECTILE, 非独立同分布等级: $NON_IID_LEVEL,test: $IS_TEST"

# 运行程序
python run.py \
    --seed $SEED \
    --save_model_dir $SAVE_MODEL_DIR \
    --lr $LR \
    --gamma $GAMMA \
    --expectile $EXPECTILE \
    --non_iid_level $NON_IID_LEVEL \
    --batch_size $BATCH_SIZE\
    --is_test $IS_TEST\
    --expectile $EXPECTILE\
    --excel_dir $EXCEL_DIR


echo "实验 ${EXPERIMENT_NAME} 已完成"