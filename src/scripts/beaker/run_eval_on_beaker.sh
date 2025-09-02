MODEL_NAME=$1 #allenai/FlexOlmo-7x7B-1T-RT
TASK_NAME=$2 #mc9
BASE_OUTPUT_DIR=$3 #/path/to/eval_results
GPUS=$4 #2

gantry run \
    --name eval-${MODEL_NAME//\//-}-${TASK_NAME} \
    --weka oe-training-default:/weka/oe-training-default \
    --install "bash src/scripts/eval/setup_eval_env.sh;" \
    --budget ai2/oe-training \
    --workspace ai2/OLMo-modular \
    --cluster ai2/jupiter-cirrascale-2 \
    --priority urgent \
    --gpus $GPUS \
    --env-secret HF_TOKEN=SEWONM_HF_TOKEN -- \
    bash src/scripts/eval/run_eval.sh \
        ${MODEL_NAME} \
        ${TASK_NAME} \
        ${BASE_OUTPUT_DIR} \
        ${GPUS}
