MODEL_PATH=$1
DOMAIN=$2
SOURCE_DATA=$3
OUTPUT_DIR=$4

NUM_PREFIXES=100 #00
PREFIX_LENGTH=32
GENERATION_LENGTH=256

python src/scripts/utils/extraction_analysis.py \
    --model_name ${MODEL_PATH} \
    --domain ${DOMAIN} \
    --source_data ${SOURCE_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --num_prefixes ${NUM_PREFIXES} \
    --num_generations_per_prefix 10 \
    --prefix_length ${PREFIX_LENGTH} \
    --generation_length ${GENERATION_LENGTH} \
    --matching_criteria "normalized_levenshtein" \
    --similarity_threshold 0.9 \
    --save_generated_samples