# This will read stream data from the public endpoints by default, but that might be a lot slower
# than reading data locally.
export DATA_ROOT="http://flexolmo-data.org"
export CHECKPOINTS=  # /path/to/checkpoints

EXPERT=news

# # With domain embeddings for router for each individual expert (e.g. news)
# DOMAIN_EMBEDDINGS_ROOT="src/data/domain_embeddings/grit"
# MODEL_PATH=${CHECKPOINTS}/OLMo2-7B-anneal-public-mix/step11921
# python src/scripts/upcycle/dense_to_expert_moe.py \
#     -m ${MODEL_PATH}-unsharded ${MODEL_PATH}-unsharded \
#     -e ${DOMAIN_EMBEDDINGS_ROOT}/public ${DOMAIN_EMBEDDINGS_ROOT}/${EXPERT} \
#     -t /path/to/checkpoints/olmoe-2x7b-public-${EXPERT}

MODEL_PATH=${CHECKPOINTS}/olmoe-2x7b-public-${EXPERT}  # upcycled from previous step
# MODEL_PATH=${CHECKPOINTS}/olmoe-2x7b-public-public  # if not using domain-specific embeddings
torchrun --nproc-per-node=8 src/scripts/train/OLMoE-2x7B-anneal.py olmoe-2x7B-${EXPERT}_top2_grit_learnbias \
    --trainer.callbacks.profiler.enabled=true \
    --dataset.mix_base_dir=${DATA_ROOT} \
    --dataset.mix=${EXPERT} \
    --trainer.max_duration.value=50_000_000_000 \
    --trainer.max_duration.unit=tokens \
    --trainer.load_path=${MODEL_PATH} \
    --model.block.feed_forward_moe.router.top_k=2 \
    --train_module.rank_microbatch_size=4096 \
    --train_module.scheduler.warmup_steps=2000 \
    --train_module.optim.lr=9e-4 \
    --trainer.save_folder=${CHECKPOINTS}/olmoe-2x7B-${EXPERT}_top2_grit_learnbias
