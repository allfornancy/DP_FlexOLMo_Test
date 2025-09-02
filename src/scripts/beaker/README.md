
# Launch experiments with Beaker

Replace `torchrun` command with beaker's launch script, e.g. 

```bash
CHECKPOINT_PATH=/path/to/checkpoint
torchrun [OPTS] src/scripts/train/OLMo2-7B-anneal.py anneal-run ${CHECKPOINT_PATH} \
    --trainer.callbacks.profiler.enabled=true \
    --trainer.callbacks.lm_evaluator.enabled=false \
    --train_module.float8_config.enabled=true \
    --trainer.hard_stop.value=20 \
    --trainer.hard_stop.unit=steps
```

is replaced by

```bash
python src/scripts/beaker/launch.py launch [cluster] \
    --launch.num_nodes=2 \
    --launch.workspace=workspace-name \
    --launch.priority=high -- src/scripts/train/OLMo2-7B-anneal.py anneal-run ${CHECKPOINT_PATH} \
    --trainer.callbacks.profiler.enabled=true \
    --trainer.callbacks.lm_evaluator.enabled=false \
    --train_module.float8_config.enabled=true \
    --trainer.hard_stop.value=20 \
    --trainer.hard_stop.unit=steps
```

