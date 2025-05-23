set -x

MODEL_PATH=Qwen2.5-VL-7B-Instruct  # replace it with your local file path

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=dataset/parquet@train \
    data.val_files=dataset/parquet@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_mmeureka_grpo \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=8
