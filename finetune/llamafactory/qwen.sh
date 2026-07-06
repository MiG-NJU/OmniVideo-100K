PYTHONWARNINGS="ignore" FORCE_TORCHRUN=1 llamafactory-cli train qwen2_5omni_full_sft.yaml
python3 ./scripts/qwen_omni_merge.py save_full \
  --model_path="Qwen/Qwen2.5-Omni-7B" \
  --thinker_path="" \
  --save_path=""

PYTHONWARNINGS="ignore" FORCE_TORCHRUN=1 llamafactory-cli train qwen3_omni_full_sft.yaml
python3 ./scripts/qwen_omni_merge.py save_full \
  --model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct" \
  --thinker_path="" \
  --save_path=""
