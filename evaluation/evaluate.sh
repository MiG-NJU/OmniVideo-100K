python evaluation.py \
    --dataset "omnivideo_test" \
    --dataset_dir "OmniVideo-Test" \
    --model_type "qwen25_omni" \
    --model_path "Qwen/Qwen2.5-Omni-7B"

python evaluation_gemini.py \
    --dataset "omnivideo_test" \
    --dataset_dir "OmniVideo-Test" \
    --model_name "gemini-3.1-pro-preview" \
    --api_key "" \
    --base_url ""
