export API_KEY=""
export MODEL_NAME=""
export BASEURL_POOL=""
export TIMEOUT_LIMIT=300
export CONCURRENCY_LIMIT=50

echo "[RUN] Starting all steps..."

python 4_seg_visual.py --root_path <root_path>

python 5_check_script.py --root_path <root_path>

echo "All steps have finished."
