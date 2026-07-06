import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
from tqdm import tqdm
from models import Qwen25_Omni, Qwen3_Omni, Uni_Moe_2_Omni, video_SALMONN2_plus, OmniVinci, MiniCPM_o_45, VITA_15, VITA_15_sft
from datasets import Video_MME, Daily_Omni, OmniVideoBench, JointAVBench, FutureOmni, OmniVideo_Test, Video_MME_v2


MAX_FRAMES = 64  # specifically, 16 for VITA-1.5, 64 for other models.


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, choices=["video_mme", "daily_omni", "omnivideobench", "jointavbench", "futureomni", "omnivideo_test", "video_mme_v2"], required=True, help="Dataset to use for evaluation.")
    args.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    args.add_argument("--model_type", type=str, choices=["qwen25_omni", "qwen3_omni", "uni_moe_2_omni", "video_salmonn2_plus", "omnivinci", "minicpm_o_45", "vita_15", "vita_15_sft"], required=True, help="Model type to use for evaluation.")
    args.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    args.add_argument("--results_file", type=str, default=None, help="Path to save the evaluation results.")
    args = args.parse_args()

    if not args.results_file:
        args.results_file = os.path.join(f"results/{args.dataset}/{args.model_path.split('/')[-1]}.jsonl")
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    if args.dataset == "video_mme":
        dataset = Video_MME(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "daily_omni":
        dataset = Daily_Omni(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "omnivideobench":
        dataset = OmniVideoBench(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "jointavbench":
        dataset = JointAVBench(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "futureomni":
        dataset = FutureOmni(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "omnivideo_test":
        dataset = OmniVideo_Test(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "video_mme_v2":
        dataset = Video_MME_v2(dataset_dir=args.dataset_dir, results_file=args.results_file)

    print(f"[INFO] Total questions: {len(dataset.data)}, Unprocessed questions: {len(dataset.unprocessed_data)}")

    if len(dataset.unprocessed_data) > 0:
        if args.model_type == "qwen25_omni":
            predictor = Qwen25_Omni(model_path=args.model_path)
        elif args.model_type == "qwen3_omni":
            predictor = Qwen3_Omni(model_path=args.model_path)
        elif args.model_type == "uni_moe_2_omni":
            predictor = Uni_Moe_2_Omni(model_path=args.model_path)
        elif args.model_type == "video_salmonn2_plus":
            predictor = video_SALMONN2_plus(model_path=args.model_path)
        elif args.model_type == "omnivinci":
            predictor = OmniVinci(model_path=args.model_path)
        elif args.model_type == "minicpm_o_45":
            predictor = MiniCPM_o_45(model_path=args.model_path)
        elif args.model_type == "vita_15":
            predictor = VITA_15(model_path=args.model_path)
        elif args.model_type == "vita_15_sft":
            predictor = VITA_15_sft(model_path=args.model_path)
        predictor.model.eval()

        with open(args.results_file, "a", encoding="utf-8") as f:
            for item in tqdm(dataset.unprocessed_data):
                try:
                    if "vita_15" in args.model_type:
                        item["model_answer"] = predictor.inference(item, max_frames=16)
                    else:
                        item["model_answer"] = predictor.inference(item, max_frames=MAX_FRAMES)
                    f.write(json.dumps(item) + "\n")
                    f.flush()
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Fail to process {item['question_id']}: {e}")
                    traceback.print_exc()

    dataset.calculate_accuracy()
