import os
import json
import random
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


root_path = "" # Replace with the actual root path of your datasets
datasets = {"dataset name": {"oe": "oe.jsonl",  # Replace with the actual dataset name and file names
                             "mcq": "mcq.jsonl"}}
save_file = os.path.join(root_path, "xxx.json")

MCQ_PROMPT = [
    ["\nSelect from the following choices.", ""],
    ["\nChoose between the following options.", ""],
    ["\nAnswer with the option's letter from the given choices directly.", ""],
    ["", "\nPlease select the correct answer from the options above."],
    ["", "\nAnswer with the option's letter from the given choices directly."],
    ["", "\nAnswer with the option's letter directly (e.g., A, B, C, or D)."],
    ["", "\nAnswer with the option's letter (A, B, C, or D) from the given choices directly."],
    ["", "\nRespond with only the letter (A, B, C, or D) of the correct option."]
]
Event_Sequence_Ordering_OE_PROMPT = "\nPlease directly answer with the correct order of all events' indices, separated by commas."
MIX_Event_Sequence_Ordering_MCQ = True


def seprate_av(item, root_path):
    id = item["id"]
    video_file = os.path.join(root_path, item["video_path"])
    sep_folder = os.path.join(root_path, "videos_sep")

    v_path = os.path.join(sep_folder, f"{id}.mp4")

    # audio codec
    result = subprocess.run([
        "ffprobe",
        "-print_format", "json",
        "-show_streams", video_file
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
    info = json.loads(result.stdout)
    audio_codec = None
    # if "streams" not in info:
    #     print(video_file)
    #     return id, None, None

    for stream in info["streams"]:
        if stream["codec_type"] == "audio":
            audio_codec = stream["codec_name"]
            break
    if audio_codec == "opus":
        a_path = os.path.join(sep_folder, f"{id}.opus")
    elif audio_codec == "aac":
        a_path = os.path.join(sep_folder, f"{id}.aac")
    else:
        print("different acodec: {audio_codec}")
        return id, None, None

    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", video_file,
            "-map", "0:v", "-c:v", "copy", v_path, # Extract video stream only
            "-map", "0:a", "-c:a", "copy", a_path, # Extract audio stream only
            "-hide_banner", "-loglevel", "error"
        ], capture_output=True, text=True, check=True)
        log = result.stderr.lower()

        bad_keywords = [
            "missing picture in access unit", 
            "no frame", 
            "invalid data", 
            "corrupt"
        ]
        for keyword in bad_keywords:
            if keyword in log:
                print(f"Found corruption: {keyword}")
                return id, None, None

        if result.returncode != 0:
            print(f"FFmpeg exited with error code {result.returncode}")
            return id, None, None

        return id, os.path.relpath(v_path, root_path), os.path.relpath(a_path, root_path)
    except Exception as e:
        print(f"error: {e}")
        return id, None, None


if __name__ == "__main__":
    random.seed(42)
    data_sharegpt = []
    for dataset, files in datasets.items():
        sub_root = os.path.join(root_path, dataset)

        for type, file in files.items():
            train_file = os.path.join(sub_root, file)
            temp_mapping_file = os.path.join(sub_root, "temp_av_mapping.json")
            sep_folder = os.path.join(sub_root, "videos_sep")
            os.makedirs(sep_folder, exist_ok=True)

            with open(train_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f.readlines()]
            data_mapping = {item["id"]: item for item in data}

            av_data = []
            if os.path.exists(temp_mapping_file):
                with open(temp_mapping_file, "r", encoding="utf-8") as f:
                    av_data = [json.loads(line) for line in f.readlines()]
            av_mapping = {item["id"]: item for item in av_data}

            pending_items = [item for item in data_mapping.values() if item["id"] not in av_mapping]
            print(len(pending_items))
            if pending_items:
                max_workers = min(16, os.cpu_count())
                with open(temp_mapping_file, "a", encoding="utf-8") as f:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(seprate_av, item, sub_root) for item in pending_items]
                        for future in tqdm(as_completed(futures), total=len(pending_items), desc="Extracting AV", mininterval=1.0):
                            vid_id, v_path, a_path = future.result()
                            if v_path and a_path:
                                av_mapping[vid_id] = {"id": vid_id, "v_path": v_path, "a_path": a_path}
                                f.write(json.dumps(av_mapping[vid_id]) + "\n")
                                f.flush()

            event_sequence_ordering_question_type = ["textual", "indexed"]
            event_sequence_ordering_question_idx = 0
            for idx, item in tqdm(enumerate(data)):
                if item["id"] in av_mapping:
                    v_path = av_mapping[item["id"]]["v_path"]
                    a_path = av_mapping[item["id"]]["a_path"]
                else:
                    print(f"skip {item['id']}")
                    continue

                if type == "oe":
                    question = item["Q"]
                    if item["task"] == "event_sequence_ordering":
                        for no, option in enumerate(item["Options"]):
                            question += f"\n({no + 1}) {option}"
                        question += Event_Sequence_Ordering_OE_PROMPT

                    if isinstance(item["A"], list):
                        if item["task"] == "event_sequence_ordering":
                            answer = ",".join([str(ord(char) - 64) for char in item["A"]])
                    else:
                        answer = item["A"]
                elif type == "mcq":
                    pmt1, pmt2 = MCQ_PROMPT[idx % len(MCQ_PROMPT)]

                    if item["task"] == "event_sequence_ordering":
                        format_type = event_sequence_ordering_question_type[event_sequence_ordering_question_idx]
                        question = item[f"Q_{format_type}"] + pmt1
                        for no, option in enumerate(item[f"Options_{format_type}"]):
                            question += f"\n{chr(65 + no)}. {option}"

                        if MIX_Event_Sequence_Ordering_MCQ:
                            event_sequence_ordering_question_idx = (event_sequence_ordering_question_idx + 1) % 2
                    else:
                        question = item["Q"] + pmt1
                        for no, option in enumerate(item["Options"]):
                            question += f"\n{chr(65 + no)}. {option}"
                    question += pmt2

                    answer = item["A"]

                data_sharegpt.append({"messages": [{"content": f"<video><audio>{question}",
                                                    "role": "user"},
                                                   {"content": answer,
                                                    "role": "assistant"}],
                                      "videos": [os.path.join(dataset, v_path)],
                                      "audios": [os.path.join(dataset, a_path)],
                                      "question_id": item["question_id"]})

    random.shuffle(data_sharegpt)
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(data_sharegpt, f, indent=2, ensure_ascii=False)
