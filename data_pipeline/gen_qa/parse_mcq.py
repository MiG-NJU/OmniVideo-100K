# coding=gbk
import os
import re
import json
import argparse
from tqdm import tqdm


single_tasks = ["fine_grained_perception", "scene_transformation_detection", "context_understanding"]
special_tasks = ["event_sequence_ordering"]
cross_tasks = ["comparison", "sentiment_analysis", "summarization",
               "causal_reasoning", "future_prediction", "hypothetical_reasoning"]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, required=True)

    args = parser.parse_args()
    args.script_file = os.path.join(args.root_path, "script.jsonl")
    args.qa_folder = os.path.join(args.root_path, "qa_files")
    return args


def get_data(qa_file):
    data_dict = {}
    if os.path.exists(qa_file):
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data_dict[item["id"]] = item

    data = []
    for item in data_dict.values():
        if item["mcq"] and item["mcq"] != "NONE":
            data.append(item)
    return data


def get_save_json(question_id, item, task, Q, Options, A, analysis, explanation, evidence=None, subtask=None):
    data = {"id": item["id"],
            "question_id": question_id,
            "video_path": item["video_path"],
            "duration": item["duration"],
            "search_tag": item.get("search_tag", None),
            "language": item.get("language", None),
            "metadata": item.get("metadata", None),
            "resolution": item.get("resolution", None),
            "task": task,
            "subtask": subtask,
            "Q": Q,
            "Options": Options,
            "A": A,
            "analysis": analysis,
            "evidence": evidence,
            "explanation": explanation}
    if subtask:
        data["subtask"] = subtask
    else:
        data.pop("subtask")
    if evidence:
        data["evidence"] = evidence
    else:
        data.pop("evidence")

    return json.dumps(data) + "\n"


if __name__ == "__main__":
    args = get_args()

    for task in single_tasks:
        mcq_file = os.path.join(args.qa_folder, f"{task}_mcq.jsonl")

        if os.path.exists(mcq_file):
            print(f"{task} begin")
            with open(mcq_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f.readlines()]

            save_file = os.path.join(args.root_path, f"{task}_mcq.jsonl")
            with open(save_file, "w", encoding="utf-8") as f_out:
                for index, item in tqdm(enumerate(data)):
                    mcq = item["mcq"][0] if isinstance(item["mcq"], list) else item["mcq"]
                    correct_pos_index = index % 4
                    final_options = list(mcq["distractors"].values())[: 3]
                    final_options.insert(correct_pos_index, mcq["correct_option"])

                    A = chr(65 + correct_pos_index)
                    explanation = mcq["explanation"] if "explanation" in mcq else mcq["distractors"]["explanation"]

                    text = get_save_json(item["question_id"], item, task, item["Q"], final_options, A, item["analysis"], explanation, subtask=item.get("subtask", None))
                    f_out.write(text)
                    f_out.flush()
        print(f"{task} completed")

    for task in cross_tasks:
        mcq_file = os.path.join(args.qa_folder, f"{task}_mcq.jsonl")
        data = get_data(mcq_file)

        if data:
            save_file = os.path.join(args.root_path, f"{task}_mcq.jsonl")
            with open(save_file, "w", encoding="utf-8") as f_out:
                index = 0
                for item in tqdm(data):
                    qa_idx = 0
                    for i in item["mcq"]:
                        if "content" not in i or not i["content"]:
                            continue

                        mcq = i["content"]
                        correct_pos_index = index % 4
                        final_options = [mcq["options"]["B"], mcq["options"]["C"], mcq["options"]["D"]]
                        final_options.insert(correct_pos_index, mcq["options"]["A"])

                        if "explanation" not in mcq:
                            print(mcq)
                        Q = mcq["question"]
                        A = chr(65 + correct_pos_index)
                        evidence = mcq["evidence"]
                        explanation = mcq["explanation"]
                        i.pop("content")
                        text = get_save_json(f"{item["id"]}_{task}_{qa_idx}", item, task, Q, final_options, A, i, explanation, evidence)
                        f_out.write(text)
                        f_out.flush()
                        qa_idx += 1
                        index += 1
        print(f"{task} completed")

    for task in special_tasks:
        if task == "event_sequence_ordering":
            mcq_file = os.path.join(args.qa_folder, f"{task}_mcq.jsonl")

            if os.path.exists(mcq_file):
                print(f"{task} begin")
                with open(mcq_file, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f.readlines()]

                save_file = os.path.join(args.root_path, f"{task}_mcq.jsonl")
                with open(save_file, "w", encoding="utf-8") as f_out:
                    for index, item in tqdm(enumerate(data)):
                        mcq = item["mcq"]
                        events = list(mcq["rewritten_events"].values())

                        correct_pos_index = index % 4
                        options = [re.findall(r"[A-Z]", distractor) for distractor in mcq["distractors"]]
                        options.insert(correct_pos_index, item["A"])
                        options_events = []
                        option_num = []
                        for option in options:
                            option_num.append(" → ".join([f"({ord(i) - 64})" for i in option]))
                            options_events.append(" → ".join([f"{events[ord(i) - 65]}" for i in option]))

                        events_str = ""
                        for no, option in enumerate(events):
                            events_str += f"\n({no + 1}) {option}"
                        data = {"id": item["id"],
                                "question_id": item["question_id"],
                                "video_path": item["video_path"],
                                "duration": item["duration"],
                                "search_tag": item.get("search_tag", None),
                                "language": item.get("language", None),
                                "metadata": item.get("metadata", None),
                                "resolution": item.get("resolution", None),
                                "task": task,
                                "Q_indexed": item["Q"] + events_str,
                                "Options_indexed": option_num,
                                "Q_textual": item["Q"].replace("following", "").replace("  ", " "),
                                "Options_textual": options_events,
                                "A": chr(65 + correct_pos_index),
                                "events": events,
                                "analysis": item["analysis"]}
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                        f_out.flush()
            print(f"{task} completed")
