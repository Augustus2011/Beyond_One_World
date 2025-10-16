import json
import re
import os
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def extract_pairs(text: str) -> List[Tuple[str, str]]:

    text = re.sub(r"[`]+", "", text).strip()
    thinking_tags = re.findall(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    acting_tags = re.findall(r"<acting>(.*?)</acting>", text, re.DOTALL)
    return list(zip(thinking_tags, acting_tags))

def embed(text: str) -> torch.Tensor:

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings

def get_best_similarity(pairs: List[Tuple[str, str]]) -> Tuple[float, Tuple[str, str]]:

    max_sim = -1.0
    best_pair = ("", "")
    for thinking, acting in pairs:
        t_embed = embed(thinking)
        a_embed = embed(acting)
        sim = cosine_similarity(t_embed, a_embed)[0][0]
        if sim > max_sim:
            max_sim = sim
            best_pair = (thinking, acting)
    return max_sim, best_pair

def process_entry(entry: dict) -> dict:

    text = entry.get("think_act_analysis", "")
    pairs = extract_pairs(text)
    
    if not pairs:
        entry["max_think_act_sim"] = 0.0
        entry["most_similar_pair"] = {"thinking": "", "acting": ""}
    else:
        max_sim, best_pair = get_best_similarity(pairs)
        entry["max_think_act_sim"] = float(max_sim)
        entry["most_similar_pair"] = {"thinking": best_pair[0], "acting": best_pair[1]}
    
    return entry

def process_file(input_file: str):
    output_file = input_file.replace(".json", "_cosim.json")
    print(f"Processing: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Warning: Skipping {input_file}, as it does not contain a JSON list.")
            return None

        processed_data = [process_entry(entry) for entry in data]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        return output_file
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}. Skipping.")
        return None
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {e}")
        return None

def main(input_dir: str):
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found at '{input_dir}'")
        return

    files_to_process = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".json") and not f.endswith("_cosim.json")
    ]

    if not files_to_process:
        print(f"No .json files to process were found in '{input_dir}'.")
        return

    print(f"Found {len(files_to_process)} files to process.")


    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in files_to_process}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result:
                    print(f"Successfully saved: {os.path.basename(result)}")
                else:
                    print(f"Skipped or failed: {os.path.basename(file_path)}")
            except Exception as exc:
                print(f'{os.path.basename(file_path)} generated an exception: {exc}')

    print("\nProcessing complete.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_directory = sys.argv[1]
    else:
        default_path = os.path.join(os.path.expanduser("~"), "charactor_ai_org", "generated_results", "dilemma", "scoring")
        print(f"Usage: python {sys.argv[0]} <directory_path>")
        print(f"No directory provided. Using default: '{default_path}'")
        input_directory = default_path
    
    main(input_directory)