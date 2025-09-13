import json
import re
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
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
    
    # Get overall cosine similarity of the full string
    #full_embed = embed(text)
    #entry["full_context_similarity"] = float(cosine_similarity(full_embed, full_embed)[0][0])  # Always 1.0, or change if compared to something else
    
    return entry

def main(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = [process_entry(entry) for entry in data]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/kunkerdthaisong/charactor_ai_org/generated_results/canon/scoring/canon_scoring_4o-mini.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/Users/kunkerdthaisong/charactor_ai_org/generated_results/canon/scoring/canon_scoring_4o-mini_cosim.json"
    main(input_path, output_path)
