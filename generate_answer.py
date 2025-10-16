import argparse
import pandas as pd
import json
import os
import time
from tools import get_model
import re


def dilemma_answer(data, model, output_path, cot=False, clean_consequence_flag=False):
    
    if isinstance(data, list):
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                cid = item.get("CID", "")
                name = item.get("name", "")
                lore = item.get("lore", "")
                
                print(f"Processing: {name} (CID: {cid})")
                
                situation = item.get("situation", "")
                choice_a = item.get("choice_A", "")
                choice_b = item.get("choice_B", "")
                consequence_a = item.get("consequence_A","")
                consequence_b = item.get("consequence_B","")
                
                if clean_consequence_flag:
                    dilemma = f"{situation}\nChoice A: {choice_a} \nChoice B: {choice_b}"
                
                else:
                    dilemma = f"{situation}\nChoice A: {choice_a} Consequence A: {consequence_a}\nChoice B: {choice_b} Consequence A: {consequence_b}"
                
                dilemma_type = item.get("dilemma_type", "").split("_")[0]
                
                if dilemma_type == "Save the One You Love vs. Save the Greater Good":
                    d_detail = "What will you choose between A) Save the One You Love vs B) Save the Greater Good"
                elif dilemma_type == "Hero or Villain":
                    d_detail = "About morality and identity crisis, what will you choose between A) Hero vs B) Villain"
                elif dilemma_type == "Duty vs. Personal Desire":
                    d_detail = "What will you choose between A) Duty vs. B) Personal Desire"
                elif dilemma_type == "The Ends Justify the Means vs. Moral Code":
                    d_detail = "What will you choose between A) The Ends Justify the Means vs. B) Moral Code"
                else:
                    d_detail = "What will you choose between A) or B) ?"

                heros_profile = pd.read_csv("heros_profile_aa.csv")
                heros_attributes = heros_profile[heros_profile["CID"]==cid]["Attributes"].values[0]
                heros_attributes = heros_attributes.split(",")
                heros_attributes = [attr.strip() for attr in heros_attributes]
                heros_attributes = ", ".join(heros_attributes)


                input_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}, {heros_attributes}, the situation is {dilemma}\n <question> {d_detail} <question/>"
                if cot:
                    input_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}, the situation is {dilemma}\n <question> {d_detail}, Let's think step by step.<question/>"
                
                try:
                    time.sleep(3)
                    answer = model(input_prompt=input_prompt)
                    time.sleep(3)
                    print(f"Response: {answer[:100]}...")
                except Exception as e:
                    print(f"Error for CID {cid}: {e}")
                    answer = f"Error: {str(e)}"
                
                json_obj = {"CID": cid, "name": name, "lore": lore, "answers": answer, "question": dilemma}
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
    
def canon_event(data, model, output_path, cot=False):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            if "Events" in item:
                cid = item.get("CID", "")
                name = item.get("Name", "")
                source = item.get("Source", [])
                lore = source

                print(f"Processing: {name} (CID: {cid})")
                for stage, events in item["Events"].items():
                    for event in events:
                        question = event.get("question", "")
                        options = event.get("options", [])
                        correct_answer = event.get("correct_answer", "")
                        
                        formatted_question = f"{question}\n" + "\n".join(options) if options else question
                        
                        input_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}\n <question> {formatted_question} <question/>"
                        if cot:
                            input_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}\n <question> {formatted_question}, Let's think step by step.<question/>"
                        
                        try:
                            time.sleep(2)
                            answer = model(input_prompt=input_prompt)
                            print(f"Response: {answer[:100]}...")
                        except Exception as e:
                            print(f"Error for CID {cid}: {e}")
                            answer = f"Error: {str(e)}"
                        
                        json_obj = {"CID": cid, "name": name, "lore": lore, "answers": answer, "question": formatted_question, "true_label": correct_answer}
                        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

def reprocess_failed_results(input_file: str, output_file: str, model_func: callable, task: str, cot: bool = False, cross_character: bool = False):
    df = pd.read_csv("heros_profile_aa.csv")
    

    dilemmas_df = pd.read_json("character_dilemmas.json", lines=False)
    canon_df = pd.read_json("characters_canon_events.json", lines=False)
    canon_by_cid = canon_df.set_index('CID').to_dict()
    
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        completed_count = 0
        total_lines = sum(1 for _ in open(input_file, 'r'))
        reprocess_count = 0
        
        infile.seek(0)
        
        for num_line, line in enumerate(infile):
            data = json.loads(line)
            if re.search(r'Error: ', data.get('answers', '')) or re.search(r'I notice that your request is incomplete',data.get('answers','')) or len(data.get('answers', '')) < 5:
                reprocess_count += 1
                input_prompt = ""
                
                if cross_character:
                    answerer_cid = data.get("answerer_CID", "")
                    answerer_name = data.get("answerer_name", "")
                    questioner_cid = data.get("questioner_CID", "")
                    questioner_name = data.get("questioner_name", "")
                    lore = str(df[df['CID']==questioner_cid]["Source"])
                    answerer_lore = lore
                    
                    print(f"Reprocessing cross-character ({reprocess_count}): {answerer_name} (CID: {answerer_cid}) answering {questioner_name}'s question")
                    
                    if task == "dilemma":
                        dilemma_type_=data.get("original_dilemma_type")
                        
                        questioner_dilemma=dilemmas_df[(dilemmas_df["CID"] == questioner_cid) &(dilemmas_df["dilemma_type"]==dilemma_type_)]
                        
                        situation = questioner_dilemma["situation"]
                        choice_a = questioner_dilemma["choice_A"]
                        choice_b = questioner_dilemma["choice_B"]
                        consequence_a = questioner_dilemma["consequence_A"]
                        consequence_b = questioner_dilemma["consequence_B"]
                        
                        dilemma = f"{situation}\nChoice A: {choice_a} Consequence A: {consequence_a}\nChoice B: {choice_b} Consequence B: {consequence_b}"
                        dilemma_type = dilemma_type_.split("_")[0]
                        if dilemma_type == "Save the One You Love vs. Save the Greater Good":
                            d_detail = "What will you choose between A) Save the One You Love vs B) Save the Greater Good"
                        elif dilemma_type == "Hero or Villain":
                            d_detail = "About morality and identity crisis, what will you choose between A) Hero vs B) Villain"
                        elif dilemma_type == "Duty vs. Personal Desire":
                            d_detail = "What will you choose between A) Duty vs. B) Personal Desire"
                        elif dilemma_type == "The Ends Justify the Means vs. Moral Code":
                            d_detail = "What will you choose between A) The Ends Justify the Means vs. B) Moral Code"
                        else:
                            d_detail = "What will you choose between A) or B) ?"
                        
                        base_prompt = f"You are playing the role of {answerer_name}, act and think as {answerer_name}, from {answerer_lore}. The situation is: {dilemma}\n<question> {d_detail} </question>"
                        
                        if cot:
                            input_prompt = base_prompt + ", Let's think step by step. </question>"
                        else:
                            input_prompt = base_prompt
                    
                    elif task == "canon":

                        questioner_canon = canon_by_cid.get(questioner_cid, {})
                        if "Events" in questioner_canon:
                            first_stage = next(iter(questioner_canon["Events"].values()))
                            if first_stage:
                                event = first_stage[0]
                                question = event.get("question", "")
                                options = event.get("options", [])
                                formatted_question = f"{question}\n" + "\n".join(options) if options else question
                                
                                base_prompt = f"You are playing the role of {answerer_name}, act and think as {answerer_name}, from {answerer_lore}.\n<question> {formatted_question} </question>"
                                
                                if cot:
                                    input_prompt = base_prompt + ", Let's think step by step. </question>"
                                else:
                                    input_prompt = base_prompt
                
                else:
                    cid = data.get("CID", "")
                    name = data.get("name", "")
                    lore = data.get("lore", "")
                    
                    print(f"Reprocessing individual ({reprocess_count}): {name} (CID: {cid})")
                    
                    if task == "dilemma":
                        dilemma= data["situation"]
                        dilemma_type= data["dilemma_type"].split("_")[0]

                        if dilemma_type == "Save the One You Love vs. Save the Greater Good":
                            d_detail = "What will you choose between A) Save the One You Love vs B) Save the Greater Good"
                        elif dilemma_type == "Hero or Villain":
                            d_detail = "About morality and identity crisis, what will you choose between A) Hero vs B) Villain"
                        elif dilemma_type == "Duty vs. Personal Desire":
                            d_detail = "What will you choose between A) Duty vs. B) Personal Desire"
                        elif dilemma_type == "The Ends Justify the Means vs. Moral Code":
                            d_detail = "What will you choose between A) The Ends Justify the Means vs. B) Moral Code"
                        else:
                            d_detail = "What will you choose between A) or B) ?"

                        input_prompt = f"You are playing the role of {name}, act and think as {name},from {lore}, the situation is {dilemma}\n <question>{d_detail}<question/>"
                    

                    elif task == "canon":
                        question = data["question"]
                        lore = data["lore"]
                    
                        
                        input_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}, the situation is {dilemma}\n <question> {d_detail} </question>"
                        
                        if cot:
                            input_prompt = base_prompt + ", Let's think step by step. </question>"
                        else:
                            input_prompt = base_prompt
                try:
                    answer = model_func(input_prompt)
                    completed_count += 1
                    
                    if cross_character:
                        print(f"Completed ({completed_count}): {data.get('answerer_name', 'Unknown')} answering {data.get('questioner_name', 'Unknown')}'s question - Response length: {len(answer)}")
                    else:
                        print(f"Completed ({completed_count}): {data.get('name', 'Unknown')} (CID: {data.get('CID', 'Unknown')}) - Response length: {len(answer)}")
                    
                    data["answers"] = answer
                    
                    if answer.startswith("Error:"):
                        if cross_character:
                            print(f"Still error for {data.get('answerer_name', 'Unknown')} -> {data.get('questioner_name', 'Unknown')}: {answer}")
                        else:
                            print(f"Still error for CID {data.get('CID', 'Unknown')}: {answer}")
                            
                except Exception as e:
                    if cross_character:
                        print(f"Model error for interaction {data.get('answerer_CID', 'Unknown')} -> {data.get('questioner_CID', 'Unknown')}: {e}")
                    else:
                        print(f"Model error for CID {data.get('CID', 'Unknown')}: {e}")
                    data["answers"] = f"Error: {str(e)}"
            
            else:
                print(f"Skipping line {num_line + 1}: already processed successfully")
            
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        print(f"Reprocessing complete. Total items: {total_lines}, Reprocessed: {reprocess_count}, Completed successfully: {completed_count}")


def load_data(data_path):
    """Load data from either CSV or JSON file"""
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

def main():
    parser = argparse.ArgumentParser(description="Process character data using different AI models.")
    parser.add_argument("--model", choices=["gemini2","gemini2-5","gemini2-5-think","sonnet3-7","sonnet3-7-think","sonnet3-5","r1","v3","4o-mini"], required=True, help="""Select model: "gemini2","gemini2-5","sonnet3-7","sonnet3-7-think","sonnet3-5","r1","v3","4o-mini""")
    parser.add_argument("--data", required=True, help="Path to input data file (CSV or JSON)")
    parser.add_argument("--output", required=True, help="Path to save JSON output")
    parser.add_argument("--task", choices=["dilemma", "canon", "dialogue"], required=True, help="Select task type")
    parser.add_argument("--apierror", action="store_true", help="Select retry run with same API")
    parser.add_argument("--inputfile", required=False, help="Select file to reprocess must be jsonl")
    parser.add_argument("--cot", action="store_true", help="Select use CoT")
    parser.add_argument("--clean_consequence", action="store_true", help="Clean consequence text from dilemmas")
    args = parser.parse_args()
    
    model = get_model(args.model)
    if model is None:
        print("Invalid model selection.")
        return
    
    #os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.apierror and args.inputfile:
        reprocess_failed_results(args.inputfile, args.output, model, args.task)
    else:
        data = load_data(args.data)
        
        if args.task == "dilemma":
            dilemma_answer(data, model, args.output, cot=args.cot, clean_consequence_flag=args.clean_consequence)

        elif args.task == "canon":
            canon_event(data, model, args.output, cot=args.cot)
        
        # elif args.task == "dialogue": future work
        #     process_dialogue(data, model, args.output)

if __name__ == "__main__":
    main()
