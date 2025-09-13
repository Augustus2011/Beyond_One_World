import argparse
import pandas as pd
import json
import os
import time
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Callable, Optional
from tools import get_model, anonymize_path


# Create a semaphore to limit concurrent API calls
MAX_CONCURRENT_REQUESTS = 6  # Adjust based on API rate limits

async def async_model_call(model_func, input_prompt, semaphore):
    """Execute a model call with rate limiting via semaphore"""
    async with semaphore:
        loop = asyncio.get_event_loop()
        try:
            # Run the model call in a thread pool to avoid blocking the event loop
            return await loop.run_in_executor(None, model_func, input_prompt)
        except Exception as e:
            print(f"Error in model call: {str(e)}")
            return f"Error: {str(e)}"


async def process_canon_batch(data: List[Dict], model_func: Callable, output_path: str, cot: bool = False):
    """Process canon event data in parallel batches"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    item_details = []
    for item in data:
        if "Events" in item:
            cid = item.get("CID", "")
            name = item.get("Name", "")
            source = item.get("Source", [])
            lore = source
            
            print(f"Queuing events for: {name} (CID: {cid})")
            
            # Process events from different life stages
            for stage, events in item["Events"].items():
                for event in events:
                    question = event.get("question", "")
                    options = event.get("options", [])
                    correct_answer = event.get("correct_answer", "")
                    
                    # Format question with options
                    formatted_question = f"{question}\n" + "\n".join(options) if options else question
                    
                    input_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}\n <question> {formatted_question} <question/>"
                    if cot:
                        input_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}\n <question> {formatted_question}, Let's think step by step.<question/>"
                    
                    # Create task for this item
                    task = asyncio.create_task(async_model_call(model_func, input_prompt, semaphore))
                    
                    # Store task with item details
                    item_details.append({
                        "CID": cid,
                        "name": name,
                        "lore": lore,
                        "question": formatted_question,
                        "true_label": correct_answer,
                        "task": task
                    })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for details in item_details:
            task = details.pop("task")
            try:
                answer = await task
                print(f"Completed: {details['name']} (CID: {details['CID']}) - Response length: {len(answer)}")
            except Exception as e:
                print(f"Task error for CID {details['CID']}: {e}")
                answer = f"Error: {str(e)}"
            
            details["answers"] = answer
            f.write(json.dumps(details, ensure_ascii=False) + "\n")


async def process_dilemma_batch(data: List[Dict], model_func: Callable, output_path: str, cot: bool = False, clean_consequence_flag: bool = False):
    """Process dilemma data in parallel batches"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    heros_profile = pd.read_csv("heros_profile_aa.csv")
                
    for item in data:
        cid = item.get("CID", "")
        name = item.get("name", "")
        lore = item.get("lore", "")
        
        print(f"Queuing: {name} (CID: {cid})")
        
        situation = item.get("situation", "")
        choice_a = item.get("choice_A", "")
        choice_b = item.get("choice_B", "")
        consequence_a = item.get("consequence_A", "")
        consequence_b = item.get("consequence_B", "")
        
        if clean_consequence_flag:
            dilemma = f"{situation}\nChoice A: {choice_a} \nChoice B: {choice_b}"
        else:
            dilemma = f"{situation}\nChoice A: {choice_a} Consequence A: {consequence_a}\nChoice B: {choice_b} Consequence B: {consequence_b}"
        
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
        


        heros_attributes = heros_profile[heros_profile["CID"]==cid]["Attributes"].values[0]
        heros_attributes = heros_attributes.split(",")
        heros_attributes = [attr.strip() for attr in heros_attributes]
        heros_attributes = ", ".join(heros_attributes)

        input_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}, {heros_attributes}, the situation is {dilemma}\n <question> {d_detail} <question/>"
        
        #input_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}, the situation is {dilemma}\n <question> {d_detail} <question/>"
        if cot:
            input_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}, the situation is {dilemma}\n <question> {d_detail}, Let's think step by step.<question/>"
        
        # Create task for this item
        task = asyncio.create_task(async_model_call(model_func, input_prompt, semaphore))
        tasks.append((item, task))
    
    # Process results as they complete and write to file
    with open(output_path, "w", encoding="utf-8") as f:
        for item, task in tasks:
            cid = item.get("CID", "")
            name = item.get("name", "")
            lore = item.get("lore", "")
            dilemma = item.get("situation", "")
            
            try:
                answer = await task
                print(f"Completed: {name} (CID: {cid}) - Response length: {len(answer)}")
                if answer.startswith("Error:"):
                    print(f"Error for CID {cid}: {answer}")
            except Exception as e:
                print(f"Task error for CID {cid}: {e}")
                answer = f"Error: {str(e)}"
            
            json_obj = {"CID": cid, "name": name, "lore": lore, "answers": answer, "question": dilemma}
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


async def process_cross_character_interaction(data: List[Dict], model_func: Callable, output_path: str, 
                                            cot: bool = False, task_type: str = "dilemma"):
    """Process cross-character interactions where characters answer each other's questions"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    all_tasks = []
    
    # Group data by UID (keep all dilemmas for each character)
    uid_groups = {}
    for item in data:
        uid = item.get("UID", "")
        if uid not in uid_groups:
            uid_groups[uid] = []
        uid_groups[uid].append(item)
    
    print(f"Processing {len(uid_groups)} UID groups")
    
    # Process each UID group and prepare all tasks
    for uid, group_items in uid_groups.items():
        # Group items by CID to get unique characters
        char_by_cid = {}
        for item in group_items:
            cid = item.get("CID", "")
            if cid not in char_by_cid:
                char_by_cid[cid] = []
            char_by_cid[cid].append(item)
        
        print(f"Processing UID {uid} with {len(char_by_cid)} unique characters")
        
        for answerer_cid, answerer_items in char_by_cid.items():
            # Get the first item to extract answerer info (name, lore are same for all items of same CID)
            answerer_char = answerer_items[0]
            
            for questioner_cid, questioner_items in char_by_cid.items():
                if questioner_cid != answerer_cid:  # Don't answer your own questions
                    if task_type == "dilemma":
                        answerer_name = answerer_char.get("name", "")
                        answerer_lore = answerer_char.get("lore", "")
                        questioner_name = questioner_items[0].get("name", "")
                        
                        print(f"Queuing: {answerer_name} (CID:{answerer_cid}) answering all {len(questioner_items)} dilemmas from {questioner_name} (CID:{questioner_cid})")
                        
                        # Process each dilemma from the questioner
                        for questioner_dilemma in questioner_items:
                            situation = questioner_dilemma.get("situation", "")
                            choice_a = questioner_dilemma.get("choice_A", "")
                            choice_b = questioner_dilemma.get("choice_B", "")
                            consequence_a = questioner_dilemma.get("consequence_A", "")
                            consequence_b = questioner_dilemma.get("consequence_B", "")
                            
                            # Format dilemma consistently with the main dilemma processing
                            dilemma = f"{situation}\nChoice A: {choice_a} Consequence A: {consequence_a}\nChoice B: {choice_b} Consequence B: {consequence_b}"
                            dilemma_type = questioner_dilemma.get("dilemma_type", "").split("_")[0]
                            
                            # Map dilemma types to appropriate questions
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
                            
                            # Create prompt with proper formatting
                            base_prompt = f"You are playing the role of {answerer_name}, act and think as {answerer_name}, from {answerer_lore}. The situation is: {dilemma}\n<question> {d_detail} </question>"
                            
                            if cot:
                                input_prompt = base_prompt+", Let's think step by step. </question>"
                            else:
                                input_prompt = base_prompt
                        
                            task = asyncio.create_task(async_model_call(model_func, input_prompt, semaphore))
                        
                            # Store task with item details including original dilemma metadata
                            task_info = {
                                "UID": uid,
                                "answerer_CID": answerer_cid,
                                "answerer_name": answerer_name,
                                "questioner_CID": questioner_cid,
                                "questioner_name": questioner_name,
                                "original_dilemma_type": questioner_dilemma.get("dilemma_type", ""),
                                "task": task
                            }
                            all_tasks.append(task_info)

                    elif task_type == "canon":
                        answerer_name = answerer_char.get("Name", "")
                        answerer_lore = answerer_char.get("Source", "")
                        questioner_name = questioner_items[0].get("Name", "")
                        print(f"Queuing: {answerer_name} (CID:{answerer_cid}) answering {questioner_name}'s canon questions (CID:{questioner_cid})")

                        for questioner_char in questioner_items:
                            if "Events" in questioner_char:
                                for stage, events in questioner_char["Events"].items():
                                    for event in events:
                                        question = event.get("question", "")
                                        options = event.get("options", [])
                                        correct_answer = event.get("correct_answer", "")
                                        formatted_question = f"{question}\n" + "\n".join(options) if options else question
                                        
                                        input_prompt = f"You are playing the role of {answerer_name}, act and think as {answerer_name}, from {answerer_lore}.\n<question> {formatted_question} </question>"
                                        if cot:
                                            input_prompt = f"You are playing the role of {answerer_name}, act and think as {answerer_name}, from {answerer_lore}.\n<question> {formatted_question}, Let's think step by step. </question>"
                                        
                                        task = asyncio.create_task(async_model_call(model_func, input_prompt, semaphore))
                        
                                        # Store task with item details
                                        task_info = {
                                            "UID": uid,
                                            "answerer_CID": answerer_cid,
                                            "answerer_name": answerer_name,
                                            "questioner_CID": questioner_cid,
                                            "questioner_name": questioner_name,
                                            #"question": formatted_question,
                                            "stage": stage,
                                            "true_label": correct_answer,
                                            "task": task
                                        }
                                        all_tasks.append(task_info)
        
        print(f"Total tasks queued: {len(all_tasks)}")
        
        # Process all results concurrently and write to file
        with open(output_path, "w", encoding="utf-8") as f:
            completed_count = 0
            for task_info in all_tasks:
                task = task_info.pop("task")
                try:
                    answer = await task
                    completed_count += 1
                    print(f"Completed ({completed_count}/{len(all_tasks)}): {task_info['answerer_name']} answering {task_info['questioner_name']}'s question")
                    
                    if answer.startswith("Error:"):
                        print(f"Error for answerer CID {task_info['answerer_CID']}: {answer}")
                    
                except Exception as e:
                    print(f"Task error for interaction {task_info['answerer_CID']} -> {task_info['questioner_CID']}: {e}")
                    answer = f"Error: {str(e)}"
                
                task_info["answers"] = answer
                f.write(json.dumps(task_info, ensure_ascii=False) + "\n")


def process_cross_character_interaction_sync(data: List[Dict], model_func: Callable[[str], str], output_path: str,cot: bool = False, task_type: str = "dilemma"):
    """Process cross-character interactions synchronously."""
    uid_groups = {}
    for item in data:
        uid = item.get("UID", "")
        if uid not in uid_groups:
            uid_groups[uid] = []
        uid_groups[uid].append(item)

    print(f"Processing {len(uid_groups)} UID groups synchronously.")

    with open(output_path, "w", encoding="utf-8") as f_out:
        processed_count = 0
        for uid, group_items in uid_groups.items():
            char_by_cid = {}
            for item in group_items:
                cid = item.get("CID", "")
                if cid not in char_by_cid:
                    char_by_cid[cid] = []
                char_by_cid[cid].append(item)

            print(f"Processing UID {uid} with {len(char_by_cid)} unique characters.")

            for answerer_cid, answerer_items in char_by_cid.items():
                answerer_char = answerer_items[0] # Get the first item for answerer info

                for questioner_cid, questioner_items in char_by_cid.items():
                    if questioner_cid == answerer_cid:  # Don't answer own questions
                        continue

                    output_item_base = {
                        "UID": uid,
                        "answerer_CID": answerer_cid,
                        "questioner_CID": questioner_cid,
                    }

                    if task_type == "dilemma":
                        answerer_name = answerer_char.get("name", "")
                        answerer_lore = answerer_char.get("lore", "")
                        questioner_name = questioner_items[0].get("name", "")

                        output_item_base["answerer_name"] = answerer_name
                        output_item_base["questioner_name"] = questioner_name
                        
                        print(f"Preparing: {answerer_name} (CID:{answerer_cid}) to answer {len(questioner_items)} dilemmas from {questioner_name} (CID:{questioner_cid})")

                        for questioner_dilemma in questioner_items:
                            situation = questioner_dilemma.get("situation", "")
                            choice_a = questioner_dilemma.get("choice_A", "")
                            choice_b = questioner_dilemma.get("choice_B", "")
                            consequence_a = questioner_dilemma.get("consequence_A", "")
                            consequence_b = questioner_dilemma.get("consequence_B", "")

                            dilemma_text = f"{situation}\nChoice A: {choice_a} Consequence A: {consequence_a}\nChoice B: {choice_b} Consequence B: {consequence_b}"
                            dilemma_type = questioner_dilemma.get("dilemma_type", "").split("_")[0]

                            d_detail_text = ""
                            if dilemma_type == "Save the One You Love vs. Save the Greater Good":
                                d_detail_text = "What will you choose between A) Save the One You Love vs B) Save the Greater Good"
                            elif dilemma_type == "Hero or Villain":
                                d_detail_text = "About morality and identity crisis, what will you choose between A) Hero vs B) Villain"
                            elif dilemma_type == "Duty vs. Personal Desire":
                                d_detail_text = "What will you choose between A) Duty vs. B) Personal Desire"
                            elif dilemma_type == "The Ends Justify the Means vs. Moral Code":
                                d_detail_text = "What will you choose between A) The Ends Justify the Means vs. B) Moral Code"
                            else:
                                d_detail_text = "What will you choose between A) or B) ?"

                            if cot:
                                d_detail_text += ", Let's think step by step."

                            input_prompt = f"You are playing the role of {answerer_name}, act and think as {answerer_name}, from {answerer_lore}. The situation is: {dilemma_text}\n<question> {d_detail_text} </question>"
                            
                            current_output_item = output_item_base.copy()
                            current_output_item["original_dilemma_type"] = questioner_dilemma.get("dilemma_type", "")
                            
                            answer = ""
                            try:
                                print(f"  Calling model for: {answerer_name} answering {questioner_name}'s dilemma...")
                                answer = model_func(input_prompt)
                                if isinstance(answer, str) and answer.startswith("Error:"): # Check if model_func returns error string
                                    print(f"  Model returned error for CID {answerer_cid} (Dilemma): {answer}")
                            except Exception as e:
                                print(f"  Model call error for CID {answerer_cid} (Dilemma {questioner_name}): {e}")
                                answer = f"Error: {str(e)}"
                            
                            current_output_item["answers"] = answer
                            f_out.write(json.dumps(current_output_item, ensure_ascii=False) + "\n")
                            processed_count += 1
                            print(f"  Completed ({processed_count} total): {answerer_name} for {questioner_name}'s dilemma.")

                    elif task_type == "canon":
                        answerer_name = answerer_char.get("Name", "") # Note: "Name" from original
                        answerer_lore = answerer_char.get("Source", "") # Note: "Source" from original
                        questioner_name = questioner_items[0].get("Name", "")

                        output_item_base["answerer_name"] = answerer_name
                        output_item_base["questioner_name"] = questioner_name

                        print(f"Preparing: {answerer_name} (CID:{answerer_cid}) to answer canon questions from {questioner_name} (CID:{questioner_cid})")

                        for questioner_event_set in questioner_items: # Each item for a questioner can have events
                            if "Events" in questioner_event_set:
                                for stage, events in questioner_event_set["Events"].items():
                                    for event in events:
                                        question_text = event.get("question", "")
                                        options = event.get("options", [])
                                        correct_answer = event.get("correct_answer", "")
                                        
                                        formatted_question = f"{question_text}\n" + "\n".join(options) if options else question_text
                                        
                                        canon_question_text = formatted_question
                                        if cot:
                                            canon_question_text += ", Let's think step by step."
                                        
                                        input_prompt = f"You are playing the role of {answerer_name}, act and think as {answerer_name}, from {answerer_lore}.\n<question> {canon_question_text} </question>"
                                        
                                        current_output_item = output_item_base.copy()
                                        current_output_item["stage"] = stage
                                        current_output_item["true_label"] = correct_answer
                                        # current_output_item["question"] = formatted_question # Was commented out in original

                                        answer = ""
                                        try:
                                            print(f"  Calling model for: {answerer_name} answering {questioner_name}'s canon question (Stage: {stage})...")
                                            answer = model_func(input_prompt)
                                            if isinstance(answer, str) and answer.startswith("Error:"):
                                                print(f"  Model returned error for CID {answerer_cid} (Canon): {answer}")
                                        except Exception as e:
                                            print(f"  Model call error for CID {answerer_cid} (Canon {questioner_name}, Stage: {stage}): {e}")
                                            answer = f"Error: {str(e)}"

                                        current_output_item["answers"] = answer
                                        f_out.write(json.dumps(current_output_item, ensure_ascii=False) + "\n")
                                        processed_count += 1
                                        print(f"  Completed ({processed_count} total): {answerer_name} for {questioner_name}'s canon question (Stage: {stage}).")
        print(f"Total items processed and written to {output_path}: {processed_count}")


from typing import Callable, List, Dict, Any

async def reprocess_failed_results_async(input_file: str, output_file: str, model_func: Callable, task: str, cot: bool = False, cross_character: bool = False, max_concurrent: int = 10):
    """Async version of reprocess_failed_results with concurrent execution while maintaining original style"""
    df = pd.read_csv("heros_profile_aa.csv")
    
    # Load original data files using pandas
    dilemmas_df = pd.read_json("character_dilemmas.json", lines=False)
    canon_df = pd.read_json("characters_canon_events.json", lines=False)
    
    # Create lookup dictionaries for faster access
    # dilemmas_by_cid = dilemmas_df.set_index('CID').to_dict()
    canon_by_cid = canon_df.set_index('CID').to_dict()
    
    # Read all lines first to identify failed ones
    failed_items = []
    all_lines = []
    
    with open(input_file, "r", encoding="utf-8") as infile:
        for num_line, line in enumerate(infile):
            data = json.loads(line)
            all_lines.append((num_line, data))
            
            # Check if this is a failed result (error message or very short response)
            if re.search(r'Error: ', data.get('answers', '')) or re.search(r'I notice that your request is incomplete',data.get('answers','')) or len(data.get('answers', '')) < 5:
                failed_items.append((num_line, data))
    
    total_lines = len(all_lines)
    reprocess_count = len(failed_items)
    completed_count = 0
    
    print(f"Found {reprocess_count} failed items out of {total_lines} total items to reprocess")
    
    if reprocess_count == 0:
        print("No failed items to reprocess!")
        return
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_failed_item(num_line: int, data: Dict[str, Any]) -> tuple:
        """Process a single failed item with the original logic"""
        async with semaphore:
            try:
                input_prompt = ""
                
                if cross_character:

                    answerer_cid = data.get("answerer_CID", "")
                    answerer_name = data.get("answerer_name", "")
                    questioner_cid = data.get("questioner_CID", "")
                    questioner_name = data.get("questioner_name", "")
                    lore = str(df[df['CID']==questioner_cid]["Source"])
                    answerer_lore = lore
                    
                    
                    print(f"Reprocessing cross-character: {answerer_name} (CID: {answerer_cid}) answering {questioner_name}'s question")
                    
                    if task == "dilemma":
                        # Get dilemma data from original file
                        dilemma_type_=data.get("original_dilemma_type")
                        
                        questioner_dilemma=dilemmas_df[(dilemmas_df["CID"] == questioner_cid) &(dilemmas_df["dilemma_type"]==dilemma_type_)]
                        situation = questioner_dilemma["situation"]
                        choice_a = questioner_dilemma["choice_A"]
                        choice_b = questioner_dilemma["choice_B"]
                        consequence_a = questioner_dilemma["consequence_A"]
                        consequence_b = questioner_dilemma["consequence_B"]
                        
                        # Format dilemma consistently
                        dilemma = f"{situation}\nChoice A: {choice_a} Consequence A: {consequence_a}\nChoice B: {choice_b} Consequence B: {consequence_b}"
                        dilemma_type = dilemma_type_.split("_")[0]
                        
                        # Map dilemma types to appropriate questions
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
                        # Get canon data from original file
                        questioner_canon = canon_by_cid.get(questioner_cid, {})
                        if "Events" in questioner_canon:
                            # Get the first event's question (you might want to modify this based on your needs)
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
                    # Handle individual character format (original logic)
                    cid = data.get("CID", "")
                    name = data.get("name", "")
                    lore = data.get("lore", "")
                    
                    print(f"Reprocessing individual: {name} (CID: {cid})")
                    
                    if task == "dilemma":
                        # Get dilemma data from original file
                        character_dilemma = dilemmas_by_cid.get(cid, {})
                        situation = character_dilemma.get("situation", "")
                        choice_a = character_dilemma.get("choice_A", "")
                        choice_b = character_dilemma.get("choice_B", "")
                        consequence_a = character_dilemma.get("consequence_A", "")
                        consequence_b = character_dilemma.get("consequence_B", "")
                        
                        # Format dilemma consistently
                        dilemma = f"{situation}\nChoice A: {choice_a} Consequence A: {consequence_a}\nChoice B: {choice_b} Consequence B: {consequence_b}"
                        dilemma_type = character_dilemma.get("dilemma_type", "").split("_")[0]
                        
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
                        
                        base_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}, the situation is {dilemma}\n <question> {d_detail} </question>"
                        
                        if cot:
                            input_prompt = base_prompt + ", Let's think step by step. </question>"
                        else:
                            input_prompt = base_prompt
                    
                    elif task == "canon":
                        # Get canon data from original file
                        character_canon = canon_by_cid.get(cid, {})
                        if "Events" in character_canon:
                            # Get the first event's question (you might want to modify this based on your needs)
                            first_stage = next(iter(character_canon["Events"].values()))
                            if first_stage:
                                event = first_stage[0]
                                question = event.get("question", "")
                                options = event.get("options", [])
                                formatted_question = f"{question}\n" + "\n".join(options) if options else question
                                
                                base_prompt = f"You are playing the role of {name}, act and think as {name}, from {lore}\n <question> {formatted_question} </question>"
                                
                                if cot:
                                    input_prompt = base_prompt + ", Let's think step by step. </question>"
                                else:
                                    input_prompt = base_prompt
                
                # Make async model call
                if asyncio.iscoroutinefunction(model_func):
                    answer = await model_func(input_prompt)
                else:
                    # If model_func is not async, run it in a thread pool
                    loop = asyncio.get_event_loop()
                    answer = await loop.run_in_executor(None, model_func, input_prompt)
                
                if cross_character:
                    print(f"Completed: {data.get('answerer_name', 'Unknown')} answering {data.get('questioner_name', 'Unknown')}'s question - Response length: {len(answer)}")
                else:
                    print(f"Completed: {data.get('name', 'Unknown')} (CID: {data.get('CID', 'Unknown')}) - Response length: {len(answer)}")
                
                data["answers"] = answer
                
                if answer.startswith("Error:"):
                    if cross_character:
                        print(f"Still error for {data.get('answerer_name', 'Unknown')} -> {data.get('questioner_name', 'Unknown')}: {answer}")
                    else:
                        print(f"Still error for CID {data.get('CID', 'Unknown')}: {answer}")
                        
                return (num_line, data, True)
                        
            except Exception as e:
                if cross_character:
                    print(f"Model error for interaction {data.get('answerer_CID', 'Unknown')} -> {data.get('questioner_CID', 'Unknown')}: {e}")
                else:
                    print(f"Model error for CID {data.get('CID', 'Unknown')}: {e}")
                data["answers"] = f"Error: {str(e)}"
                return (num_line, data, False)
    
    # Process all failed items concurrently
    print(f"Processing {reprocess_count} failed items with max {max_concurrent} concurrent requests...")
    
    tasks = [process_single_failed_item(num_line, data) for num_line, data in failed_items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Create a dictionary to store processed results
    processed_data = {}
    
    for result in results:
        if isinstance(result, tuple) and len(result) == 3:
            num_line, data, success = result
            processed_data[num_line] = data
            if success:
                completed_count += 1
        else:
            print(f"Unexpected result: {result}")
    
    # Write results back to file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for num_line, original_data in all_lines:
            # Check if this line was a failed result (error message or very short response)
            if re.search(r'Error: ', original_data.get('answers', '')) or re.search(r'I notice that your request is incomplete',original_data.get('answers','')) or len(original_data.get('answers', '')) < 5:
                # Use processed data if available, otherwise keep original
                data_to_write = processed_data.get(num_line, original_data)
            else:
                # For non-failed items, just keep the original data
                print(f"Skipping line {num_line + 1}: already processed successfully")
                data_to_write = original_data
            
            # Write the result (whether reprocessed or original)
            outfile.write(json.dumps(data_to_write, ensure_ascii=False) + "\n")
    
    print(f"Reprocessing complete. Total items: {total_lines}, Reprocessed: {reprocess_count}, Completed successfully: {completed_count}")



def load_data(data_path: str) -> List[Dict]:
    """Load data from either CSV or JSON file"""
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path).to_dict(orient='records')
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

async def main_async():
    parser = argparse.ArgumentParser(description="Process character data using different AI models with async execution.")
    parser.add_argument("--model", choices=["gemini2","gemini2-5","gemini2-5-think","sonnet3-7","sonnet3-7-think","sonnet3-5","r1","v3","4o-mini"], required=True, help="""Select model: "gemini2","gemini2-5","sonnet3-7","sonnet3-7-think","sonnet3-5","r1","v3","4o-mini""")
    parser.add_argument("--data", required=False, help="Path to input data file (CSV or JSON)")
    parser.add_argument("--output", required=True, help="Path to save JSON output")
    parser.add_argument("--task", choices=["dilemma", "canon"], required=True, help="Select task type")
    parser.add_argument("--apierror", action="store_true", help="Select retry run with same API")
    parser.add_argument("--inputfile", required=False, help="Select file to reprocess must be jsonl")
    parser.add_argument("--cot", action="store_true", help="Select use CoT")
    parser.add_argument("--clean_consequence", action="store_true", help="Clean consequence text from dilemmas")
    parser.add_argument("--max_con", type=int, default=4, help="Maximum concurrent API requests")
    parser.add_argument("--cross_character", action="store_true", help="Enable cross-character interactions")
    args = parser.parse_args()
    
    # Update the global concurrency limit
    global MAX_CONCURRENT_REQUESTS
    MAX_CONCURRENT_REQUESTS = args.max_con
    
    model_func = get_model(args.model)
    if model_func is None:
        print("Invalid model selection.")
        return
    
    if args.apierror and args.inputfile:
        # Call the non-async version directly (no await needed)
        await reprocess_failed_results_async(args.inputfile, args.output, model_func, args.task, 
                                cot=args.cot, cross_character=args.cross_character)
    else:
        data = load_data(args.data)
        
        if args.cross_character:
            process_cross_character_interaction_sync(data, model_func, args.output, cot=args.cot, task_type=args.task) #await process_cross_character_interaction(data, model_func, args.output, cot=args.cot, task_type=args.task)
        else:
            if args.task == "dilemma":
                await process_dilemma_batch(data, model_func, args.output, cot=args.cot, 
                                           clean_consequence_flag=args.clean_consequence)
            elif args.task == "canon":
                await process_canon_batch(data, model_func, args.output, cot=args.cot)
        
if __name__ == "__main__":
    asyncio.run(main_async())
