import glob
import os 
import pandas
import json
import re
import argparse
import asyncio
from typing import Dict, Any, Tuple, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from tools import get_model


construct_think_act = get_model("gen-think")
judge = get_model("judge")
API_SEMAPHORE = asyncio.Semaphore(10)

async def load_character_data(data_path: str) -> Dict[str, Dict[str, Any]]:
    async with aiofiles.open(data_path, 'r', encoding='utf-8') as f:
        content = await f.read()
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        df = await loop.run_in_executor(executor, lambda: pandas.read_csv(data_path))
    
    character_dict = {}
    for _, row in df.iterrows():
        cid = str(row['CID'])
        character_dict[cid] = {
            'name': row['Name'],
            'source': row['Source'],
            'attributes': row['Attributes']
        }
    
    return character_dict

def extract_scores(score_text: str) -> Tuple[float, float]:
    try:
        match = re.search(r'(\d+),(\d+)', score_text)
        if match:
            think_score = float(match.group(1))
            act_score = float(match.group(2))
            return think_score, act_score
        
    except:
        return 0, 0

async def call_model_async(model, text: str) -> str:
    async with API_SEMAPHORE:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, model, text)
        return result

async def process_and_score(input_json: Dict[str, Any], character_data: Dict[str, Dict[str, Any]], 
                           output_dir: str = None, index: int = None) -> Dict[str, Any]:

    text_to_process = input_json.get('answers', '')
    cid = str(input_json.get('CID', ''))
    
    char_ref = character_data.get(cid, {})
    name = char_ref.get('name', 'Unknown')
    source = char_ref.get('source', '')
    attributes = char_ref.get('attributes', '')
    
    reference_context = f"Character: {name}\nSource: {source}\nAttributes: {attributes}"
    think_act_task = call_model_async(construct_think_act, text_to_process)
    think_act_result = await think_act_task
    judge_input = f"reference :\n{reference_context}\n\nResponse to evaluate:\n{think_act_result}"
    score_result = await call_model_async(judge, judge_input)
    think_score, act_score = extract_scores(score_result)
    print(f"Processed: CID={cid}, Name={name}, Thinking={think_score}, Acting={act_score}")
    result = input_json.copy()
    result['think_act_analysis'] = think_act_result
    result['score'] = {
        'thinking': think_score,
        'acting': act_score
    }
    result['character_reference'] = char_ref

    if output_dir and index is not None:
        await save_individual_result(result, output_dir, cid, name, index)
    
    return result
        


async def save_individual_result(result: Dict[str, Any], output_dir: str, cid: str, name: str, index: int):
    """Save individual result file asynchronously."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    individual_file = os.path.join(output_dir, f"{cid}_{name.replace(' ', '_')}_{index}_{timestamp}.json")
    
    async with aiofiles.open(individual_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(result, indent=2, ensure_ascii=False))

async def read_jsonl_file(input_file: str) -> List[Dict[str, Any]]:
    """Read JSONL file asynchronously."""
    data = []
    async with aiofiles.open(input_file, 'r', encoding='utf-8') as f:
        async for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

async def process_json_file(input_file: str, output_file: str, character_data: Dict[str, Dict[str, Any]], save_individual: bool = True, max_concurrent: int = 5):
    try:

        individual_dir = None
        if save_individual:
            individual_dir = os.path.join(os.path.dirname(output_file), "individual_results2")
            os.makedirs(individual_dir, exist_ok=True)
            print(f"Saving individual results to: {individual_dir}")
            
        print(f"\nProcessing file: {input_file}")
        print("-" * 60)
        

        input_data_list = await read_jsonl_file(input_file)
        processing_semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(data_and_index):
            async with processing_semaphore:
                data, index = data_and_index
                return await process_and_score(data, character_data, individual_dir, index)
        tasks = [
            process_with_semaphore((data, i)) 
            for i, data in enumerate(input_data_list)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing item {i}: {result}")
                # Create default result for failed items
                default_result = input_data_list[i].copy()
                default_result['score'] = {'thinking': 2.5, 'acting': 2.5}
                default_result['error'] = str(result)
                valid_results.append(default_result)
            else:
                valid_results.append(result)

        # Save all results as a JSON array
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(valid_results, indent=2, ensure_ascii=False))
            
        print("-" * 60)
        print(f"Completed processing {len(valid_results)} items. Full results saved to: {output_file}")

    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")

async def process_json_directory(input_dir: str, output_dir: str, character_data: Dict[str, Dict[str, Any]], 
                                save_individual: bool = True, max_concurrent_files: int = 3):
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    print(f"Found {len(json_files)} JSON files to process in {input_dir}")

    file_semaphore = asyncio.Semaphore(max_concurrent_files)
    
    async def process_file_with_semaphore(json_file, file_index):
        async with file_semaphore:
            print(f"\nProcessing file {file_index+1}/{len(json_files)}: {os.path.basename(json_file)}")
            output_file = os.path.join(output_dir, os.path.basename(json_file))
            await process_json_file(json_file, output_file, character_data, save_individual)

    tasks = [
        process_file_with_semaphore(json_file, i) 
        for i, json_file in enumerate(json_files)
    ]
    
    await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    parser = argparse.ArgumentParser(description="Process and score character responses using AI models.")
    parser.add_argument("--input", required=True, help="Path to input JSON file or directory")
    parser.add_argument("--output", required=True, help="Path to save output JSON file or directory")
    parser.add_argument("--cdata", required=False, help="Path to character data CSV file")
    parser.add_argument("--no-individual", action="store_true", help="Disable saving individual JSON files")
    parser.add_argument("--max-concurrent", type=int, default=6, help="Maximum concurrent processing tasks per file")
    parser.add_argument("--max-concurrent-files", type=int, default=1, help="Maximum concurrent files to process")
    parser.add_argument("--api-limit", type=int, default=6, help="Maximum concurrent API calls")
    args = parser.parse_args()
    

    global API_SEMAPHORE
    API_SEMAPHORE = asyncio.Semaphore(args.api_limit)
    

    print(f"Loading character data from: ./heros_profile_aa.csv")
    character_data = await load_character_data("./heros_profile_aa.csv")
    print(f"Loaded data for {len(character_data)} characters")
    
    save_individual = not args.no_individual
    if os.path.isfile(args.input):
        await process_json_file(args.input, args.output, character_data, save_individual, args.max_concurrent)
    elif os.path.isdir(args.input):
        await process_json_directory(args.input, args.output, character_data, save_individual, args.max_concurrent_files)
    else:
        print(f"Error: Input path {args.input} does not exist")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    asyncio.run(main())