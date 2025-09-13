
import glob
import os 
import pandas
import json
import re
import argparse
from typing import Dict, Any, Tuple
from datetime import datetime

from tools import get_model, initialize_vertexai, anonymize_path


# Get models
construct_think_act = get_model("gen-think")
judge = get_model("judge")

def load_character_data(data_path: str) -> Dict[str, Dict[str, Any]]:
    """Load character data from CSV file and create a lookup dictionary."""
    df = pandas.read_csv(data_path)
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
    """Extract thinking and acting scores from judge model output."""
    try:
        # Use regex to find two numbers separated by comma
        match = re.search(r'(\d+),(\d+)', score_text)
        if match:
            think_score = float(match.group(1))
            act_score = float(match.group(2))
            return think_score, act_score
        return 2.5, 2.5  # Default average scores if pattern not found
    except:
        return 2.5, 2.5  # Default average scores if any error occurs

def process_and_score(input_json: Dict[str, Any], character_data: Dict[str, Dict[str, Any]], 
                      output_dir: str = None, index: int = None) -> Dict[str, Any]:
    """
    Process input JSON through the pipeline and add scores.
    
    Args:
        input_json: Dictionary containing the input data
        character_data: Dictionary containing character reference data
        output_dir: Optional directory to save individual result files
        index: Optional index for naming individual result files
        
    Returns:
        Dictionary with added scores
    """
    try:
        # Get the text to process
        text_to_process = input_json.get('answers', '')
        cid = str(input_json.get('CID', ''))
        
        # Get character reference data
        char_ref = character_data.get(cid, {})
        name = char_ref.get('name', 'Unknown')
        source = char_ref.get('source', '')
        attributes = char_ref.get('attributes', '')
        
        # Create reference context
        reference_context = f"Character: {name}\nSource: {source}\nAttributes: {attributes}"
        
        # Get thinking/acting separation
        think_act_result = construct_think_act(text_to_process)
        
        # Get scoring from judge with reference context
        score_result = judge(f"reference :\n{reference_context}\n\nResponse to evaluate:\n{text_to_process}")
        
        # Extract scores
        think_score, act_score = extract_scores(score_result)
        
        # Print progress information
        print(f"Processed: CID={cid}, Name={name}, Thinking={think_score}, Acting={act_score}")
        
        # Add results to input JSON
        result = input_json.copy()
        result['think_act_analysis'] = think_act_result
        result['score'] = {
            'thinking': think_score,
            'acting': act_score
        }
        result['character_reference'] = char_ref
        
        # Save individual result file if output_dir is provided
        if output_dir and index is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            individual_file = os.path.join(output_dir, f"{cid}_{name.replace(' ', '_')}_{index}_{timestamp}.json")
            with open(individual_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
        
    except Exception as e:
        # If any error occurs, return input with default scores and print error
        print(f"Error processing CID={cid}: {str(e)}")
        result = input_json.copy()
        result['score'] = {
            'thinking': 2.5,
            'acting': 2.5
        }
        result['error'] = str(e)
        return result

def process_json_file(input_file: str, output_file: str, character_data: Dict[str, Dict[str, Any]], save_individual: bool = True):
    """
    Process a JSONL file (one JSON object per line) through the pipeline and save results.
    """
    try:
        results = []
        
        # Create directory for individual results if needed
        individual_dir = None
        if save_individual:
            individual_dir = os.path.join(os.path.dirname(output_file), "individual_results")
            os.makedirs(individual_dir, exist_ok=True)
            print(f"Saving individual results to: {individual_dir}")
            
        print(f"\nProcessing file: {anonymize_path(input_file)}")
        print("-" * 60)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue  # skip empty lines
                input_data = json.loads(line)
                result = process_and_score(input_data, character_data, individual_dir, i)
                results.append(result)

        # Save all results as a JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print("-" * 60)
        print(f"Completed processing {len(results)} items. Full results saved to: {anonymize_path(output_file)}")

    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")


def process_json_directory(input_dir: str, output_dir: str, character_data: Dict[str, Dict[str, Any]], save_individual: bool = True):
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save output JSON files
        character_data: Dictionary containing character reference data
        save_individual: Whether to save individual result files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    print(f"Found {len(json_files)} JSON files to process in {input_dir}")
    
    for i, json_file in enumerate(json_files):
        print(f"\nProcessing file {i+1}/{len(json_files)}: {os.path.basename(json_file)}")
        output_file = os.path.join(output_dir, os.path.basename(json_file))
        process_json_file(json_file, output_file, character_data, save_individual)

def main():
    parser = argparse.ArgumentParser(description="Process and score character responses using AI models.")
    parser.add_argument("--input", required=True, help="Path to input JSON file or directory")
    parser.add_argument("--output", required=True, help="Path to save output JSON file or directory")
    parser.add_argument("--cdata", required=False, help="Path to character data CSV file")
    parser.add_argument("--no-individual", action="store_true", help="Disable saving individual JSON files")
    args = parser.parse_args()
    
    # Load character data
    character_data_path = args.cdata if args.cdata else "./heros_profile_aa.csv"
    print(f"Loading character data from: {character_data_path}")
    character_data = load_character_data(character_data_path)
    print(f"Loaded data for {len(character_data)} characters")
    
    # Process single file or directory
    save_individual = not args.no_individual
    if os.path.isfile(args.input):
        process_json_file(args.input, args.output, character_data, save_individual)
    elif os.path.isdir(args.input):
        process_json_directory(args.input, args.output, character_data, save_individual)
    else:
        print(f"Error: Input path {args.input} does not exist")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()