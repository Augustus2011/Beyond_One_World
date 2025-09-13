import asyncio
import aiofiles
from tools import (get_model)
import time
import pandas as pd
import random
import json
import re
import logging
from typing import List, Dict, Any
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_number(text):
    text = re.sub(r'CID:\s*\d+,?', '', text)
    text = text.encode('utf-8').decode('unicode_escape')
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(' ,', ',')
    return text

class AsyncDialogueGenerator:
    def __init__(self, model_name="gemini2-gen", model_name2="v3", max_concurrent_requests=5):
        self.model1 = get_model(model_name)
        self.model2 = get_model(model_name2)
        self.df = pd.read_csv("heros_profile.csv")
        self.df = self.df.rename(columns={
            "Name": "name", 
            "Source": "lore"
        })
        self.df["CID"] = pd.to_numeric(self.df["CID"], errors='coerce')
        self.dialogue = []
        self.situations = ["first meeting"]
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _call_model_async(self, model, prompt: str, delay: float = 0) -> str:
        """Async wrapper for model calls with semaphore for rate limiting"""
        async with self.semaphore:
            if delay > 0:
                await asyncio.sleep(delay)
            

            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, model, prompt)
                return result
            except Exception as e:
                logger.error(f"Model call failed: {e}")
                return f"Error: {str(e)}"

    async def generate_dialogue_async(self, target_uid: int) -> Dict[str, Any]:
        """Generate dialogue asynchronously"""
        group_chars = self.df[self.df["UID"] == target_uid].copy()
        if len(group_chars) < 3:
            raise ValueError(f"Not enough characters in UID {target_uid} for dialogue. Need at least 3 characters.")
        
        self.dialogue = []
        situation = self.situations[0]
        
        temp_chars = group_chars.copy().reset_index(drop=True)
        interest_char = temp_chars.iloc[0]
        non_interest_chars = temp_chars[temp_chars["CID"] != interest_char["CID"]].copy().reset_index(drop=True)
        
        available_non_interest = non_interest_chars.sample(frac=1).reset_index(drop=True)
        char_index = 0
        last_cid = None
        dialogue_tasks = []
        
        for turn in range(1, 31):
            if turn == 1:
                current_char = available_non_interest.iloc[0]
                last_cid = current_char["CID"]
            elif turn == 30:
                current_char = interest_char
                last_cid = current_char["CID"]
            else:
                if turn % 3 == 0 and turn != 30:
                    current_char = interest_char
                    last_cid = current_char["CID"]
                else:
                    next_index = (char_index + 1) % len(available_non_interest)
                    candidate = available_non_interest.iloc[next_index]
                    
                    if candidate["CID"] == last_cid:
                        next_index = (next_index + 1) % len(available_non_interest)
                        candidate = available_non_interest.iloc[next_index]
                    
                    current_char = candidate
                    char_index = next_index
                    last_cid = current_char["CID"]
            
            char_info = {
                "name": current_char["name"],
                "lore": current_char["lore"],
                "CID": current_char["CID"]
            }
            
            dialogue_tasks.append({
                "turn": turn,
                "char_info": char_info,
                "situation": situation,
                "group_chars": group_chars.head(3)
            })
        
        # Process dialogue turns in batches to maintain conversation flow
        batch_size = 3  # Process a few turns at a time to maintain some order
        for batch_start in range(0, len(dialogue_tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(dialogue_tasks))
            batch_tasks = dialogue_tasks[batch_start:batch_end]
            
            # Create async tasks for this batch
            async_tasks = []
            for task_data in batch_tasks:
                if task_data["turn"] < 30:  # Skip last turn as it's empty
                    async_tasks.append(
                        self._generate_single_turn_async(task_data)
                    )
                else:
                    # Handle turn 30 (empty response)
                    async_tasks.append(
                        self._create_empty_turn_async(task_data)
                    )
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # Add results to dialogue in order
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Turn generation failed: {result}")
                    continue
                self.dialogue.append(result)
        
        return await self._create_output_json_async()

    async def _generate_single_turn_async(self, task_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate a single dialogue turn asynchronously"""
        turn = task_data["turn"]
        char_info = task_data["char_info"]
        situation = task_data["situation"]
        group_chars = task_data["group_chars"]
        
        if turn == 1:
            input_prompt = self._create_first_turn_prompt(char_info, situation, group_chars)
        
        else:
            input_prompt = self._create_character_prompt(char_info, situation, group_chars)
        

        # Choose model based on turn (alternating)
        model_to_use = self.model1 if turn % 2 == 0 else self.model2
        
        # Add small delay to respect rate limits
        delay = (turn - 1) * 0.1  # Stagger requests
        
        response = await self._call_model_async(model_to_use, input_prompt, delay)
        
        return {
            "from": f"Name:{char_info['name']} Lore:{char_info['lore']} CID:{char_info['CID']}",
            "value": response
        }

    async def _create_empty_turn_async(self, task_data: Dict[str, Any]) -> Dict[str, str]:
        """Create empty turn for turn 30"""
        char_info = task_data["char_info"]
        return {
            "from": f"Name:{char_info['name']} Lore:{char_info['lore']} CID:{char_info['CID']}",
            "value": ""
        }

    def _create_first_turn_prompt(self, character, situation, group_chars) -> str:
        """Create prompt for the first turn with full context about other characters"""
        other_chars = group_chars[group_chars["name"] != character["name"]]
        other_chars_desc = ""
        for _, row in other_chars.iterrows():
            other_chars_desc += f"- {row['name']}: from {row['lore']}"
        
        name = character["name"]
        lore = character["lore"]

        prompt = f"""You are playing the role of {name}, from {lore}.
        The situation is: {situation}
        Other characters present: {other_chars_desc}, remember answer in very short sentence
        """
        return prompt
    
    def _create_character_prompt(self, character, situation, group_chars) -> str:
        """Create prompt for subsequent turns with minimal context"""
        name = character["name"]
        lore = character["lore"]
        
        # Get last dialogue if available
        last_dialogue = ""
        last_speaker = ""
        if self.dialogue:
            last_dialogue = self.dialogue[-1]["value"]
            last_speaker = self.dialogue[-1]["from"]

        other_chars = group_chars[group_chars["name"] != character["name"]]
        other_chars_desc = ""
        for _, row in other_chars.iterrows():
            other_chars_desc += f"- {row['name']}: from {row['lore']}\n"
        
        prompt = f"""You are playing the role of {name}, from {lore}.\n Last dialogue: {last_dialogue} from {clean_number(last_speaker)} try answer diferently.\n
Other characters present: {other_chars_desc} , remember answer in very short sentence
"""
        return prompt
    
    async def _create_output_json_async(self) -> Dict[str, Any]:
        """Create output JSON with async choice generation"""
        choices_prompts = [
            "Continue the conversation with a surprising revelation",
            "Introduce a conflict between the characters",
            "Add a moment of cooperation between the characters",
            "Bring up a topic related to the characters' backstories",
            "Do nothing",
            "Run away",
            "Challenge each other to a friendly competition",
            "Formulate a plan to tackle a common enemy",
            "Share a personal fear or vulnerability",
            "Accidentally reveal a secret to the other",
        ]
        
        # Generate all choices concurrently
        choice_tasks = []
        for i, prompt_letter in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]):
            choice_tasks.append(
                self._generate_choice_async(f"must do something like you ..{choices_prompts[i]}", prompt_letter)
            )
        
        choice_results = await asyncio.gather(*choice_tasks, return_exceptions=True)
        
        choices = {}
        for result in choice_results:
            if isinstance(result, Exception):
                logger.error(f"Choice generation failed: {result}")
                continue
            if isinstance(result, tuple) and len(result) == 2:
                letter, choice_text = result
                choices[letter] = choice_text
        
        # Get target_uid from dialogue context (you may need to pass this differently)
        target_uid = 1  # This should be passed as parameter
        
        output = {
            "dialogue": self.dialogue,
            "first_instruction": f"situation: {self.situations[0]}, interest character: {clean_number(self.df[self.df['UID'] == target_uid].iloc[0]['name']) if not self.df[self.df['UID'] == target_uid].empty else 'Unknown'} CID:{self.df[self.df['UID'] == target_uid].iloc[0]['CID'] if not self.df[self.df['UID'] == target_uid].empty else 'Unknown'}",
            "choices": choices
        }
        
        return output
    
    async def _generate_choice_async(self, prompt_direction: str, choice_letter: str) -> tuple:
        """Generate a single choice asynchronously"""
        if self.dialogue:
            last_speaker_info = self.dialogue[-1]["from"]
            
            # Get the dialogue from turn 29 (index 28) if available
            last_dialogue = ""
            if len(self.dialogue) >= 29:
                last_dialogue = self.dialogue[28]["value"]
            
            cid_match = re.search(r'CID:(\d+)', last_speaker_info)
            
            if cid_match:
                cid = int(cid_match.group(1))
                char_row = self.df[self.df["CID"] == cid]
                
                if not char_row.empty:
                    char_data = char_row.iloc[0]
                    lore = char_data["lore"]
                    name = char_data["name"]
                    prompt = f"""You are playing the role of {name}, from {lore}, answer in very short sentence {prompt_direction}. Last response from another person: {last_dialogue}"""
                    
                    response = await self._call_model_async(self.model1, prompt)
                    return (choice_letter, response)
        
        return (choice_letter, f"Option related to {prompt_direction}")


async def process_single_character_dialogue(generator: AsyncDialogueGenerator, 
                                            target_uid: int,
                                            character_index: int,
                                            all_chars: pd.DataFrame) -> Dict[str, Any]:
    """Process dialogue generation for a single character as interest character"""
    
    current_interest_char = all_chars.iloc[character_index]
    logger.info(f"Generating dialogue with {current_interest_char['name']} (CID: {current_interest_char['CID']}) as the interest character ({character_index+1}/{len(all_chars)})")
    
    # Create a new ordering of characters with current character as first (interest character)
    ordered_chars = all_chars.copy()
    interest_char = ordered_chars.iloc[character_index].copy()
    ordered_chars = ordered_chars.drop(character_index).reset_index(drop=True)
    ordered_chars = pd.concat([pd.DataFrame([interest_char]), ordered_chars], ignore_index=True)
    
    # Update the dataframe temporarily for this generation
    temp_df = generator.df.copy()
    temp_df = temp_df[temp_df["UID"] != target_uid]  # Remove all chars with this UID
    temp_df = pd.concat([temp_df, ordered_chars], ignore_index=True)  # Add back with new order
    generator.df = temp_df
    
    try:
        # Generate dialogue
        dialogue_data = await generator.generate_dialogue_async(target_uid)
        
        # Add UID and CID information for easy filtering
        dialogue_data["uid"] = target_uid
        dialogue_data["cid"] = int(current_interest_char["CID"])
        
        # Save individual file
        filename = f"dialogue_uid_{target_uid}_cid_{int(current_interest_char['CID'])}.json"
        
        # Ensure directory exists
        #os.makedirs("generated_results/multiversal_dialogue/1/", exist_ok=True)
        
        async with aiofiles.open(f"generated_results/multiversal_dialogue/re_2/{filename}", "w") as f:
            await f.write(json.dumps(dialogue_data, indent=2))
        
        logger.info(f"Saved dialogue to {filename}")
        return dialogue_data
        
    except Exception as e:
        logger.error(f"Error generating dialogue for UID {target_uid}, CID {current_interest_char['CID']}: {e}")
        return None

async def main():
    """Main async function to orchestrate the entire pipeline"""
    generator = AsyncDialogueGenerator(model_name="gemini2", model_name2="v3", max_concurrent_requests=3)
    all_dialogues = []
    
    # Process all UIDs
    uid_tasks = []
    for target_uid in range(1, 31):  # UIDs 1 to 30
        all_chars = generator.df[generator.df["UID"] == target_uid].copy().reset_index(drop=True)
        
        if len(all_chars) < 3:
            logger.info(f"Skipping UID {target_uid}: has fewer than 3 characters. Need at least 3 characters.")
            continue
        else:
            logger.info(f"Found {len(all_chars)} characters in UID {target_uid}")
            
            # Create tasks for each character as interest character
            for i in range(len(all_chars)):
                uid_tasks.append(
                    process_single_character_dialogue(generator, target_uid, i, all_chars)
                )
    
    # Process all tasks with controlled concurrency
    batch_size = 5  # Process 5 dialogues concurrently
    
    for batch_start in range(0, len(uid_tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(uid_tasks))
        batch_tasks = uid_tasks[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(uid_tasks) + batch_size - 1)//batch_size}")
        
        # Process batch
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect successful results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch task failed: {result}")
                continue
            if result is not None:
                all_dialogues.append(result)
        
        # Small delay between batches to be nice to the API
        await asyncio.sleep(1)
    
    # Save all dialogues to a single file
    combined_filename = "all_dialogues.json"
    #os.makedirs("generated_results/multiversal_dialogue/re_2/", exist_ok=True)
    
    async with aiofiles.open(f"generated_results/multiversal_dialogue/re_2/{combined_filename}", "w") as f:
        await f.write(json.dumps(all_dialogues, indent=2))
    
    logger.info(f"All dialogues combined and saved to {combined_filename}")
    logger.info(f"Total dialogues generated: {len(all_dialogues)}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())