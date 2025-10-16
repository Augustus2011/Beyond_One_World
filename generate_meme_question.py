from tools import get_model
import time
import pandas as pd
import json
import re
from scripts.clean import clean_number
    

class DialogueGenerator:
    def __init__(self, model_name="gemini2-gen",model_name2="v3"):
        self.model1 = get_model(model_name)
        self.model2= get_model(model_name2)
        self.df = pd.read_csv("heros_profile.csv")#.reset_index(drop=True)
        self.df = self.df.rename(columns={
            "Name": "name", 
            "Source": "lore"
        })

        self.df["CID"] = pd.to_numeric(self.df["CID"], errors='coerce')
        self.dialogue = []
        self.situations = [
            "first meeting",
        ]

    def generate_dialogue(self, target_uid):
        group_chars = self.df[self.df["UID"] == target_uid].copy()
        if len(group_chars) < 3:
            raise ValueError(f"Not enough characters in UID {target_uid} for dialogue. Need at least 3 characters.")
        
        self.dialogue = []
        situation = self.situations[0]
        
        temp_chars = group_chars.copy().reset_index(drop=True)
        interest_char = temp_chars.iloc[0]
        non_interest_chars = temp_chars[temp_chars["CID"] != interest_char["CID"]].copy().reset_index(drop=True)
        
        available_non_interest = non_interest_chars.sample(frac=1).reset_index(drop=True)  # Shuffle once
        char_index = 0

        last_cid = None
        
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
            
            if turn == 1:
                input_prompt = self._create_first_turn_prompt(char_info, situation, group_chars.head(3))
            else:
                input_prompt = self._create_character_prompt(char_info, situation, group_chars.head(3))
            
            
            if turn <30:
                if turn%2==0:
                    response = self.model1(input_prompt)
                else:
                    response = self.model2(input_prompt)
            else:
                response=""
                
            
            self.dialogue.append({
                "from": f"Name:{char_info['name']} Lore:{char_info['lore']} CID:{char_info['CID']}",
                "value": response
            })
            
            time.sleep(2)
        
        return self._create_output_json()
    
    def _create_first_turn_prompt(self, character, situation, group_chars)->str:
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
    
    def _create_character_prompt(self, character, situation, group_chars)->str:
        name = character["name"]
        lore = character["lore"]
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
    
    def _create_output_json(self):
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
        
        choices = {}
        for i, prompt_letter in enumerate(["A", "B", "C", "D","E","F","G","H","I","J"]):
            choices[prompt_letter] = self._generate_choice("must do something like you .."+choices_prompts[i])
        
        output = {
            "dialogue": self.dialogue,
            "first_instruction": f"situation: {self.situations[0]}, interest character: {clean_number(self.df[self.df['UID'] == target_uid].iloc[0]['name'])} CID:{self.df[self.df['UID'] == target_uid].iloc[0]['CID']}",
            "choices": choices
        }
        
        return json.dumps(output, indent=2)
    
    def _generate_choice(self, prompt_direction):
        if self.dialogue:
            last_speaker_info = self.dialogue[-1]["from"]
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
                    response = self.model1(prompt)

                    return response
        
        return f"Option related to {prompt_direction}"

if __name__ == "__main__":
    generator = DialogueGenerator(model_name="gemini2",model_name2="v3")
    all_dialogues = []
    
    for target_uid in range(1,31):  # UIDs 1 to 30
        all_chars = generator.df[generator.df["UID"] == target_uid].copy().reset_index(drop=True)
        print(all_chars.columns)
        if len(all_chars) < 3:
            print(f"Skipping UID {target_uid}: has fewer than 3 characters. Need at least 3 characters.")
            continue
        else:
            print(f"Found {len(all_chars)} characters in UID {target_uid}")
            for i in range(len(all_chars)):
                current_interest_char = all_chars.iloc[i]
                print(f"\nGenerating dialogue with {current_interest_char['name']} (CID: {current_interest_char['CID']}) as the interest character ({i+1}/{len(all_chars)})")
                ordered_chars = all_chars.copy()
                interest_char = ordered_chars.iloc[i].copy()
                ordered_chars = ordered_chars.drop(i).reset_index(drop=True)
                ordered_chars = pd.concat([pd.DataFrame([interest_char]), ordered_chars], ignore_index=True)
                temp_df = generator.df.copy()
                temp_df = temp_df[temp_df["UID"] != target_uid]
                temp_df = pd.concat([temp_df, ordered_chars], ignore_index=True)
                generator.df = temp_df
                
                try:
                    dialogue_json_str = generator.generate_dialogue(target_uid)
                    dialogue_data = json.loads(dialogue_json_str)
                    dialogue_data["uid"] = target_uid
                    dialogue_data["cid"] = int(current_interest_char["CID"])
                    all_dialogues.append(dialogue_data)
                    filename = f"dialogue_uid_{target_uid}_cid_{int(current_interest_char['CID'])}.json"
                    with open("generated_results/multiversal_dialogue/1/"+filename, "w") as f:
                        f.write(dialogue_json_str)
                    
                    print(f"Saved dialogue to {filename}")
                except Exception as e:
                    print(f"Error generating dialogue for UID {target_uid}, CID {current_interest_char['CID']}: {e}")
                
                time.sleep(0.5)
    
    combined_filename = "all_dialogues.json"
    with open("generated_results/multiversal_dialogue/1/"+combined_filename, "w") as f:
        json.dump(all_dialogues, f, indent=2)
    
    print(f"\nAll dialogues combined and saved to {combined_filename}")



