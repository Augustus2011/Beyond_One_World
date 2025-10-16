
#other apis
import requests
import os

#gemini
from google import genai
from google.genai.types import GenerationConfig
from google.genai import types

#anthropic
import anthropic

#nebius
from openai import OpenAI

#env
from dotenv import load_dotenv
import os 


load_dotenv()

# clean string
import json
import os
from scripts.clean import clean_markdown


# other apis
import requests

# gemini
from google import genai
from google.genai import types

# env
from dotenv import load_dotenv


# any model
from model.LLM import LLM

load_dotenv()


def get_model(model_name):
    models = {
        "your_model":your_model("your_huggingface_model_path").generate,
        "gemini2":google_gemini.gemini2_flash,
        "gemini2-5":google_gemini.gemini2_5_flash,
        "gemini2-5-think":google_gemini.gemini2_5_flash_thinking,
        "sonnet3-7-gen":sonnet.sonnet_37_gen,
        "sonnet3-7":sonnet.sonnet_37,
        "sonnet3-7-think":sonnet.sonnet_37_thinking,
        "sonnet3-5":sonnet.sonnet_35,
        "judge":sonnet.sonnet_37_judge,
        "gen-think":gpt4o_mini,
        "r1":hyperbolic.r1,
        "v3":hyperbolic.deepseek_v3,
    }

    return models.get(model_name, None)




# ---------------- GEMINI CONFIGS ---------------- #
CONFIG_4_ANSWER = types.GenerateContentConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=2,
    seed=0,
    max_output_tokens=1024,
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
)

CONFIG_4_GEN = types.GenerateContentConfig(
    temperature=1.2,
    top_p=0.90,
    top_k=10,
    seed=0,
    max_output_tokens=1024,
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
)


SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]



def get_processed_cids(jsonl_path):
    processed_cids = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_cids.add(data["CID"])
                except json.JSONDecodeError:
                    print("Warning: Skipping corrupted line in JSONL file")
    return processed_cids





#helper function for cr. anthropic cookbook
def print_thinking_response(response):
    """Pretty print a message response with thinking blocks."""
    print("\n==== FULL RESPONSE ====")
    for block in response.content:
        if block.type == "thinking":
            print("\n🧠 THINKING BLOCK:")
            # Show truncated thinking for readability
            print(block.thinking[:500] + "..." if len(block.thinking) > 500 else block.thinking)


        elif block.type == "redacted_thinking":
            print("\n🔒 REDACTED THINKING BLOCK:")
            print(f"[Data length: {len(block.data) if hasattr(block, 'data') else 'N/A'}]")
        elif block.type == "text":
            print("\n✓ FINAL ANSWER:")
            print(block.text)
    
    print("\n==== END RESPONSE ====")


#helper function for cr. anthropic cookbook
def return_thinking_response(response):
    """Pretty print a message response with thinking blocks and return formatted output."""
    output = []
    output.append("\n==== FULL RESPONSE ====")
    for block in response.content:
        if block.type == "thinking":
            output.append("\n🧠 THINKING BLOCK:")
            thinking_text = block.thinking[:500] + "..." if len(block.thinking) > 500 else block.thinking
            output.append(thinking_text)
        elif block.type == "redacted_thinking":
            output.append("\n🔒 REDACTED THINKING BLOCK:")
            output.append(f"[Data length: {len(block.data) if hasattr(block, 'data') else 'N/A'}]")
        elif block.type == "text":
            output.append("\n✓ FINAL ANSWER:")
            output.append(block.text)
    
    output.append("\n==== END RESPONSE ====")
    return "\n".join(output)

# custom you trained model
class your_model(LLM):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    def __init__(self,hf_path:str=""):
        self.hf_path=hf_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_path)

    def generate(self, prompt: str, **kwargs) -> str:
        #llama3 template adjust your model here
        messages = [
            {"role": "user", "content": f"{prompt}"},
        ]
        inputs = self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to("gpu")
        outputs = self.model.generate(**inputs, max_new_tokens=40)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    


class sonnet:
    def sonnet_37(input_prompt:str="")->str:
        client = anthropic.Anthropic(api_key=os.environ["CLAUDE_KEY"])
        try:
            response=client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    messages=[
                        {
                            "role": "user",
                            "content": input_prompt,
                        },
                    ],
                    max_tokens=1024,
                    temperature=0.6,
                    top_p = 0.95,
                )
            return response.content[0].text
        except:
            return "Error: API"
        
    def sonnet_37_gen(input_prompt:str="")->str:
        client = anthropic.Anthropic(api_key=os.environ["CLAUDE_KEY"])
        try:
            response=client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        messages=[
                            {
                                "role": "user",
                                "content": input_prompt,
                            },
                        ],
                        max_tokens=1024,
                        temperature=1,
                        top_p = 0.90,
                    )
            return response.content[0].text
        except:
            return "Error: API"
        
    def sonnet_37_judge(input_prompt:str="")->str:
        client = anthropic.Anthropic(api_key=os.environ["CLAUDE_KEY"])
        try:
            response=client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        messages=[
                            {
                                "role": "user",
                                "content": "Help me scoring character role-playing, score point between 0-5 , score have two type: thinking(doec thiking response look like reference character) ,acting(does response acting like the character reference), the output must be this format: think_score,act_score example 3,2,"+input_prompt,
                            },
                        ],
                        max_tokens=1024,
                        temperature=0.1,
                        top_p = 0.90,
                    )
            return response.content[0].text
        except:
            return "Error: API"

        
    def sonnet_37_thinking(input_prompt:str="")->str:
        client = anthropic.Anthropic(api_key=os.environ["CLAUDE_KEY"])
        try:
            response=client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 1024
                        },
                    max_tokens=2000,
                    temperature=1,
                    #top_p = 0.95,

                    messages=[
                        {
                            "role": "user",
                            "content": input_prompt,
                        },
                    ],
                )
            return return_thinking_response(response)
        except:
            return "Error: API"

        
    def sonnet_35(input_prompt:str="")->str:
        client = anthropic.Anthropic(api_key=os.environ["CLAUDE_KEY"])
        try:
            response=client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    messages=[
                        {
                            "role": "user",
                            "content": input_prompt,
                        },
                    ],
                    max_tokens= 1024,
                    temperature= 0.6,
                    top_p = 0.95,

                )
            return response.content[0].text
        except:
            return "Error: API"
        
    def think_gen(input_prompt:str="")->str:
        
        client = anthropic.Anthropic(api_key=os.environ["CLAUDE_KEY"])
        try:
            response=client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    messages=[
                        {
                            "role": "user",
                            "content": """Your task is to **separate** internal thought("thinking") from external behavior ("acting") in the text response.
                            Wrap the internal thought process in `<thinking>...</thinking>` and the final response or action in `<acting>...</acting>`.                  
                            Now process text response:"""+input_prompt,
                        },
                    ],
                    max_tokens= 1024,
                    temperature= 0.6,
                    top_p = 0.95,

                )
            return response.content[0].text
        except:
            return "Error: API"

class hyperbolic:
    def r1(input_prompt: str = "") -> str:
        url = "https://api.hyperbolic.xyz/v1/chat/completions"
        key = os.getenv("R1_API_KEY")
        if not key:
            raise ValueError("R1_API_KEY is missing from environment variables")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        data = {
            "messages": [{"role": "user", "content": input_prompt}],
            "model": "deepseek-ai/DeepSeek-R1",
            "temperature": 0.6,
            "max_tokens": 1024,
            "top_p": 0.95,
            "top_k": 2,
        }

        try:
            response = requests.post(url, headers=headers, json=data,timeout=1000)
            response.raise_for_status()
            if not response.text.strip():
                print("Warning: API returned an empty response")
                return "Error: API Empty response from API"

            response_data = response.json()


            if "choices" not in response_data or not response_data["choices"]:
                print(f"Error: Unexpected API response format: {response_data}")
                return f"Error: API Invalid response format - {response_data}"

            return clean_markdown(response_data["choices"][0]["message"]["content"])

        except requests.exceptions.Timeout:
            return "Error: API request timed out"
        except requests.exceptions.RequestException as e:
            return f"Error: API request failed - {str(e)}"
        except ValueError as e:
            return f"Error: API JSON decoding failed - {str(e)}"
        

    def gen_think_v3(input_prompt:str="")->str:
        url = "https://api.hyperbolic.xyz/v1/chat/completions"
        key = os.getenv("R1_API_KEY")
        if not key:
            raise ValueError("R1_API_KEY is missing from environment variables")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        data = {
            "messages": [{"role": "user", "content": f"""Your task is to **separate** internal thought("thinking") from external behavior ("acting") in the text response.Wrap the internal thought process in `<thinking>...</thinking>` and the final response or action in `<acting>...</acting>`.
        Now process text response: {input_prompt}"""}],
            "model": "deepseek-ai/DeepSeek-V3-0324",
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_p": 0.70,
            "top_k": 1,
        }

        try:
            response = requests.post(url, headers=headers, json=data,timeout=1000)
            response.raise_for_status()
            if not response.text.strip():
                return "Error: API Empty response from API"

            response_data = response.json()

            if "choices" not in response_data or not response_data["choices"]:
                print(f"Error: API Unexpected API response format: {response_data}")
                return f"Error: API Invalid response format - {response_data}"

            return clean_markdown(response_data["choices"][0]["message"]["content"])

        except requests.exceptions.Timeout:
            return "Error: API request timed out"
        except requests.exceptions.RequestException as e:
            return f"Error: API request failed - {str(e)}"
        except ValueError as e:
            return f"Error: API JSON decoding failed - {str(e)}"

    
    def deepseek_v3(input_prompt:str="")->str:
        url = "https://api.hyperbolic.xyz/v1/chat/completions"
        key = os.getenv("R1_API_KEY")
        if not key:
            raise ValueError("R1_API_KEY is missing from environment variables")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        data = {
            "messages": [{"role": "user", "content": input_prompt}],
            "model": "deepseek-ai/DeepSeek-V3-0324",
            "temperature": 0.6,
            "max_tokens": 1024,
            "top_p": 0.95,
            "top_k": 2,
        }

        try:
            response = requests.post(url, headers=headers, json=data,timeout=1000)
            response.raise_for_status()
            if not response.text.strip():
                return "Error: API Empty response from API"

            response_data = response.json()

            if "choices" not in response_data or not response_data["choices"]:
                print(f"Error: API Unexpected API response format: {response_data}")
                return f"Error: API Invalid response format - {response_data}"

            return clean_markdown(response_data["choices"][0]["message"]["content"])

        except requests.exceptions.Timeout:
            return "Error: API request timed out"
        except requests.exceptions.RequestException as e:
            return f"Error: API request failed - {str(e)}"
        except ValueError as e:
            return f"Error: API JSON decoding failed - {str(e)}"


def deepseek_v3(input_prompt:str="")->str:
    url = "https://api.deepseek.com/chat/completions"
    key = os.getenv("R1_API_KEY")
    if not key:
        raise ValueError("R1_API_KEY is missing from environment variables")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    data = {
        "messages": [{"role": "user", "content": input_prompt}],
        "model": "deepseek-chat",
        'stream': False,
        "temperature": 0.6,
        "max_tokens": 1024,
        "top_p": 0.95,
        "top_k": 2,
    }

    try:
        response = requests.post(url, headers=headers, json=data,timeout=1000)
        response.raise_for_status()
        if not response.text.strip():
            return "Error: API Empty response from API"

        response_data = response.json()

        if "choices" not in response_data or not response_data["choices"]:
            print(f"Error: API Unexpected API response format: {response_data}")
            return f"Error: API Invalid response format - {response_data}"

        return clean_markdown(response_data["choices"][0]["message"]["content"])

    except requests.exceptions.Timeout:
        return "Error: API request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed - {str(e)}"
    except ValueError as e:
        return f"Error: API JSON decoding failed - {str(e)}"




def gpt4o_mini(input_prompt:str="")->str:
    client=client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": input_prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


class google_gemini:
    def gemini2_flash_lite(input_prompt:str="")->str:
        
        client = genai.Client(api_key=os.getenv("GEMINI_API"))
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=input_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=2,
                    safety_settings=SAFETY_SETTINGS,
                    
                )
                
            )
            return response.text
        except:
            return "Error: API"

    def gemini2_flash_thinking(input_prompt:str="")->str:
        client = genai.Client(api_key=os.getenv("GEMINI_API"))

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=input_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.6,
                top_p=0.95,
                top_k=2,
                thinking_config=types.ThinkingConfig(thinking_budget=512),
                safety_settings=SAFETY_SETTINGS,
            )
            
        )
        return response.text

    def gemini2_flash(input_prompt:str="")->str:

        client = genai.Client(api_key=os.getenv("GEMINI_API"))
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=input_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=2,
                    safety_settings=SAFETY_SETTINGS,

                )
                
            )
            return response.text
        except:
            return "Error: API"
    

    def gemini2_5_flash_thinking(input_prompt:str="")->str:

        client = genai.Client(api_key=os.getenv("GEMINI_API"))
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=input_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=2,
                    thinking_config=types.ThinkingConfig(thinking_budget=512),
                    safety_settings=SAFETY_SETTINGS,

                )
            )
            return response.text
        except:
            return "Error: API"

    def gemini2_5_flash(input_prompt:str="")->str:

        client = genai.Client(api_key=os.getenv("GEMINI_API"))
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=input_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=2,
                    safety_settings=SAFETY_SETTINGS,

                )
            )
            return response.text
        
        except:
            return "Error: API"
    