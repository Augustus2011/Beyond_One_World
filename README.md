#Beyond One World

A comprehensive framework for evaluating and generating character-based AI responses using multiple large language models. This research tool enables systematic evaluation of character role-playing capabilities across different AI models through dilemma resolution, canonical event generation, and multiversal dialogue creation.

## 🎯 Overview

- **Character Dilemma**: Evaluate how different AI models handle moral and ethical dilemmas in character-specific contexts.
- **Canonical Event**: Evaluate characters responses to canonical events.
- **Multiversal Dialogue**: Create cross-character interactions and dialogues #future work.
- **Automated Scoring**: Evaluate character consistency and role-playing quality using AI judges.
- **label platform**: for experts to construct dataset.


## 📁 Project Structure

```
charactor_ai/
├── tools.py                          # Core utilities and model management
├── generate_answer.py                # Main generation script (synchronous)
├── generate_answer_con.py           # Concurrent generation script
├── generate_meme_question.py        # Dialogue generation (synchronous)
├── generate_meme_question_con.py    # Concurrent dialogue generation
├── scoring.py                       # Character response scoring
├── scoring_con.py                   # Concurrent scoring
├── match_think_act.py               # Think-act pair extraction
├── match_think_act_con.py           # Concurrent think-act matching
├── label_platform1.py               # Annotation platform
├── label_platform2.py               # Enhanced annotation platform
├── model/
│   └── LLM.py        #abstract class to be implemented with your llm-model
├── scripts/
│   └── clean.py                  # clean text
├── generated_results/               # Output directory
│   ├── canon/                       # Canonical event results
│   ├── dilemma/                     # Dilemma resolution results
│   └── multiversal_dialogue/        # Dialogue results
├── annotated_results/               # Human annotations
├── all_character_data.csv          # Character dataset 
├── character_dilemmas.json         # required dataset on our hf
├── characters_canon_events.json    # required dataset on our hf
├── heros_profile_aa.csv            # required dataset on our hf
└── requirements.txt                 # Dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+

### Setup
1. Clone the repository:
```bash
https://github.com/Augustus2011/Beyond_One_World.git
cd Beyond_One_World
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file with your API keys
CLAUDE_KEY=your_claude_api_key
GEMINI_API=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
R1_API_KEY=your_r1_api_key
```

## 📊 Usage

### Character Dilemma Resolution

Generate character responses to moral dilemmas:

```bash
# Basic dilemma generation
python generate_answer.py --model gemini2 --data character_dilemmas.json --output ./generated_results/dilemma_gemini2.jsonl --task dilemma

# With chain-of-thought reasoning
python generate_answer.py --model sonnet3-7-think --data character_dilemmas.json --output ./generated_results/dilemma_claude_thinking.jsonl --task dilemma --cot

# Clean consequence text
python generate_answer.py --model r1 --data character_dilemmas.json --output ./generated_results/dilemma_r1_clean.jsonl --task dilemma --clean_consequence
```

### Canonical Event Generation

Generate character responses to canonical events:

```bash
# Basic canon event generation
python generate_answer.py --model sonnet3-7 --data characters_canon_events.json --output ./generated_results/canon_claude.jsonl --task canon

# With thinking capabilities
python generate_answer.py --model gemini2-5-think --data characters_canon_events.json --output ./generated_results/canon_gemini_thinking.jsonl --task canon
```

### Concurrent Processing

For large-scale processing, use the concurrent versions:

```bash
# Concurrent dilemma generation
python generate_answer_con.py --model r1 --data character_dilemmas.json --output ./generated_results/dilemma_r1_concurrent.jsonl --task dilemma --max_con 10

# Concurrent canon generation
python generate_answer_con.py --model sonnet3-7 --data character_dilemmas.json --output ./generated_results/canon_claude_concurrent.jsonl --task canon --max_con 8
```

### Create character dialogues(future work):

```bash
# Synchronous dialogue generation
python generate_meme_question.py

# Concurrent dialogue generation
python generate_meme_question_con.py
```

### Character Response Scoring

Evaluate character consistency and role-playing quality:

```bash
# Basic scoring
python scoring.py --input ./generated_results/dilemma_gemini2.jsonl --output ./scored_results/dilemma_gemini2_scored.json --cdata ./heros_profile_aa.csv

# Concurrent scoring
python scoring_con.py --input ./generated_results/dilemma_gemini2.jsonl --output ./scored_results/dilemma_gemini2_scored_concurrent.json --cdata ./heros_profile_aa.csv --max-concurrent 6
```

### Think-Act Pair Extraction

Extract thinking and acting components from responses:

```bash
# Basic extraction
python match_think_act.py input_file.json output_file.json

# Concurrent extraction
python match_think_act_con.py input_directory/
```

### Reprocessing Failed Results

Retry failed API calls:

```bash
python generate_answer.py --model r1 --output ./generated_results/dilemma_r1_fixed.jsonl --task dilemma --apierror --inputfile ./generated_results/dilemma_r1_failed.jsonl
```

## 🔧 Configuration

### Model Configuration

Models are configured in `tools.py`:

```python
def get_model(model_name):
    models = {
        "your_model":your_model("your_huggingface_model_path").generate,
        "gemini2": google_gemini.gemini2_flash,
        "gemini2-5": google_gemini.gemini2_5_flash,
        "gemini2-5-think": google_gemini.gemini2_5_flash_thinking,
        "sonnet3-7": sonnet.sonnet_37,
        "sonnet3-7-think": sonnet.sonnet_37_thinking,
        "sonnet3-5": sonnet.sonnet_35,
        "judge": sonnet.sonnet_37_judge,
        "gen-think": gpt4o_mini,
        "r1": hyperbolic.r1,
        "v3": hyperbolic.deepseek_v3,
    }
    return models.get(model_name, None)
```

<!-- 
## 📈 Data Format

### Input Data Format

Character data should be in CSV format with the following columns:
- `CID`: Character ID
- `Name`: Character name
- `Source`: Source material (e.g., "Marvel", "DC")
- `Attributes`: Character attributes and background

### Output Data Format

Results are saved in JSONL format with the following structure:

```json
{
  "CID": "character_id",
  "name": "Character Name",
  "task": "dilemma|canon|dialogue",
  "model": "model_name",
  "response": "AI generated response",
  "thinking": "Internal reasoning (if available)",
  "acting": "External behavior (if available)",
}
``` -->

## Label Platform

```
docker-compose up --build
```





## 📚 Citation

```bibtex
@misc{2510.14351,
Author = {Perapard Ngokpol and Kun Kerdthaisong and Pasin Buakhaw and Pitikorn Khlaisamniang and Supasate Vorathammathorn and Piyalitt Ittichaiwong and Nutchanon Yongsatianchot},
Title = {Beyond One World: Benchmarking Super Heros in Role-Playing Across Multiversal Contexts},
Year = {2025},
Eprint = {arXiv:2510.14351},
}
---


