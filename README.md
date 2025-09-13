# Character AI Research Framework

A comprehensive framework for evaluating and generating character-based AI responses using multiple large language models. This research tool enables systematic evaluation of character role-playing capabilities across different AI models through dilemma resolution, canonical event generation, and multiversal dialogue creation.

## 🎯 Overview

This framework provides a standardized approach to:
- **Character Dilemma Resolution**: Evaluate how different AI models handle moral and ethical dilemmas in character-specific contexts
- **Canonical Event Generation**: Generate character-appropriate responses to canonical events from their source material
- **Multiversal Dialogue**: Create cross-character interactions and dialogues
- **Automated Scoring**: Evaluate character consistency and role-playing quality using AI judges

## 🚀 Features

### Multi-Model Support
- **Google Gemini**: 2.0 Flash, 2.5 Flash (with thinking capabilities)
- **Anthropic Claude**: 3.5 Sonnet, 3.7 Sonnet (with thinking capabilities)
- **OpenAI**: GPT-4o Mini
- **DeepSeek**: R1, V3 models
- **Hyperbolic**: Various model integrations

### Core Capabilities
- **Concurrent Processing**: Async and concurrent processing for large-scale evaluations
- **Path Anonymization**: Privacy-preserving file operations with anonymized logging
- **Flexible Data Formats**: Support for CSV, JSON, and JSONL input/output
- **Automated Scoring**: AI-powered evaluation of character consistency and role-playing quality
- **Reprocessing Pipeline**: Automatic retry and error handling for failed API calls

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
├── scripts/
│   └── analysis.py                  # Data analysis utilities
├── generated_results/               # Output directory
│   ├── canon/                       # Canonical event results
│   ├── dilemma/                     # Dilemma resolution results
│   └── multiversal_dialogue/        # Dialogue results
├── annotated_results/               # Human annotations
├── all_character_data.csv          # Character dataset
├── heros_profile_aa.csv            # Character profiles
└── requirements.txt                 # Dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- API keys for supported models

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd charactor_ai
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
python generate_answer.py --model gemini2 --data all_character_data.csv --output ./generated_results/dilemma_gemini2.jsonl --task dilemma

# With chain-of-thought reasoning
python generate_answer.py --model sonnet3-7-think --data all_character_data.csv --output ./generated_results/dilemma_claude_thinking.jsonl --task dilemma --cot

# Clean consequence text
python generate_answer.py --model r1 --data all_character_data.csv --output ./generated_results/dilemma_r1_clean.jsonl --task dilemma --clean_consequence
```

### Canonical Event Generation

Generate character responses to canonical events:

```bash
# Basic canon event generation
python generate_answer.py --model sonnet3-7 --data all_character_data.csv --output ./generated_results/canon_claude.jsonl --task canon

# With thinking capabilities
python generate_answer.py --model gemini2-5-think --data all_character_data.csv --output ./generated_results/canon_gemini_thinking.jsonl --task canon
```

### Concurrent Processing

For large-scale processing, use the concurrent versions:

```bash
# Concurrent dilemma generation
python generate_answer_con.py --model r1 --data all_character_data.csv --output ./generated_results/dilemma_r1_concurrent.jsonl --task dilemma --max_con 10

# Concurrent canon generation
python generate_answer_con.py --model sonnet3-7 --data all_character_data.csv --output ./generated_results/canon_claude_concurrent.jsonl --task canon --max_con 8
```

### Create cross-character dialogues:

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
```

## 🧪 Evaluation Metrics

### Scoring System

The framework uses AI-powered evaluation with the following metrics:

1. **Thinking Score (0-5)**: How well the internal reasoning matches the character
2. **Acting Score (0-5)**: How well the external behavior matches the character

### Evaluation Process

1. **Response Generation**: Generate character responses using various models
2. **Think-Act Separation**: Extract internal reasoning and external behavior
3. **AI Scoring**: Use AI judges to evaluate character consistency
4. **Analysis**: Aggregate and analyze results

## 🔬 Research Applications

This framework is designed for research in:

- **Character Consistency**: How well AI models maintain character consistency across different scenarios
- **Moral Reasoning**: How different models handle ethical dilemmas in character-specific contexts
- **Cross-Model Comparison**: Systematic comparison of different AI models' character role-playing abilities
- **Human-AI Interaction**: Understanding how AI characters might interact in multiversal scenarios

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{character_ai_framework,
  title={Character AI Research Framework},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/charactor_ai}
}
```
---

**Note**: This framework is designed for research purposes. Please ensure responsible use of AI models and respect for intellectual property rights when working with character data.# Beyond_One_World
# Beyond_One_World
