# Autograder Agent

An AI-powered automatic grading system for student submissions. Supports Python scripts, Jupyter notebooks, ZIP files, and PowerPoint presentations.

## Features

- **Multiple File Formats**: Supports `.py`, `.ipynb`, `.zip`, `.pptx`
- **Flexible Rubrics**: Parse rubrics from text with automatic point extraction
- **Dataset Awareness**: Upload datasets to provide context for grading
- **Batch Grading**: Grade multiple submissions at once
- **Detailed Feedback**: Get criterion-by-criterion feedback
- **Export Results**: Download grades as CSV or Excel
- **Visualizations**: Score distribution and criterion breakdown charts

## Supported LLM Providers

| Provider | Type | Best For |
|----------|------|----------|
| **Ollama** | Local | Privacy, no cost |
| **OpenAI** | Cloud | GPT-4o accuracy |
| **Anthropic** | Cloud | Claude models |

## How to Use

### Step 1: Setup Grading

1. **Problem Statement**: Enter or upload the assignment description
2. **Dataset** (optional): Upload the dataset students were given
3. **Rubric**: Enter grading criteria in the format:
   ```
   Criterion Name (points): Description
   ```

### Step 2: Upload Submissions

Upload student submissions in any supported format:
- Individual `.py` or `.ipynb` files
- `.zip` files containing multiple files
- `.pptx` presentations

### Step 3: Grade

Click "Start Grading" to evaluate all submissions.

### Step 4: Review & Export

- View the score summary DataFrame
- Check detailed feedback per submission
- Download results as CSV or Excel

## Rubric Format Examples

```
# Simple format
Data Loading (10): Correctly loads the dataset
EDA (15): Performs exploratory data analysis
Model (20): Implements the required model
Results (10): Presents results clearly

# Alternative format
- Data Loading - 10 points: Load and inspect data
- Feature Engineering (15 pts): Create relevant features
- Model Training: Train a classification model (20)
```

## Project Structure

```
Autograder/
├── evaluators/
│   ├── __init__.py
│   └── grading_engine.py    # Core grading logic
├── processors/
│   ├── __init__.py
│   └── file_processor.py    # File parsing
├── utils/
│   ├── __init__.py
│   └── llm_client.py        # LLM API client
├── app.py                   # Streamlit interface
├── config.py                # Configuration
├── requirements.txt
└── README.md
```

## Output Format

The grading results DataFrame includes:

| Filename | Criterion 1 | Criterion 2 | ... | Total | Max Points | Percentage |
|----------|-------------|-------------|-----|-------|------------|------------|
| student1.ipynb | 8 | 12 | ... | 78 | 100 | 78.0% |
| student2.zip | 10 | 14 | ... | 92 | 100 | 92.0% |

## Tips for Best Results

1. **Clear Problem Statement**: Be specific about requirements
2. **Detailed Rubric**: Include what you're looking for in each criterion
3. **Provide Dataset**: Helps the grader understand expected analysis
4. **Use Larger Models**: For more accurate grading, use GPT-4o or Llama 70B

## Requirements

- Python 3.8+
- Ollama (for local models) or API keys
- See `requirements.txt` for dependencies

