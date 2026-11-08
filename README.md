# Selective Context for Efficient LLM-Based News Summarisation

A Prompt Engineering course project exploring how context compression can improve the efficiency of Large Language Model (LLM)-based news summarisation while preserving summary quality.

## Overview

Large Language Models can generate informative summaries of lengthy news articles, but processing large amounts of input context can increase computational requirements and inference time.

This project investigates the effect of **context compression on LLM-based news summarisation** using a phrase-level Selective Context approach.

The objective is to understand the relationship between the amount of context retained, summarisation quality, and computational efficiency.

The project uses locally hosted language models to perform context compression, generate summaries, and evaluate the results without relying on paid LLM APIs.

## Research Question

**How does context compression using Selective Context affect the quality and efficiency of LLM-based news summarisation?**

## Objectives

- Implement a local context compression pipeline using GPT-2 self-information.
- Apply phrase-level context compression to BBC News articles.
- Generate summaries using a consistent LLM summarisation model.
- Evaluate summary quality using ROUGE and BERTScore.
- Measure input-token reduction and summarisation generation time.
- Analyse the trade-off between compression level, summary quality, and computational efficiency.

## Methodology

The project follows a pipeline consisting of five major stages:

### 1. Dataset Preparation

BBC News articles are used as the input dataset.

The experiment uses 29 usable articles from a locally prepared dataset.

### 2. Context Compression

The Selective Context approach is used to identify and remove less informative phrases.

GPT-2 is used locally to calculate token-level self-information scores. These scores help determine which phrases are retained or removed during compression.

The experiment evaluates the following configured compression levels:

- 0% – Original, uncompressed context
- 20%
- 35%
- 50%
- 65%
- 80%

### 3. Summary Generation

The compressed and original article contexts are passed to the same summarisation model:

`sshleifer/distilbart-cnn-6-6`

This ensures that the summarisation model remains consistent across experimental conditions.

### 4. Evaluation

Summary quality and computational efficiency are evaluated using:

| Metric | Purpose |
|---|---|
| ROUGE-1 | Unigram overlap |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Longest common subsequence overlap |
| BERTScore | Semantic similarity |
| Input Tokens | Measures context length |
| Token Reduction (%) | Measures input compression |
| Generation Time | Measures summarisation inference time |

BERTScore uses `distilroberta-base`.

### 5. Statistical Analysis

Paired Wilcoxon signed-rank tests are used to compare compressed conditions against the original baseline.

Holm correction is applied to account for multiple comparisons.

## Experimental Results

The completed experiment evaluates 29 articles across six conditions, producing 174 generated summaries.

The main observations are:

- The 20% compression condition achieved the highest average ROUGE and BERTScore values in the evaluated sample.
- Summary quality generally declined as compression became more aggressive.
- The 80% compression condition achieved approximately 79% average input-token reduction.
- Higher compression levels resulted in measurable losses in summary quality.
- Generation-time measurements varied across conditions and require cautious interpretation.

These observations highlight the trade-off between context compression, information preservation, and summarisation efficiency.

Detailed results and visualisations are available in the `results/` directory.

## Technology Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Transformers:** Hugging Face Transformers
- **Context Compression:** GPT-2
- **Summarisation:** DistilBART CNN
- **Semantic Evaluation:** BERTScore / DistilRoBERTa
- **Lexical Evaluation:** ROUGE
- **Statistical Analysis:** SciPy
- **Data Processing:** Pandas, NumPy
- **NLP Processing:** spaCy

## Project Structure

```text
Selective_Context/
│
├── Dataset/
│   └── .gitkeep
│
├── results/
│   ├── final_metrics.csv
│   ├── aggregate_results.csv
│   ├── statistical_tests.csv
│   └── graphs/
│       ├── rouge_quality.png
│       ├── bertscore.png
│       ├── token_reduction.png
│       └── runtime.png
│
├── context_manager.py
├── run_experiment.py
├── evaluate_bertscore.py
├── analyse_results.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/arfanasgar/Selective_Context.git
cd Selective_Context
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```cmd
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

Place the prepared BBC News dataset at:

```text
Dataset/articles.json
```

The dataset is not distributed with this repository. Obtain it from its source and follow its applicable usage and redistribution terms.

## Running the Project

Run the commands from the repository root.

**Step 1 – Run the summarisation experiment**

```bash
python run_experiment.py
```

**Step 2 – Evaluate using BERTScore**

```bash
python evaluate_bertscore.py
```

**Step 3 – Perform statistical analysis**

```bash
python analyse_results.py
```

The models are downloaded from Hugging Face on their first use and cached locally. The current implementation uses CPU inference, so execution time depends on the hardware.

## Limitations

- The experiment uses a sample of 29 usable BBC News articles.
- The dataset's `description` field is used as the evaluation reference. It is an editorial description and is not independently verified as a gold-standard summary.
- Configured phrase-removal ratios do not correspond exactly to actual token reduction.
- Generation time is hardware-dependent and can vary due to system load.
- BERTScore results depend on the selected evaluation model.
- Further evaluation on larger and more diverse datasets is required to establish generalisability.
