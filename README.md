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
- Extend the evaluation to question answering to investigate how compression affects information retrieval from news articles.

## Methodology

The project follows a pipeline consisting of five major stages:

### 1. Dataset Preparation

BBC News articles are used as the input dataset.

The summarisation experiment uses 29 usable articles from a locally prepared dataset.

The additional question-answering experiment uses a subset of 10 articles from the same dataset.

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

The completed summarisation experiment evaluates 29 articles across six conditions, producing 174 generated summaries.

The main observations are:

- The 20% compression condition achieved the highest average ROUGE and BERTScore values in the evaluated sample.
- Summary quality generally declined as compression became more aggressive.
- The 80% compression condition achieved approximately 79% average input-token reduction.
- Higher compression levels resulted in measurable losses in summary quality.
- Generation-time measurements varied across conditions and require cautious interpretation.

These observations highlight the trade-off between context compression, information preservation, and summarisation efficiency.

Detailed results and visualisations are available in the `results/` directory.

## Additional Experiment: Question Answering

To extend the project beyond summarisation, a question-answering experiment was conducted to investigate whether compressed news articles retain sufficient information for answering factual questions.

The experiment reuses the existing Selective Context compression pipeline while keeping the original summarisation experiment files and results unchanged.

### 1. Question Dataset Preparation

- A subset of 10 BBC News articles was selected.
- Three factual questions per article were prepared, giving 30 question-answer pairs.
- Questions and reference answers were generated using a local question-generation model and reviewed before evaluation.

### 2. Question Answering Model

The experiment uses:

`deepset/roberta-base-squad2`

This extractive question-answering model receives a question and its corresponding article context and predicts an answer from that context.

### 3. Experimental Conditions

The same six conditions were evaluated:

- 0% – Original article context
- 20% compression
- 35% compression
- 50% compression
- 65% compression
- 80% compression

The experiment produced:

- 10 articles
- 30 questions
- 6 compression conditions
- 180 QA predictions

### 4. Evaluation Metrics

| Metric | Purpose |
|---|---|
| Exact Match (EM) | Measures whether the predicted answer exactly matches the reference answer after normalisation |
| Token-level F1 | Measures token overlap between predicted and reference answers |
| Input Tokens | Measures the length of the context supplied to the QA model |
| QA Inference Time | Measures the time taken to generate an answer |

### 5. Question Answering Results

The following table presents the average results across the 30 questions.

| Compression Ratio | Exact Match | Token F1 |
|---|---:|---:|
| 0% (Baseline) | 63.33% | 87.16% |
| 20% | 56.67% | 83.56% |
| 35% | 40.00% | 68.06% |
| 50% | 36.67% | 57.49% |
| 65% | 36.67% | 53.88% |
| 80% | 20.00% | 32.38% |

The main observations are:

- At 20% compression, QA performance remained relatively close to the baseline.
- At 35% compression, both Exact Match and F1 declined.
- At 50% and 65% compression, token-level F1 showed statistically significant declines.
- At 80% compression, both Exact Match and F1 showed statistically significant declines.
- Higher compression levels generally resulted in reduced answer quality.

### 6. Statistical Testing

Paired Wilcoxon signed-rank tests were conducted to compare each compressed condition against the baseline.

Holm correction was applied across the 10 comparisons (five compression levels and two evaluation metrics).

| Compression | Metric | Holm-adjusted p-value | Significant |
|---|---|---:|---|
| 20% | Exact Match | 0.2176 | No |
| 20% | Token F1 | 0.2176 | No |
| 35% | Exact Match | 0.0785 | No |
| 35% | Token F1 | 0.0785 | No |
| 50% | Exact Match | 0.0685 | No |
| 50% | Token F1 | 0.0210 | Yes |
| 65% | Exact Match | 0.0685 | No |
| 65% | Token F1 | 0.0114 | Yes |
| 80% | Exact Match | 0.0028 | Yes |
| 80% | Token F1 | 0.0007 | Yes |

Significance threshold: p < 0.05.

### 7. Question Answering Visualisations

The QA experiment generates the following graphs:

- `qa_f1.png` – Question-answering F1 against compression ratio.
- `qa_exact_match.png` – Exact Match against compression ratio.

These graphs are available in `results/qa/graphs/`.

The results indicate that moderate compression can preserve a substantial amount of information needed for question answering, while aggressive compression can significantly affect answer quality.

## Technology Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Transformers:** Hugging Face Transformers
- **Context Compression:** GPT-2
- **Summarisation:** DistilBART CNN
- **Question Answering:** RoBERTa SQuAD2
- **Question Generation:** T5 QA Question Generation
- **Semantic Evaluation:** BERTScore / DistilRoBERTa
- **Lexical Evaluation:** ROUGE
- **QA Evaluation:** Exact Match and Token-level F1
- **Statistical Analysis:** SciPy
- **Data Processing:** Pandas, NumPy
- **NLP Processing:** spaCy
- **Visualisation:** Matplotlib

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
│   ├── graphs/
│   │   ├── rouge_quality.png
│   │   ├── bertscore.png
│   │   ├── token_reduction.png
│   │   └── runtime.png
│   │
│   └── qa/
│       ├── qa_dataset.json
│       ├── qa_results.csv
│       ├── qa_aggregate_results.csv
│       ├── qa_statistical_tests.csv
│       └── graphs/
│           ├── qa_f1.png
│           └── qa_exact_match.png
│
├── context_manager.py
├── run_experiment.py
├── evaluate_bertscore.py
├── analyse_results.py
│
├── generate_qa_dataset.py
├── run_qa_experiment.py
├── analyse_qa_results.py
├── statistical_qa_tests.py
│
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

### Summarisation Experiment

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

### Question Answering Experiment

**Step 1 – Generate the QA dataset**

```bash
python generate_qa_dataset.py
```

Review the generated questions and reference answers in `results/qa/qa_dataset.json` before proceeding.

**Step 2 – Run the QA experiment**

```bash
python run_qa_experiment.py
```

**Step 3 – Analyse QA results**

```bash
python analyse_qa_results.py
```

**Step 4 – Perform QA statistical testing**

```bash
python statistical_qa_tests.py
```

The models are downloaded from Hugging Face on their first use and cached locally. The current implementation uses CPU inference, so execution time depends on the hardware.

## Limitations

- The summarisation experiment uses a sample of 29 usable BBC News articles.
- The question-answering experiment uses a smaller subset of 10 articles and 30 questions.
- The dataset's `description` field is used as the summarisation evaluation reference. It is an editorial description and is not independently verified as a gold-standard summary.
- QA reference answers are generated and reviewed, but are not independently verified human annotations.
- Configured phrase-removal ratios do not correspond exactly to actual token reduction.
- The QA model uses a maximum input length of 512 tokens, which limits the context available for answering questions, including in the baseline condition.
- Generation time is hardware-dependent and can vary due to system load.
- QA inference-time measurements can also vary between experimental conditions.
- BERTScore results depend on the selected evaluation model.
- Further evaluation on larger and more diverse datasets is required to establish generalisability.

