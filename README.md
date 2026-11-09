# Selective Context for Efficient LLM-Based News Summarisation

A Prompt Engineering course project exploring how context compression affects the quality and computational efficiency of LLM-based news summarisation, question answering, and response generation.

## Overview

Large Language Models (LLMs) can process lengthy documents and generate informative outputs, but processing large amounts of context can increase computational requirements and inference time.

This project investigates the effect of **context compression using a phrase-level Selective Context approach** across three NLP tasks:

1. News summarisation
2. Question answering
3. Multi-turn response generation

The project uses locally hosted language models to perform context compression, generate outputs, and evaluate results without relying on paid LLM APIs.

## Research Question

**How does context compression using Selective Context affect the quality and efficiency of LLM-based language generation tasks?**

## Objectives

- Implement a local context compression pipeline using GPT-2 self-information.
- Apply phrase-level context compression to BBC News articles.
- Generate summaries using a consistent summarisation model.
- Evaluate summarisation quality using ROUGE and BERTScore.
- Investigate the effect of compression on factual question answering.
- Extend the evaluation to multi-turn response generation using article context and conversation history.
- Measure context-token reduction and generation time.
- Analyse the relationship between compression level, output quality, and computational efficiency.

## Methodology

The project evaluates three language-generation tasks using six experimental conditions.

### 1. Dataset Preparation

BBC News articles are used as the source dataset.

The experiments use the following subsets:

| Experiment | Articles | Examples |
|---|---:|---:|
| Summarisation | 29 | 29 articles |
| Question Answering | 10 | 30 questions |
| Response Generation | 10 | 20 dialogue examples |

The summarisation experiment uses the `description` field of each article as its reference summary.

The question-answering and response-generation experiments use manually reviewed question-answer pairs derived from the selected articles.

### 2. Context Compression

The project adapts the Selective Context approach to identify and remove less informative portions of text.

GPT-2 is used locally to calculate token-level self-information scores. These scores help identify informative text for retention during compression.

The experiments evaluate six configured compression conditions:

- 0% – Original, uncompressed context
- 20%
- 35%
- 50%
- 65%
- 80%

Actual token reduction is measured and reported separately from the target compression ratio.

### 3. Experimental Tasks

#### A. News Summarisation

The original and compressed article contexts are passed to:

`sshleifer/distilbart-cnn-6-6`

The model generates a summary under each compression condition.

The experiment evaluates 29 articles across six conditions, producing **174 summaries**.

#### B. Question Answering

The question-answering experiment investigates whether compressed news articles retain enough information to answer factual questions.

The experiment uses:

`deepset/roberta-base-squad2`

The model receives a question and its corresponding article context and predicts an extractive answer.

The experiment evaluates:

- 10 articles
- 30 manually reviewed questions
- 6 compression conditions
- 180 QA predictions

#### C. Multi-turn Response Generation

The response-generation experiment extends the project to conversational responses.

Dialogue examples are constructed from the reviewed BBC question-answer pairs.

Each example includes:

- The corresponding BBC article context.
- Previous question-answer turns as conversation history.
- A new user question.
- A reviewed reference response.

The experiment uses:

- Context compressor: GPT-2
- Response-generation model: `google/flan-t5-base`

The full article context is supplied alongside the conversation history. Compression is applied to the article context, and the model generates a response to the latest question.

The experiment evaluates:

- 10 articles
- 20 constructed dialogue examples
- 6 compression conditions
- 120 generated responses

Results are checkpointed after every prediction, allowing interrupted experiments to resume without regenerating completed examples.

### 4. Evaluation Metrics

| Metric | Purpose |
|---|---|
| ROUGE-1 | Measures unigram overlap |
| ROUGE-2 | Measures bigram overlap |
| ROUGE-L | Measures longest common subsequence overlap |
| BERTScore F1 | Measures semantic similarity |
| Exact Match | Measures exact agreement with a reference answer |
| Token-level F1 | Measures token overlap between predicted and reference answers |
| Input Tokens | Measures the length of the supplied context |
| Actual Token Reduction (%) | Measures the achieved context compression |
| Generation Time | Measures model inference time |

BERTScore uses `distilroberta-base`.

## Experimental Results

### 1. Summarisation Results

The summarisation experiment evaluates 29 articles across six conditions, producing 174 summaries.

| Compression Ratio | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| 0% (Baseline) | 0.2664 | 0.0973 | 0.1940 | 0.8955 |
| 20% | 0.2762 | 0.1116 | 0.2139 | 0.8971 |
| 35% | 0.2481 | 0.0837 | 0.1850 | 0.8913 |
| 50% | 0.2269 | 0.0631 | 0.1699 | 0.8842 |
| 65% | 0.2082 | 0.0488 | 0.1460 | 0.8791 |
| 80% | 0.1442 | 0.0222 | 0.1110 | 0.8668 |

Main observations:

- The 20% compression condition achieved the highest average ROUGE and BERTScore values in the evaluated sample.
- Summary quality generally declined as compression became more aggressive.
- The 80% compression condition achieved approximately 79% average input-token reduction.
- Higher compression levels resulted in measurable losses in summary quality.

### 2. Summarisation Statistical Analysis

Paired Wilcoxon signed-rank tests were used to compare compressed conditions against the original baseline.

Holm correction was applied to account for multiple comparisons.

The statistical results are available in:

- `results/statistical_tests.csv`

### 3. Question Answering Results

The QA experiment evaluates 30 questions across six compression conditions.

| Compression Ratio | Exact Match | Token F1 |
|---|---:|---:|
| 0% (Baseline) | 63.33% | 87.16% |
| 20% | 56.67% | 83.56% |
| 35% | 40.00% | 68.06% |
| 50% | 36.67% | 57.49% |
| 65% | 36.67% | 53.88% |
| 80% | 20.00% | 32.38% |

Main observations:

- At 20% compression, QA performance remained relatively close to baseline.
- At 35% compression, both Exact Match and Token F1 declined.
- At 50% and 65% compression, Token F1 showed statistically significant declines after Holm correction.
- At 80% compression, both Exact Match and Token F1 showed statistically significant declines.
- Aggressive compression generally resulted in reduced answer quality.

### 4. Question Answering Statistical Analysis

Paired Wilcoxon signed-rank tests were conducted against the 0% baseline.

Holm correction was applied across the 10 comparisons.

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

QA visualisations are available in:

`results/Graphs/qa_graphs/`

- `qa_f1.png`
- `qa_exact_match.png`

### 5. Response Generation Results

The corrected response-generation experiment evaluates 20 dialogue examples across six compression conditions, producing 120 evaluated responses.

The full article context is included in each dialogue prompt, alongside previous conversation turns and the latest question.

#### Average Evaluation Results

| Compression Ratio | BERTScore F1 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| 0% (Baseline) | 0.9126 | 0.4754 | 0.3827 | 0.4714 |
| 20% | 0.9097 | 0.4787 | 0.3378 | 0.4787 |
| 35% | 0.9094 | 0.4779 | 0.3441 | 0.4779 |
| 50% | 0.9032 | 0.4236 | 0.3049 | 0.4181 |
| 65% | 0.9025 | 0.4374 | 0.3255 | 0.4374 |
| 80% | 0.8890 | 0.3491 | 0.2111 | 0.3465 |

Main observations:

- BERTScore remained relatively close to the baseline at lower compression levels.
- ROUGE-1 and ROUGE-L were slightly higher at 20% and 35% compression than at baseline.
- Quality generally declined at higher compression levels.
- The 80% condition produced the lowest average scores across all four quality metrics.
- The observed relationship is not perfectly monotonic across every metric and condition.

#### Actual Context Compression

The following table reports measured average GPT-2 token reduction in the corrected experiment.

| Target Compression | Actual Average Context Reduction |
|---|---:|
| 0% | 0.00% |
| 20% | 20.91% |
| 35% | 35.80% |
| 50% | 50.68% |
| 65% | 65.56% |
| 80% | 80.30% |

These measurements show that the corrected compression pipeline achieved reductions close to the configured targets.

#### Response Generation Runtime

| Compression Ratio | Mean Generation Time |
|---|---:|
| 0% | 4.94 seconds |
| 20% | 3.99 seconds |
| 35% | 3.40 seconds |
| 50% | 2.88 seconds |
| 65% | 2.04 seconds |
| 80% | 1.74 seconds |

The anomalous 20% runtime measurement was investigated and rerun. The original measurement was preserved in a backup CSV, and the corrected measurement was used for the final aggregate results.

Generation time is hardware-dependent and should be interpreted cautiously.

#### Response Generation Statistical Analysis

Paired Wilcoxon signed-rank tests were conducted for each compression condition against the 0% baseline.

Four metrics were evaluated:

- BERTScore F1
- ROUGE-1
- ROUGE-2
- ROUGE-L

Holm correction was applied across all 20 comparisons (four metrics × five compression conditions).

**Statistical results:**

All 20 comparisons had Holm-adjusted p-values of 1.0000.

| Metric | Compression | Holm-adjusted p-value | Significant |
|---|---:|---:|---|
| BERTScore F1 | 20% | 1.0000 | No |
| BERTScore F1 | 35% | 1.0000 | No |
| BERTScore F1 | 50% | 1.0000 | No |
| BERTScore F1 | 65% | 1.0000 | No |
| BERTScore F1 | 80% | 1.0000 | No |
| ROUGE-1 | 20% | 1.0000 | No |
| ROUGE-1 | 35% | 1.0000 | No |
| ROUGE-1 | 50% | 1.0000 | No |
| ROUGE-1 | 65% | 1.0000 | No |
| ROUGE-1 | 80% | 1.0000 | No |
| ROUGE-2 | 20% | 1.0000 | No |
| ROUGE-2 | 35% | 1.0000 | No |
| ROUGE-2 | 50% | 1.0000 | No |
| ROUGE-2 | 65% | 1.0000 | No |
| ROUGE-2 | 80% | 1.0000 | No |
| ROUGE-L | 20% | 1.0000 | No |
| ROUGE-L | 35% | 1.0000 | No |
| ROUGE-L | 50% | 1.0000 | No |
| ROUGE-L | 65% | 1.0000 | No |
| ROUGE-L | 80% | 1.0000 | No |

Significance threshold: p < 0.05.

**Interpretation:**

Although average response-generation scores generally declined at higher compression levels, none of the differences from the baseline reached statistical significance after Holm correction in this sample.

This does not establish that compression has no effect. The experiment uses only 20 paired dialogue examples, so its statistical power is limited. The findings should be treated as exploratory.

Response-generation results and graphs are available in:

`results/response_generation_corrected/`

The statistical test outputs are:

- `response_statistical_tests.csv`
- `response_statistical_tests_summary.txt`

The graphs are available in:

`results/response_generation_corrected/graphs/`

### 6. Overall Experimental Observations

Across the three tasks, the experiments suggest that:

- Moderate compression can preserve useful information for several language-generation tasks.
- Aggressive compression generally reduces output quality.
- The relationship between compression and quality varies across tasks and evaluation metrics.
- Summarisation and QA show evidence of quality degradation at higher compression levels.
- Response generation shows lower average scores at aggressive compression, but the observed differences were not statistically significant after Holm correction.
- Context reduction can reduce generation time, although runtime measurements depend on hardware and system conditions.

## Technology Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Transformers:** Hugging Face Transformers
- **Context Compression:** GPT-2
- **Summarisation:** DistilBART CNN
- **Question Answering:** RoBERTa SQuAD2
- **Response Generation:** FLAN-T5 Base
- **Question Generation:** T5 QA Question Generation
- **Semantic Evaluation:** BERTScore / DistilRoBERTa
- **Lexical Evaluation:** ROUGE
- **QA Evaluation:** Exact Match and Token-level F1
- **Statistical Analysis:** SciPy, Statsmodels
- **Data Processing:** Pandas, NumPy
- **NLP Processing:** spaCy
- **Visualisation:** Matplotlib

## Project Structure

```text
Selective_Context/
│
├── Dataset/
│   └── articles.json                 # Local dataset; not tracked by Git
│
├── results/
│   ├── final_metrics.csv
│   ├── aggregate_results.csv
│   ├── statistical_tests.csv
│   │
│   ├── Graphs/
│   │   ├── summerization_graphs/
│   │   │   ├── rouge_quality.png
│   │   │   ├── bertscore.png
│   │   │   ├── token_reduction.png
│   │   │   └── runtime.png
│   │   │
│   │   └── qa_graphs/
│   │       ├── qa_f1.png
│   │       └── qa_exact_match.png
│   │
│   ├── qa/
│   │   ├── qa_results.csv
│   │   ├── qa_aggregate_results.csv
│   │   ├── qa_statistical_tests.csv
│   │   └── qa_dataset.json            # Local working data
│   │
│   └── response_generation_corrected/
│       ├── response_dataset.json
│       ├── response_results.csv
│       ├── response_results_backup.csv
│       ├── response_evaluated.csv
│       ├── response_aggregate_results.csv
│       ├── response_statistical_tests.csv
│       ├── response_statistical_tests_summary.txt
│       └── graphs/
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
├── prepare_response_dataset.py
├── run_response_generation.py
├── evaluate_response_generation.py
├── analyse_response_generation.py
├── statistical_response_tests.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

The project structure lists the main experiment outputs. Intermediate files and local datasets may be excluded from version control through `.gitignore`.

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

Install dependencies:

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

Run all commands from the repository root.

### A. Summarisation Experiment

**Step 1 – Run the summarisation experiment**

```bash
python run_experiment.py
```

**Step 2 – Evaluate using BERTScore**

```bash
python evaluate_bertscore.py
```

**Step 3 – Analyse results**

```bash
python analyse_results.py
```

### B. Question Answering Experiment

**Step 1 – Generate the QA dataset**

```bash
python generate_qa_dataset.py
```

Review the questions and reference answers in `results/qa/qa_dataset.json` before proceeding.

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

### C. Response Generation Experiment

**Step 1 – Prepare the response-generation dataset**

```bash
python prepare_response_dataset.py --output results/response_generation_corrected/response_dataset.json
```

**Step 2 – Run corrected response generation**

```bash
python run_response_generation.py --input results/response_generation_corrected/response_dataset.json --output results/response_generation_corrected/response_results.csv
```

The script saves a checkpoint after each prediction and resumes automatically if interrupted.

**Step 3 – Evaluate generated responses**

```bash
python evaluate_response_generation.py --input results/response_generation_corrected/response_results.csv --output results/response_generation_corrected/response_evaluated.csv
```

**Step 4 – Generate aggregate results and graphs**

```bash
python analyse_response_generation.py --input results/response_generation_corrected/response_evaluated.csv --output-dir results/response_generation_corrected
```

**Step 5 – Perform statistical significance analysis**

```bash
python statistical_response_tests.py
```

The response-generation statistical script performs paired Wilcoxon tests and applies Holm correction across 20 comparisons.

The models are downloaded from Hugging Face on their first use and cached locally. The current implementation uses CPU inference, so execution time depends on the hardware.

## Limitations

- The summarisation experiment uses a sample of 29 usable BBC News articles.
- The QA and response-generation experiments use smaller subsets of 10 articles.
- The summarisation `description` field is an editorial description and is not independently verified as a gold-standard summary.
- QA reference answers are manually reviewed but are not independently verified human annotations.
- Response-generation dialogues are constructed from reviewed QA pairs, not naturally collected conversations.
- Response-generation reference answers are short factual answers rather than independently authored conversational responses.
- The QA model uses a maximum input length of 512 tokens, limiting the context available for answering questions, including in the baseline condition.
- Configured compression ratios and actual token reductions may differ slightly.
- BERTScore and ROUGE do not fully capture factual faithfulness, helpfulness, or conversational naturalness.
- The response-generation statistical analysis has only 20 paired examples and should be interpreted as exploratory.
- Generation time is hardware-dependent and can vary due to system load.
- Further evaluation on larger and more diverse datasets is required to establish generalisability.

