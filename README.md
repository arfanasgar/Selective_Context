# Selective Context: Prompt Compression for Efficient LLM-Based News Understanding

A Prompt Engineering course project investigating how phrase-level
context compression affects the quality and efficiency of LLM-based news
understanding across summarisation, question answering, multi-turn
response generation, and original-context reconstruction.

## Overview

Large Language Models (LLMs) can process lengthy documents and generate
informative outputs, but supplying large amounts of context can increase
input-token usage, computational requirements, and inference time.

This project implements a local phrase-level context compression
pipeline and evaluates its effects across four tasks:

1.  News summarisation
2.  Question answering
3.  Multi-turn response generation
4.  Original-context reconstruction

The pipeline uses locally hosted models for phrase-importance
estimation, task execution, and evaluation. It does not rely on paid LLM
APIs.

## Research Question

**How does phrase-level context compression affect the quality and
efficiency of LLM-based news understanding tasks?**

## Objectives

-   Implement a local phrase-level context compression pipeline using
    GPT-2 self-information.
-   Apply compression to BBC News article contexts.
-   Evaluate summarisation quality using ROUGE and BERTScore.
-   Investigate the effect of compression on factual question answering.
-   Extend evaluation to multi-turn response generation using article
    context and conversation history.
-   Assess whether original article content can be reconstructed from
    compressed contexts.
-   Measure input-token reduction and generation time where available.
-   Analyse the relationship between compression level, output quality,
    and computational efficiency.

## Dataset

The source corpus is the RealTimeData BBC News March 2023 dataset. The
local source dataset contains 1,892 article records and is not
distributed in this repository.

The experiments use the following subsets:

  Experiment                 Articles                    Examples / outputs
  ------------------------ ---------- -------------------------------------
  Summarisation                    29     174 summaries (29 × 6 conditions)
  Question answering               10         30 questions; 180 predictions
  Response generation              10   20 dialogue examples; 120 responses
  Context reconstruction           10                    60 reconstructions

The QA, response-generation, and reconstruction experiments use the same
aligned subset of 10 articles. QA reference answers were manually
reviewed but are not independently verified human annotations.
Response-generation examples are constructed from reviewed
question-answer material rather than naturally collected conversations.

## Methodology

### 1. Context Compression

The project estimates phrase importance using self-information scores
calculated locally with GPT-2. Lower-priority phrases are removed
according to the selected compression condition. The resulting
compressed context is passed to the model for the relevant downstream
task.

Six conditions are evaluated:

-   **0%** -- Original, uncompressed context
-   **20%**
-   **35%**
-   **50%**
-   **65%**
-   **80%**

The configured ratio represents the target phrase-removal level. Actual
token reduction is measured separately because phrases have different
token lengths.

### 2. Experimental Tasks

#### A. News Summarisation

The original and compressed article contexts are passed to:

`sshleifer/distilbart-cnn-6-6`

The model generates a summary under each compression condition. The
experiment evaluates 29 articles across six conditions, producing **174
summaries**. The `description` field of each article is used as the
reference summary.

#### B. Question Answering

The QA experiment investigates whether compressed news articles retain
enough information to answer factual questions.

Models:

-   Question generation: `valhalla/t5-base-qa-qg-hl`
-   Question answering: `deepset/roberta-base-squad2`

The experiment evaluates 10 articles, 30 questions (three per article),
six compression conditions, and 180 predictions. The QA model has a
maximum input length of 512 tokens; long contexts may therefore be
truncated, including in the uncompressed baseline.

#### C. Multi-turn Response Generation

The response-generation experiment evaluates 20 constructed dialogue
examples associated with 10 source articles. Each example contains
article context, previous question-answer turns as conversation history,
a new user question, and a reference response.

The article context is compressed under each condition, while the
conversation history and latest question are supplied to the
response-generation pipeline. The experiment produces 120 responses
across six conditions. Results are checkpointed after predictions so
interrupted runs can resume.

#### D. Original-Context Reconstruction

The reconstruction experiment investigates whether a model can reproduce
the original article from a compressed version of that article.

Model:

`google/flan-t5-base`

Ten articles are evaluated under all six conditions, producing **60
reconstructions**. The model input and generated output are each capped
at 512 tokens. This is therefore a constrained reconstruction
experiment, not an unrestricted test of recovering complete long-form
articles.

### 3. Evaluation Metrics

  -----------------------------------------------------------------------
  Metric                              Purpose
  ----------------------------------- -----------------------------------
  ROUGE-1                             Unigram overlap

  ROUGE-2                             Bigram overlap

  ROUGE-L                             Longest common subsequence overlap

  BERTScore F1                        Semantic similarity

  Exact Match                         Exact agreement with a reference
                                      answer

  Token-level F1                      Token overlap between predicted and
                                      reference answers

  Input tokens                        Length of the supplied context

  Actual token reduction (%)          Achieved reduction relative to
                                      original context

  Generation time                     Measured model inference time
  -----------------------------------------------------------------------

BERTScore uses `distilroberta-base`. Statistical comparisons use paired
Wilcoxon signed-rank tests with Holm correction where paired
observations are available.

## Experimental Results

### 1. News Summarisation

The experiment evaluates 29 articles across six conditions.

  ------------------------------------------------------------------------------
  Compression        ROUGE-1      ROUGE-2      ROUGE-L BERTScore F1   Mean token
                                                                       reduction
  ------------- ------------ ------------ ------------ ------------ ------------
  0% (Baseline)       0.2664       0.0973       0.1940       0.8955        0.00%

  20%                 0.2762       0.1116       0.2139       0.8971       18.18%

  35%                 0.2481       0.0837       0.1850       0.8913       29.81%

  50%                 0.2269       0.0631       0.1699       0.8842       44.11%

  65%                 0.2082       0.0488       0.1460       0.8791       61.36%

  80%                 0.1442       0.0222       0.1110       0.8668       79.20%
  ------------------------------------------------------------------------------

**Observations**

-   The 20% condition achieved the highest mean ROUGE and BERTScore
    values in this sample.
-   Mean input-token reduction at 20% was approximately 18.18%.
-   Summary quality generally declined as compression became more
    aggressive.
-   At 80%, average token reduction was approximately 79.20%,
    accompanied by the largest quality decline.

Paired statistical comparisons found no significant quality difference
from baseline at 20% and 35% after Holm correction. At 50%, the adjusted
BERTScore comparison showed a significant decline, while ROUGE
differences did not remain significant. At 65% and 80%, declines were
significant across the reported quality metrics.

Statistical output: `results/statistical_tests.csv`.

### 2. Question Answering

  Compression       Exact Match   Token F1
  --------------- ------------- ----------
  0% (Baseline)          63.33%     87.16%
  20%                    56.67%     83.56%
  35%                    40.00%     68.06%
  50%                    36.67%     57.49%
  65%                    36.67%     53.88%
  80%                    20.00%     32.38%

**Observations**

-   QA performance decreased as compression increased.
-   At 20% and 35%, neither metric showed a statistically significant
    difference from baseline after Holm correction.
-   Token F1 declines at 50% and 65% were significant after correction;
    Exact Match did not cross the adjusted threshold at those levels.
-   At 80%, both Exact Match and Token F1 declined significantly.

  Compression   Metric          Holm-adjusted p-value Significant
  ------------- ------------- ----------------------- -------------
  20%           Exact Match                    0.2176 No
  20%           Token F1                       0.2176 No
  35%           Exact Match                    0.0785 No
  35%           Token F1                       0.0785 No
  50%           Exact Match                    0.0685 No
  50%           Token F1                       0.0210 Yes
  65%           Exact Match                    0.0685 No
  65%           Token F1                       0.0114 Yes
  80%           Exact Match                    0.0028 Yes
  80%           Token F1                       0.0007 Yes

Significance threshold: p \< 0.05. The experiment contains 30
question-answer pairs from 10 articles, and the 512-token model input
limit constrains the available context.

QA visualisations: `results/Graphs/qa_graphs/`.

-   `qa_f1.png`
-   `qa_exact_match.png`

### 3. Multi-turn Response Generation

The response-generation experiment evaluates 20 dialogue examples across
six conditions, producing 120 responses.

  Compression       BERTScore F1   ROUGE-1   ROUGE-2   ROUGE-L
  --------------- -------------- --------- --------- ---------
  0% (Baseline)           0.9126    0.4754    0.3827    0.4714
  20%                     0.9097    0.4787    0.3378    0.4787
  35%                     0.9094    0.4779    0.3441    0.4779
  50%                     0.9032    0.4236    0.3049    0.4181
  65%                     0.9025    0.4374    0.3255    0.4374
  80%                     0.8890    0.3491    0.2111    0.3465

**Observations**

-   BERTScore remained relatively close to baseline at lower compression
    levels.
-   ROUGE-1 and ROUGE-L were slightly higher at 20% and 35% than at
    baseline.
-   Quality generally declined at higher compression levels, although
    the trend was not perfectly monotonic for every metric.
-   The 80% condition produced the lowest mean scores across the four
    reported quality metrics.

The measured average context-token reductions were:

  Target compression     Actual average context reduction
  -------------------- ----------------------------------
  0%                                                0.00%
  20%                                              20.91%
  35%                                              35.80%
  50%                                              50.68%
  65%                                              65.56%
  80%                                              80.30%

Mean generation times:

  Compression     Mean generation time
  ------------- ----------------------
  0%                            4.94 s
  20%                           3.99 s
  35%                           3.40 s
  50%                           2.88 s
  65%                           2.04 s
  80%                           1.74 s

The 20% runtime measurement was investigated and rerun; the corrected
measurement is used in the aggregate results. Runtime remains hardware-
and system-dependent.

Paired Wilcoxon tests were conducted for four metrics across five
compressed conditions (20 comparisons), followed by Holm correction. All
20 comparisons had adjusted p-values of 1.0000; none was statistically
significant at p \< 0.05. This does not establish equivalence or prove
that compression has no effect. The sample contains only 20 paired
dialogue examples, so the analysis is exploratory.

Response-generation outputs and graphs:
`results/response_generation_corrected/`.

### 4. Original-Context Reconstruction

  Compression       BERTScore F1   ROUGE-1   ROUGE-L
  --------------- -------------- --------- ---------
  0% (Baseline)           0.8600    0.0595    0.0520
  20%                     0.8417    0.0451    0.0410
  35%                     0.8428    0.0494    0.0464
  50%                     0.8411    0.0595    0.0477
  65%                     0.8503    0.0949    0.0754
  80%                     0.7545    0.1409    0.1198

**Observations**

-   BERTScore F1 remained around 0.84 through 65% compression and
    declined to 0.7545 at 80%.
-   ROUGE-1 and ROUGE-L increased at higher compression levels, showing
    a different trend from BERTScore.
-   No metric-by-condition comparison remained significant after Holm
    correction.
-   The disagreement between lexical and semantic metrics makes the
    outcome inconclusive.

The experiment uses 10 articles and a 512-token input/output limit. It
should not be interpreted as evidence that full original articles can be
reliably recovered from heavily compressed contexts.

### 5. Overall Findings

Across the four tasks:

-   Light compression produced the clearest quality-efficiency trade-off
    in summarisation. At 20%, input tokens fell by approximately 18.18%
    while mean summarisation metrics were maintained or slightly
    improved.
-   QA was more sensitive to aggressive compression, particularly at 50%
    and above for token F1.
-   Response-generation metrics were comparatively stable at lower
    compression levels, but the small sample did not yield statistically
    significant differences after correction.
-   Reconstruction results were inconclusive because semantic and
    lexical metrics showed different trends and the model was
    constrained by 512-token input/output limits.
-   Token reduction is a direct measure of shorter input context, but it
    does not by itself establish lower end-to-end latency, cost, or
    energy use.

## Technology Stack

-   **Programming language:** Python
-   **Deep learning:** PyTorch
-   **Model framework:** Hugging Face Transformers
-   **Phrase-importance scoring:** GPT-2
-   **Summarisation:** DistilBART CNN
-   **Question answering:** RoBERTa SQuAD2
-   **Question generation:** T5 QA Question Generation
-   **Response generation and reconstruction:** FLAN-T5 Base
-   **Semantic evaluation:** BERTScore / DistilRoBERTa
-   **Lexical evaluation:** ROUGE, BLEU, METEOR (reconstruction)
-   **QA evaluation:** Exact Match and token-level F1
-   **Statistical analysis:** SciPy, Statsmodels
-   **Data processing:** Pandas, NumPy
-   **NLP processing:** spaCy
-   **Visualisation:** Matplotlib

## Project Structure

``` text
Selective_Context/
│
├── Dataset/
│   └── articles.json                         # Local source dataset; not tracked
│
├── results/
│   ├── summerization/
│   │   └── ...                               # Summarisation outputs
│   ├── qa/
│   │   ├── qa_results.csv
│   │   ├── qa_aggregate_results.csv
│   │   └── qa_statistical_tests.csv
│   ├── response_generation_corrected/
│   │   ├── response_results.csv
│   │   ├── response_evaluated.csv
│   │   ├── response_aggregate_results.csv
│   │   ├── response_statistical_tests.csv
│   │   └── graphs/
│   ├── context_reconstruction/
│   │   ├── reconstruction_aggregate_results.csv
│   │   ├── reconstruction_statistical_tests.csv
│   │   └── graphs/
│   └── Graphs/
│       ├── summerization_graphs/
│       ├── qa_graphs/
│       └── context_reconstruction_graphs/
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
├── run_context_reconstruction.py
├── evaluate_context_reconstruction.py
├── analyse_context_reconstruction.py
├── statistical_context_reconstruction.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

This is a logical overview of the main project files and result
artifacts. Local datasets, generated caches, full article-text outputs,
and working-only files may be excluded from version control through
`.gitignore`. Exact output filenames should be checked against the
current repository contents.

## Installation and Setup

Clone the repository:

``` bash
git clone https://github.com/arfanasgar/Selective_Context.git
cd Selective_Context
```

Create and activate a virtual environment:

``` bash
python -m venv .venv
```

Windows CMD:

``` cmd
.venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Download the spaCy English language model:

``` bash
python -m spacy download en_core_web_sm
```

Place the prepared BBC News dataset at:

``` text
Dataset/articles.json
```

The dataset is not distributed with this repository. Obtain it from its
source and follow its applicable usage and redistribution terms.

## Running the Experiments

Run commands from the repository root. Model weights are downloaded from
Hugging Face on first use and cached locally. The current implementation
uses CPU inference, so execution time depends on hardware.

### A. Summarisation

``` bash
python run_experiment.py
python evaluate_bertscore.py
python analyse_results.py
```

### B. Question Answering

Generate the QA dataset:

``` bash
python generate_qa_dataset.py
```

Review the generated questions and reference answers before running the
experiment. The reviewed working dataset is kept locally and may be
excluded from Git.

Run QA, analyse, and test:

``` bash
python run_qa_experiment.py
python analyse_qa_results.py
python statistical_qa_tests.py
```

### C. Multi-turn Response Generation

Prepare the response dataset:

``` bash
python prepare_response_dataset.py --output results/response_generation_corrected/response_dataset.json
```

Run response generation:

``` bash
python run_response_generation.py --input results/response_generation_corrected/response_dataset.json --output results/response_generation_corrected/response_results.csv
```

Evaluate generated responses:

``` bash
python evaluate_response_generation.py --input results/response_generation_corrected/response_results.csv --output results/response_generation_corrected/response_evaluated.csv
```

Generate aggregate results and graphs:

``` bash
python analyse_response_generation.py --input results/response_generation_corrected/response_evaluated.csv --output-dir results/response_generation_corrected
```

Run statistical testing:

``` bash
python statistical_response_tests.py
```

The generation script checkpoints predictions so an interrupted run can
resume. The exact command-line arguments should be checked with the
current script's `--help` output if the local version differs.

### D. Original-Context Reconstruction

Place the prepared reconstruction dataset in:

``` text
results/context_reconstruction/context_reconstruction_dataset.json
```

Run reconstruction:

``` bash
python run_context_reconstruction.py
```

Evaluate the generated reconstructions:

``` bash
python evaluate_context_reconstruction.py
```

Generate aggregate results and graphs:

``` bash
python analyse_context_reconstruction.py
```

Run paired statistical tests:

``` bash
python statistical_context_reconstruction.py
```

The reconstruction scripts use FLAN-T5-base with 512-token input/output
limits. Full article-text datasets and detailed generated text outputs
are local working files and should not be committed if excluded by
`.gitignore`.

## Limitations

-   The summarisation experiment uses 29 articles; each of the other
    tasks uses 10 articles.
-   The `description` field is an editorial description and is not
    independently verified as a gold-standard summary.
-   QA reference answers are manually reviewed but are not independently
    verified human annotations.
-   Response-generation dialogues are constructed examples, not
    naturally collected conversations.
-   The response-generation experiment has only 20 paired dialogue
    examples; statistical power is limited.
-   The QA model has a maximum input length of 512 tokens, which can
    constrain both original and compressed contexts.
-   Reconstruction uses 512-token input and output limits, so complete
    long-form article recovery is not tested.
-   Configured phrase-removal ratios and actual token reductions are
    different quantities.
-   ROUGE, BERTScore, Exact Match, and token F1 do not fully measure
    factual faithfulness, completeness, usefulness, or conversational
    naturalness.
-   Generation time is hardware-dependent and may vary with system load.
-   Larger and more diverse datasets, independently annotated
    references, human evaluation, and controlled runtime measurements
    are needed to establish broader conclusions.
