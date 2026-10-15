# DSA4213 Assignment 1

**Student ID:** A0211187L  
**Main notebook:** `Assignment_1_A0211187L_full.ipynb`
**PDF file:** `Assignment_1_A0211187L_full.pdf`

This project uses AG News to compare next-token language models, examine input-embedding choices, and evaluate learned representations for a topic classification task.

## 1. Project contents

```text
Assignment_1/
├── README.md
├── Assignment_1_A0211187L_full.ipynb
├── Assignment_1_A0211187L_full.html
├── Assignment_1_A0211187L_full.pdf
├── DSA4213_Assignment_1_Report_A0211187L.pdf
├── dsa4213_text.py
├── data/
│   ├── ag_news_experiment/
│   ├── ag_news_manifest.json
│   ├── ag_news_tokenized/
│   ├── vocabulary.json
│   ├── preprocessing_manifest.json
│   └── pretrained_embeddings/
└── results/
    ├── ngram/
    ├── rnn/
    ├── lstm/
    ├── transformer/
    ├── part1_comparison.csv
    ├── part2/
    └── part3/
```

`dsa4213_text.py` is a self-made function package provides shared cleaning, tokenization, vocabulary construction, encoding, and decoding functions. Keep it beside the notebook so the project-specific module can be imported.

## 2. Environment and dependencies

Recorded environment:

- Python 3.13.5, using Anaconda.
- macOS on Apple Silicon (`arm64`).
- PyTorch 2.9.1.
- Neural-model training and feature extraction on MPS.
- N-gram counting and scikit-learn classifiers on CPU.

Recorded package versions:

| Package | Version |
|---|---|
| torch | 2.9.1 |
| datasets | 4.2.0 |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| scipy | 1.15.3 |
| scikit-learn | 1.6.1 |
| joblib | 1.4.2 |
| jupyterlab | 4.3.4 |
| ipykernel | 6.29.5 |

Example setup on macOS, with Python 3.13 installed:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install \
  torch==2.9.1 datasets==4.2.0 numpy==2.1.3 \
  pandas==2.2.3 matplotlib==3.10.0 scipy==1.15.3 \
  scikit-learn==1.6.1 joblib==1.4.2 \
  jupyterlab==4.3.4 ipykernel==6.29.5
python -m ipykernel install --user \
  --name dsa4213-a1 \
  --display-name "DSA4213 Assignment 1"
```

## 3. Data access and preprocessing

### AG News

English corpus dataset: [AG News on Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news).

The notebook downloads the dataset using:

```python
load_dataset("fancyzhx/ag_news", cache_dir=str(DATA_DIR / "hf_cache"))
```

The experiment uses four topics: World, Sports, Business, and Sci/Tech.

| Split | News items | Items per topic |
|---|---:|---:|
| Training | 20,000 | 5,000 |
| Validation | 2,000 | 500 |
| Test | 7,600 | 1,900 |

Training and validation are sampled from the original training split using seed **2026**. Before sampling, the notebook removes empty items, normalized exact duplicates, and training candidates matching official test text. The complete official test split is retained.

The overlap check ignores case, HTML escapes, and whitespace differences. It does not detect every near-duplicate or related news story. Saved manifests record source indices, dataset fingerprints, split sizes, and sampling settings.

The same splits are used throughout the project. Topic labels are used for balanced sampling and the Part III classification task, but are not inputs to the language models.

### Text processing

The helper module applies Unicode normalization, HTML cleanup, typography normalization, lowercasing, and word/punctuation tokenization.

The vocabulary is fitted on training text only:

- Maximum size: 10,000 entries, including special tokens.
- Minimum ordinary-token frequency: 2.
- Special tokens: `<pad>`, `<unk>`, `<bos>`, and `<eos>`.
- Out-of-vocabulary tokens map to `<unk>`.

Each news item is encoded separately with beginning and end markers. Sequences are not truncated or joined across news items. Padding is added only when forming batches.

### Public embeddings

Part II and III uses **GloVe 6B, 100-dimensional vectors**, trained on Wikipedia 2014 and Gigaword 5.

The notebook downloads the approximately 134.3 MB compressed Gensim distribution from this [release](https://github.com/piskvorky/gensim-data/releases/tag/glove-wiki-gigaword-100). It verifies the MD5 checksum before using the file:

```text
40ec481866001177b8cd4cb0df92924f
```

Matched vectors retain their original values. Missing ordinary entries and `<unk>`, `<bos>`, and `<eos>` receive seeded standard-normal vectors; `<pad>` is zero. The complete input-embedding table remains frozen during the corresponding RNN training.

## 4. Experimental configurations

### Part I: Language-model comparison

- **N-gram:** interpolated unigram, bigram, and trigram probabilities; unigram smoothing with alpha 0.1. Validation selects interpolation weights; the recorded choice is `(0.2, 0.4, 0.4)`.
- **RNN:** 100-dimensional embeddings, one forward recurrent layer, 128 hidden units.
- **LSTM:** 100-dimensional embeddings, one forward recurrent layer, 128 hidden units.
- **Transformer:** 100-dimensional embeddings, two layers, four attention heads, feedforward dimension 256, sinusoidal positions, causal attention, and dropout 0.

All neural models start with random trainable embeddings and use five epochs, batch size 32, Adam with learning rate 0.001, and gradient clipping at norm 1.0. Checkpoints are selected using validation cross-entropy.

Evaluation averages next-token loss over non-padding targets, including `<unk>` and `<eos>`. Cross-entropy uses natural logarithms; perplexity is its exponential.

Generation uses the same three prompts, top-50 sampling, temperature 1.0, and at most 40 new tokens. Generation stops earlier if `<eos>` is sampled.

Inference timing uses 100 shared validation prefixes of at most 64 tokens, a warm-up pass, and the median of three timed repetitions. Each request produces a full next-token distribution. Neural timings synchronize the device, with inputs prepared beforehand. Complete prefixes are recomputed without caching.

Training timings exclude validation and checkpoint copying. Stored model sizes measure serialized files rather than runtime memory.

### Part II: Embedding comparison

The RNN architecture and language-model training settings remain unchanged:

1. Reuse the Part I random/trainable baseline.
2. Train a fresh RNN with locally trained, frozen CBOW embeddings.
3. Train a fresh RNN with public, frozen GloVe embeddings.

CBOW uses 100-dimensional embeddings, a context window of two tokens on each side, three epochs, batch size 256, Adam with learning rate 0.001, and gradient clipping at 1.0. Contexts stay within individual news items.

CBOW training cost is recorded separately from RNN training cost. Public embedding preparation is measured locally; original external pretraining cost is unknown.

### Part III: Topic classification

Compare representations from the random/trainable RNN and public GloVe RNN.

Both language models are frozen. Each document is represented by the mean of its 128-dimensional token hidden states, excluding padding and boundary positions but retaining punctuation and unknown-token positions.

Each representation uses a training-fitted StandardScaler and logistic regression with the `lbfgs` solver and a maximum of 1,000 iterations. Validation macro-F1 selects `C` from `[0.1, 1.0, 10.0]`.

The selected classifiers are evaluated on the test split using accuracy, macro-F1, per-topic metrics, confusion matrices, and illustrative errors.

## 5. Recorded results and outputs

Recorded test perplexities:

| Language-model setting | Perplexity |
|---|---:|
| N-gram | 162.59 |
| RNN, random/trainable | 172.45 |
| LSTM | 184.54 |
| Transformer | 173.48 |
| RNN, own CBOW/frozen | 175.71 |
| RNN, public GloVe/frozen | 156.99 |

Recorded classification results:

| Representation | Selected C | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| Random/trainable RNN | 0.1 | 68.11% | 0.6803 |
| Public GloVe RNN | 10.0 | 87.37% | 0.8734 |

Results are from one experiment seed. Fixed seeds support reproducibility but do not guarantee identical numerical results or timings across hardware and software environments.
