# Merging Probes for Cross-lingual Transfer

Ananya Agarwal (2024114007) and K S Saisankalp Davey (2024114010).

A Hindi Nom/Acc case probe on mBERT recovers Marathi noun case almost as well as a Marathi probe, yet the two point in different directions. We test whether the two probes are redundant or complementary by merging their weights, training on pooled data, and routing between them noun by noun.

## Setup

```bash
uv sync
```

`uv sync` creates `.venv` with Python 3.14 and the exact packages in `uv.lock` (the Intel XPU build of torch on Linux and Windows, which runs on the CPU when there is no Intel GPU).

Open `data.ipynb` with the `.venv` kernel and run all cells. It downloads UD Hindi HDTB and UD Marathi CMUPAN at the release pinned in `config.yaml`, prints a SHA-256 digest for each file, and writes `results/audit.json`: noun counts, label balance, excluded nouns, subword fertility, and word-type overlap between splits and languages. The first run also downloads mBERT.

## Protocol

- **Targets:** tokens with UPOS `NOUN` and Case exactly `Nom` or `Acc`.
- **Splits:** the official dev and test splits stay fixed; the five seeds in
  `config.yaml` choose the training sentences.
- **Model:** frozen `google-bert/bert-base-multilingual-cased`, all 13 layers, first- and last-piece pooling.
