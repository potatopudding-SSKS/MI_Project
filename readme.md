# Merging Probes for Cross-lingual Transfer

Ananya Agarwal (2024114007) and K S Saisankalp Davey (2024114010).

A Hindi Nom/Acc case probe on mBERT recovers Marathi noun case almost as well as a Marathi probe, yet the two point in different directions. We test whether the two probes are redundant or complementary by merging their weights, training on pooled data, and routing between them noun by noun.

## Setup

```bash
uv sync
```

`uv sync` creates `.venv` with Python 3.14 and the exact packages in `uv.lock` (the Intel XPU build of torch on Linux and Windows, which runs on the CPU when there is no Intel GPU).

Open `data.ipynb` with the `.venv` kernel and run all cells. It downloads UD Hindi HDTB and UD Marathi CMUPAN at the release pinned in `config.yaml`, prints a SHA-256 digest for each file, and writes `results/audit.json`: noun counts, label balance, excluded nouns, subword fertility, and word-type overlap between splits and languages. It then samples the training sentences for each seed and caches mBERT's first- and last-piece vectors for every target noun at all 13 layers in `data/cache/` (about 4 GB). The first run also downloads mBERT.

`probes.ipynb` trains probes on the cached features and scores them on the Marathi dev split. It reproduces the class result at layer 8 (Hindi→Marathi and Marathi→Marathi accuracy, and probe cosines) in `results/reproduction.json`. It then measures the headroom for combining the probes at every layer and pooling (both probes' accuracy, how often they disagree, and the oracle router's accuracy, with two Marathi probes trained on disjoint sentences as the baseline) in `results/headroom.json` and `results/headroom.png`.

## Protocol

- **Targets:** tokens with UPOS `NOUN` and Case exactly `Nom` or `Acc`.
- **Splits:** the official dev and test splits stay fixed; the five seeds in
  `config.yaml` choose the training sentences.
- **Model:** frozen `google-bert/bert-base-multilingual-cased`, all 13 layers, first- and last-piece pooling.
- **Main setting:** layer 7, first-piece pooling. Layer 7 had the best average over the five seeds on Marathi dev for both probes. Last-piece pooling is left out of the main experiments: the last piece of a Marathi noun is often just its case ending, so a Marathi probe on it scores 88% before mBERT has seen any context, and the Hindi probe does no better than chance.
- **Marathi budget:** with n labelled Marathi sentences, every Marathi probe trains on the same 80% and the router (and the blend weight) learns from the other 20%. The Marathi-only baseline is reported trained on the 80% and on all n.
- **Combining probes:** the features are not rescaled, so every probe's weights are in the same units. The weight blend and the router use the same two probes, each scaled to length 1, and refit one shared scale and bias on the router's 20%. The router mixes the two probes' scores, so the weight blend is a router that gives every noun the same mix, and any difference between them comes from choosing per noun.
- **Evaluation:** the official Marathi dev split is used only for choices such as the layer; the test split is scored once, at the end.
