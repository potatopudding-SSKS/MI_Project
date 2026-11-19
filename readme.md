# Merging Probes for Cross-lingual Transfer

Ananya Agarwal (2024114007) and K S Saisankalp Davey (2024114010).

A Hindi Nom/Acc case probe on mBERT recovers Marathi noun case almost as well as a Marathi probe, yet the two point in different directions. We test whether the two probes are redundant or complementary by merging their weights, training on pooled data, and routing between them noun by noun.

## Setup

```bash
uv sync
```

`uv sync` creates `.venv` with Python 3.14 and the exact packages in `uv.lock` (the Intel XPU build of torch on Linux and Windows, which runs on the CPU when there is no Intel GPU).

Open `data.ipynb` with the `.venv` kernel and run all cells. It downloads UD Hindi HDTB and UD Marathi CMUPAN at the release pinned in `config.yaml`, prints a SHA-256 digest for each file, and writes `results/audit.json`: noun counts, label balance, excluded nouns, subword fertility, and word-type overlap between splits and languages. It then samples the training sentences for each seed and caches mBERT's first- and last-piece vectors for every target noun at all 13 layers in `data/cache/` (about 5 GB, including Hindi dev and test for the reverse direction and the ablation); splits that are already cached are skipped. The first run also downloads mBERT.

`probes.ipynb` trains probes on the cached features and scores them on the Marathi dev split. It reproduces the class result at layer 8 (Hindi→Marathi and Marathi→Marathi accuracy, and probe cosines) in `results/reproduction.json`. It then measures the headroom for combining the probes at every layer and pooling (both probes' accuracy, how often they disagree, and the oracle router's accuracy, with two Marathi probes trained on disjoint sentences as the baseline) in `results/headroom.json` and `results/headroom.png`.

`combine.ipynb` runs the combination methods (pooled data, weight blend, router, a router whose gate also learns from Hindi, and the Marathi + Marathi control) at the main setting for every Marathi budget from 25 to 800 sentences, and writes accuracy and macro-F1 to `results/combination.json` and a plot to `results/combination.png`. It then tests whether the differences between methods are real (`results/significance.json`, and `results/significance_gate.json` for the second router). It plots accuracy for every fixed Hindi share of the blend, next to a shuffled-label Hindi probe and a second Marathi probe (`results/mixes.json` and `results/mixes.png`), and compares the probes' directions (`results/cosines.json`). It runs three controls at 25, 100 and 800 sentences (random word vectors, shuffled labels, and a control task with a random label per word form) in `results/controls.json`. It scores the same probes once on the Marathi test split (`results/test.json`, `results/test_significance.json` and `results/test_significance_gate.json`). Finally, it breaks accuracy down by kind of noun (case, whether the form was in the Marathi training sentences, whether it is also a Hindi noun, and whether a postposition follows) and shows what the router's gate depends on, on dev and test (`results/nouns.json`).

`reverse.ipynb` runs the same experiment in the other direction: a Marathi probe trained on 800 sentences combined with Hindi probes trained on 25 to 800 Hindi sentences, at the same main setting, scored on Hindi dev and once on Hindi test. It writes `results/reverse.json` and `results/reverse.png`, the significance tests (`results/reverse_significance.json` and `results/reverse_test_significance.json`), the test scores (`results/reverse_test.json`), and the noun breakdown (`results/reverse_nouns.json`). About half of Hindi nouns are followed by a separate postposition, so here that group is large enough to compare.

`ablation.ipynb` asks whether a probe still works once the other language's case directions are taken out of mBERT's features, in both directions. It finds up to 64 directions with a Hindi (or Marathi) probe trained on 800 sentences, removing one at a time and retraining (iterative nullspace projection). It then trains a Marathi (or Hindi) probe from scratch on what is left, with 25, 100 or 800 sentences, and scores it on dev. Two controls remove the same number of random directions: one picks them evenly in every direction of the space, the other in proportion to how much the features vary along them. It writes `results/ablation.json`, `results/ablation.png`, and tests of whether removing the other language's directions hurts more than removing random ones in `results/ablation_significance.json`. It caches Hindi dev itself if `data.ipynb` has not.

## Protocol

- **Targets:** tokens with UPOS `NOUN` and Case exactly `Nom` or `Acc`.
- **Splits:** the official dev and test splits stay fixed; the five seeds in
  `config.yaml` choose the training sentences.
- **Model:** frozen `google-bert/bert-base-multilingual-cased`, all 13 layers, first- and last-piece pooling.
- **Main setting:** layer 7, first-piece pooling. Layer 7 had the best average over the five seeds on Marathi dev for both probes. Last-piece pooling is left out of the main experiments: the last piece of a Marathi noun is often just its case ending, so a Marathi probe on it scores 88% before mBERT has seen any context, and the Hindi probe does no better than chance.
- **Marathi budget:** with n labelled Marathi sentences, every Marathi probe trains on the same 80% and the router (and the blend weight) learns from the other 20%. The Marathi-only baseline is reported trained on the 80% and on all n.
- **Regularisation:** every probe chooses its strength (C) from 0.001 to 10 by 5-fold cross-validation over its own training sentences, keeping each sentence's nouns in one fold.
- **Combining probes:** the features are not rescaled, so every probe's weights are in the same units. The weight blend and the router use the same two probes, each scaled to length 1, and refit one shared scale and bias on the router's 20%. The router mixes the two probes' scores, so the weight blend is a router that gives every noun the same mix, and any difference between them comes from choosing per noun. A second router learns its gate from the 20% plus 200 Hindi sentences that the Hindi probe never saw.
- **Evaluation:** the official Marathi dev split is used only for choices such as the layer; the test split is scored once, at the end. Macro-F1, the second router and the noun breakdown were added after that first test run; they change no setting of the methods scored then.

## Limitations

- **Smallest Marathi budgets:** with 25 or 50 labelled Marathi sentences, the 20% held back for the weight blend and the router is only 5 or 10 sentences, so the learned mix is unstable (about ±4–5 points across seeds at 25 sentences). At these sizes, pooled data or the Hindi probe alone does better than either.
- **Postposition split:** UD Marathi CMUPAN attaches almost every case marker to the noun, so only 31 dev and 22 test nouns are followed by a separate postposition. That group of the noun breakdown is too small to compare against Hindi, where about half of all nouns are followed by one. Forms seen with both cases in the Marathi budget are rarer still (at most 18 nouns).
