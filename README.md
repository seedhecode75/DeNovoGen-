# DeNovoGen

DeNovoGen learns the syntax of SMILES with a two-layer PyTorch LSTM and generates molecular candidates one token at a time. Six REINFORCE experiments measure how rewarding chemical validity, novelty, QED, substructure matches and scaffold diversity changes the output at a fixed sampling budget. RDKit verifies each generated structure.

[Open in Colab](https://colab.research.google.com/github/seedhecode75/DeNovoGen-/blob/main/DeNovoGen.ipynb) · [Source code](denovogen/) · [Example results](examples/smoke_results/RESULTS.md)

## Start in Kaggle or Colab

Open **DeNovoGen.ipynb**. The notebook contains the project source, configs, tests, and a small synthetic fixture. It can run independently of the source ZIP. Enable a GPU for training. In the first cell choose the profile, data path, split type and output location. Run the cells in order. With internet enabled the notebook can fetch the pinned upstream dataset. Offline, provide a mounted directory containing the raw .smi shards and an environment with the requirements installed.

| Profile | Selected source rows before cleaning | Pretraining | RL comparisons | Final evaluation | Scaling |
|---|---:|---|---|---|---|
| smoke | All 882 synthetic input rows | 2 variants × 3 short epochs, hidden 32 | 6 × 2 updates × 8 samples × 1 seed | 32 attempts per policy/seed | 12, 24 |
| small | Up to 62,500 from across shards | 2 × 3 epochs, hidden 256 | 6 × 100 × 32 × 1 | 1,000 | 1k, 10k |
| medium | Up to 625,000 from across shards | 2 × 5 epochs, hidden 256 | 6 × 300 × 64 × 3 | 10,000 | 1k, 10k, 100k |
| full | All supplied source rows | 2 × 3 epochs, hidden 256 | Same as medium | 10,000 | 1k, 10k, 100k |

Actual training identity counts depend on deduplication, split, and the length cap. Each epoch traverses all eligible training identities. Small builds a reference from selected rows and leaves full-source novelty blank when sampling a subset. Medium and full index the entire supplied corpus, so preprocessing can be the longest CPU stage. Full-source always refers to the files in the saved manifest. The notebook can enable full indexing for small. A GPU session with one T4 is sufficient to start. Measure your pilot before scheduling the medium or full matrix. For Kaggle, save run outputs as a notebook version before the session ends, attach them as an input next session, then copy the run into a writable directory. For Colab, set the run root to a mounted persistent drive or export the run before disconnecting.

## Local quick start

Use Python 3.11 or 3.12 for the pinned RDKit release. The requirements include a newer RDKit range for Python 3.13; the manifest records the actual version. Install a compatible PyTorch build for your GPU first. From this project directory:

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m denovogen.cli all --config configs/smoke.json --source examples/smoke.smi --run runs/smoke --device cpu
python -m denovogen.cli all --config configs/small.json --source /path/to/dataset --run runs/small
~~~

For offline data preparation, point --source to the upstream dataset directory, or to a single SMILES-first file. Optional second field is the source ID. Headers, comments and blank lines are skipped. Source IDs, file hashes and shard counts are recorded. The synthetic fixture exists for software checks. The original dataset is downloaded separately; its raw files contain ZINC IDs and the source repository does not document its exact upstream release.

## Run in stages

All seven stages use the same --config and --run. Only prepare and all need --source.

~~~bash
python -m denovogen.cli prepare --config configs/medium.json --source /path/to/dataset --run runs/medium
python -m denovogen.cli pretrain --config configs/medium.json --run runs/medium
python -m denovogen.cli select --config configs/medium.json --run runs/medium
python -m denovogen.cli rl --config configs/medium.json --run runs/medium
python -m denovogen.cli evaluate --config configs/medium.json --run runs/medium
python -m denovogen.cli benchmark --config configs/medium.json --run runs/medium
python -m denovogen.cli report --config configs/medium.json --run runs/medium
~~~

To spread work over sessions, pretrain accepts --variant canonical or --variant randomized. The rl stage accepts e.g. --arm 4 --seed 11. Run the missing arms before evaluation. One process owns a run directory at a time. Rerunning a stage resumes. Data indexing commits per shard; an interrupted token-cache build restarts that cache. Supervised checkpoints commit after each epoch. RL checkpoints commit at validation intervals and include optimizer, schedule, sample RNG, scaffold memory and update history. Output chunks commit atomically; retries keep attempt IDs stable. Keep all files in the run directory together. The runner refuses changes to settings, source hashes or code within an existing run. Use a fresh directory for another split or ablation. Resume on the same device type and compatible versions. For comparable throughput, use the same GPU model and environment.

Generate more molecules with a completed checkpoint, at any sampling temperature or batch size:

~~~bash
python -m denovogen.generate --run runs/medium --policy R4 --training-seed 11 --sample-seed 1234 --count 10000 --batch 128 --top-p 0.9 --out runs/custom-R4
~~~

## Data and architecture

1. Read shard checksums and row counts. Allocate a fixed row budget proportionally across shards, then sample without replacement within each shard. Canonicalize with RDKit. The exact identity is canonical isomeric SMILES: retain stereo, disconnected components and tautomer form. Store full strings as SQLite primary keys. Records failing parsing, strict tokenization or the length bound are counted. Keep representative rejects and per-shard counters. Valid source identities remain eligible for source matching even when too long for model training.
2. Deduplicate eligible identities and assign an approximately 80/10/10 deterministic random or scaffold-group split. All molecules with the same Murcko scaffold share a split in scaffold mode. Acyclic molecules share one explicit group; inspect group sizes and resulting imbalance. Choose SPLIT = "scaffold" in a fresh run to measure structural generalization. Vocabulary comes only from training. OOV held-out targets use UNK and remain in masked metrics. The generator cannot emit UNK, PAD or BOS.
3. Memory-map token arrays with 64-bit offsets. Length-bucket each shuffled block, dynamically pad and enforce a batch-token limit. Train both variants from the same initial seed, with identical identity order and meaningful target count. The randomized variant uses one seeded, different RDKit enumeration of the same token length on odd epochs and the canonical string on even epochs. For each identity, try up to four alternatives then fall back to canonical if necessary. The number of changed views is logged. This limits augmentation variety but controls length and token-budget effects.
4. A 128-dimensional embedding feeds two 256-unit LSTM layers and a linear vocabulary head. The loss is summed next-token cross-entropy divided by the number of non-PAD targets, including EOS. Log masked top-1 and top-5 accuracy, loss, perplexity, unknown-token rate and token-category counts. Unmasked accuracy uses this run’s dynamic padding and is a diagnostic.
5. Select the best epoch of each variant by validation NLL. Pick the common RL prior on independent validation draws ranked by validity, QED-qualified distinct scaffolds per 1k and distinct novel yield, then by lower validation NLL. Record the checkpoint hash. Final samples and teacher-forced test scores are separate.

## Six reward configurations

V is 1 for a sanitized nonempty molecule; N is exact train novelty; Q is QED; F is a hit for the configured SMARTS, default CN. For R6, c is the count of a scaffold in a bounded queue of previous valid samples. Freeze this queue within the batch. Acyclic molecules share the empty scaffold. Log descriptor failures; a failed QED contributes zero to the reward.

| Arm | Terminal reward | Controlled addition |
|---|---|---|
| R1 | V / 4 | Validity-only control |
| R2 | V(1 + N) / 4 | Novelty |
| R3 | V(1 + Q) / 4 | QED |
| R4 | V(1 + N + Q) / 4 | Novelty and QED jointly |
| R5 | V(1 + N + Q + F) / 4 | Verified substructure |
| R6 | R5 × (1 − λ × min(c / cap, 1)) | Fixed scaffold repetition penalty |

The underlying algorithm is REINFORCE. Sample from the unrestricted categorical distribution over permitted tokens at temperature 1. Replay the sampled trajectory through the same policy with identical action masks. Let ℓ be the sum of the sampled action log probabilities through EOS, or through the cap for a truncated episode. Return = reward − β(ℓ − ℓ_prior). A sequence-level prior KL penalty discourages large deviations. Use detached returns with a leave-one-out batch baseline b, and minimize mean[−(return − b)ℓ]. Clip gradients and decay the RL learning rate with a cosine schedule. All six arms have identical budgets and share one pretrained checkpoint. The final comparison uses fixed-budget last policies, including runs that perform poorly. Validation curves are available for diagnosing collapse. The frozen prior is the no-RL control. Evaluation uses the same temperature and top-p across arms. Reward definitions and sampling settings are independent.

## Evaluation and output files

| Location under the run directory | Contents |
|---|---|
| data/manifest.json, data/ids.sqlite | Source inventory, identity/split reference, vocabulary, rejection counts and augmentation coverage |
| pretrain/{canonical,randomized}/ | Best and resumable last checkpoints, training traces, actual unique identities and token exposures |
| selection.json, rl/R*/seed_*/ | Common prior evidence, RL checkpoints, component diagnostics, independent periodic validation |
| final/POLICY/seed_*/ | All attempted generations in chunk CSVs, exact IDs and properties, per-run metrics |
| quality/ | Actual reference panels and held-out/property comparisons, SA, PAINS alerts and nearest similarity to a fixed train sample |
| bench/ and report/scaling.csv | Batched versus serial throughput at the smallest workload and batched scaling; latency, memory, chemistry time and valid unique throughput |
| report/RESULTS.md, per_seed.csv, comparisons.csv | Means, seed variation, t intervals and paired changes from the frozen prior |
| report/*.png | Supervised/chemical comparisons, RL diagnostics and first valid structures in output order when available |

Validity = RDKit-valid attempts / all attempts. A cap-truncated string may parse; its EOS status and cap rate are saved. Uniqueness = distinct valid identities / valid attempts. Train and source novelty = distinct valid identities absent from the relevant index / all distinct valid identities. Distinct novel yield divides by attempts. Empty denominators are undefined. QED, substructure satisfaction and descriptor distributions are conditional on valid attempts unless explicitly called yield. Diversity averages 1 − Tanimoto over distinct sample-index pairs including duplicate identities if drawn; Morgan radius 2, 1,024 bits. Small sets use every pair; large sets use uniform independent index pairs with replacement and a reported Monte Carlo SE. The PAINS/SA and nearest-neighbor panels are bounded diagnostic samples. The SNN lookup is exact against that panel only.

A benchmark excludes warm-up and session downtime. It includes generation, chemistry, actual output writing and final aggregation, and records active wall time and component timing. Batch latency quantiles refer to generation/decode batches. Host RSS is sampled during generation plus an end-of-aggregation reading; GPU memory is peak PyTorch allocation. Streaming chunks prevents losing completed work; the final summary and fingerprint aggregation retain O(N) records in host memory. A 100k profile is a workload to measure on your hardware.

For three seeds, inspect individual points alongside the exploratory t interval. Seeds vary RL and final generation while sharing one pretrained model. A single-seed run has no training-seed confidence interval. The report records absolute differences, percentage points for rates and relative changes from nonzero prior means. QED and SA describe modeled properties. Biological activity and synthesis require independent evidence.

## Example outputs and checkpoints

The [CPU smoke report](examples/smoke_results/RESULTS.md) shows what the pipeline produces on the synthetic fixture. The [example run archive](examples/smoke_run.zip) contains the checkpoints, prepared reference data, raw attempts and diagnostics. To sample from the example checkpoint:

~~~bash
unzip examples/smoke_run.zip -d examples
python -m denovogen.generate --run examples/verified_smoke_run --policy R4 --count 100 --out runs/demo
~~~

## Data and license

The notebook accepts your own SMILES files or downloads the [pinned source dataset](https://github.com/jitendra663735/molgen-lstm-rl/tree/bd1bb100871c9c19567fd05a9e5d41b010fe3010/dataset). The package is distributed under [MPL 2.0](LICENSE).

## Method references

[Randomized SMILES](https://doi.org/10.1186/s13321-019-0393-0) · [Prior-guided molecular generation](https://doi.org/10.1186/s13321-017-0235-x) · [Diversity memory](https://doi.org/10.1186/s13321-020-00473-0) · [MOSES](https://github.com/molecularsets/moses) · [GuacaMol](https://doi.org/10.1021/acs.jcim.8b00839) · [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html).

