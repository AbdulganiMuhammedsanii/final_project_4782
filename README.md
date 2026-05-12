# Wide Residual Networks: CIFAR-10 Replication

## 1. Introduction

This repository is a coursework-style **re-implementation and replication study** built around *[Wide Residual Networks](http://arxiv.org/abs/1605.07146)* (Zagoruyko & Komodakis, BMVC 2016). The authors show that making residual blocks **wider** (rather than arbitrarily deeper) yields better accuracy-compute trade-offs; we train their WRN family on **CIFAR-10** in PyTorch and compare test error to reported numbers.

## 2. Chosen Result

We target **CIFAR-10 test-error numbers** on four WRNs that show how widening interacts with depth. **References:** *[Table 5](http://arxiv.org/abs/1605.07146)* (flip/translation + mean/std): **WRN-28-10 = 4.00%**; *[Table 4](http://arxiv.org/abs/1605.07146)* (width sweep on WRN-40-*k*, ZCA preprocessing in the paper): **WRN-40-1 / 40-2 / 40-4 = 6.85 / 5.33 / 4.97%**. Our trainer uses Table-5-*style* mean/std preprocessing, so comparisons to the WRN-40-*k* figures are directional (full apples-to-apples would replicate each table’s preprocessing).

## 3. GitHub Contents

| Path | Role |
|------|------|
| `final_content/code/wrn_cifar10.ipynb` | End-to-end training, plots, summary table export |
| `final_content/results/summary.csv` | Recorded comparison vs paper baseline for the four architectures |

## 4. Re-implementation Details

**Approach:** Pre-activation wide residual blocks, SGD **momentum 0.9**, **weight decay 5e−4**, **batch size 128**, **200 epochs**, base LR **0.1** with multiplicative decay **×0.2** at epochs **60, 120, 160**; training augments with pad+random crop + horizontal flip and uses CIFAR-10 mean/std normalization. **Toolkit:** PyTorch + torchvision dataloaders. **Metric:** classification **test error (%)**. **Extras vs paper:** optional **BF16 mixed precision** when the GPU supports it (slightly different numerics vs full FP32). We did **not** reproduce multi-seed median aggregation or dropout variants from later tables.

## 5. Reproduction Steps

### Environment

Python 3.10+ recommended. Install PyTorch matching your CUDA build (CPU-only runs are possible but very slow):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision pandas matplotlib notebook
```

### Data (automatic download)

**You do not need a separate manual dataset drop for CIFAR-10.** Running the notebook’s data cell calls `torchvision.datasets.CIFAR10(..., download=True)`, which pulls the official tarball (~170 MB) into a `data/` folder **next to the notebook’s working directory** (typically `final_content/code/data/` if you execute from that folder).

- **Firewall / offline machines:** preload the tarball and place it where torchvision expects (`data/cifar-10-python.tar.gz` under the dataset root). See [PyTorch torchvision CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html) for the expected layout after extraction.
- **Custom cache location:** before launching Python, point caches with `TORCH_HOME` / `XDG_CACHE_HOME` if you want data outside the repo.

### Running

```bash
cd final_content/code
jupyter notebook wrn_cifar10.ipynb
```

Execute cells top to bottom: the notebook performs a short smoke run, then trains **four** configurations sequentially (hours of GPU time). There are no CLI args; hyperparameters live in **Cell 1** (`CONFIGS`, `EPOCHS`, etc.).

### Compute

Training is intended for **NVIDIA GPUs** with sufficient memory (WRN-28-10 is the largest configuration). Expect on the order of **tens of minutes per model × 200 epochs on a datacenter-grade GPU** in our logged runs (~38-48 min/model on an **H100 80 GB** profile in `summary.csv`); commodity GPUs will scale roughly with throughput; plan for **many hours total** across all four widths if memory or clock speed is lower.

## 6. Results / Insights

| Model | Params (M) | Our test err. (%) | Paper baseline in code (%) | Δ |
|-------|-----------:|------------------:|----------------------------:|-----:|
| WRN-28-10 | 36.48 | 3.93 | 4.00 | −0.07 |
| WRN-40-1  | 0.56 | 6.61 | 6.85 | −0.24 |
| WRN-40-2  | 2.24 | 5.12 | 5.33 | −0.21 |
| WRN-40-4  | 8.95 | 4.32 | 4.97 | −0.65 |

(Table from [`final_content/results/summary.csv`](final_content/results/summary.csv); “Paper baseline” follows `CONFIGS` targets in [`wrn_cifar10.ipynb`](final_content/code/wrn_cifar10.ipynb).) **Expectation:** you should land near the cited paper percentages for the **28-10 mean/std regime** (Table 5); widths 40-{1,2,4} are compared to Table 4’s published medians despite the preprocessing mismatch noted above, plus single-seed AMP noise vs median-of-five runs.

## 7. Conclusion

The replication supports the paper’s headline claim for this subset: wide residual networks reach **published ballpark errors** without structural shortcuts. Practical lessons: aligning LR schedule + augmentation matters as much as the block definition; mixed precision is convenient but contributes small run-to-run drift vs the original experiments.

## 8. References

- S. Zagoruyko, N. Komodakis. *Wide Residual Networks.* BMVC 2016. [`arXiv:1605.07146`](http://arxiv.org/abs/1605.07146)
- Official Torch/Lua training release (historical reference): [`szagoruyko/wide-residual-networks`](https://github.com/szagoruyko/wide-residual-networks)
- Learning original ResNet preactivation blocks: He et al., *Identity Mappings in Deep Residual Networks* (as cited in WRN).

## 9. Acknowledgements

This project was prepared as **coursework / final assessment** materials (repository name `final_project_4782`), which provided structure and grading context for the replication. Thanks to the original authors for clear architecture specifications and baseline numbers.
