# What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://arxiv.org/abs/2602.01334"><img src="https://img.shields.io/badge/Paper-arXiv-red.svg" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/Med2026/Med-eval-logs"><img src="https://img.shields.io/badge/Eval%20Logs-HuggingFace-yellow.svg" alt="Eval Logs"></a>
</p>

<p align="center">
  <img src="assets/framework.png" width="100%" alt="MED Framework">
</p>

## Overview

**TL;DR**: *Vision tool-use RL enhances model performance by reducing tool-induced harm, but does not significantly improve tool-based correction of intrinsic failures.*

This repository provides the **MED (Measure-Explain-Diagnose)** framework for analyzing vision tool-use reinforcement learning. We decompose performance improvements into **intrinsic capability changes** and **tool-induced effects**, providing fine-grained insights into what vision RL truly learns.

### Key Findings

- **Performance gains are primarily driven by intrinsic learning** - Models improve their base reasoning capabilities
- **Tool-use RL mainly reduces tool-induced harm** - Reduces errors from tool invocation and weakens tool pattern interference
- **Limited improvement in tool-based correction** - Tools don't significantly improve correction of intrinsic failures
- **Current vision RL learns to "safely coexist with tools"** - Rather than fully mastering their strategic use

## The MED Framework

The MED framework provides a **coarse-to-fine analysis** of vision tool-use reinforcement learning through three sequential steps:

<table>
<tr>
<td width="33%" align="center">
  <img src="assets/measure.png" alt="Measure"><br>
  <b>Measure</b><br>
  Quantify tool-induced drift by decomposing<br>tool-available drift into intrinsic and tool-induced components
</td>
<td width="33%" align="center">
  <img src="assets/explain.png" alt="Explain"><br>
  <b>Explain</b><br>
  Decompose tool-induced performance gap<br>into Gross Gain and Gross Harm via 4-term analysis
</td>
<td width="33%" align="center">
  <img src="assets/diagnose.png" alt="Diagnose"><br>
  <b>Diagnose</b><br>
  Factorize each term into Mass, Policy, and Quality<br>to probe root causes of term evolution
</td>
</tr>
</table>

This repository contains the **core methodology** from our paper (Section 3), including:

- **4-term decomposition** - Call Gain, Schema Gain, Call Harm, Schema Harm
- **Factor analysis** - Decompose each term into Mass (domain size), Policy (when to call), Quality (how to use)
- **Visualization tools** - Generate all figures (Measure, Explain, Diagnose) from the paper

## Installation

```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/GAIR-NLP/Med.git
cd Med

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

**Requirements**: Python 3.11+, [uv](https://github.com/astral-sh/uv) package manager

## Reproducing Paper Figures

### Step 1: Download Evaluation Logs

Download the evaluation logs from HuggingFace:

```bash
# Using HuggingFace CLI
hf download Med2026/Med-eval-logs --repo-type dataset --local-dir evals/

# Or using Python API
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Med2026/Med-eval-logs", repo_type="dataset", local_dir="evals/")
```

This downloads evaluation results for 6 perception benchmarks across 21 training checkpoints:
- VStar
- HRBench (4k)
- HRBench (8k)
- VisualProb (easy)
- VisualProb (medium)
- VisualProb (hard)

### Step 2: Generate CSV Data

Extract metrics from evaluation logs:

```bash
bash scripts/run_create_csv.sh
```

This creates CSV files in each eval logs with performance metrics, 4-term decomposition, and factor analysis across all checkpoints.

### Step 3: Generate Paper Figures

Generate all figures using the plotting script:

```bash
bash scripts/run_plot_paper_figures.sh
```

This generates two types of figures in the `figures/` directory:

**Aggregated figures** (averaged across all 6 benchmarks):
- `{exp_name}_measure.pdf` - MEASURE: Intrinsic vs tool-induced drift over training
- `{exp_name}_explain.pdf` - EXPLAIN: 4-term decomposition (Call/Schema Gain/Harm)
- `{exp_name}_diagnose.pdf` - DIAGNOSE: Factor analysis (Mass × Policy × Quality)

**Per-benchmark figures** (individual benchmark breakdowns):
- `{exp_name}_per_bench_exp{N}_measure.pdf` - MEASURE for each benchmark
- `{exp_name}_per_bench_exp{N}_explain.pdf` - EXPLAIN for each benchmark
- `{exp_name}_per_bench_exp{N}_diagnose.pdf` - DIAGNOSE for each benchmark

## Understanding the Results

The MED framework provides three levels of analysis, each visualized in separate figures:

### MEASURE: Quantifying Drift Components

<p align="center">
  <img src="assets/measure.png" width="100%" alt="Measure">
</p>

The MEASURE figure decomposes tool-available drift f<sub>w</sub>(t) into two components:

- **Grey area**: Intrinsic drift f<sub>wo</sub>(t) - performance change without tool access
- **Colored area**: Tool-induced drift Δ<sub>tool</sub>(t) - change in tool-induced performance gap
  - Green: positive relative gain (f<sub>w</sub> > f<sub>wo</sub>)
  - Red: negative relative drift (f<sub>wo</sub> > f<sub>w</sub>)
  - Color intensity: tool call rate

**Tool contribution ratio S<sub>tool</sub>** (top progress bar): fraction of total drift magnitude from tool effects

**Key finding**: Tool-induced effects account for only ~20-30% of total improvement. Most gains come from intrinsic capability improvements.

### EXPLAIN: 4-Term Decomposition

<p align="center">
  <img src="assets/explain.png" width="100%" alt="Explain">
</p>

The EXPLAIN figure decomposes the tool-induced performance gap G(t) = Acc<sub>w</sub>(t) - Acc<sub>wo</sub>(t) into:

**Gross Gain** (green, positive contributions):
- **Call Gain (Term 1)**: Intrinsic failures corrected by tool execution
- **Schema Gain (Term 2)**: Intrinsic failures recovered under tool schema without invocation

**Gross Harm** (red, negative contributions):
- **Call Harm (Term 3)**: Intrinsic successes lost due to tool calls
- **Schema Harm (Term 4)**: Intrinsic successes lost under tool schema without invocation

**Net gap G(t)** (yellow diamonds): Call Gain + Schema Gain - Call Harm - Schema Harm

**Key finding**: Gross Gain stagnates (Call Gain plateaus) while Gross Harm decreases consistently, indicating RL primarily reduces tool-induced harm rather than maximizing tool-based correction.

### DIAGNOSE: Factor Analysis

<p align="center">
  <img src="assets/diagnose.png" width="100%" alt="Diagnose">
</p>

The DIAGNOSE figure factorizes each of the four terms into:

- **Mass** (grey): Domain size P(D) - capacity for gain/harm
- **Policy** (blue): Calling probability P(call|D) - when to use the tool
- **Quality** (orange): Success rate P(✓|call,D) - how well the tool is used

**Thick line**: Term value (left axis)
**Thin lines**: Individual factors (right axis)

**Key findings**:
- **Limited failure correction**: Call Gain quality P(✓|call, failures) shows little improvement on current and persistent failure sets
- **Reduced breakage**: Call Harm quality P(✗|call, successes) decreases, indicating fewer errors on already-solved instances
- **Schema interference mitigation**: Schema Harm decreases as model becomes less sensitive to tool prompt

### Bottom Line

Current vision tool-use RL learns to **safely coexist** with tools rather than **master** them:
1. Tool effects contribute minimally (~20-30%) compared to intrinsic improvements
2. RL primarily reduces harm (fewer tool-induced errors) rather than increasing gain (better failure correction)
3. Models improve at not breaking existing capabilities, but show limited progress in using tools to fix hard cases

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{ma2026does,
  title={What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom},
  author={Ma, Yan and Zhang, Weiyu and Li, Tianle and Du, Linge and Shen, Xuyang and Liu, Pengfei},
  journal={arXiv preprint arXiv:2602.01334},
  year={2026}
}
```

## Roadmap

We are progressively open-sourcing components of the MED project:

- [x] **Evaluation logs** - Available at [HuggingFace](https://huggingface.co/datasets/Med2026/Med-eval-logs)
- [x] **Analysis code** - MED framework implementation (`recipe/med/analysis_plot/`)
- [ ] **Training data** - RL training dataset (~15k samples)
- [ ] **Training code** - GRPO-based RL training pipeline
- [ ] **Evaluation data** - Benchmark datasets (6 perception tasks)
- [ ] **Evaluation code** - Evaluation pipeline for tool-free and tool-available protocols

Stay tuned for updates!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
