# MED: Measure-Explain-Diagnose Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **What Does Vision Tool-Use Reinforcement Learning Really Learn?**
> Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom

## Overview

MED (Measure-Explain-Diagnose) is a fine-grained analysis framework for understanding vision tool-use reinforcement learning. We decompose performance improvements into intrinsic capability changes and tool-induced effects, revealing what vision RL truly learns.

### Key Findings

- Performance gains are primarily driven by intrinsic learning
- Tool-use RL mainly reduces tool-induced harm (e.g., reducing errors from tool invocation, weakening tool pattern interference)
- Limited improvement in tool-based correction of intrinsic failures
- Current vision tool-use RL learns to "safely coexist with tools" rather than fully mastering their use

## Installation

### Requirements
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/GAIR-NLP/Med.git
cd Med

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Project Structure

```
Med/
├── verl/                    # VERL training framework (independent package)
├── recipe/med/              # MED project implementation
│   ├── crop_zoom.py         # Vision tool implementations
│   └── analysis_plot/       # Evaluation analysis and plotting
├── scripts/                 # Utility scripts
└── assets/                  # Paper and resources
```

## Usage

### Evaluation and Analysis

Generate CSV from evaluation results:
```bash
python -m recipe.med.analysis_plot.data.create_csv --exp-name qwen25vl_instruct_75_50
```

Generate paper figures:
```bash
python recipe/med/analysis_plot/plotting/paper_figures.py
```

### Using Claude Code Skills

If you're using Claude Code, we provide pre-defined skills for common tasks:

```bash
/init-project          # Initialize project infrastructure
/create-csv            # Generate CSV data
/generate-figures      # Generate paper figures
```

See [CLAUDE.md](CLAUDE.md) for detailed documentation (Chinese).

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{med2024,
  title={What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom},
  author={[Authors TBD]},
  journal={[Journal TBD]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Paper: [assets/paper.pdf](assets/paper.pdf)
- Issues: [GitHub Issues](https://github.com/GAIR-NLP/Med/issues)
