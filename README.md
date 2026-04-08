# Topo Monosemanticity Research Project

## Project Overview
Implementation of research plan for studying monosemanticity in neural networks using topological data analysis.

## Quick Links
- [Research Plan](RESEARCH_PLAN.md) - Full research plan extracted from original document
- [Progress Tracker](PROGRESS.md) - **MAIN TRACKING FILE** - All results, graphs, and progress
- [Experimental Results](results/)
- [Code](src/)
- [Notebooks](notebooks/)

## Rules & Best Practices

### 1. Version Control
- ✅ All code committed to this repository
- ✅ Each experiment gets its own branch: `experiment/<name>`
- ✅ Regular commits with descriptive messages
- ✅ Tag major milestones: `v0.1-experiment-1`, etc.

### 2. Project Structure
```
topo/
├── README.md                    # This file
├── RESEARCH_PLAN.md            # Detailed research plan
├── PROGRESS.md                 # 🎯 MAIN TRACKING FILE - All results here
├── .gitignore
├── src/                        # Source code
│   ├── __init__.py
│   ├── models/                 # Model implementations
│   ├── experiments/            # Experiment code
│   ├── analysis/               # Analysis utilities
│   └── visualization/          # Visualization code
├── notebooks/                  # Jupyter notebooks
├── results/                    # All experimental results
│   ├── figures/               # Generated graphs and plots
│   ├── data/                  # Raw data files
│   └── summaries/             # Summary reports
├── configs/                    # Configuration files
└── docs/                       # Documentation
```

### 3. Results Tracking
- 🎯 **PROGRESS.md is the single source of truth**
- All results, graphs, and findings go in PROGRESS.md
- Graphs embedded as images with descriptions
- Tables for quantitative results
- Timestamps on all entries

### 4. Code Standards
- Python 3.9+
- Type hints required
- Docstrings for all public functions
- PEP 8 compliance
- Unit tests for critical components

### 5. Experiment Protocol
1. Create experiment branch
2. Document hypothesis in PROGRESS.md
3. Run experiment
4. Log results with graphs in PROGRESS.md
5. Analyze and conclude
6. Merge back to main

### 6. Naming Conventions
- Experiments: `exp_<number>_<short_description>`
- Branches: `experiment/<name>` or `feature/<name>`
- Files: snake_case.py
- Results folders: YYYY-MM-DD_<experiment_name>

## Current Status
**Phase:** Setup ✅
**Last Updated:** 2026-04-09

## Next Steps
1. [ ] Extract and document full research plan from DOCX
2. [ ] Set up development environment
3. [ ] Begin implementing experiments

## GitHub Integration
- Repository: (Add your GitHub repo URL here)
- Remote setup: `git remote add origin <url>`
- Push regularly: `git push -u origin main`
