# Development Guidelines

## Code Style
- Follow PEP 8
- Use type hints: `def function(param: int) -> str:`
- Docstrings in Google format
- Max line length: 88 characters (use black formatter)

## Git Workflow
```bash
# Start new experiment
git checkout -b experiment/exp_001_name

# Commit regularly
git add .
git commit -m "feat: implement XYZ"

# Push to remote (after setup)
git push -u origin experiment/exp_001_name

# Merge when done
git checkout main
git merge experiment/exp_001_name
```

## Commit Message Format
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `exp:` experiment
- `refactor:` code refactoring
- `test:` tests

## Experiment Checklist
- [ ] Define hypothesis in PROGRESS.md
- [ ] Create experiment branch
- [ ] Implement code with tests
- [ ] Run experiment
- [ ] Log results in PROGRESS.md with graphs
- [ ] Analyze results
- [ ] Write conclusion
- [ ] Merge branch
- [ ] Update PROGRESS.md summary

## Results Format
All results in PROGRESS.md must include:
1. Date
2. Hypothesis/Objective
3. Configuration
4. Results (with metrics)
5. Graphs (embedded as images)
6. Key insights
7. Status

## Visualization Standards
- Use matplotlib/seaborn for plots
- High resolution (300 DPI for figures)
- Clear labels and titles
- Colorblind-friendly palettes
- Save to results/figures/
- Embed in PROGRESS.md with caption

## Data Management
- Raw data: results/data/
- Processed data: results/data/processed/
- Figures: results/figures/
- Summaries: results/summaries/
- Never commit large files (>10MB) without LFS
