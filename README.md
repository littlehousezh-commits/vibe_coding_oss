
# Supplementary Materials for the Vibe-Coding OSS Study

This repository contains supplementary code and survey materials for our study of how AI coding assistants are used in open-source software (OSS) repositories and how repository activity changes after observable AI adoption. Data collection and labeling largely follow prior studies, so the corresponding scripts are relatively simple. We organize the data-analysis scripts in one Python file to support clearer and easier interpretation.

The supplementary materials include:

- `supplementary_data_collection.py`: data-collection pipeline
- `supplementary_data_analysis.py`: data-analysis pipeline
- survey file: developer survey responses

## Overview

The study analyzes public OSS repositories containing observable AI-chat history artifacts, such as SpecStory chat logs from AI coding assistants. We link these chat histories to GitHub repository activity and analyze commit activity, file changes, issues, pull requests, CI/check outcomes, contributor participation, communication/review patterns, and developer survey responses.

The final filtered analysis cohorts used in the paper are:

- **RQ1 full filtered sample:** 1,240 repositories
- **RQ2 main comparison sample:** 608 repositories whose first observed AI chat occurred at or after GitHub repository publication
- **RQ2 validation sample:** 114 older repositories created before 2025

## Data Collection

The data-collection script consolidates the collection workflow used in the paper:

```bash
python supplementary_data_collection.py \
  --output-dir data_collection_output \
  --prefixes-file prefixes.txt \
  --repo-list repositories.csv
```

The script performs the following steps:

1. Searches GitHub Code Search for SpecStory chat-history files under `.specstory/history/`.
2. Downloads Git blob contents for matched chat-history files.
3. Parses standard SpecStory and CLI-style chat logs.
4. Maps chat sessions to GitHub repositories.
5. Collects GitHub repository history using the GitHub REST API:
   - repository metadata
   - commits
   - changed-file metadata
   - issues
   - pull requests
   - issue/PR comments
   - PR reviews
   - CI/check runs
6. Classifies changed files into source code, tests, documentation, dependencies, configuration/build files, AI-chat artifacts, generated/binary files, or other files.

### GitHub Credentials

The script does not contain API keys. Set a GitHub token in the environment before running:

```bash
export GITHUB_TOKEN=your_token_here
```

or, for multiple tokens:

```bash
export GITHUB_TOKENS="token1 token2 token3"
```

## Data Analysis

The data-analysis script reproduces the main analysis structure used in the paper:

```bash
python supplementary_data_analysis.py --root /path/to/project/root
```

The project root should contain the cleaned data tables and cohort files expected by the script, including the filtered cohort lists under:

```text
outputs/toy_homework_audit/
```

Expected cohort files:

```text
complete_history_repos_excluding_strict_name_toy_homework_1240.csv
main_comparison_chat_after_start_excluding_toy_homework_608.csv
older_pre2025_excluding_toy_homework_114.csv
```

The script outputs analysis tables under:

```text
outputs/supplementary_reproducible_analysis/
```

The analysis covers:

- AI-related commit prevalence and temporal trends
- AI-related file-type composition
- file/module concentration of AI-related changes
- chat-purpose distribution and chat-associated activity
- repository-relative-month commit activity
- defect-related activity
- testing and CI signals
- issue and pull-request activity
- contributor participation and concentration
- communication/review intensity and concentration
- survey demographics and Likert/construct summaries

## Survey Analysis

The analysis script also summarizes survey responses when a survey CSV is available:

```bash
python supplementary_data_analysis.py \
  --root /path/to/project/root \
  --survey vibe-coding_survey.csv
```

Survey analyses include:

- demographic distributions
- Likert-item summaries
- construct-level summaries
- perceptions of AI-assisted OSS contribution
- concerns about AI-generated code
- disclosure willingness and appropriate/risky use cases

## Privacy and Ethics

The supplementary scripts do not include API keys, access tokens, raw email addresses, IP addresses, or institution-specific identifiers.

When contributor identity is needed, the collection script stores hashed email identifiers rather than raw email addresses. Bot and automated accounts are excluded from human-contributor analyses where possible.

## Requirements

Python 3.10+ is recommended.

Core Python packages:

```bash
pip install pandas numpy scipy statsmodels
```

For data collection through GitHub:

```bash
pip install tqdm
```
