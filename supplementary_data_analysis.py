#!/usr/bin/env python3
"""
Supplementary analysis code for the vibe-coding OSS study.

This file consolidates the final analysis logic corresponding to the paper's
methodology and results. It intentionally excludes exploratory and deprecated
notebook fragments.

Canonical cohorts used in the paper:
  - RQ1 full filtered sample: 1,240 repositories.
  - RQ2 main comparison sample: 608 repositories whose first observed AI chat
    occurred at or after GitHub publication.
  - RQ2 validation sample: 114 older repositories created before 2025.

The script assumes the cleaned data tables and final cohort CSVs are available
under a project root. Set VIBE_CODING_ROOT to override the current directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover - statsmodels is needed for model outputs.
    sm = None
    smf = None


ROOT = Path(os.environ.get("VIBE_CODING_ROOT", Path(__file__).resolve().parents[1])).resolve()
CLEAN = ROOT / "clean_data"
REPO_ACTIVITY = ROOT / "data_complete" / "repo_activity"
OUTPUTS = ROOT / "outputs"
SUPP_OUT = OUTPUTS / "supplementary_reproducible_analysis"

COHORT_DIR = OUTPUTS / "toy_homework_audit"
FILTERED_ALL = COHORT_DIR / "complete_history_repos_excluding_strict_name_toy_homework_1240.csv"
FILTERED_MAIN = COHORT_DIR / "main_comparison_chat_after_start_excluding_toy_homework_608.csv"
FILTERED_OLD = COHORT_DIR / "older_pre2025_excluding_toy_homework_114.csv"

BUG_FIX_RE = re.compile(r"\b(fix|bug|defect|fault|error|issue|patch|hotfix|regression)\b", re.I)
REVERT_RE = re.compile(r"\b(revert|reverted|rollback|back\s*out|backout)\b", re.I)
BUG_LABEL_RE = re.compile(r"\b(bug|defect|regression|error)\b", re.I)
BOT_RE = re.compile(r"(\[bot\]|bot$|github-actions|dependabot|renovate)", re.I)
DEV_CATEGORIES = {"source_code", "test", "documentation", "dependency", "config_build", "generated_binary", "other"}
AI_ARTIFACT_CATEGORIES = {"ai_chat_artifact", "chat_history_artifact"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def repo_key_series(df: pd.DataFrame) -> pd.Series:
    if "repo_full_name" in df.columns:
        return df["repo_full_name"].astype(str)
    if "full_name" in df.columns:
        return df["full_name"].astype(str)
    if "repo_name" in df.columns:
        return df["repo_name"].astype(str)
    raise KeyError("No repository identifier column found")


def normalize_repo_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "repo_key" not in out.columns:
        out["repo_key"] = repo_key_series(out)
    return out


def load_canonical_cohorts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load final toy/homework-filtered cohorts used in the paper."""
    all_repos = normalize_repo_column(read_csv(FILTERED_ALL))
    main = normalize_repo_column(read_csv(FILTERED_MAIN))
    old = normalize_repo_column(read_csv(FILTERED_OLD))
    assert all_repos["repo_key"].nunique() == 1240, "Expected 1,240 filtered repositories"
    assert main["repo_key"].nunique() == 608, "Expected 608 main comparison repositories"
    assert old["repo_key"].nunique() == 114, "Expected 114 older validation repositories"
    return all_repos, main, old


def load_clean_tables() -> dict[str, pd.DataFrame]:
    """Load cleaned tables; aliases cover naming differences across snapshots."""
    candidates = {
        "sessions": ["chat_sessions.csv"],
        "commits": ["repository_commit_activity.csv", "repo_commits_with_churn_full.csv", "repo_commits_full.csv"],
        "commit_files": ["repository_commit_files.csv", "repo_commit_files_full.csv"],
        "metadata": ["repository_metadata.csv", "repo_metadata.csv"],
        "issues": ["repo_issues_full.csv", "repository_issues.csv"],
        "pulls": ["repo_pull_requests_full.csv", "repository_pull_requests.csv"],
        "comments": ["repo_comments_full.csv", "repository_comments.csv"],
        "reviews": ["repo_pr_reviews_full.csv", "repository_pr_reviews.csv"],
        "ci": ["repo_ci_checks_full.csv", "repository_ci_checks.csv"],
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, names in candidates.items():
        search_roots = [CLEAN, REPO_ACTIVITY] if name == "sessions" else [REPO_ACTIVITY, CLEAN]
        for root in search_roots:
            for rel in names:
                path = root / rel
                if path.exists():
                    tables[name] = read_csv(path)
                    break
            if name in tables:
                break
    return tables


def extract_commit_or_ref_from_blob_url(url: Any) -> str | None:
    if pd.isna(url):
        return None
    parts = [p for p in urlparse(str(url)).path.split("/") if p]
    if len(parts) >= 5 and parts[2] == "blob":
        return parts[3]
    return None


def standardize_file_category(value: Any) -> str:
    if pd.isna(value):
        return "other"
    text = str(value).strip().lower()
    aliases = {
        "source": "source_code",
        "source_file": "source_code",
        "source_code": "source_code",
        "test": "test",
        "tests": "test",
        "documentation": "documentation",
        "docs": "documentation",
        "dependency": "dependency",
        "dependencies": "dependency",
        "config": "config_build",
        "configuration": "config_build",
        "config_build": "config_build",
        "generated": "generated_binary",
        "generated_or_binary": "generated_binary",
        "generated_binary": "generated_binary",
        "chat_history_artifact": "ai_chat_artifact",
        "ai_chat_artifact": "ai_chat_artifact",
    }
    return aliases.get(text, "other")


def add_file_category_flags(files: pd.DataFrame) -> pd.DataFrame:
    out = normalize_repo_column(files)
    if "path_category" in out.columns:
        out["file_category"] = out["path_category"].map(standardize_file_category)
    elif "file_category" not in out.columns:
        out["file_category"] = "other"
    out["is_source"] = out["file_category"].eq("source_code") | out.get("is_source_file", False).fillna(False).astype(bool)
    out["is_test"] = out["file_category"].eq("test") | out.get("is_test_file", False).fillna(False).astype(bool)
    out["is_ai_artifact"] = out["file_category"].isin(AI_ARTIFACT_CATEGORIES)
    out["is_development_file"] = out["file_category"].isin(DEV_CATEGORIES)
    for col in ["additions", "deletions", "changes"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def add_commit_flags(commits: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    commits = normalize_repo_column(commits).copy()
    files = add_file_category_flags(files)
    if "commit_timestamp" in commits.columns:
        commits["commit_time"] = to_utc(commits["commit_timestamp"])
    elif "author_timestamp" in commits.columns:
        commits["commit_time"] = to_utc(commits["author_timestamp"])
    else:
        raise KeyError("Commit table needs commit_timestamp or author_timestamp")
    commits["commit_message"] = commits.get("commit_message", "").fillna("").astype(str)
    commits["is_bug_fix_commit"] = commits["commit_message"].str.contains(BUG_FIX_RE, na=False)
    commits["is_revert_commit"] = commits["commit_message"].str.contains(REVERT_RE, na=False)

    agg = (
        files.groupby(["repo_key", "commit_sha"], dropna=False)
        .agg(
            has_source=("is_source", "any"),
            has_test=("is_test", "any"),
            has_ai_artifact=("is_ai_artifact", "any"),
            has_development_file=("is_development_file", "any"),
            total_additions=("additions", "sum"),
            total_deletions=("deletions", "sum"),
            total_changes=("changes", "sum"),
            files_changed=("file_path", "nunique"),
            source_files_changed=("is_source", "sum"),
        )
        .reset_index()
    )
    agg["total_churn"] = agg["total_additions"] + agg["total_deletions"]
    out = commits.merge(agg, on=["repo_key", "commit_sha"], how="left")
    for col in ["has_source", "has_test", "has_ai_artifact", "has_development_file"]:
        out[col] = out[col].fillna(False).astype(bool)
    out["is_source_commit"] = out["has_source"]
    out["is_substantive_commit"] = out["has_development_file"]
    return out


def build_ai_related_commits(sessions: pd.DataFrame, commits: pd.DataFrame, files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify artifact-containing and mixed AI-artifact development commits.

    The session table should contain either a `commit_or_ref` column or a
    GitHub blob URL column (`search_html_url`, `chat_blob_url`, or `html_url`).
    The ref is matched against collected commit SHAs in the mapped repository.
    """
    sessions = normalize_repo_column(sessions).copy()
    commits = add_commit_flags(commits, files)
    if "commit_or_ref" not in sessions.columns:
        url_col = next((c for c in ["chat_blob_url", "search_html_url", "html_url"] if c in sessions.columns), None)
        sessions["commit_or_ref"] = sessions[url_col].map(extract_commit_or_ref_from_blob_url) if url_col else pd.NA
    sessions["session_time"] = to_utc(
        sessions.get("session_timestamp", sessions.get("session_timestamp_heuristic", pd.Series(index=sessions.index)))
    )

    linked = sessions.merge(
        commits,
        left_on=["repo_key", "commit_or_ref"],
        right_on=["repo_key", "commit_sha"],
        how="left",
        suffixes=("_session", "_commit"),
    )
    linked["has_chat_associated_commit"] = linked["commit_sha"].notna()
    linked["is_ai_related_commit"] = linked["has_chat_associated_commit"]
    linked["is_ai_related_development_commit"] = linked["is_ai_related_commit"] & linked["has_development_file"]

    ai_commit_keys = linked.loc[linked["is_ai_related_commit"], ["repo_key", "commit_sha"]].drop_duplicates()
    ai_dev_keys = linked.loc[linked["is_ai_related_development_commit"], ["repo_key", "commit_sha"]].drop_duplicates()
    commits = commits.merge(ai_commit_keys.assign(is_ai_related_commit=True), on=["repo_key", "commit_sha"], how="left")
    commits = commits.merge(ai_dev_keys.assign(is_ai_related_development_commit=True), on=["repo_key", "commit_sha"], how="left")
    commits["is_ai_related_commit"] = commits["is_ai_related_commit"].fillna(False)
    commits["is_ai_related_development_commit"] = commits["is_ai_related_development_commit"].fillna(False)
    return commits, linked


def repository_adoption_times(sessions: pd.DataFrame) -> pd.DataFrame:
    sessions = normalize_repo_column(sessions).copy()
    time_col = "session_timestamp" if "session_timestamp" in sessions.columns else "session_timestamp_heuristic"
    sessions["session_time"] = to_utc(sessions[time_col])
    return (
        sessions.dropna(subset=["session_time"])
        .groupby("repo_key", as_index=False)
        .agg(first_ai_session_time=("session_time", "min"))
    )


def relative_month(event_time: pd.Series, adoption_time: pd.Series) -> pd.Series:
    event_period = event_time.dt.to_period("M")
    adoption_period = adoption_time.dt.to_period("M")
    return (event_period.dt.year - adoption_period.dt.year) * 12 + (event_period.dt.month - adoption_period.dt.month)


def add_period(df: pd.DataFrame, time_col: str, adoption: pd.DataFrame) -> pd.DataFrame:
    out = normalize_repo_column(df).merge(adoption, on="repo_key", how="left")
    out[time_col] = to_utc(out[time_col])
    out["relative_month"] = relative_month(out[time_col], out["first_ai_session_time"])
    out["period"] = np.where(out[time_col] < out["first_ai_session_time"], "pre", "post")
    out.loc[out["relative_month"].eq(0), "period"] = "adoption_month"
    return out


def wilcoxon_pre_post(values: pd.DataFrame, measure: str, cohort: str) -> dict[str, Any]:
    paired = values.pivot_table(index="repo_key", columns="period", values=measure, aggfunc="first")
    paired = paired.dropna(subset=["pre", "post"])
    if paired.empty:
        return {"cohort": cohort, "measure": measure, "n": 0}
    diffs = paired["post"] - paired["pre"]
    try:
        stat, p = stats.wilcoxon(paired["post"], paired["pre"], zero_method="wilcox")
    except ValueError:
        stat, p = np.nan, 1.0
    return {
        "cohort": cohort,
        "measure": measure,
        "n": int(len(paired)),
        "pre_mean": float(paired["pre"].mean()),
        "post_mean": float(paired["post"].mean()),
        "pre_median": float(paired["pre"].median()),
        "post_median": float(paired["post"].median()),
        "median_change": float(diffs.median()),
        "pct_increase": float((diffs > 0).mean()),
        "pct_decrease": float((diffs < 0).mean()),
        "p": float(p),
    }


def add_bh_q(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    if "p" in out.columns and out["p"].notna().any():
        mask = out["p"].notna()
        out.loc[mask, "q"] = multipletests(out.loc[mask, "p"], method="fdr_bh")[1]
    return out


def fit_its(panel: pd.DataFrame, outcome: str, cohort: str) -> pd.DataFrame:
    """Fit the repository-relative-month ITS model used in the paper."""
    if smf is None:
        return pd.DataFrame()
    data = panel.dropna(subset=[outcome, "relative_month", "repo_key"]).copy()
    data = data[~data["relative_month"].eq(0)]
    if data.empty:
        return pd.DataFrame()
    data["post"] = data["relative_month"].gt(0).astype(int)
    data["pre_time"] = np.where(data["relative_month"] < 0, data["relative_month"], 0)
    data["post_time"] = np.where(data["relative_month"] > 0, data["relative_month"], 0)
    if "repo_age_months" not in data.columns:
        data["repo_age_months"] = 0
    data["log_age"] = np.log1p(pd.to_numeric(data["repo_age_months"], errors="coerce").fillna(0))
    data["y"] = np.log1p(pd.to_numeric(data[outcome], errors="coerce").fillna(0))
    model = smf.ols("y ~ pre_time + post + post_time + log_age + C(repo_key)", data=data)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": data["repo_key"]})
    rows = []
    for term, label in [("pre_time", "pre_slope"), ("post", "immediate_level"), ("post_time", "post_slope")]:
        coef = result.params.get(term, np.nan)
        p = result.pvalues.get(term, np.nan)
        rows.append(
            {
                "cohort": cohort,
                "outcome": outcome,
                "term": label,
                "coef_log": coef,
                "relative_change": math.exp(coef) - 1 if pd.notna(coef) else np.nan,
                "p": p,
            }
        )
    contrast = np.array([0.0] * len(result.params))
    names = list(result.params.index)
    if "post_time" in names and "pre_time" in names:
        contrast[names.index("post_time")] = 1.0
        contrast[names.index("pre_time")] = -1.0
        test = result.t_test(contrast)
        coef = float(test.effect[0])
        rows.append(
            {
                "cohort": cohort,
                "outcome": outcome,
                "term": "post_minus_pre_slope",
                "coef_log": coef,
                "relative_change": math.exp(coef) - 1,
                "p": float(test.pvalue),
            }
        )
    return pd.DataFrame(rows)


#RQ1 AI-related commit analysis
def rq1_ai_use(tables: dict[str, pd.DataFrame], all_repos: pd.DataFrame) -> None:
    sessions = tables["sessions"]
    commits, linked_sessions = build_ai_related_commits(sessions, tables["commits"], tables["commit_files"])
    files = add_file_category_flags(tables["commit_files"])
    adoption = repository_adoption_times(sessions)
    repos = set(all_repos["repo_key"])
    commits = commits[commits["repo_key"].isin(repos)].merge(adoption, on="repo_key", how="left")
    commits = commits[commits["commit_time"] >= commits["first_ai_session_time"]]

    repo_summary = (
        commits.groupby("repo_key")
        .agg(
            post_adoption_commits=("commit_sha", "nunique"),
            ai_related_commits=("is_ai_related_commit", "sum"),
            ai_related_development_commits=("is_ai_related_development_commit", "sum"),
        )
        .reset_index()
    )
    repo_summary["ai_related_commit_share"] = repo_summary["ai_related_commits"] / repo_summary["post_adoption_commits"]
    repo_summary.to_csv(SUPP_OUT / "rq1_repo_ai_related_commit_summary.csv", index=False)

    commits["relative_month"] = relative_month(commits["commit_time"], commits["first_ai_session_time"])
    month = (
        commits.groupby("relative_month")
        .agg(commits=("commit_sha", "nunique"), ai_related=("is_ai_related_commit", "sum"))
        .reset_index()
    )
    month["ai_related_commit_share"] = month["ai_related"] / month["commits"]
    month.to_csv(SUPP_OUT / "rq1_ai_related_commit_share_by_relative_month.csv", index=False)

    # File-type composition among development files in AI-related development commits.
    ai_keys = commits.loc[commits["is_ai_related_development_commit"], ["repo_key", "commit_sha"]].drop_duplicates()
    ai_files = files.merge(ai_keys, on=["repo_key", "commit_sha"], how="inner")
    dev_files = ai_files[~ai_files["is_ai_artifact"]].copy()
    file_dist = dev_files["file_category"].value_counts(normalize=True).rename_axis("file_category").reset_index(name="share")
    file_dist.to_csv(SUPP_OUT / "rq1_ai_related_development_file_type_distribution.csv", index=False)

    # Concentration by file and top-level module, per repository.
    dev_files["module"] = dev_files["file_path"].fillna("(missing)").astype(str).str.replace("\\", "/", regex=False).map(
        lambda p: p.split("/", 1)[0] if "/" in p else "(root)"
    )
    concentration = []
    for repo, grp in dev_files.groupby("repo_key"):
        for level, col in [("file", "file_path"), ("module", "module")]:
            counts = grp[col].value_counts()
            shares = counts / counts.sum()
            concentration.append(
                {
                    "repo_key": repo,
                    "level": level,
                    "unique_units": int(len(counts)),
                    "top1_share": float(shares.iloc[0]) if len(shares) else np.nan,
                    "top3_share": float(shares.iloc[:3].sum()) if len(shares) else np.nan,
                    "hhi": float((shares**2).sum()) if len(shares) else np.nan,
                }
            )
    pd.DataFrame(concentration).to_csv(SUPP_OUT / "rq1_ai_related_file_module_concentration.csv", index=False)
    linked_sessions.to_csv(SUPP_OUT / "rq1_chat_associated_commit_session_level.csv", index=False)


def load_chat_labels() -> pd.DataFrame:
    candidates = [
        OUTPUTS / "chat_label_correction_1240" / "combined_session_dominant_labels_1240.csv",
        OUTPUTS / "chat_label_correction_1240" / "combined_seven_category_session_distribution_1240.csv",
    ]
    for path in candidates:
        if path.exists():
            return read_csv(path)
    return pd.DataFrame()


def rq1_chat_purpose(tables: dict[str, pd.DataFrame], all_repos: pd.DataFrame) -> None:
    labels = load_chat_labels()
    if labels.empty:
        return
    labels = labels.rename(columns={"dominant_category": "chat_purpose", "category": "chat_purpose"})
    if "session_sha" not in labels.columns or "chat_purpose" not in labels.columns:
        return
    sessions = normalize_repo_column(tables["sessions"])
    labeled = sessions.merge(labels[["session_sha", "chat_purpose"]].drop_duplicates(), on="session_sha", how="inner")
    labeled = labeled[labeled["repo_key"].isin(set(all_repos["repo_key"]))]
    labeled.groupby("repo_key")["chat_purpose"].nunique().reset_index(name="distinct_chat_purposes").to_csv(
        SUPP_OUT / "rq1_distinct_chat_purposes_per_repo.csv", index=False
    )
    dominant = (
        labeled.groupby(["repo_key", "chat_purpose"]).size().reset_index(name="sessions")
        .sort_values(["repo_key", "sessions"], ascending=[True, False])
        .drop_duplicates("repo_key")
    )
    dominant.to_csv(SUPP_OUT / "rq1_repository_dominant_chat_purpose.csv", index=False)

    commits, linked = build_ai_related_commits(labeled, tables["commits"], tables["commit_files"])
    files = add_file_category_flags(tables["commit_files"])
    linked_dev = linked[linked["is_ai_related_development_commit"]].copy()
    linked_dev.to_csv(SUPP_OUT / "rq1_chat_purpose_associated_development_commits.csv", index=False)

    # Summarize file-type composition of chat-associated development commits.
    associated_keys = linked_dev[["session_sha", "repo_key", "commit_sha", "chat_purpose"]].drop_duplicates()
    associated_files = files.merge(associated_keys, on=["repo_key", "commit_sha"], how="inner")
    associated_files = associated_files[~associated_files["is_ai_artifact"]]
    file_types = (
        associated_files.groupby(["chat_purpose", "file_category"]).size().reset_index(name="file_changes")
    )
    file_types["share_within_purpose"] = file_types["file_changes"] / file_types.groupby("chat_purpose")["file_changes"].transform("sum")
    file_types.to_csv(SUPP_OUT / "rq1_chat_purpose_associated_file_types.csv", index=False)


#RQ2 Commit quality and collaboration analysis
def build_relative_month_commit_panel(commits: pd.DataFrame, adoption: pd.DataFrame, repos: pd.DataFrame, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    commits = commits[commits["repo_key"].isin(set(repos["repo_key"]))].copy()
    commits = commits.merge(adoption, on="repo_key", how="left")
    commits = commits.dropna(subset=["first_ai_session_time", "commit_time"])
    commits["relative_month"] = relative_month(commits["commit_time"], commits["first_ai_session_time"])
    grp = (
        commits.groupby(["repo_key", "relative_month"])
        .agg(
            total_commits=("commit_sha", "nunique"),
            substantive_commits=("is_substantive_commit", "sum"),
            source_commits=("is_source_commit", "sum"),
            bug_fix_commits=("is_bug_fix_commit", "sum"),
            revert_commits=("is_revert_commit", "sum"),
            test_touching_commits=("has_test", "sum"),
        )
        .reset_index()
    )
    grp["bug_fix_commit_share"] = grp["bug_fix_commits"] / grp["total_commits"].replace(0, np.nan)
    grp["revert_commit_share"] = grp["revert_commits"] / grp["total_commits"].replace(0, np.nan)
    grp["test_touching_commit_share"] = grp["test_touching_commits"] / grp["total_commits"].replace(0, np.nan)
    if metadata is not None and "created_at" in metadata.columns:
        meta = normalize_repo_column(metadata)[["repo_key", "created_at"]].drop_duplicates()
        grp = grp.merge(adoption, on="repo_key", how="left").merge(meta, on="repo_key", how="left")
        created = to_utc(grp["created_at"])
        adoption_month_time = grp["first_ai_session_time"] + pd.to_timedelta(grp["relative_month"] * 30, unit="D")
        grp["repo_age_months"] = ((adoption_month_time - created).dt.days / 30.4375).clip(lower=0)
    else:
        grp["repo_age_months"] = np.nan
    return grp


def average_pre_post_from_panel(panel: pd.DataFrame, measures: list[str], cohort: str) -> pd.DataFrame:
    panel = panel[~panel["relative_month"].eq(0)].copy()
    panel["period"] = np.where(panel["relative_month"] < 0, "pre", "post")
    repo_period = panel.groupby(["repo_key", "period"])[measures].mean().reset_index()
    rows = [wilcoxon_pre_post(repo_period, measure, cohort) for measure in measures]
    return add_bh_q(rows)


def rq2_commit_activity_and_quality(tables: dict[str, pd.DataFrame], main: pd.DataFrame, old: pd.DataFrame) -> None:
    sessions = tables["sessions"]
    adoption = repository_adoption_times(sessions)
    commits, _ = build_ai_related_commits(sessions, tables["commits"], tables["commit_files"])
    metadata = tables.get("metadata")
    rows = []
    its_rows = []
    for cohort_name, repos in [("main_608", main), ("older_114", old)]:
        panel = build_relative_month_commit_panel(commits, adoption, repos, metadata)
        panel.to_csv(SUPP_OUT / f"rq2_relative_month_commit_quality_panel_{cohort_name}.csv", index=False)
        measures = [
            "total_commits", "substantive_commits", "source_commits",
            "bug_fix_commits", "bug_fix_commit_share", "revert_commit_share",
            "test_touching_commit_share",
        ]
        rows.append(average_pre_post_from_panel(panel, measures, cohort_name))
        for outcome in ["total_commits", "substantive_commits", "source_commits", "bug_fix_commits", "bug_fix_commit_share"]:
            its_rows.append(fit_its(panel, outcome, cohort_name))
    pd.concat(rows, ignore_index=True).to_csv(SUPP_OUT / "rq2_commit_quality_pre_post_tests.csv", index=False)
    if its_rows:
        add_bh_q(pd.concat(its_rows, ignore_index=True).to_dict("records")).to_csv(
            SUPP_OUT / "rq2_commit_quality_its_models.csv", index=False
        )


def issue_is_bug(labels: Any) -> bool:
    if pd.isna(labels):
        return False
    return bool(BUG_LABEL_RE.search(str(labels)))


def rq2_issues_prs(tables: dict[str, pd.DataFrame], main: pd.DataFrame, old: pd.DataFrame) -> None:
    adoption = repository_adoption_times(tables["sessions"])
    issues = normalize_repo_column(tables.get("issues", pd.DataFrame()))
    pulls = normalize_repo_column(tables.get("pulls", pd.DataFrame()))
    if issues.empty and pulls.empty:
        return
    if not issues.empty:
        issues["created_at"] = to_utc(issues["created_at"])
        issues["closed_at"] = to_utc(issues.get("closed_at", pd.NaT))
        issues["is_bug_labeled"] = issues.get("labels", "").map(issue_is_bug)
    if not pulls.empty:
        pulls["created_at"] = to_utc(pulls["created_at"])
        pulls["merged_at"] = to_utc(pulls.get("merged_at", pd.NaT))

    all_results = []
    for cohort_name, repos in [("main_608", main), ("older_114", old)]:
        repo_set = set(repos["repo_key"])
        issue_events = issues[issues["repo_key"].isin(repo_set)].merge(adoption, on="repo_key", how="left") if not issues.empty else pd.DataFrame()
        pr_events = pulls[pulls["repo_key"].isin(repo_set)].merge(adoption, on="repo_key", how="left") if not pulls.empty else pd.DataFrame()
        period_rows = []
        for repo in repo_set:
            row = {"repo_key": repo}
            for period in ["pre", "post"]:
                if not issue_events.empty:
                    repo_issues = issue_events[issue_events["repo_key"].eq(repo)]
                    cutoff = repo_issues["first_ai_session_time"].dropna().min() if not repo_issues.empty else pd.NaT
                    if pd.notna(cutoff):
                        opened = repo_issues[repo_issues["created_at"].lt(cutoff) if period == "pre" else repo_issues["created_at"].ge(cutoff)]
                        closed = repo_issues[repo_issues["closed_at"].lt(cutoff) if period == "pre" else repo_issues["closed_at"].ge(cutoff)]
                        row[f"{period}_issue_openings"] = len(opened)
                        row[f"{period}_issue_closings"] = closed["closed_at"].notna().sum()
                        row[f"{period}_bug_labeled_issues"] = opened["is_bug_labeled"].sum()
                if not pr_events.empty:
                    repo_prs = pr_events[pr_events["repo_key"].eq(repo)]
                    cutoff = repo_prs["first_ai_session_time"].dropna().min() if not repo_prs.empty else pd.NaT
                    if pd.notna(cutoff):
                        opened = repo_prs[repo_prs["created_at"].lt(cutoff) if period == "pre" else repo_prs["created_at"].ge(cutoff)]
                        merged = repo_prs[repo_prs["merged_at"].lt(cutoff) if period == "pre" else repo_prs["merged_at"].ge(cutoff)]
                        row[f"{period}_pr_openings"] = len(opened)
                        row[f"{period}_merged_prs"] = merged["merged_at"].notna().sum()
            period_rows.append(row)
        wide = pd.DataFrame(period_rows)
        for period in ["pre", "post"]:
            denom = wide[f"{period}_issue_openings"].fillna(0) + wide[f"{period}_pr_openings"].fillna(0)
            wide[f"{period}_issue_open_share"] = wide[f"{period}_issue_openings"] / denom.replace(0, np.nan)
            wide[f"{period}_pr_open_share"] = wide[f"{period}_pr_openings"] / denom.replace(0, np.nan)
            wide[f"{period}_issue_closure_to_opening_ratio"] = wide[f"{period}_issue_closings"] / wide[f"{period}_issue_openings"].replace(0, np.nan)
            wide[f"{period}_pr_merge_share"] = wide[f"{period}_merged_prs"] / wide[f"{period}_pr_openings"].replace(0, np.nan)
            wide[f"{period}_bug_labeled_issue_share"] = wide[f"{period}_bug_labeled_issues"] / wide[f"{period}_issue_openings"].replace(0, np.nan)
        long_rows = []
        for _, row in wide.iterrows():
            for period in ["pre", "post"]:
                vals = {"repo_key": row["repo_key"], "period": period}
                for measure in [
                    "issue_openings", "issue_closings", "pr_openings", "merged_prs",
                    "issue_open_share", "pr_open_share", "issue_closure_to_opening_ratio",
                    "pr_merge_share", "bug_labeled_issue_share",
                ]:
                    vals[measure] = row.get(f"{period}_{measure}")
                long_rows.append(vals)
        long = pd.DataFrame(long_rows)
        measures = [c for c in long.columns if c not in {"repo_key", "period"}]
        all_results.extend(wilcoxon_pre_post(long, measure, cohort_name) for measure in measures)
        wide.to_csv(SUPP_OUT / f"rq2_issue_pr_whole_period_values_{cohort_name}.csv", index=False)
    add_bh_q(all_results).to_csv(SUPP_OUT / "rq2_issue_pr_whole_period_tests.csv", index=False)


def hhi_from_counts(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return np.nan
    shares = counts / total
    return float((shares**2).sum())


def top_share(counts: pd.Series) -> float:
    total = counts.sum()
    return float(counts.max() / total) if total > 0 else np.nan


def rq2_collaboration_review(tables: dict[str, pd.DataFrame], main: pd.DataFrame, old: pd.DataFrame) -> None:
    adoption = repository_adoption_times(tables["sessions"])
    commits, _ = build_ai_related_commits(tables["sessions"], tables["commits"], tables["commit_files"])
    comments = normalize_repo_column(tables.get("comments", pd.DataFrame()))
    reviews = normalize_repo_column(tables.get("reviews", pd.DataFrame()))
    rows = []
    for cohort_name, repos in [("main_608", main), ("older_114", old)]:
        repo_set = set(repos["repo_key"])
        c = commits[commits["repo_key"].isin(repo_set)].merge(adoption, on="repo_key", how="left")
        c["period"] = np.where(c["commit_time"] < c["first_ai_session_time"], "pre", "post")
        c = c[c["period"].isin(["pre", "post"]) & c["is_substantive_commit"]]
        author_col = "commit_author_login" if "commit_author_login" in c.columns else "commit_author_name"
        contrib = []
        for (repo, period), grp in c.groupby(["repo_key", "period"]):
            human = grp[~grp[author_col].fillna("").astype(str).str.contains(BOT_RE)]
            counts = human[author_col].fillna("unknown").value_counts()
            contrib.append(
                {
                    "repo_key": repo,
                    "period": period,
                    "active_contributors": counts.size,
                    "top_contributor_share": top_share(counts),
                    "contributor_hhi": hhi_from_counts(counts),
                }
            )
        contrib_long = pd.DataFrame(contrib)
        for measure in ["active_contributors", "top_contributor_share", "contributor_hhi"]:
            rows.append(wilcoxon_pre_post(contrib_long, measure, cohort_name))

        if not comments.empty:
            comments["created_at"] = to_utc(comments["created_at"])
            cm = comments[comments["repo_key"].isin(repo_set)].merge(adoption, on="repo_key", how="left")
            cm["period"] = np.where(cm["created_at"] < cm["first_ai_session_time"], "pre", "post")
            cm = cm[cm["period"].isin(["pre", "post"])]
            user_col = "user_login"
            comm_rows = []
            for (repo, period), grp in cm.groupby(["repo_key", "period"]):
                human = grp[~grp[user_col].fillna("").astype(str).str.contains(BOT_RE)]
                counts = human[user_col].fillna("unknown").value_counts()
                comm_rows.append(
                    {
                        "repo_key": repo,
                        "period": period,
                        "unique_commenters": counts.size,
                        "top_commenter_share": top_share(counts),
                        "commenter_hhi": hhi_from_counts(counts),
                    }
                )
            comm_long = pd.DataFrame(comm_rows)
            for measure in ["unique_commenters", "top_commenter_share", "commenter_hhi"]:
                rows.append(wilcoxon_pre_post(comm_long, measure, cohort_name))

        if not reviews.empty:
            reviews["created_at"] = to_utc(reviews["created_at"])
            rv = reviews[reviews["repo_key"].isin(repo_set)].merge(adoption, on="repo_key", how="left")
            rv["period"] = np.where(rv["created_at"] < rv["first_ai_session_time"], "pre", "post")
            rv = rv[rv["period"].isin(["pre", "post"])]
            review_rows = []
            for (repo, period), grp in rv.groupby(["repo_key", "period"]):
                human = grp[~grp["user_login"].fillna("").astype(str).str.contains(BOT_RE)]
                counts = human["user_login"].fillna("unknown").value_counts()
                review_rows.append(
                    {
                        "repo_key": repo,
                        "period": period,
                        "unique_reviewers": counts.size,
                        "reviewer_hhi": hhi_from_counts(counts),
                    }
                )
            review_long = pd.DataFrame(review_rows)
            for measure in ["unique_reviewers", "reviewer_hhi"]:
                rows.append(wilcoxon_pre_post(review_long, measure, cohort_name))
    add_bh_q(rows).to_csv(SUPP_OUT / "rq2_collaboration_communication_tests.csv", index=False)

#RQ3 survey analysis

LIKERT_MAP = {
    "Strongly disagree": 1,
    "Somewhat disagree": 2,
    "Neither agree nor disagree": 3,
    "Somewhat agree": 4,
    "Strongly agree": 5,
}


def survey_likert_to_num(series: pd.Series) -> pd.Series:
    return series.map(lambda x: LIKERT_MAP.get(str(x).strip(), np.nan))


def rq3_survey(survey_path: Path) -> None:
    if not survey_path.exists():
        return
    raw = pd.read_csv(survey_path, dtype=str)
    df = raw.iloc[2:].copy() if len(raw) > 2 else raw.copy()
    finished = df[(df.get("Finished") == "True") & (df.get("Progress") == "100")].copy()
    valid = finished if not finished.empty else df

    # Demographics.
    demo_cols = {
        "Q1": "age",
        "Q2": "gender",
        "Q3": "role",
        "Q4": "programming_experience",
        "Q5": "years_programming",
        "Q6": "oss_experience",
        "Q7": "oss_contribution_frequency",
        "Q8": "vibe_coding_familiarity",
        "Q9": "vibe_coding_use_for_oss",
    }
    demo_rows = []
    for col, label in demo_cols.items():
        if col not in valid.columns:
            continue
        counts = valid[col].dropna().value_counts()
        total = counts.sum()
        for category, count in counts.items():
            demo_rows.append({"measure": label, "category": category, "n": int(count), "share": float(count / total)})
    pd.DataFrame(demo_rows).to_csv(SUPP_OUT / "rq3_survey_demographics.csv", index=False)

    # Likert item summaries and constructs.
    likert_cols = [c for c in valid.columns if re.match(r"Q(11|12|13|14|15)_\d+", c)]
    item_rows = []
    scores = pd.DataFrame(index=valid.index)
    for col in likert_cols:
        score = survey_likert_to_num(valid[col])
        scores[col] = score
        item_rows.append(
            {
                "item": col,
                "n": int(score.notna().sum()),
                "mean": float(score.mean()),
                "median": float(score.median()),
                "pct_somewhat_or_strongly_agree": float(score.ge(4).mean()),
            }
        )
    pd.DataFrame(item_rows).to_csv(SUPP_OUT / "rq3_survey_likert_item_summary.csv", index=False)

    constructs = {
        "oss_contribution_access": ["Q11_3", "Q11_4", "Q11_5", "Q11_6"],
        "own_ai_code_concern": ["Q12_1", "Q12_2", "Q12_4", "Q12_6", "Q12_7"],
        "others_ai_code_concern": ["Q13_1", "Q13_2", "Q13_4", "Q13_6", "Q13_7"],
        "social_image_concern": ["Q14_1", "Q14_2", "Q14_3", "Q14_4", "Q14_5", "Q14_8"],
        "responsibility_governance": ["Q15_3", "Q15_4"],
    }
    construct_rows = []
    for name, cols in constructs.items():
        available = [c for c in cols if c in scores.columns]
        if not available:
            continue
        score = scores[available].mean(axis=1)
        try:
            _, p = stats.wilcoxon(score.dropna() - 3)
        except ValueError:
            p = np.nan
        construct_rows.append(
            {
                "construct": name,
                "items": ";".join(available),
                "n": int(score.notna().sum()),
                "mean": float(score.mean()),
                "median": float(score.median()),
                "pct_above_neutral": float(score.gt(3).mean()),
                "p_vs_neutral": p,
            }
        )
    pd.DataFrame(construct_rows).to_csv(SUPP_OUT / "rq3_survey_construct_summary.csv", index=False)


def main() -> None:
    global ROOT, CLEAN, REPO_ACTIVITY, OUTPUTS, SUPP_OUT, COHORT_DIR, FILTERED_ALL, FILTERED_MAIN, FILTERED_OLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--survey", type=Path, default=ROOT / "vibe-coding_survey.csv")
    args = parser.parse_args()

    ROOT = args.root.resolve()
    CLEAN = ROOT / "clean_data"
    REPO_ACTIVITY = ROOT / "data_complete" / "repo_activity"
    OUTPUTS = ROOT / "outputs"
    SUPP_OUT = OUTPUTS / "supplementary_reproducible_analysis"
    COHORT_DIR = OUTPUTS / "toy_homework_audit"
    FILTERED_ALL = COHORT_DIR / "complete_history_repos_excluding_strict_name_toy_homework_1240.csv"
    FILTERED_MAIN = COHORT_DIR / "main_comparison_chat_after_start_excluding_toy_homework_608.csv"
    FILTERED_OLD = COHORT_DIR / "older_pre2025_excluding_toy_homework_114.csv"
    ensure_dir(SUPP_OUT)

    all_repos, main_repos, old_repos = load_canonical_cohorts()
    tables = load_clean_tables()
    required = {"sessions", "commits", "commit_files"}
    missing = required - set(tables)
    if missing:
        raise SystemExit(f"Missing required clean tables: {sorted(missing)}")

    rq1_ai_use(tables, all_repos)
    rq1_chat_purpose(tables, all_repos)
    rq2_commit_activity_and_quality(tables, main_repos, old_repos)
    rq2_issues_prs(tables, main_repos, old_repos)
    rq2_collaboration_review(tables, main_repos, old_repos)
    rq3_survey(args.survey)


if __name__ == "__main__":
    main()
