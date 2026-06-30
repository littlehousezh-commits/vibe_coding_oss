#!/usr/bin/env python3
"""
Supplementary data-collection code for the vibe-coding OSS study.

This file consolidates the data-collection procedures reported in the paper:

1. Locate SpecStory chat-history files on GitHub using GitHub Code Search.
2. Download chat-history blobs and parse chat sessions/messages.
3. Map chat sessions to GitHub repositories.
4. Collect repository development histories through the GitHub REST API.
5. Classify changed files and export sanitized tables.

The SpecStory scraping and parsing logic follows the workflow in the
vibe-coding-scraper archive used by the prior Programming by Chat study:
GitHub Code Search -> Git blob download -> Markdown parsing using SpecStory
role markers, with a separate parser for CLI-style traces.

Set GITHUB_TOKEN or GITHUB_TOKENS in the environment before running.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


GITHUB_API = "https://api.github.com"
SPECSTORY_PATH = ".specstory/history"

SOURCE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cc",
    ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".kts",
    ".scala", ".sh", ".bash", ".zsh", ".ps1", ".sql", ".r", ".m", ".mm",
    ".vue", ".svelte", ".html", ".css", ".scss", ".sass", ".less", ".dart",
    ".lua", ".pl", ".pm", ".ex", ".exs", ".erl", ".hrl", ".fs", ".fsx",
    ".clj", ".cljs", ".jl", ".zig", ".sol", ".tf",
}
DOC_EXTS = {".md", ".rst", ".adoc", ".txt", ".tex", ".org"}
TEST_HINTS = ("/test/", "/tests/", "__tests__/", "/spec/", ".spec.", ".test.", "_test.")
DEPENDENCY_FILES = {
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "requirements.txt", "pipfile", "pipfile.lock", "poetry.lock",
    "pyproject.toml", "go.mod", "go.sum", "cargo.toml", "cargo.lock",
    "gemfile", "gemfile.lock", "composer.json", "composer.lock",
    "pom.xml", "build.gradle", "gradle.lockfile",
}
CONFIG_HINTS = (
    ".github/workflows/", "dockerfile", "docker-compose", "makefile",
    "cmakelists.txt", "webpack.config", "vite.config", "rollup.config",
    "tsconfig", "eslint", "prettier", "babel.config", "jest.config",
    "tailwind.config", "next.config", "nuxt.config", ".gitignore",
)
CHAT_ARTIFACT_HINTS = (
    ".specstory/", "specstory/history", "chat-history", "chat_history",
    "conversation-log", "conversation_log", "ai-conversation",
    "ai_conversation", ".aider.chat.history", ".cursor/chat",
)
GENERATED_OR_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
    ".tar", ".tgz", ".mp4", ".mov", ".mp3", ".wav", ".woff", ".woff2",
    ".ttf", ".eot", ".lockb", ".min.js", ".min.css",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_tokens() -> list[str]:
    raw = os.environ.get("GITHUB_TOKENS") or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not raw:
        return []
    return [tok.strip() for tok in re.split(r"[\s,]+", raw) if tok.strip()]


def hash_email(email: Any, salt: str = "") -> str | None:
    if email is None:
        return None
    text = str(email).strip().lower()
    if not text or text in {"none", "null", "nan"}:
        return None
    return hashlib.sha256((salt + text).encode("utf-8")).hexdigest()[:16]


def classify_path(path: Any) -> str:
    """Classify a changed file path into the categories used in the paper."""
    if not path:
        return "other"
    p = str(path).replace("\\", "/")
    low = p.lower()
    name = low.rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()

    if any(hint in low for hint in CHAT_ARTIFACT_HINTS):
        return "ai_chat_artifact"
    if low.startswith(("dist/", "build/", "coverage/", ".next/", "out/")):
        return "generated_binary"
    if any(low.endswith(ext) for ext in GENERATED_OR_BINARY_EXTS):
        return "generated_binary"
    if name in DEPENDENCY_FILES:
        return "dependency"
    if any(hint in low for hint in CONFIG_HINTS) or low.endswith((".yml", ".yaml", ".toml", ".ini", ".cfg")):
        return "config_build"
    if any(hint in low for hint in TEST_HINTS):
        return "test"
    if suffix in DOC_EXTS or low.startswith(("docs/", "doc/")):
        return "documentation"
    if suffix in SOURCE_EXTS:
        return "source_code"
    return "other"


def file_flags(category: str) -> dict[str, bool]:
    return {
        "is_source_file": category == "source_code",
        "is_test_file": category == "test",
        "is_doc_file": category == "documentation",
        "is_config_file": category == "config_build",
        "is_dependency_file": category == "dependency",
        "is_ai_chat_artifact_file": category == "ai_chat_artifact",
    }


class RateLimitError(RuntimeError):
    def __init__(self, message: str, reset_epoch: int | None = None):
        super().__init__(message)
        self.reset_epoch = reset_epoch


@dataclass
class GitHubClient:
    tokens: list[str]
    sleep_on_rate_limit: bool = True
    max_rate_sleep_seconds: int = 3600
    user_agent: str = "vibe-coding-oss-study"

    def __post_init__(self) -> None:
        self._token_index = 0
        self._reset_epoch_by_token: dict[str | None, int] = {}

    def _token(self) -> str | None:
        if not self.tokens:
            return None
        return self.tokens[self._token_index % len(self.tokens)]

    def _advance_token(self) -> None:
        if self.tokens:
            self._token_index = (self._token_index + 1) % len(self.tokens)

    def request_json(self, url: str, *, accept: str = "application/vnd.github+json") -> Any:
        last_error: Exception | None = None
        attempts = max(1, len(self.tokens)) + 1
        for _ in range(attempts):
            token = self._token()
            headers = {
                "Accept": accept,
                "User-Agent": self.user_agent,
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"
            req = Request(url, headers=headers)
            try:
                with urlopen(req, timeout=60) as resp:
                    payload = resp.read().decode("utf-8")
                    return json.loads(payload) if payload else None
            except HTTPError as err:
                last_error = err
                if err.code in {403, 429}:
                    reset = err.headers.get("X-RateLimit-Reset")
                    reset_epoch = int(reset) if reset and reset.isdigit() else None
                    self._reset_epoch_by_token[token] = reset_epoch or int(time.time()) + 60
                    self._advance_token()
                    continue
                if err.code == 404:
                    return None
                raise
            except URLError as err:
                last_error = err
                time.sleep(2)
        if self.sleep_on_rate_limit and self._reset_epoch_by_token:
            sleep_until = min(v for v in self._reset_epoch_by_token.values() if v)
            wait = max(1, min(self.max_rate_sleep_seconds, sleep_until - int(time.time()) + 2))
            time.sleep(wait)
            return self.request_json(url, accept=accept)
        raise RateLimitError(f"GitHub request failed after token rotation: {url}") from last_error

    def paginate(self, url: str) -> list[Any]:
        """Paginate GitHub REST endpoints using page/per_page parameters."""
        rows: list[Any] = []
        sep = "&" if "?" in url else "?"
        page = 1
        while True:
            page_url = f"{url}{sep}per_page=100&page={page}"
            data = self.request_json(page_url)
            if not data:
                break
            if isinstance(data, dict) and "items" in data:
                batch = data["items"]
            else:
                batch = data
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return rows


def github_code_search(client: GitHubClient, query: str) -> list[dict[str, Any]]:
    """Search GitHub code and return result items.

    The scraper used date-prefix queries such as:
    `.specstory/history/2025-01 in:path language:markdown`
    to reduce GitHub Code Search's result cap.
    """
    url = f"{GITHUB_API}/search/code?{urlencode({'q': query, 'per_page': 100})}"
    data = client.request_json(url, accept="application/vnd.github.text-match+json")
    if not data:
        return []
    return data.get("items", [])


def scrape_specstory_searches(prefixes: Iterable[str], output_dir: Path, client: GitHubClient) -> None:
    ensure_dir(output_dir / "searches")
    all_items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for prefix in prefixes:
        query = f"{SPECSTORY_PATH}/{prefix} in:path language:markdown"
        items = github_code_search(client, query)
        (output_dir / "searches" / f"{prefix}.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
        for item in items:
            key = (str(item.get("sha")), str(item.get("html_url")))
            if key not in seen:
                all_items.append(item)
                seen.add(key)
    (output_dir / "searches.json").write_text(json.dumps(all_items, indent=2), encoding="utf-8")


def download_chat_blobs(output_dir: Path, client: GitHubClient) -> None:
    ensure_dir(output_dir / "contents")
    ensure_dir(output_dir / "markdowns")
    items = json.loads((output_dir / "searches.json").read_text(encoding="utf-8"))
    for item in items:
        sha = item.get("sha")
        git_url = item.get("git_url")
        if not sha or not git_url:
            continue
        blob = client.request_json(git_url)
        if not blob:
            continue
        (output_dir / "contents" / str(sha)).write_text(json.dumps(blob), encoding="utf-8")
        content = blob.get("content", "")
        if blob.get("encoding") == "base64":
            text = base64.b64decode(content).decode("utf-8", errors="replace")
            (output_dir / "markdowns" / f"{sha}.md").write_text(text, encoding="utf-8")


TIMESTAMP_RE = re.compile(r"(\d{4}\D+\d{2}\D+\d{2}\D+\d{2}\D+\d{2})")


def extract_timestamp_from_search_name(name: str | None) -> str | None:
    if not name:
        return None
    match = TIMESTAMP_RE.search(name)
    if not match:
        return None
    normalized = re.sub(r"\D+", "-", match.group(1)).strip("-") + "-00"
    try:
        return datetime.strptime(normalized, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def parse_title(markdown: str) -> str:
    for pattern in [r"^# (.+)", r"^## (?!SpecStory)(.+)", r"^\*\*(.+?)\*\*$"]:
        match = re.search(pattern, markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
    return "Untitled"


def clean_message_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def normalize_role(role: str) -> str:
    text = role.strip().lower()
    if text.startswith("user"):
        return "User"
    if text.startswith(("assistant", "agent")):
        return "Assistant"
    return "Assistant"


def parse_standard_specstory_messages(markdown: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"_\*\*((?:User|Assistant|Agent)[^\n]*?)\*\*_\n+(.*?)(?=(?:_\*\*(?:User|Assistant|Agent)[^\n]*?\*\*_\n+)|\Z)",
        re.DOTALL,
    )

    def inside_fence(pos: int) -> bool:
        return markdown[:pos].count("```") % 2 == 1

    messages: list[dict[str, str]] = []
    for match in pattern.finditer(markdown):
        if inside_fence(match.start()):
            continue
        role = normalize_role(match.group(1))
        content = clean_message_text(re.sub(r"\n\n---\n\n", "\n", match.group(2)))
        if content:
            messages.append({"role": role, "content": content})
    return messages


def parse_cli_messages(markdown: str) -> list[dict[str, str]]:
    """Heuristic parser for CLI-style chat logs such as Claude Code traces."""
    heading_pattern = re.compile(
        r"^### (User|Assistant|Agent)\s*\n(.*?)(?=^### (?:User|Assistant|Agent)\s*\n|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    messages = [
        {"role": normalize_role(m.group(1)), "content": clean_message_text(m.group(2))}
        for m in heading_pattern.finditer(markdown)
        if clean_message_text(m.group(2))
    ]
    if messages:
        return messages
    request = re.search(r"\*\*User Request:\*\*\s*(.*)", markdown, re.DOTALL)
    if request:
        return [{"role": "User", "content": clean_message_text(request.group(1))}]
    return [{"role": "Assistant", "content": clean_message_text(markdown)}] if markdown.strip() else []


def parse_chat_markdowns(output_dir: Path) -> None:
    ensure_dir(output_dir / "parsed_chats_simple")
    searches = json.loads((output_dir / "searches.json").read_text(encoding="utf-8"))
    by_sha = {str(item.get("sha")): item for item in searches if item.get("sha")}
    for path in sorted((output_dir / "markdowns").glob("*.md")):
        sha = path.stem
        item = by_sha.get(sha, {})
        markdown = path.read_text(encoding="utf-8", errors="replace")
        is_cli = "Claude Code Session Log" in markdown or "WORK SESSION" in markdown or "\n_**User" not in markdown
        messages = parse_cli_messages(markdown) if is_cli else parse_standard_specstory_messages(markdown)
        parsed = {
            "session_sha": sha,
            "title": parse_title(markdown),
            "timestamp": extract_timestamp_from_search_name(item.get("name")),
            "platform": "CLI-style" if is_cli else "SpecStory standard",
            "messages": messages,
        }
        (output_dir / "parsed_chats_simple" / f"{sha}.json").write_text(
            json.dumps(parsed, ensure_ascii=False), encoding="utf-8"
        )


def extract_commit_or_ref_from_blob_url(url: str | None) -> str | None:
    """Return the `<commit_or_ref>` component from a GitHub `/blob/<ref>/...` URL."""
    if not url:
        return None
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) >= 5 and parts[2] == "blob":
        return parts[3]
    return None


def build_session_repo_mapping(output_dir: Path) -> None:
    """Create the session-to-repository mapping used by later analyses."""
    items = json.loads((output_dir / "searches.json").read_text(encoding="utf-8"))
    rows = []
    for item in items:
        repo = item.get("repository") or {}
        rows.append(
            {
                "session_sha": item.get("sha"),
                "repo_id": repo.get("id"),
                "repo_full_name": repo.get("full_name"),
                "chat_path": item.get("path"),
                "chat_blob_url": item.get("html_url"),
                "commit_or_ref": extract_commit_or_ref_from_blob_url(item.get("html_url")),
                "session_timestamp": extract_timestamp_from_search_name(item.get("name")),
            }
        )
    # Deduplicate by session SHA while retaining the first observed repository mapping.
    seen: set[str] = set()
    deduped = []
    for row in rows:
        sha = str(row.get("session_sha"))
        if sha and sha not in seen:
            deduped.append(row)
            seen.add(sha)
    write_csv(output_dir / "session_repo_mapping.csv", deduped)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_repo_metadata(repo_full_name: str, client: GitHubClient) -> dict[str, Any] | None:
    data = client.request_json(f"{GITHUB_API}/repos/{repo_full_name}")
    if not data:
        return None
    return {
        "repo_id": data.get("id"),
        "repo_full_name": data.get("full_name"),
        "owner_login": (data.get("owner") or {}).get("login"),
        "owner_type": (data.get("owner") or {}).get("type"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "pushed_at": data.get("pushed_at"),
        "size": data.get("size"),
        "stargazers_count": data.get("stargazers_count"),
        "forks_count": data.get("forks_count"),
        "language": data.get("language"),
        "default_branch": data.get("default_branch"),
        "archived": data.get("archived"),
        "fork": data.get("fork"),
        "html_url": data.get("html_url"),
    }


def collect_commits_all_refs(repo_full_name: str, client: GitHubClient) -> list[dict[str, Any]]:
    """Collect API-visible commits from default history, branches, and tags."""
    refs: set[str | None] = {None}
    branches = client.paginate(f"{GITHUB_API}/repos/{repo_full_name}/branches")
    tags = client.paginate(f"{GITHUB_API}/repos/{repo_full_name}/tags")
    refs.update(b.get("name") for b in branches if isinstance(b, dict) and b.get("name"))
    refs.update(t.get("name") for t in tags if isinstance(t, dict) and t.get("name"))

    by_sha: dict[str, dict[str, Any]] = {}
    for ref in refs:
        base = f"{GITHUB_API}/repos/{repo_full_name}/commits"
        url = base if ref is None else f"{base}?{urlencode({'sha': ref})}"
        for row in client.paginate(url):
            sha = row.get("sha") if isinstance(row, dict) else None
            if sha:
                by_sha[str(sha)] = row
    return list(by_sha.values())


def collect_commit_detail(repo_full_name: str, sha: str, client: GitHubClient, email_salt: str = "") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detail = client.request_json(f"{GITHUB_API}/repos/{repo_full_name}/commits/{sha}")
    if not detail:
        return {}, []
    commit = detail.get("commit") or {}
    author = commit.get("author") or {}
    committer = commit.get("committer") or {}
    row = {
        "repo_full_name": repo_full_name,
        "commit_sha": detail.get("sha"),
        "commit_timestamp": author.get("date") or committer.get("date"),
        "commit_author_login": (detail.get("author") or {}).get("login"),
        "commit_author_name": author.get("name"),
        "commit_author_email_hash": hash_email(author.get("email"), salt=email_salt),
        "committer_login": (detail.get("committer") or {}).get("login"),
        "commit_message": commit.get("message"),
        "parent_count": len(detail.get("parents") or []),
        "html_url": detail.get("html_url"),
        "api_url": detail.get("url"),
    }
    file_rows: list[dict[str, Any]] = []
    for file_obj in detail.get("files") or []:
        file_path = file_obj.get("filename")
        category = classify_path(file_path)
        file_rows.append(
            {
                "repo_full_name": repo_full_name,
                "commit_sha": detail.get("sha"),
                "commit_timestamp": row["commit_timestamp"],
                "file_path": file_path,
                "status": file_obj.get("status"),
                "additions": file_obj.get("additions"),
                "deletions": file_obj.get("deletions"),
                "changes": file_obj.get("changes"),
                "previous_filename": file_obj.get("previous_filename"),
                "raw_url": file_obj.get("raw_url"),
                "blob_url": file_obj.get("blob_url"),
                "patch_available": bool(file_obj.get("patch")),
                "path_category": category,
                **file_flags(category),
            }
        )
    return row, file_rows


def collect_issues(repo_full_name: str, client: GitHubClient) -> list[dict[str, Any]]:
    rows = []
    for item in client.paginate(f"{GITHUB_API}/repos/{repo_full_name}/issues?state=all"):
        rows.append(
            {
                "repo_full_name": repo_full_name,
                "issue_number": item.get("number"),
                "issue_id": item.get("id"),
                "title": item.get("title"),
                "state": item.get("state"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "closed_at": item.get("closed_at"),
                "user_login": (item.get("user") or {}).get("login"),
                "author_association": item.get("author_association"),
                "labels": json.dumps([lab.get("name") for lab in item.get("labels") or []]),
                "comments_count": item.get("comments"),
                "is_pull_request": "pull_request" in item,
                "html_url": item.get("html_url"),
            }
        )
    return rows


def collect_pull_requests(repo_full_name: str, client: GitHubClient) -> list[dict[str, Any]]:
    rows = []
    for pr in client.paginate(f"{GITHUB_API}/repos/{repo_full_name}/pulls?state=all"):
        rows.append(
            {
                "repo_full_name": repo_full_name,
                "pr_number": pr.get("number"),
                "pr_id": pr.get("id"),
                "title": pr.get("title"),
                "state": pr.get("state"),
                "created_at": pr.get("created_at"),
                "updated_at": pr.get("updated_at"),
                "closed_at": pr.get("closed_at"),
                "merged_at": pr.get("merged_at"),
                "merged": bool(pr.get("merged_at")),
                "user_login": (pr.get("user") or {}).get("login"),
                "author_association": pr.get("author_association"),
                "base_branch": (pr.get("base") or {}).get("ref"),
                "head_branch": (pr.get("head") or {}).get("ref"),
                "head_repo": ((pr.get("head") or {}).get("repo") or {}).get("full_name"),
                "merge_commit_sha": pr.get("merge_commit_sha"),
                "comments_count": pr.get("comments"),
                "review_comments_count": pr.get("review_comments"),
                "html_url": pr.get("html_url"),
            }
        )
    return rows


def collect_comments_and_reviews(repo_full_name: str, client: GitHubClient) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comments: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []

    for item in client.paginate(f"{GITHUB_API}/repos/{repo_full_name}/issues/comments"):
        comments.append(
            {
                "repo_full_name": repo_full_name,
                "comment_id": item.get("id"),
                "comment_type": "issue_or_pr_comment",
                "user_login": (item.get("user") or {}).get("login"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "body_length": len(item.get("body") or ""),
                "author_association": item.get("author_association"),
                "html_url": item.get("html_url"),
            }
        )

    for pr in collect_pull_requests(repo_full_name, client):
        number = pr.get("pr_number")
        if not number:
            continue
        for item in client.paginate(f"{GITHUB_API}/repos/{repo_full_name}/pulls/{number}/comments"):
            comments.append(
                {
                    "repo_full_name": repo_full_name,
                    "comment_id": item.get("id"),
                    "comment_type": "pr_review_comment",
                    "user_login": (item.get("user") or {}).get("login"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "body_length": len(item.get("body") or ""),
                    "author_association": item.get("author_association"),
                    "html_url": item.get("html_url"),
                }
            )
        for review in client.paginate(f"{GITHUB_API}/repos/{repo_full_name}/pulls/{number}/reviews"):
            reviews.append(
                {
                    "repo_full_name": repo_full_name,
                    "review_id": review.get("id"),
                    "pr_number": number,
                    "comment_type": "pr_review",
                    "user_login": (review.get("user") or {}).get("login"),
                    "created_at": review.get("submitted_at"),
                    "updated_at": review.get("submitted_at"),
                    "body_length": len(review.get("body") or ""),
                    "author_association": review.get("author_association"),
                    "state": review.get("state"),
                    "html_url": review.get("html_url"),
                }
            )
    return comments, reviews


def collect_ci_checks(repo_full_name: str, commit_shas: Iterable[str], client: GitHubClient) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sha in commit_shas:
        data = client.request_json(f"{GITHUB_API}/repos/{repo_full_name}/commits/{sha}/check-runs")
        if isinstance(data, dict):
            for check in data.get("check_runs") or []:
                rows.append(
                    {
                        "repo_full_name": repo_full_name,
                        "commit_sha": sha,
                        "check_type": "check_run",
                        "name_context": check.get("name"),
                        "status": check.get("status"),
                        "conclusion_state": check.get("conclusion"),
                        "started_at": check.get("started_at"),
                        "completed_at": check.get("completed_at"),
                        "created_at": check.get("created_at"),
                        "updated_at": check.get("updated_at"),
                        "html_url": check.get("html_url"),
                        "app_name": (check.get("app") or {}).get("name"),
                    }
                )
    return rows


def collect_repository_history(repo_full_name: str, output_dir: Path, client: GitHubClient, email_salt: str = "") -> None:
    """Collect raw and tabular history for one repository."""
    repo_dir = output_dir / "raw_collected" / repo_full_name.replace("/", "__")
    ensure_dir(repo_dir)

    metadata = collect_repo_metadata(repo_full_name, client)
    if metadata:
        (repo_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    commit_refs = collect_commits_all_refs(repo_full_name, client)
    commit_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for ref in commit_refs:
        sha = ref.get("sha")
        if not sha:
            continue
        commit_row, commit_files = collect_commit_detail(repo_full_name, sha, client, email_salt=email_salt)
        if commit_row:
            commit_rows.append(commit_row)
            file_rows.extend(commit_files)

    write_csv(repo_dir / "commits.csv", commit_rows)
    write_csv(repo_dir / "commit_files.csv", file_rows)
    write_csv(repo_dir / "issues.csv", collect_issues(repo_full_name, client))
    write_csv(repo_dir / "pull_requests.csv", collect_pull_requests(repo_full_name, client))
    comments, reviews = collect_comments_and_reviews(repo_full_name, client)
    write_csv(repo_dir / "comments.csv", comments)
    write_csv(repo_dir / "pr_reviews.csv", reviews)
    write_csv(repo_dir / "ci_checks.csv", collect_ci_checks(repo_full_name, [r["commit_sha"] for r in commit_rows], client))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data_collection_output"))
    parser.add_argument("--prefixes-file", type=Path, help="One GitHub Code Search date prefix per line.")
    parser.add_argument("--repo-list", type=Path, help="CSV with repo_full_name for repository-history collection.")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-repo-history", action="store_true")
    parser.add_argument("--email-salt", default=os.environ.get("EMAIL_HASH_SALT", ""))
    args = parser.parse_args()

    client = GitHubClient(read_tokens())
    ensure_dir(args.output_dir)

    if not args.skip_search:
        if not args.prefixes_file:
            raise SystemExit("--prefixes-file is required unless --skip-search is set")
        prefixes = [line.strip() for line in args.prefixes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        scrape_specstory_searches(prefixes, args.output_dir, client)
        download_chat_blobs(args.output_dir, client)
        parse_chat_markdowns(args.output_dir)
        build_session_repo_mapping(args.output_dir)

    if not args.skip_repo_history:
        if not args.repo_list:
            raise SystemExit("--repo-list is required unless --skip-repo-history is set")
        with args.repo_list.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            repos = sorted({row["repo_full_name"] for row in reader if row.get("repo_full_name")})
        for repo in repos:
            collect_repository_history(repo, args.output_dir, client, email_salt=args.email_salt)


if __name__ == "__main__":
    main()
