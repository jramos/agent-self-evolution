"""Import session data from external AI tools into golden eval datasets.

Bridges the gap between existing tool usage (Claude Code, GitHub Copilot)
and Hermes self-evolution by mining real session history for skill-relevant
evaluation examples. Solves the cold-start problem: new Hermes users don't
have golden datasets, but they do have session history from tools they
already use.

Supported sources:
  - Claude Code (~/.claude/history.jsonl) — user inputs only
  - GitHub Copilot (~/.copilot/session-state/*/events.jsonl) — full conversations
  - Hermes Agent (~/.hermes/sessions/*.json) — user + assistant + tool context

Usage as standalone CLI:
    python -m evolution.core.external_importers \\
        --source all --skill my-skill --dry-run

    python -m evolution.core.external_importers \\
        --source claude-code --skill my-skill --model openrouter/google/gemini-2.5-flash

Usage from evolve_skill.py:
    python -m evolution.skills.evolve_skill --skill my-skill --eval-source sessiondb
"""

import json
import re
from functools import lru_cache
import random
import sqlite3
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.progress import Progress

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalExample, EvalDataset, split_examples
from evolution.core.hermes_provider import resolve_default_lm

console = Console()

# Patterns are anchored to known key formats so prose-mining doesn't drop
# legitimate text containing the substring "password" or "token".
SECRET_PATTERNS = re.compile(
    r'('
    r'sk-ant-api\S+'           # Anthropic API keys
    r'|sk-or-v1-\S+'          # OpenRouter API keys
    r'|sk-\S{20,}'            # Generic OpenAI-style keys (20+ chars after sk-)
    r'|ghp_\S+'               # GitHub personal access tokens
    r'|ghu_\S+'               # GitHub user tokens
    r'|xoxb-\S+'              # Slack bot tokens
    r'|xapp-\S+'              # Slack app tokens
    r'|ntn_\S+'               # Notion integration tokens
    r'|AKIA[0-9A-Z]{16}'      # AWS access key IDs
    r'|Bearer\s+\S{20,}'      # Bearer auth headers (20+ char tokens)
    r'|-----BEGIN\s+(RSA\s+)?PRIVATE\sKEY-----'  # PEM private keys
    r'|ANTHROPIC_API_KEY'      # Known env var names (exact match)
    r'|OPENAI_API_KEY'
    r'|OPENROUTER_API_KEY'
    r'|SLACK_BOT_TOKEN'
    r'|GITHUB_TOKEN'
    r'|AWS_SECRET_ACCESS_KEY'
    r'|DATABASE_URL'
    r'|\bpassword\s*[=:]\s*\S+' # password assignments (password=xxx, password: xxx)
    r'|\bsecret\s*[=:]\s*\S+'   # secret assignments (secret=xxx, secret: xxx)
    r'|\btoken\s*[=:]\s*\S{10,}' # token assignments with 10+ char values
    r')',
    re.IGNORECASE,
)


VALID_DIFFICULTIES = {"easy", "medium", "hard"}

MIN_DATASET_SIZE = 3  # Minimum examples needed to produce a meaningful split


def contains_secret(text: str) -> bool:
    """Check if text contains potential API keys or tokens.

    Public helper — `evolution.tools.session_mining` imports it for the
    tool-mining path's secret scrub.
    """
    return bool(SECRET_PATTERNS.search(text))


def _validate_eval_example(
    task_input: str,
    expected_behavior: str,
    difficulty: str,
    category: str,
) -> Optional[dict]:
    """Validate and normalize fields before creating an EvalExample.

    Returns:
        Dict of validated fields, or None if the example should be skipped.
    """
    if not task_input or not task_input.strip():
        return None
    if not expected_behavior or not expected_behavior.strip():
        return None

    difficulty = difficulty.strip().lower() if difficulty else "medium"
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "medium"

    category = category.strip() if category else "general"
    if not category:
        category = "general"

    task_input = task_input[:2000]

    return {
        "task_input": task_input,
        "expected_behavior": expected_behavior.strip(),
        "difficulty": difficulty,
        "category": category,
    }


@lru_cache(maxsize=8)
def _skill_keywords(skill_text: str) -> frozenset[str]:
    """Vocabulary a message can overlap with, drawn from the skill's opening.

    Cached because it depends only on ``skill_text`` while its caller runs once
    per message. The previous boolean pre-filter short-circuited before building
    this set whenever the skill name matched; scoring every tier removes that
    escape, so without the cache a large corpus would rebuild an identical set
    tens of thousands of times.
    """
    keywords = set()
    for word in skill_text[:500].lower().split():
        word = re.sub(r'[^a-z]', '', word)
        if len(word) > 4:
            keywords.add(word)
    return frozenset(keywords)


def _relevance_score(text: str, skill_name: str, skill_text: str) -> tuple[int, int, int]:
    """Graded relevance of a message to a skill, strongest signal first.

    Returns ``(name_match, name_words, keyword_overlap)`` so candidates sort
    lexicographically by signal tier. A tuple rather than a weighted sum,
    because ``skill_name`` is caller-supplied and unbounded in length: no fixed
    set of weights can stop a long name's word count from outranking a
    full-name match.

    An all-zero tuple means "not relevant". The tiers reproduce the conditions
    of the original boolean pre-filter exactly, so scoring changes only the
    *order* of candidates, never which ones qualify. Note that
    ``keyword_overlap`` is thresholded rather than raw: it is 0 unless at least
    two keywords overlap, matching the pre-filter's requirement.
    """
    text_lower = text.lower()
    skill_lower = skill_name.lower().replace("-", " ").replace("_", " ")

    name_match = 1 if skill_lower in text_lower else 0

    # Words ≤ 3 chars are skipped to avoid matching "run", "use", etc.
    name_words = sum(
        1 for word in skill_lower.split()
        if len(word) > 3 and word in text_lower
    )

    message_words = set(re.sub(r'[^a-z\s]', '', text_lower).split())
    skill_keywords = _skill_keywords(skill_text)
    overlap = len(message_words & skill_keywords)
    # One overlapping keyword was never a match; awarding partial credit for it
    # would widen the qualifying set rather than reorder it.
    keyword_overlap = overlap if overlap >= 2 else 0

    return (name_match, name_words, keyword_overlap)


def _is_relevant_to_skill(text: str, skill_name: str, skill_text: str) -> bool:
    """Boolean view of :func:`_relevance_score`: does this message qualify at all?

    Has no production callers — :meth:`RelevanceFilter.filter_and_score` needs the
    graded score for ordering. It is kept as the equivalence surface for the
    pre-filter's original tests, which are what pin the guarantee that adding the
    score changed only candidate *order* and never which messages qualify. Delete
    it and that guarantee stops being tested.
    """
    return any(_relevance_score(text, skill_name, skill_text))


class ClaudeCodeImporter:
    """Import user prompts from Claude Code history.jsonl.

    Claude Code stores a flat JSONL of user messages at ~/.claude/history.jsonl.
    Each line has: display (user text), timestamp, project, sessionId.
    Only user inputs are available — no assistant responses.
    """

    HISTORY_PATH = Path.home() / ".claude" / "history.jsonl"

    @staticmethod
    def extract_messages(limit: int = 0) -> list[dict]:
        """Read user messages from Claude Code history.

        Args:
            limit: Maximum messages to return (0 = no limit).

        Returns:
            List of dicts with keys: source, task_input, project, session_id, timestamp.
        """
        if not ClaudeCodeImporter.HISTORY_PATH.exists():
            return []

        messages = []
        with open(ClaudeCodeImporter.HISTORY_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = entry.get("display", "")
                if not text or len(text) < 10:
                    continue
                if contains_secret(text):
                    continue

                messages.append({
                    "source": "claude-code",
                    "task_input": text,
                    "project": entry.get("project", ""),
                    "session_id": entry.get("sessionId", ""),
                    "timestamp": entry.get("timestamp", 0),
                })

                if limit and len(messages) >= limit:
                    break

        return messages


class CopilotImporter:
    """Import conversations from GitHub Copilot session events.

    Copilot stores sessions at ~/.copilot/session-state/<session-id>/.
    Each session has workspace.yaml (project context) and events.jsonl
    (chronological stream of user.message / assistant.message events).
    Files can be 100MB+ so we stream line-by-line.

    Note: This path is the default Copilot CLI session storage location.
    Override SESSION_DIR for non-standard installations.
    """

    SESSION_DIR = Path.home() / ".copilot" / "session-state"

    @staticmethod
    def extract_messages(limit: int = 0) -> list[dict]:
        """Read user/assistant message pairs from Copilot sessions.

        Args:
            limit: Maximum messages to return (0 = no limit).

        Returns:
            List of dicts with keys: source, task_input, assistant_response,
            project, session_id.
        """
        if not CopilotImporter.SESSION_DIR.exists():
            return []

        messages = []
        event_files = list(CopilotImporter.SESSION_DIR.glob("*/events.jsonl"))

        with Progress() as progress:
            task = progress.add_task("Reading Copilot sessions...", total=len(event_files))

            for events_path in event_files:
                session_id = events_path.parent.name
                project = _read_copilot_workspace(events_path.parent / "workspace.yaml")

                pairs = _parse_copilot_events(events_path, session_id, project)
                messages.extend(pairs)

                progress.update(task, advance=1)

                if limit and len(messages) >= limit:
                    messages = messages[:limit]
                    break

        return messages


def _read_copilot_workspace(workspace_path: Path) -> str:
    """Extract cwd from a Copilot workspace.yaml file."""
    if not workspace_path.exists():
        return ""
    try:
        for line in workspace_path.read_text().split("\n"):
            if line.startswith("cwd:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def _parse_copilot_events(
    events_path: Path, session_id: str, project: str,
) -> list[dict]:
    """Parse a single Copilot events.jsonl into user/assistant pairs."""
    pairs = []
    current_user_msg = None
    current_assistant_msg = None

    try:
        with open(events_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")
                data = event.get("data", {})

                if event_type == "user.message":
                    if current_user_msg and current_assistant_msg:
                        if not contains_secret(current_user_msg) and not contains_secret(current_assistant_msg):
                            pairs.append({
                                "source": "copilot",
                                "task_input": current_user_msg,
                                "assistant_response": current_assistant_msg,
                                "project": project,
                                "session_id": session_id,
                            })

                    current_user_msg = data.get("content", "")
                    current_assistant_msg = None

                elif event_type == "assistant.message":
                    content = data.get("content", "")
                    if content and current_user_msg:
                        if current_assistant_msg:
                            current_assistant_msg += "\n" + content
                        else:
                            current_assistant_msg = content

        if current_user_msg and current_assistant_msg:
            if not contains_secret(current_user_msg) and not contains_secret(current_assistant_msg):
                pairs.append({
                    "source": "copilot",
                    "task_input": current_user_msg,
                    "assistant_response": current_assistant_msg,
                    "project": project,
                    "session_id": session_id,
                })

    except Exception as e:
        console.print(f"[dim]Skipped {session_id}: {e}[/dim]")

    return pairs


class HermesSessionImporter:
    """Import conversations from Hermes Agent session files.

    Hermes stores session transcripts as JSON files in ~/.hermes/sessions/.
    Each file contains an OpenAI-format message list with user, assistant,
    and tool messages — providing richer signal than Claude Code (user-only)
    or Copilot (user+assistant without tool context).

    This mines user messages paired with the assistant's final response,
    giving the LLM judge both the task and how it was actually handled.
    """

    SESSION_DIR = Path.home() / ".hermes" / "sessions"
    STATE_DB = Path.home() / ".hermes" / "state.db"

    @staticmethod
    def extract_messages(limit: int = 0) -> list[dict]:
        """Read user/assistant pairs from Hermes session files.

        Args:
            limit: Maximum messages to return (0 = no limit).

        Returns:
            List of dicts with keys: source, task_input, assistant_response,
            session_id.
        """
        messages = []
        for session_id, msg_list in iter_hermes_sessions():
            for i, msg in enumerate(msg_list):
                if msg.get("role") != "user":
                    continue
                user_text = msg.get("content", "")
                if not user_text or len(user_text) < 10:
                    continue
                if contains_secret(user_text):
                    continue

                assistant_text = ""
                for j in range(i + 1, len(msg_list)):
                    if msg_list[j].get("role") == "assistant":
                        content = msg_list[j].get("content", "")
                        if content:
                            assistant_text = content
                            break
                    elif msg_list[j].get("role") == "user":
                        break

                if assistant_text and contains_secret(assistant_text):
                    continue

                messages.append({
                    "source": "hermes",
                    "task_input": user_text,
                    "assistant_response": assistant_text,
                    "session_id": session_id,
                })

                if limit and len(messages) >= limit:
                    return messages

        return messages


# Hermes prepends a model-switch note onto the user's turn (not a standalone
# message), e.g. "[Note: model was just switched from X to Y ...]\n\n<real text>".
# The note carries no nested ']'. Strip it and keep the genuine instruction.
_MODEL_SWITCH_NOTE = re.compile(r"^\s*\[Note: model was just switched from[^\]]*\]\s*")


def _strip_model_switch_note(content: str) -> str:
    """Drop a leading Hermes model-switch note, preserving the real user text."""
    return _MODEL_SWITCH_NOTE.sub("", content, count=1)


def _iter_hermes_sessions_from_db(db_path: Path) -> list[tuple[str, list[dict]]]:
    """Read ``(session_id, messages)`` for every session in a Hermes ``state.db``.

    ``state.db`` is the canonical store — modern ``hermes`` persists sessions here,
    not to ``sessions/*.json``. Mines ALL sessions: it never reads or filters
    ``sessions.source`` (that column records launch origin, not relevance; filtering
    it would starve mining). Each message is ``{"role", "content", "tool_calls"}``
    with ``tool_calls`` decoded to a list (``None`` if absent/malformed) — the same
    shape the JSON path yields, so both the skill and tool consumers are unchanged.
    Read-only; any SQLite error abstains (returns ``[]``) so a locked/corrupt db
    falls back to the JSON path.
    """
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    out: list[tuple[str, list[dict]]] = []
    try:
        conn.row_factory = sqlite3.Row
        session_rows = conn.execute(
            "SELECT id FROM sessions ORDER BY started_at DESC"
        ).fetchall()
        for srow in session_rows:
            session_id = srow["id"]
            msg_rows = conn.execute(
                "SELECT role, content, tool_calls FROM messages "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            msgs: list[dict] = []
            for m in msg_rows:
                role = m["role"]
                if role not in ("user", "assistant"):
                    continue
                content = m["content"] or ""
                if role == "user":
                    content = _strip_model_switch_note(content)
                raw_tc = m["tool_calls"]
                tool_calls = None
                if raw_tc:
                    try:
                        tool_calls = json.loads(raw_tc)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        tool_calls = None
                msgs.append({"role": role, "content": content, "tool_calls": tool_calls})
            if msgs:
                out.append((session_id, msgs))
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return out


def iter_hermes_sessions():
    """Yield ``(session_id, messages)`` for each Hermes session.

    Reads the canonical SQLite ``state.db`` first; if it yields no sessions
    (absent, empty, or unreadable) falls back to the legacy
    ``~/.hermes/sessions/*.json`` files (newest-first by mtime). The yielded
    ``messages`` is a list of ``{"role", "content", "tool_calls"}`` dicts —
    callers do their own pair / tool-call extraction.

    Shared by ``HermesSessionImporter`` (skill-path) and
    ``evolution.tools.session_mining`` (tool-path).
    """
    db_sessions = _iter_hermes_sessions_from_db(HermesSessionImporter.STATE_DB)
    if db_sessions:
        yield from db_sessions
        return

    if not HermesSessionImporter.SESSION_DIR.exists():
        return

    session_files = sorted(
        HermesSessionImporter.SESSION_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,  # newest first
    )

    for session_file in session_files:
        try:
            data = json.loads(session_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        msg_list = data.get("messages", [])
        if not msg_list:
            continue

        session_id = data.get("session_id", session_file.stem)
        yield session_id, msg_list


class RelevanceFilter:
    """Use LLM-as-judge to determine which messages are relevant to a skill.

    Two-stage pipeline:
      1. Cheap heuristic pre-filter, graded for ordering (_relevance_score)
      2. LLM scoring for final relevance + eval metadata generation
    """

    class ScoreRelevance(dspy.Signature):
        """Score whether a user message is relevant to a specific agent skill.

        Return a JSON object with:
        - relevant: boolean (true if the message relates to what this skill does)
        - expected_behavior: string (if relevant, what should a good response do?)
        - difficulty: string (easy, medium, or hard)
        - category: string (what aspect of the skill this tests)
        """
        skill_name: str = dspy.InputField(desc="Name of the skill")
        skill_description: str = dspy.InputField(desc="First 800 chars of the skill file")
        user_message: str = dspy.InputField(desc="The user's message to evaluate")
        assistant_response: str = dspy.InputField(desc="The assistant's actual response (may be empty)")
        scoring: str = dspy.OutputField(desc="JSON object with: relevant, expected_behavior, difficulty, category")

    # Class-level default load-bearing for tests that bypass __init__ via
    # __new__; runtime instances overwrite it.
    seed: int = 42

    def __init__(self, model: str, seed: int = 42):
        self.scorer = dspy.ChainOfThought(self.ScoreRelevance)
        self.model = model
        self.seed = seed

    def filter_and_score(
        self,
        messages: list[dict],
        skill_name: str,
        skill_text: str,
        max_examples: int = 50,
    ) -> list[EvalExample]:
        """Filter messages by relevance and generate eval examples.

        Args:
            messages: Raw messages from importers.
            skill_name: Name of the target skill.
            skill_text: Full text of the SKILL.md file.
            max_examples: Maximum eval examples to produce.

        Returns:
            List of EvalExample objects for relevant messages.
        """
        skill_desc = skill_text[:800]

        messages = [m for m in messages if m.get("task_input") and m.get("source")]

        # Strongest first. The candidate-count cap (max_examples * 3) and the
        # examples-count cap (the scoring loop's early break) both consume this
        # list in order, so its ordering decides which messages ever reach the
        # LLM scorer. sorted() is stable, so equally-scored messages keep their
        # import order and source priority survives the caps.
        scored = [
            (_relevance_score(m["task_input"], skill_name, skill_text), m)
            for m in messages
        ]
        candidates = [
            m for _, m in sorted(
                (pair for pair in scored if any(pair[0])),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ]

        # Backfill from random non-matching messages so the LLM sees a useful
        # sample even when the cheap heuristic misses everything.
        if len(candidates) < max_examples:
            candidate_ids = {id(m) for m in candidates}
            remaining = [m for m in messages if id(m) not in candidate_ids]
            random.Random(self.seed).shuffle(remaining)
            candidates.extend(remaining[:max_examples * 2])

        candidates = candidates[:max_examples * 3]

        console.print(f"  Pre-filtered to {len(candidates)} candidates (from {len(messages)} total)")

        examples = []
        errors = 0
        _lm = resolve_default_lm(role="judge", explicit_model=self.model)
        lm = dspy.LM(_lm.model, **_lm.lm_kwargs, temperature=0.0, max_tokens=2000)

        with Progress() as progress:
            task = progress.add_task("Scoring relevance...", total=len(candidates))

            for msg in candidates:
                try:
                    with dspy.context(lm=lm):
                        result = self.scorer(
                            skill_name=skill_name,
                            skill_description=skill_desc,
                            user_message=msg["task_input"][:1000],
                            assistant_response=msg.get("assistant_response", "")[:1000],
                        )

                    scoring = parse_scoring_json(result.scoring)
                    if scoring is None:
                        errors += 1
                        progress.update(task, advance=1)
                        continue

                    if scoring.get("relevant", False):
                        validated = _validate_eval_example(
                            task_input=msg["task_input"],
                            expected_behavior=scoring.get("expected_behavior", ""),
                            difficulty=scoring.get("difficulty", "medium"),
                            category=scoring.get("category", "general"),
                        )
                        if validated:
                            examples.append(EvalExample(
                                source=msg["source"],
                                **validated,
                            ))

                except Exception:
                    errors += 1

                progress.update(task, advance=1)

                if len(examples) >= max_examples:
                    break

        total_scored = len(candidates)
        if errors > 0:
            console.print(
                f"  [yellow]LLM scoring: {errors}/{total_scored} failed "
                f"({errors / max(1, total_scored) * 100:.0f}% error rate)[/yellow]"
            )

        return examples


def parse_scoring_json(text: str) -> Optional[dict]:
    """Extract a JSON object from LLM scoring output.

    First tries ``json.loads`` for clean output, then falls back to a
    balanced-brace walk. Regex extraction was rejected — ``r'\\{[^}]+\\}'``
    breaks on nested braces (e.g. "handle {edge} cases" inside a string).
    Returns None when no valid JSON object is found.

    Public helper — `evolution.tools.session_mining` imports it for the
    tool-mining path's judge-response parse.
    """
    if not text:
        return None

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None

    return None


def build_dataset_from_external(
    skill_name: str,
    skill_text: str,
    sources: list[str],
    output_path: Path,
    model: str,
    max_examples: int = 50,
    seed: int = 42,
) -> EvalDataset:
    """Extract messages from external tools, filter for relevance, and save.

    This is the main entry point called by both the standalone CLI and
    evolve_skill.py when --eval-source sessiondb is used.

    Args:
        skill_name: Name of the target skill.
        skill_text: Full text of the SKILL.md file.
        sources: List of source names ("claude-code", "copilot").
        output_path: Directory to write train/val/holdout JSONL files.
        model: LiteLLM model string for relevance scoring.
        max_examples: Maximum eval examples to generate.

    Returns:
        EvalDataset with train/val/holdout splits.
    """
    all_messages = []

    importers = {
        "claude-code": ("Claude Code", ClaudeCodeImporter),
        "copilot": ("Copilot", CopilotImporter),
        "hermes": ("Hermes Agent", HermesSessionImporter),
    }

    for source in sources:
        if source not in importers:
            continue
        label, importer_cls = importers[source]
        console.print(f"\n[bold]Importing from {label}...[/bold]")
        msgs = importer_cls.extract_messages()
        console.print(f"  Found {len(msgs)} messages")
        all_messages.extend(msgs)

    if not all_messages:
        console.print("[red]No messages found from any source.[/red]")
        return EvalDataset()

    console.print(f"\n[bold]Total messages: {len(all_messages)}[/bold]")
    console.print(f"[bold]Filtering for relevance to skill: {skill_name}[/bold]")

    relevance_filter = RelevanceFilter(model=model, seed=seed)
    examples = relevance_filter.filter_and_score(
        all_messages, skill_name, skill_text, max_examples=max_examples,
    )

    console.print(f"\n[bold green]Found {len(examples)} relevant examples[/bold green]")

    if not examples:
        console.print("[yellow]No relevant examples found. Try a different skill or broader sources.[/yellow]")
        return EvalDataset()

    if len(examples) < MIN_DATASET_SIZE:
        console.print(
            f"[yellow]⚠ Only {len(examples)} examples found (minimum {MIN_DATASET_SIZE} "
            f"recommended for meaningful train/val/holdout split)[/yellow]"
        )

    # Source ratios from EvolutionConfig defaults so synthetic + sessiondb
    # produce the same splits at the same N (was: hardcoded 50/25/25).
    config = EvolutionConfig()
    dataset = split_examples(
        examples,
        seed=seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        holdout_ratio=config.holdout_ratio,
    )

    dataset.save(output_path)
    console.print(f"\n[bold]Saved to {output_path}/[/bold]")
    console.print(f"  train: {len(dataset.train)}  val: {len(dataset.val)}  holdout: {len(dataset.holdout)}")

    source_counts: dict[str, int] = {}
    for ex in examples:
        source_counts[ex.source] = source_counts.get(ex.source, 0) + 1
    for src, count in sorted(source_counts.items()):
        console.print(f"  {src}: {count}")

    return dataset


def _load_skill_text(skill_name: str, skills_dir: Optional[Path] = None) -> tuple[str, str]:
    """Load skill text from the installed Hermes skills directory.

    This is used by the standalone CLI only. When called via evolve_skill.py,
    skill loading goes through skill_module.find_skill() + load_skill() instead,
    which searches the hermes-agent repo path rather than installed skills.

    Args:
        skill_name: Name of the skill directory.
        skills_dir: Override skills directory (default: ~/.hermes/skills).

    Returns:
        Tuple of (skill_name, skill_file_contents).

    Raises:
        FileNotFoundError: If no SKILL.md found for the given name.
    """
    if skills_dir is None:
        skills_dir = Path.home() / ".hermes" / "skills"

    for pattern in [skill_name, f"*/{skill_name}"]:
        for skill_dir in skills_dir.glob(pattern):
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                return skill_name, skill_file.read_text()

    raise FileNotFoundError(f"Skill '{skill_name}' not found in {skills_dir}")


@click.command()
@click.option(
    "--source",
    type=click.Choice(["claude-code", "copilot", "hermes", "all"]),
    default="all",
    help="Which tool to import from",
)
@click.option("--skill", required=True, help="Skill name to generate eval data for")
@click.option("--output", type=click.Path(), default=None,
              help="Output directory (default: datasets/skills/<skill>/)")
@click.option("--model", default="openrouter/google/gemini-2.5-flash",
              help="LiteLLM model string for relevance scoring")
@click.option("--max-examples", default=50, help="Max eval examples to generate")
@click.option("--dry-run", is_flag=True, help="Show message counts without LLM scoring")
def main(source, skill, output, model, max_examples, dry_run):
    """Import external session data into golden eval datasets for self-evolution."""
    console.print(f"\n[bold cyan]External Session Importer[/bold cyan] — skill: [bold]{skill}[/bold]\n")

    try:
        skill_name, skill_text = _load_skill_text(skill)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    console.print(f"  Loaded skill: {skill_name} ({len(skill_text):,} chars)")

    sources = [source] if source != "all" else ["claude-code", "copilot", "hermes"]

    if dry_run:
        importers = {
            "claude-code": ClaudeCodeImporter,
            "copilot": CopilotImporter,
            "hermes": HermesSessionImporter,
        }
        for src in sources:
            msgs = importers[src].extract_messages()
            console.print(f"  {src}: {len(msgs)} messages")
        console.print("\n[bold green]DRY RUN — no LLM calls made.[/bold green]")
        return

    if output is None:
        output = Path(__file__).parent.parent.parent / "datasets" / "skills" / skill_name
    else:
        output = Path(output)

    build_dataset_from_external(
        skill_name=skill_name,
        skill_text=skill_text,
        sources=sources,
        output_path=output,
        model=model,
        max_examples=max_examples,
    )


if __name__ == "__main__":
    main()
