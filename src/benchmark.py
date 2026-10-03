from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return proportion (0.0 to 1.0) of expected facts that appear in answer."""
    if not expected:
        return 1.0
    lower_ans = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in lower_ans)
    return round(matches / len(expected), 2)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline / benchmark mode."""
    rec = recall_points(answer, expected)
    length = len(answer.strip())
    # Length appropriateness: between 30 and 400 chars is good
    len_score = 0.3 if 30 <= length <= 600 else 0.1
    # Structure bonus: presence of bullet points or clear sentences
    struct_score = 0.2 if ("-" in answer or "\n" in answer) else 0.1
    quality = (0.5 * rec) + len_score + struct_score
    return round(min(1.0, max(0.0, quality)), 2)


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over multiple conversations and recall questions.

    Flow:
    1. Feed all turns to the agent per conversation.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a brand-new thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """
    all_threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    users_seen: set[str] = set()

    for conv in conversations:
        user_id = conv["user_id"]
        users_seen.add(user_id)
        conv_id = conv["id"]
        all_threads.append(conv_id)

        # 1. Feed regular conversation turns
        for turn in conv.get("turns", []):
            agent.reply(user_id=user_id, thread_id=conv_id, message=turn)

        # 2. Ask recall questions in fresh threads (testing cross-session memory)
        recall_questions = conv.get("recall_questions", [])
        for q_idx, q_item in enumerate(recall_questions):
            fresh_thread = f"{conv_id}-recall-{q_idx}"
            all_threads.append(fresh_thread)
            res = agent.reply(
                user_id=user_id,
                thread_id=fresh_thread,
                message=q_item["question"],
            )
            ans = res.get("response", "")
            rec = recall_points(ans, q_item.get("expected_contains", []))
            qual = heuristic_quality(ans, q_item.get("expected_contains", []))
            recall_scores.append(rec)
            quality_scores.append(qual)

    # Calculate token metrics across all threads
    total_agent_tokens = sum(agent.token_usage(t) for t in all_threads)
    total_prompt_tokens = sum(agent.prompt_token_usage(t) for t in all_threads)
    total_compactions = sum(agent.compaction_count(t) for t in all_threads)

    # Memory growth
    if hasattr(agent, "memory_file_size"):
        memory_bytes = sum(agent.memory_file_size(u) for u in users_seen)
    else:
        memory_bytes = 0

    avg_recall = round(sum(recall_scores) / len(recall_scores), 2) if recall_scores else 0.0
    avg_quality = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_bytes,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{int(r.recall_score * 100)}%",
            r.response_quality,
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both standard benchmark and long-context stress benchmark."""
    config = load_config(Path(__file__).resolve().parent.parent)

    std_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    print("=" * 80)
    print("PHASE 2, TRACK 3, DAY 17: MEMORY SYSTEMS BENCHMARK")
    print("=" * 80)

    # 1. Standard Benchmark Suite
    print("\n### 1. Standard Benchmark (10 conversations, user 'dungct')")
    std_convs = load_conversations(std_path)

    # Baseline Agent
    baseline_agent_std = BaselineAgent(config, force_offline=True)
    row_base_std = run_agent_benchmark("Baseline Agent", baseline_agent_std, std_convs, config)

    # Clean profiles before running advanced
    profiles_dir = config.state_dir / "profiles"
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)

    # Advanced Agent
    advanced_agent_std = AdvancedAgent(config, force_offline=True)
    row_adv_std = run_agent_benchmark("Advanced Agent", advanced_agent_std, std_convs, config)

    print(format_rows([row_base_std, row_adv_std]))

    # 2. Long-Context Stress Benchmark Suite
    print("\n### 2. Long-Context Stress Benchmark (16-turn heavy thread, user 'dungct_stress')")
    stress_convs = load_conversations(stress_path)

    # Baseline Agent
    baseline_agent_stress = BaselineAgent(config, force_offline=True)
    row_base_stress = run_agent_benchmark("Baseline Agent", baseline_agent_stress, stress_convs, config)

    # Clean profiles before running advanced stress
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)

    # Advanced Agent
    advanced_agent_stress = AdvancedAgent(config, force_offline=True)
    row_adv_stress = run_agent_benchmark("Advanced Agent", advanced_agent_stress, stress_convs, config)

    print(format_rows([row_base_stress, row_adv_stress]))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
