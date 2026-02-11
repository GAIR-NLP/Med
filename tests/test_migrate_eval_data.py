#!/usr/bin/env python3
"""
Tests for the evaluation data migration tool.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import the script
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from migrate_eval_data import (
    extract_pass_at_1,
    extract_summary,
    filter_trajectory_line,
    parse_step_range,
)


def test_extract_pass_at_1():
    """Test extracting pass@1 accuracy from formatted strings."""
    assert extract_pass_at_1("0.5610 (56.10%)") == 0.561
    assert extract_pass_at_1("0.4790 (47.90%)") == 0.479
    assert extract_pass_at_1("1.0000 (100.00%)") == 1.0
    assert extract_pass_at_1("0.0000 (0.00%)") == 0.0
    print("✅ test_extract_pass_at_1 passed")


def test_extract_summary():
    """Test extracting simplified summary from evaluation results."""
    eval_results = {
        "summary": {
            "charxiv2rq_bench_single_turn_agent": {
                "format": "0.0000 (0.00%)",
                "accuracy": "0.5560 (55.60%)",
                "final": "0.5560 (55.60%)",
                "pass@1_accuracy": "0.5610 (56.10%)",
            },
            "charxiv2rq_bench_tool_agent": {
                "format": "0.0000 (0.00%)",
                "accuracy": "0.4650 (46.50%)",
                "final": "0.4650 (46.50%)",
                "pass@1_accuracy": "0.4790 (47.90%)",
            },
        }
    }

    result = extract_summary(eval_results)
    assert result == {"w/o tool": 0.561, "w/ tool": 0.479}
    print("✅ test_extract_summary passed")


def test_filter_trajectory_line():
    """Test filtering trajectory JSON line to keep only essential fields."""
    # Test with normal uid
    original = {
        "uid": "test_123",
        "category": "reasoning",
        "response_length": 150,
        "tool_call_counts": 2,
        "full_valid_text": "some long text...",
        "marked_text": "marked text...",
        "position_ids": [1, 2, 3],
        "messages": [{"role": "user", "content": "question"}],
        "format_reward": 0.0,
        "accuracy_reward": 1.0,
        "tool_consistency_reward": 0.5,
        "tool_intrinsic_reward": 0.0,
        "final_reward": 0.75,
        "accuracy_reward_original": 0.8,
        "accuracy_reward_llm": 0.9,
        "result_dict": {"key": "value"},
    }

    filtered_line = filter_trajectory_line(json.dumps(original))
    filtered = json.loads(filtered_line)

    # Should only have these 5 fields
    assert set(filtered.keys()) == {
        "uid",
        "response_length",
        "tool_call_counts",
        "messages",
        "accuracy_reward",
    }

    # Values should match
    assert filtered["uid"] == "test_123"
    assert filtered["response_length"] == 150
    assert filtered["tool_call_counts"] == 2
    assert filtered["messages"] == [{"role": "user", "content": "question"}]
    assert filtered["accuracy_reward"] == 1.0

    # Test with None uid - should generate new one
    original_with_none_uid = original.copy()
    original_with_none_uid["uid"] = None

    filtered_line_none = filter_trajectory_line(json.dumps(original_with_none_uid))
    filtered_none = json.loads(filtered_line_none)

    # Should have generated a new uid
    assert filtered_none["uid"] is not None
    assert isinstance(filtered_none["uid"], str)
    assert len(filtered_none["uid"]) == 32  # UUID hex length

    # Test with string 'None' uid - should also generate new one
    original_with_str_none = original.copy()
    original_with_str_none["uid"] = "None"

    filtered_line_str_none = filter_trajectory_line(json.dumps(original_with_str_none))
    filtered_str_none = json.loads(filtered_line_str_none)

    # Should have generated a new uid
    assert filtered_str_none["uid"] != "None"
    assert isinstance(filtered_str_none["uid"], str)
    assert len(filtered_str_none["uid"]) == 32  # UUID hex length

    # Test with generate_uid=False - should keep None
    filtered_line_keep_none = filter_trajectory_line(
        json.dumps(original_with_none_uid), generate_uid=False
    )
    filtered_keep_none = json.loads(filtered_line_keep_none)
    assert filtered_keep_none["uid"] is None

    print("✅ test_filter_trajectory_line passed")


def test_parse_step_range():
    """Test parsing step range strings."""
    # None returns None
    assert parse_step_range(None) is None

    # Range format
    assert parse_step_range("10-50") == [10, 20, 30, 40, 50]

    # Comma-separated
    assert parse_step_range("10,20,30") == [10, 20, 30]

    # Single value
    assert parse_step_range("10") == [10]

    print("✅ test_parse_step_range passed")


def test_integration_dry_run():
    """Test dry run on actual data (if available)."""
    import subprocess

    source_path = Path(
        "/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results/qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1"
    )

    if not source_path.exists():
        print("⚠️  test_integration_dry_run skipped (source data not found)")
        return

    # Test dry run
    result = subprocess.run(
        [
            "python3",
            "scripts/migrate_eval_data.py",
            "qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1",
            "--benchmarks",
            "charxiv2rq",
            "--steps",
            "10",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Dry run failed: {result.stderr}"
    assert "[DRY RUN]" in result.stdout
    assert "charxiv2rq" in result.stdout
    print("✅ test_integration_dry_run passed")


def test_integration_actual_migration():
    """Test actual migration with a small dataset."""
    import subprocess
    import shutil

    source_path = Path(
        "/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results/qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1"
    )

    if not source_path.exists():
        print("⚠️  test_integration_actual_migration skipped (source data not found)")
        return

    # Clean up previous test data
    test_output = Path(
        "evals/qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1/global_step_0000010/charxiv2rq"
    )
    if test_output.exists():
        shutil.rmtree(test_output.parent.parent.parent)

    # Run actual migration
    result = subprocess.run(
        [
            "python3",
            "scripts/migrate_eval_data.py",
            "qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1",
            "--benchmarks",
            "charxiv2rq",
            "--steps",
            "10",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Migration failed: {result.stderr}"
    assert "✅ Migrated: 1" in result.stdout

    # Verify output structure
    assert test_output.exists(), "Output directory not created"

    eval_results_file = test_output / "evaluation_results.json"
    assert eval_results_file.exists(), "evaluation_results.json not created"

    # Verify content
    with eval_results_file.open() as f:
        data = json.load(f)
        assert "w/o tool" in data
        assert "w/ tool" in data
        assert isinstance(data["w/o tool"], float)
        assert isinstance(data["w/ tool"], float)

    # Verify trajectory files
    traj_dir = test_output / "trajectories"
    assert traj_dir.exists(), "trajectories directory not created"

    single_turn_file = (
        traj_dir / "charxiv2rq_bench_single_turn_agent_trajectories.jsonl"
    )
    tool_agent_file = traj_dir / "charxiv2rq_bench_tool_agent_trajectories.jsonl"

    assert single_turn_file.exists(), "single_turn trajectory file not created"
    assert tool_agent_file.exists(), "tool_agent trajectory file not created"

    # Verify trajectory content (first line)
    with single_turn_file.open() as f:
        first_line = f.readline()
        traj_data = json.loads(first_line)
        # Should only have 5 fields
        assert set(traj_data.keys()) == {
            "uid",
            "response_length",
            "tool_call_counts",
            "messages",
            "accuracy_reward",
        }

    # Clean up
    shutil.rmtree(test_output.parent.parent.parent)

    print("✅ test_integration_actual_migration passed")


def test_bash_wrapper():
    """Test bash wrapper script."""
    import subprocess

    result = subprocess.run(
        ["bash", "scripts/migrate_eval_data.sh", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1  # Help exits with 1
    assert "Usage:" in result.stdout
    assert "EXP_PATH" in result.stdout
    print("✅ test_bash_wrapper passed")


def run_all_tests():
    """Run all tests."""
    print("\n🧪 Running migration tool tests...\n")

    test_extract_pass_at_1()
    test_extract_summary()
    test_filter_trajectory_line()
    test_parse_step_range()
    test_integration_dry_run()
    test_integration_actual_migration()
    test_bash_wrapper()

    print("\n✅ All tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
