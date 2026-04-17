#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_eval_info(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize code validation eval outputs")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    rows: list[dict] = []

    for info_path in sorted(eval_root.glob("**/eval_info.json")):
        info = load_eval_info(info_path)
        relative = info_path.relative_to(eval_root)
        parts = relative.parts
        if len(parts) < 4:
            continue
        policy = parts[0]
        dataset_size = int(parts[1])
        split = parts[2]
        overall = info.get("overall", {})
        rows.append(
            {
                "policy": policy,
                "dataset_size": dataset_size,
                "eval_split": split,
                "success_rate": overall.get("pc_success"),
                "n_episodes": overall.get("n_episodes"),
                "avg_sum_reward": overall.get("avg_sum_reward"),
                "avg_max_reward": overall.get("avg_max_reward"),
                "eval_info_path": str(info_path),
            }
        )

    rows.sort(key=lambda row: (row["policy"], row["dataset_size"], row["eval_split"]))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "dataset_size",
                "eval_split",
                "success_rate",
                "n_episodes",
                "avg_sum_reward",
                "avg_max_reward",
                "eval_info_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
