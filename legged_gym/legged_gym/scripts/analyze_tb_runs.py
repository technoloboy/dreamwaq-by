#!/usr/bin/env python3
"""Compare every scalar in two DreamWaQ TensorBoard runs.

The report is intentionally machine-readable JSON so reward-shaping decisions
can be reproduced instead of being based on a few hand-picked screenshots.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(run_dir: Path) -> Dict[str, List[dict]]:
    accumulator = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    return {
        tag: [
            {"step": int(event.step), "value": float(event.value)}
            for event in accumulator.Scalars(tag)
        ]
        for tag in accumulator.Tags()["scalars"]
    }


def nearest(events: List[dict], step: int) -> Optional[dict]:
    if not events:
        return None
    return min(events, key=lambda event: abs(event["step"] - step))


def linear_slope(events: Iterable[dict], per_steps: int = 1000) -> Optional[float]:
    points = list(events)
    if len(points) < 2:
        return None
    x = np.asarray([point["step"] for point in points], dtype=np.float64)
    y = np.asarray([point["value"] for point in points], dtype=np.float64)
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0] * per_steps)


def summary(events: List[dict]) -> dict:
    if not events:
        return {}
    values = np.asarray([event["value"] for event in events], dtype=np.float64)
    tail_size = max(2, math.ceil(len(events) * 0.1))
    tail = events[-tail_size:]
    return {
        "first_step": events[0]["step"],
        "last_step": events[-1]["step"],
        "first": float(values[0]),
        "last": float(values[-1]),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "min_step": events[int(np.argmin(values))]["step"],
        "max": float(np.max(values)),
        "max_step": events[int(np.argmax(values))]["step"],
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
        "tail_10pct_mean": float(np.mean([event["value"] for event in tail])),
        "tail_10pct_std": float(np.std([event["value"] for event in tail])),
        "tail_10pct_slope_per_1k": linear_slope(tail),
        "tail_10pct_positive_fraction": float(
            np.mean([event["value"] > 0 for event in tail])
        ),
        "tail_10pct_negative_fraction": float(
            np.mean([event["value"] < 0 for event in tail])
        ),
    }


def synchronized_delta(baseline_event: Optional[dict], candidate_event: Optional[dict]):
    if baseline_event is None or candidate_event is None:
        return None
    baseline = baseline_event["value"]
    candidate = candidate_event["value"]
    relative = None if baseline == 0 else (candidate - baseline) / abs(baseline)
    return {
        "baseline_step": baseline_event["step"],
        "candidate_step": candidate_event["step"],
        "baseline": baseline,
        "candidate": candidate,
        "absolute": candidate - baseline,
        "relative": relative,
    }


def compare(
    baseline: Dict[str, List[dict]],
    candidate: Dict[str, List[dict]],
    milestones: List[int],
) -> dict:
    report = {}
    for tag in sorted(set(baseline) | set(candidate)):
        baseline_events = baseline.get(tag, [])
        candidate_events = candidate.get(tag, [])
        report[tag] = {
            "baseline_summary": summary(baseline_events),
            "candidate_summary": summary(candidate_events),
            "milestones": {
                str(step): synchronized_delta(
                    nearest(baseline_events, step),
                    nearest(candidate_events, step),
                )
                for step in milestones
            },
        }
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[200, 500, 1000, 3000, 5000, 8000, 12000, 15000, 20000, 25000, 29999],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_dir = args.root / args.baseline
    candidate_dir = args.root / args.candidate
    payload = {
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
        "milestones": args.milestones,
        "tags": compare(
            load_scalars(baseline_dir),
            load_scalars(candidate_dir),
            args.milestones,
        ),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
