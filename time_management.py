"""Timing helpers used by the prediction workflow."""

import time
from contextlib import contextmanager


BASELINE_TIMES = {
    "LOAD DATA": 2.06,
    "CLEAN DATA": 0.08,
    "CHECK NEW PARAMETERS": 0.01,
    "PREPARE FEATURES": 0.08,
    "SPLIT TRAIN / TEST": 0.01,
    "BUILD X / Y": 0.58,
    "TRAIN MODEL": 8.31,
    "CHARTS": 0.12,
}


def timing_marker(elapsed, baseline):
    if not baseline:
        return ""
    ratio = elapsed / baseline
    if 0.8 <= ratio <= 1.2:
        return "🟨"
    if ratio < 0.8:
        return "🟩" * min(8, 1 + int((1 - ratio) / 0.25))
    return "🟥" * min(8, 1 + int((ratio - 1) / 0.25))


@contextmanager
def timed_step(name):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        key = name.split(" [")[0]
        marker = timing_marker(elapsed, BASELINE_TIMES.get(key))
        print(f"⏱  {name:<40} {elapsed:7.2f} s  {marker}")
