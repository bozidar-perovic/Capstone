"""Shared raw static-code features for learned and heuristic analyzers.

These regex counts are indicators, not a Java parser: comments and string
literals can still match. Keep transformations separate from extraction.
"""

import re

import pandas as pd


FEATURE_PATTERNS = {
    "sleep_count": re.compile(r"\bThread\s*\.\s*sleep\s*\("),
    "thread_creation_count": re.compile(r"\bnew\s+Thread\s*\("),
    "await_count": re.compile(r"\bawait\b"),
    "thread_start_count": re.compile(
        r"\b(?:new\s+Thread\s*\([^\n;]*?\)|"
        r"(?:\w*[Tt]hread\w*|t\d*|worker\w*|runner\w*|"
        r"consumer\w*|producer\w*))\s*\.\s*start\s*\("
    ),
    "executor_count": re.compile(
        r"\b(?:Executors|ExecutorService|Executor)\b"
    ),
    "future_count": re.compile(r"\b(?:Future|CompletableFuture)\b"),
    "latch_barrier_count": re.compile(
        r"\b(?:CountDownLatch|CyclicBarrier|Semaphore|Phaser)\b"
    ),
    "time_api_count": re.compile(
        r"\b(?:Date|Instant|LocalDateTime|Clock)\b"
        r"|\bSystem\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\("
    ),
    "random_api_count": re.compile(
        r"\b(?:Random|ThreadLocalRandom)\b"
        r"|\bMath\s*\.\s*random\s*\("
        r"|\bshuffle\s*\("
    ),
    "unordered_collection_count": re.compile(
        r"\b(?:HashMap|HashSet|Hashtable)\b"
    ),
    "file_network_count": re.compile(
        r"\b(?:File|Path|Files|Socket|URL|HttpClient|URLConnection)\b"
    ),
    "assertion_count": re.compile(
        r"\bassert(?:That|Equals|NotEquals|True|False|Null|NotNull|"
        r"Same|NotSame|Throws)\s*\("
        r"|\bAssertions\s*\.\s*assert\w*\s*\("
    ),
    "loop_count": re.compile(r"\b(?:for|while|do)\b"),
    "branch_count": re.compile(r"\b(?:if|switch)\b"),
}


def extract_features(data, patterns=None):
    """Count patterns per row, preserving input indices and column order."""
    if patterns is None:
        patterns = FEATURE_PATTERNS
    rows = []
    for code in data["full_code"]:
        if not isinstance(code, str):
            code = ""
        rows.append({name: len(pattern.findall(code))
                     for name, pattern in patterns.items()})
    return pd.DataFrame(rows, columns=list(patterns), index=data.index)
