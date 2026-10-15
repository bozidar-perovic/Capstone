"""Shared raw static-code features for learned and heuristic analyzers.

Extracts regex-based keyword counts from Java test method source code without
requiring a full Java AST parser. The extracted patterns target common
root causes of test flakiness established in software engineering literature:
- Concurrency and asynchrony (threads, awaits, executors, futures, latches)
- Time and clock dependencies (System.currentTimeMillis, nanoTime, Clock)
- Randomness and nondeterministic ordering (Random, Math.random, shuffle)
- Unordered collections (HashMap, HashSet, Hashtable)
- External resource and I/O access (files, paths, sockets, URLs, HTTP)
- Test structure and complexity (assertions, loops, branches)

Note:
    These regex counts serve as lightweight static indicators. Because they
    operate directly on source text, occurrences inside comments or string
    literals may still match. Feature scaling and transformations belong to
    individual analyzer pipelines, not raw extraction.
"""

from typing import Dict, Optional, Pattern
import re

import pandas as pd

# Centralized collection of precompiled regex patterns representing
# static flakiness indicators and code complexity signals.
FEATURE_PATTERNS: Dict[str, Pattern[str]] = {
    # Concurrency: explicit thread sleeping
    "sleep_count": re.compile(r"\bThread\s*\.\s*sleep\s*\("),
    # Concurrency: explicit thread instantiation
    "thread_creation_count": re.compile(r"\bnew\s+Thread\s*\("),
    # Asynchrony: async wait keywords and libraries (e.g. Awaitility)
    "await_count": re.compile(r"\bawait\b"),
    # Concurrency: thread starting calls on common variable name patterns
    "thread_start_count": re.compile(
        r"\b(?:new\s+Thread\s*\([^\n;]*?\)|"
        r"(?:\w*[Tt]hread\w*|t\d*|worker\w*|runner\w*|"
        r"consumer\w*|producer\w*))\s*\.\s*start\s*\("
    ),
    # Concurrency: managed thread pools and execution services
    "executor_count": re.compile(
        r"\b(?:Executors|ExecutorService|Executor)\b"
    ),
    # Concurrency: asynchronous future results
    "future_count": re.compile(r"\b(?:Future|CompletableFuture)\b"),
    # Concurrency: coordination primitives and synchronization barriers
    "latch_barrier_count": re.compile(
        r"\b(?:CountDownLatch|CyclicBarrier|Semaphore|Phaser)\b"
    ),
    # Time: date, clock, and system timer APIs
    "time_api_count": re.compile(
        r"\b(?:Date|Instant|LocalDateTime|Clock)\b"
        r"|\bSystem\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\("
    ),
    # Randomness: random number generators and shuffling
    "random_api_count": re.compile(
        r"\b(?:Random|ThreadLocalRandom)\b"
        r"|\bMath\s*\.\s*random\s*\("
        r"|\bshuffle\s*\("
    ),
    # Nondeterminism: hash-based collections without guaranteed iteration order
    "unordered_collection_count": re.compile(
        r"\b(?:HashMap|HashSet|Hashtable)\b"
    ),
    # External resources: file system and network I/O
    "file_network_count": re.compile(
        r"\b(?:File|Path|Files|Socket|URL|HttpClient|URLConnection)\b"
    ),
    # Complexity: JUnit, TestNG, and AssertJ assertion statements
    "assertion_count": re.compile(
        r"\bassert(?:That|Equals|NotEquals|True|False|Null|NotNull|"
        r"Same|NotSame|Throws)\s*\("
        r"|\bAssertions\s*\.\s*assert\w*\s*\("
    ),
    # Complexity: loop constructs
    "loop_count": re.compile(r"\b(?:for|while|do)\b"),
    # Complexity: conditional branching
    "branch_count": re.compile(r"\b(?:if|switch)\b"),
}


def extract_features(
    data: pd.DataFrame,
    patterns: Optional[Dict[str, Pattern[str]]] = None,
) -> pd.DataFrame:
    """Count pattern occurrences per test row, preserving indices and order.

    Iterates through the 'full_code' column of the input DataFrame and counts
    the occurrences of each compiled regex pattern. Non-string or missing code
    entries are treated as empty strings.

    Args:
        data: Input DataFrame containing a 'full_code' column with source text.
        patterns: Optional dictionary mapping feature names to compiled regex
            patterns. Defaults to FEATURE_PATTERNS if None.

    Returns:
        A pandas DataFrame where each column corresponds to a feature pattern
        and each row contains the integer count of matches in the test method.
        Preserves the exact index of the input DataFrame.
    """
    if patterns is None:
        patterns = FEATURE_PATTERNS
    rows = []
    for code in data["full_code"]:
        if not isinstance(code, str):
            code = ""
        rows.append({
            name: len(pattern.findall(code))
            for name, pattern in patterns.items()
        })
    return pd.DataFrame(rows, columns=list(patterns), index=data.index)
