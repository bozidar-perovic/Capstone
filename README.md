# Static Analysis for Flaky Test Prioritization

An undergraduate computer science capstone project exploring lightweight static source-code analysis to detect and prioritize flaky tests in Java projects.

---

## Overview

Flaky tests pass or fail nondeterministically without changes to the underlying source code. Detecting flaky tests through repeated dynamic execution is computationally expensive and significantly delays continuous integration (CI) pipelines.

This project investigates whether lightweight static analysis can prioritize tests by their risk of being flaky before executing them:
1. **Rule-Based Baseline (`src/RuleBasedAnalyzer.py`)**: Uses static heuristics (e.g., thread sleeps, async waits, time APIs, concurrency constructs) to assign transparent, keyword-based risk scores.
2. **Machine Learning Baselines**:
   - **Logistic Regression (`src/MLAnalyzer.py`)**: Extracts regex-based code pattern counts, applies feature clipping (`max_count=5`) and logarithmic normalization, and trains a class-weighted logistic regression model.
   - **Random Forest (`src/RandomForestAnalyzer.py`)**: Non-linear ensemble model capturing feature interactions and providing Gini feature importances.
3. **Adaptive / Hybrid Analysis (`src/AdaptiveAnalyzer.py`)**: Implements a 3-zone tiered routing architecture that triages tests using fast static heuristics, bypassing ML for ~83.7% of the test suite while reserving model evaluation for ambiguous cases.
4. **Shared Feature Extraction (`src/features.py`)**: 14 centralized, precompiled regex indicators for concurrency, timing, assertions, and control flow.
5. **Prioritization Metrics Suite (`src/metrics.py`)**: Evaluation framework tailored for test prioritization (ROC-AUC, PR-AUC, Recall@K%, and Precision@K%).

Experiments are conducted on the cleaned FlakeBench benchmark using 4-fold project-based cross-validation (Group K-Fold by repository) so models are strictly evaluated on held-out projects they have not seen during training.

---

## 4-Fold Benchmark Results

Each model was evaluated across all 4 held-out project folds. Ranking metrics reflect early detection performance in CI pipelines (e.g., identifying flakiness within the top 5%, 10%, or 20% highest-risk tests):

### Summary Performance Across 4 Folds

| Metric | Rule-Based | Logistic Reg (Capped) | Random Forest | Adaptive Hybrid |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.6309 | **0.6798** | 0.6745 | 0.6618 |
| **PR-AUC** | 0.1137 | 0.1213 | **0.1247** | 0.1053 |
| **Recall@5%** | 21.6% | **24.5%** | 20.4% | 24.0% |
| **Recall@10%** | 33.1% | 33.4% | 28.5% | **33.5%** |
| **Recall@20%** | 33.1% | 42.8% | 39.4% | **46.6%** |
| **Precision@5%** | 14.1% | **16.2%** | 13.9% | 16.0% |
| **Precision@10%** | 11.1% | 11.2% | 9.5% | **11.3%** |

### Adaptive Efficiency Breakdown (Stage 1 Routing)

The Adaptive Hybrid analyzer triages tests into three operational zones:

| Routing Zone | Criteria | Avg. % of Tests | Operational Action |
| :--- | :--- | :---: | :--- |
| **Fast Clean** | `rule_score <= 0.0` | **82.6%** | Assigned risk `0.0`; completely skips ML feature extraction & inference |
| **Ambiguous (ML Deep)** | `0.0 < rule_score < 0.5` | **16.3%** | Evaluated by trained ML model for fine-grained ranking |
| **Fast Flaky** | `rule_score >= 0.5` | **1.1%** | Assigned risk `0.95` immediately based on strong concurrency indicators |
| **Total ML Workload Saved** | Bypassed ML | **83.7%** | **83.7% reduction in ML evaluation overhead** |

### Key Findings
- **High Efficiency with No Loss in Recall**: The Adaptive Hybrid analyzer achieves the highest **Recall@10% (33.5%)** and **Recall@20% (46.6%)**, outperforming both full-suite ML models while executing ML inference on only **16.3%** of the tests.
- **Assertion Count Capping (`max_count=5`)**: Raw assertion counts in complex integration tests previously generated false positives. Capping counts at 5 and applying `log1p` normalization stabilized Logistic Regression and improved ranking quality.
- **Explainability**: Random Forest and Rule-Based feature importances confirm that thread manipulation (`Thread.sleep`, `Thread.start`), asynchronous waiting (`await`), and random generation (`Random`, `Math.random`) are the dominant static predictors of test flakiness.

---

## Architecture & Analyzers

### 1. Shared Feature Extraction (`src/features.py`)
Features are defined as precompiled regex patterns to ensure fast, deterministic extraction across all analyzers:
- **Concurrency & Asynchrony**: `Thread.sleep`, `new Thread()`, `.start()`, `await`, `ExecutorService`, `Future` / `CompletableFuture`, `CountDownLatch` / `CyclicBarrier` / `Semaphore` / `Phaser`.
- **Nondeterminism & Time**: `Date`, `Instant`, `Clock`, `System.currentTimeMillis()`, `System.nanoTime()`, `Random`, `Math.random()`, `shuffle()`.
- **Collections & I/O**: `HashMap`, `HashSet`, `Hashtable`, `File`, `Path`, `Socket`, `URL`, `HttpClient`.
- **Complexity & Structure**: JUnit/AssertJ assertions, loops (`for`, `while`, `do`), and branches (`if`, `switch`).

### 2. Rule-Based Analyzer (`src/RuleBasedAnalyzer.py`)
- Evaluates indicator presence (binary indicator rather than unbounded counts) to avoid skew from repeated calls.
- Jointly handles `new Thread()` and `.start()` to prevent double-counting.
- Includes `time_passed()` benchmarking to measure microsecond-level scoring latency.

### 3. Machine Learning Baselines (`src/MLAnalyzer.py`, `src/RandomForestAnalyzer.py`)
- **Logistic Regression**: Scikit-Learn `Pipeline` with:
  1. `FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": 5})` (caps assertion and loop counts)
  2. `FunctionTransformer(np.log1p)` (smooths heavy-tailed feature distributions)
  3. `StandardScaler()`
  4. `LogisticRegression(class_weight="balanced", max_iter=1000)`
- **Random Forest**: `RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)` providing non-linear combinations and feature importance ranking via `get_feature_importances()`.

### 4. Adaptive Hybrid Analyzer (`src/AdaptiveAnalyzer.py`)
- Orchestrates a tiered two-stage evaluation pipeline.
- Provides `get_routing_summary()` to inspect the exact proportion of tests routed to fast paths vs. deep ML inference.
- Modular design: supports pluggable ML backends (defaults to Logistic Regression, compatible with Random Forest).

### 5. Prioritization Metrics (`src/metrics.py`)
Standard binary classification metrics (accuracy, raw F1) are unsuitable for flaky test prioritization because test suites are heavily imbalanced (~2–3% flaky tests) and continuous integration teams typically inspect only the top K% highest-risk tests. `src/metrics.py` implements:
- **`recall_at_k_percent(y_true, y_scores, k_percent)`**: Measures the percentage of all actual flaky tests caught within the top K% of the ranked test suite.
- **`precision_at_k_percent(y_true, y_scores, k_percent)`**: Measures the density of true flaky tests within the top K% cutoff.
- **ROC-AUC & PR-AUC**: Global discriminative power across all thresholds, with guards for single-class edge cases.

---

## Dataset Audit & Detection Methodology

To guarantee experimental validity and eliminate data leakage, the repository includes a custom audit tool in `src/dataset_audit.py`.

### How the Python Audit Tool Detects Issues

The audit script inspects source files and cross-validation splits to detect code duplication, cross-split leakage, label contradictions, and syntax anomalies.

#### 1. Tokenization and Normalization
Raw string comparisons fail on superficial differences like indentation or comments. The script tokenizes Java source code using regular expressions:
- Identifies string literals, character literals, text blocks, and language tokens.
- Strips single-line (`//`) and multi-line (`/* ... */`) comments.
- Standardizes token sequences while preserving exact literal contents and identifiers.
- Masks string literals during structural checks so quotes containing braces or annotations do not skew syntax heuristics.

#### 2. Source Code Hashing (`hash_source`)
Normalized tokens are serialized into JSON and hashed using SHA-256:
```python
def hash_source(source):
    tokens = source_tokens(source)
    if not tokens:
        return None
    normalized = json.dumps(tokens, ensure_ascii=True)
    return hashlib.sha256(normalized.encode()).hexdigest()
```
This produces an identical hash for two test methods whenever their executable Java instructions, variable names, method calls, and literals match, even if their indentation (e.g. tabs vs. spaces) or developer comments differ.

#### 3. Cross-Split Leakage Detection (`source_hash`)
The audit script evaluates `train_set_k.csv` against `test_set_k.csv` within each fold:
- Compares the sets of source hashes present in the training fold versus the held-out test fold.
- Identifies any test method in the test set that is already present in the training set.
- Detected verbatim test method overlap between `opensearch-project_OpenSearch` and `strapdata_elassandra` (forks of Elasticsearch assigned to opposing splits).

#### 4. Binary Label Conflict Detection (`conflicting_source_hash`)
The audit groups identical source hashes within files and across splits to verify label consistency:
- Flags cases where identical code is labeled both flaky (`category` 0–4) and non-flaky (`category` 5).
- Identified 47 duplicate test pairs where an order-dependent flaky test (category 4) had an identical non-flaky twin in the baseline run.

#### 5. Identifier Collisions and Shared Names
- **ID Collisions**: Flags non-unique IDs reused across different repositories in FlakeBench.
- **Method Name Overlap**: Tracks shared test method names (e.g., `testHashCode`, `testToString`) across projects.

#### 6. Syntax Heuristics
- Checks for balanced curly braces to catch truncated method bodies.
- Detects lifecycle annotations (`@Before`, `@After`, `@BeforeEach`, `@AfterEach`).
- Verifies recognized test annotations (`@Test`, `@ParameterizedTest`, etc.).

### Audit Reports Generated
Running `python3 -m src.dataset_audit` outputs reports to `src/results/dataset_audit/`:
- `summary.json`: Per-file row counts, category distributions, issue counts, and fold overlap counts grouped by kind.
- `row_issues.csv`: Heuristic flags (unbalanced braces, missing bodies, lifecycle methods).
- `duplicate_sources.csv`: Duplicate source code groups within each split.
- `fold_overlap.csv`: Cross-split overlaps between train and test sets (`id`, `test_name`, `source_hash`, `project`).

---

## Specific Rows Deleted and Rationale

Following the audit findings, exactly **50 rows** were removed from `FlakeBench_dataset.csv` and propagated to the corresponding train/test fold CSVs. Full documentation is recorded in `DELETED_ROWS.csv`, and unmodified originals are preserved in `src/data_original_backup/`.

### 1. Cross-Split Code Leakage (3 Rows Dropped)
- **Specific Rows Deleted:**
  - ID 77454: `PreBuiltTransportClientTests.testPluginInstalled` (`opensearch-project_OpenSearch`)
  - ID 77455: `PreBuiltTransportClientTests.testInstallPluginTwice` (`opensearch-project_OpenSearch`)
  - ID 77460: `SuiteScopeClusterIT.testReproducible` (`opensearch-project_OpenSearch`)
- **Why They Were Deleted:** `opensearch-project_OpenSearch` and `strapdata_elassandra` are both forks of Elasticsearch and contain identical test implementations. In Fold 2, Elassandra was in the training set while OpenSearch was in the test set; in Fold 3, this was reversed. Evaluating a machine learning model on test code it was trained on artificially inflates evaluation metrics (data leakage). Dropping OpenSearch's duplicate copies eliminated the leakage while keeping Elassandra's copies (IDs 91436, 91437, 91445) intact in the dataset.

### 2. Contradictory Non-Flaky Duplicates (47 Rows Dropped)
- **Specific Rows Deleted:** 47 test records labeled non-flaky (`category 5`). 40 occurred in `wildfly_wildfly`, and 7 in `spring-projects_spring-data-envers`, `Wikidata_wikidata-toolkit`, `ctco_cukes`, `tbsalling_aismessages`, `undertow-io_undertow`, and `ktuukkan_marine-api`.
- **Key Examples:**
  - ID 26864 (`wildfly`): `NamingEventCoordinatorTestCase.testFireObjectEvent` (non-flaky duplicate of flaky test ID 16: `b19048...testFireObjectEvent`)
  - ID 26880 (`wildfly`): `NamingContextTestCase.testCreateSubcontext` (non-flaky duplicate of flaky test ID 29: `b19048...testCreateSubcontext`)
  - ID 76912 (`spring-data-envers`): `QueryDslRepositoryIntegrationTests.testWithRevisions` (non-flaky duplicate of flaky test ID 43: `563799...testWithRevisions`)
  - ID 110156 (`wikidata-toolkit`): `DirectoryManagerFactoryTest.createDefaultDirectoryManagerPath` (non-flaky duplicate of flaky test ID 148: `20de6f...createDefaultDirectoryManagerPath`)
  - ID 176806 (`ctco_cukes`): `HttpAssertionFacadeImplTest.shouldReturnBodyWhenEnabledAndNoMax` (non-flaky duplicate of flaky test ID 352: `b483e1...shouldReturnBodyWhenEnabledAndNoMax`)
- **Why They Were Deleted:** In FlakeBench, these tests passed during standard baseline runs (recorded as non-flaky, category 5) but failed under test execution order permutations (recorded as flaky, category 4: test order dependency). Because static analyzers extract features strictly from test source code, having identical code with opposite labels ($y=0$ and $y=1$) creates contradictory ground truth on identical feature vectors. Deleting the non-flaky duplicate resolved the contradiction while preserving 100% of all verified flaky tests.

### 3. What Was Kept (Not Deleted)
- **Zero Flaky Tests Were Deleted:** All 280 original flaky test cases remain intact in the dataset.
- **ID Collisions Were Kept:** Independent projects sharing the same numeric ID (such as ID 282 in `apache_hadoop` and ID 282 in `apache_cassandra`) were left untouched because their code and test logic are completely unrelated.

### Dataset Row Counts (Before vs. After)

| Dataset File | Original Rows | Cleaned Rows | Dropped Rows | Conflicting Groups |
| :--- | :---: | :---: | :---: | :---: |
| `src/data/FlakeBench_dataset.csv` | 8,574 | **8,524** | 50 | 0 |
| `src/data/train_set_1.csv` | 6,345 | **6,296** | 49 | 0 |
| `src/data/test_set_1.csv` | 2,229 | **2,228** | 1 | 0 |
| `src/data/train_set_2.csv` | 6,393 | **6,386** | 7 | 0 |
| `src/data/test_set_2.csv` | 2,181 | **2,138** | 43 | 0 |
| `src/data/train_set_3.csv` | 6,375 | **6,329** | 46 | 0 |
| `src/data/test_set_3.csv` | 2,199 | **2,195** | 4 | 0 |
| `src/data/train_set_4.csv` | 6,609 | **6,561** | 48 | 0 |
| `src/data/test_set_4.csv` | 1,965 | **1,963** | 2 | 0 |

---

## Project Structure

```
.
├── src/
│   ├── BaseAnalyzer.py           # Abstract base class interface for all analyzers
│   ├── RuleBasedAnalyzer.py      # Static heuristic risk scorer with timing benchmark
│   ├── MLAnalyzer.py             # Feature capping, log1p scaling, and balanced logistic regression
│   ├── RandomForestAnalyzer.py   # Non-linear ensemble model with Gini feature importances
│   ├── AdaptiveAnalyzer.py       # 3-zone tiered hybrid analyzer (fast clean / ambiguous / fast flaky)
│   ├── features.py               # Centralized regex patterns and raw feature extraction
│   ├── metrics.py                # Prioritization metrics (ROC-AUC, PR-AUC, Recall@K%, Precision@K%)
│   ├── dataset_audit.py          # Tokenizer-based data quality, duplicate, and overlap audit
│   ├── utility.py                # Data loading, unique ID generation, and CSV export helpers
│   ├── main.py                   # 4-fold cross-validation runner and model comparison benchmark
│   ├── data/                     # Cleaned FlakeBench dataset and 4 CV fold splits
│   ├── data_original_backup/     # Unmodified backup of original raw CSV files
│   └── results/                  # Generated predictions (*_results.csv) and audit reports
├── tests/
│   ├── test_rule_based_analyzer.py    # Unit tests for rule scoring, indicator presence, and timing
│   ├── test_ml_analyzer.py            # Unit tests for ML feature extraction, clipping, and pipeline
│   ├── test_random_forest_analyzer.py # Unit tests for RF training, probabilities, and importances
│   ├── test_adaptive_analyzer.py      # Unit tests for 3-zone routing, training, and backend swapping
│   ├── test_metrics.py                # Unit tests for Recall@K%, Precision@K%, and ROC/PR-AUC
│   ├── test_dataset_audit.py          # Unit tests for audit, tokenizing, and overlap detection
│   └── test_main.py                   # Unit tests for data loading and pipeline helpers
├── DELETED_ROWS.csv              # Machine-readable log of dropped duplicate/leakage rows
```

---

## Setup and Usage

### Prerequisites
- Python 3.11 or later
- Dependencies: `pandas`, `scikit-learn`, `numpy`, `pytest`, `flake8`

Install dependencies:
```bash
pip install pandas scikit-learn numpy pytest flake8
```

### Running 4-Fold Model Comparison
To run the full cross-validation benchmark across all 4 project folds for Rule-Based, Logistic Regression, Random Forest, and Adaptive Hybrid:
```bash
python3 -m src.main
```
This executes:
1. Training and evaluation across all 4 held-out project folds silently without console clutter.
2. Export of individual fold predictions to `src/results/test_set_{k}_{model}_results.csv`.
3. Export of consolidated 4-fold benchmark metrics and adaptive routing efficiency to `src/benchmark_summary.csv`.
4. Outputs confirmation: `Process done, saved in the "src/benchmark_summary.csv"`.

### Running the Dataset Audit
To audit the 4 cross-validation folds:
```bash
python3 -m src.dataset_audit
```

To audit a specific standalone file:
```bash
python3 -m src.dataset_audit --files src/data/FlakeBench_dataset.csv
```

To check specific folds:
```bash
python3 -m src.dataset_audit --folds 1 2
```

### Running Tests and Linting
Run the complete unit test suite (37 tests across all 7 test suites):
```bash
python3 -m pytest tests
```

Run style and linting checks:
```bash
flake8 src tests
```
