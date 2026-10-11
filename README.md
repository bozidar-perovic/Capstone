# Static Analysis for Flaky Test Prioritization

An undergraduate computer science capstone project exploring lightweight static source-code analysis to detect and prioritize flaky tests in Java projects.

---

## Overview

Flaky tests pass or fail nondeterministically without changes to the underlying source code. Detecting flaky tests through repeated dynamic execution is computationally expensive and significantly delays continuous integration pipelines.

This project investigates whether lightweight static analysis can prioritize tests by their risk of being flaky before executing them:
1. **Rule-Based Baseline (`src/RuleBasedAnalyzer.py`)**: Uses static heuristics (e.g., thread sleeps, async waits, time APIs, concurrency constructs) to assign transparent risk scores.
2. **Machine Learning Baseline (`src/MLAnalyzer.py`)**: Extracts regex-based code pattern counts and trains a balanced logistic regression model to predict flaky risk.
3. **Adaptive / Hybrid Analysis (`src/AdaptiveAnalyzer.py`)**: Explores tiered scoring that combines fast rule-based filtering with deeper model evaluation.

Experiments are conducted on the FlakeBench benchmark using 4-fold project-based cross-validation (Group K-Fold by repository) so models are evaluated on held-out projects they have not seen during training.

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
- **Why They Were Deleted:** In FlakeBench, these tests passed during standard baseline runs (recorded as non-flaky, category 5) but failed under test execution order permutations (recorded as flaky, category 4: test order dependency). Because `src/MLAnalyzer.py` extracts features strictly from test source code, having identical code with opposite labels ($y=0$ and $y=1$) creates contradictory ground truth on identical feature vectors. Deleting the non-flaky duplicate resolved the contradiction while preserving 100% of all verified flaky tests.

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
│   ├── MLAnalyzer.py           # Feature extraction and logistic regression model
│   ├── RuleBasedAnalyzer.py    # Static heuristic risk scorer
│   ├── AdaptiveAnalyzer.py     # Multi-stage adaptive analyzer
│   ├── BaseAnalyzer.py         # Base class interface
│   ├── dataset_audit.py        # Tokenizer-based data quality and overlap audit
│   ├── utility.py              # Data loading and result serialization helpers
│   ├── main.py                 # Training and evaluation runner for CV folds
│   ├── data/                   # Cleaned FlakeBench dataset and 4 CV fold splits
│   └── data_original_backup/   # Unmodified backup of original raw CSV files
├── tests/
│   ├── test_dataset_audit.py   # Unit tests for audit and overlap detection
│   ├── test_ml_analyzer.py     # Unit tests for ML feature extraction
│   └── test_main.py            # Unit tests for core pipeline execution
└── DELETED_ROWS.csv            # Machine-readable log of dropped duplicate rows
```

---

## Setup and Usage

### Prerequisites
- Python 3.11 or later
- Dependencies: `pandas`, `scikit-learn`, `pytest`, `flake8`

Install dependencies:
```bash
pip install pandas scikit-learn pytest flake8
```

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

### Running Experiments
To train the ML analyzer on project-based folds and score test sets:
```bash
python3 -m src.main
```

### Running Tests and Linting
Run the test suite:
```bash
python3 -m pytest tests
```

Run style checks:
```bash
flake8 src tests
```
