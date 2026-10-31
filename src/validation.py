"""Project-grouped setting selection, separate from outer-test evaluation.

Run ``python -B -m src.validation validate --output-dir RUN`` first. This
command reads training CSVs only. A later, explicit ``evaluate --run-dir RUN``
refits the selected configurations on complete outer training folds.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from .AdaptiveAnalyzer import (
    AdaptiveAnalyzer, AdaptiveRuleScreen, DEFAULT_RULE_WEIGHTS,
)
from .benchmark import DEFAULT_DATA_DIR
from .metrics import compute_ranking_metrics
from .MLAnalyzer import MLAnalyzer
from .RandomForestAnalyzer import RandomForestAnalyzer
from .RuleBasedAnalyzer import RuleBasedAnalyzer


MODEL_DEFAULTS = {
    "Rule-Based": {"source_mode": "raw"},
    "Logistic Reg": {"max_count": 5},
    "Random Forest": {
        "n_estimators": 100, "max_depth": 5,
        "random_state": 42, "class_weight": "balanced",
    },
    "Adaptive Hybrid": {
        "low_threshold": 0.0, "high_threshold": 0.5, "high_score": 0.95,
        "rule_weights": dict(DEFAULT_RULE_WEIGHTS), "ml_max_count": 5,
        "bypass_low": True, "bypass_high": True,
    },
}
# The matched screen is opt-in; default validation still runs four analyzers.
SUPPORTED_DEFAULTS = {
    **MODEL_DEFAULTS,
    "Adaptive Rule Screen": {"rule_weights": dict(DEFAULT_RULE_WEIGHTS)},
}
RANKING_METRICS = (
    "pr_auc", "roc_auc", "recall@5%", "recall@10%", "recall@20%",
    "precision@5%", "precision@10%", "precision@20%",
)


@dataclass
class Candidate:
    """A named configuration of an existing, unchanged analyzer."""

    candidate_id: str
    analyzer: str
    parameters: Dict[str, Any]

    def build(self):
        """Create a fresh analyzer, including a fresh adaptive ML backend."""
        parameters = dict(self.parameters)
        if self.analyzer == "Rule-Based":
            return RuleBasedAnalyzer(**parameters)
        if self.analyzer == "Logistic Reg":
            return MLAnalyzer(**parameters)
        if self.analyzer == "Random Forest":
            return RandomForestAnalyzer(**parameters)
        if self.analyzer == "Adaptive Rule Screen":
            return AdaptiveRuleScreen(**parameters)
        max_count = parameters.pop("ml_max_count")
        return AdaptiveAnalyzer(
            ml_analyzer=MLAnalyzer(max_count=max_count), **parameters
        )

    def as_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "analyzer": self.analyzer,
            "parameters": self.parameters,
        }


def parse_candidates(records=None):
    """Expand constructor defaults so saved configurations are explicit.

    JSON configuration files contain a list of objects with candidate_id,
    analyzer, and optional parameters. Candidate declaration order breaks
    exact ties in mean validation AP; outer test scores are never consulted.
    """
    if records is None:
        records = [
            {"candidate_id": name.lower().replace("-", "_").replace(
                " ", "_"
            ) + "_default", "analyzer": name}
            for name in MODEL_DEFAULTS
        ]
    if not isinstance(records, list) or not records:
        raise ValueError("Candidate configuration must be a nonempty list.")
    candidates, identifiers = [], set()
    for record in records:
        if set(record) - {"candidate_id", "analyzer", "parameters"}:
            raise ValueError("Unknown candidate configuration fields.")
        identifier = record["candidate_id"]
        name = record["analyzer"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Each candidate needs a nonempty candidate_id.")
        if identifier in identifiers:
            raise ValueError("Candidate IDs must be unique.")
        if name not in SUPPORTED_DEFAULTS:
            raise ValueError(f"Unknown analyzer: {name}")
        parameters = dict(SUPPORTED_DEFAULTS[name])
        supplied = record.get("parameters", {})
        if not isinstance(supplied, dict) or set(supplied) - set(parameters):
            raise ValueError(f"Unsupported parameters for {name}.")
        parameters.update(supplied)
        if name == "Adaptive Hybrid":
            if not 0 <= parameters["low_threshold"] < parameters[
                "high_threshold"
            ]:
                raise ValueError(
                    "Routing thresholds must satisfy 0 <= low < high."
                )
            if not 0 <= parameters["high_score"] <= 1:
                raise ValueError("high_score must be between 0 and 1.")
        if name in ("Adaptive Hybrid", "Adaptive Rule Screen"):
            weights = parameters["rule_weights"]
            if not isinstance(weights, dict) or not weights:
                raise ValueError("rule_weights must be a nonempty mapping.")
            if set(weights) - set(DEFAULT_RULE_WEIGHTS):
                raise ValueError("Unknown adaptive rule weight.")
            if any(not np.isfinite(v) or v < 0 for v in weights.values()):
                raise ValueError("Rule weights must be finite and >= 0.")
            parameters["rule_weights"] = {**DEFAULT_RULE_WEIGHTS, **weights}
        candidate = Candidate(identifier, name, parameters)
        candidate.build()
        candidates.append(candidate)
        identifiers.add(identifier)
    return candidates


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_checksums():
    """Detect source changes between validation and final evaluation."""
    return {
        path.name: checksum(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def runtime_versions():
    return {
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_dataset(path):
    """Reject unusable splits instead of generating fallback scores."""
    data = pd.read_csv(path)
    required = {"project", "category", "full_code"}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError(f"Missing data or required columns in {path}.")
    if data[list(required)].isna().any().any():
        raise ValueError(f"Missing project, category, or source in {path}.")
    if not data.category.isin(range(6)).all():
        raise ValueError(f"Invalid category in {path}.")
    require_both_classes(data, str(path))
    return data


def require_both_classes(data, description):
    if (data.category != 5).nunique() != 2:
        raise ValueError(f"{description} must contain both classes.")


def class_counts(data):
    flaky = int((data.category != 5).sum())
    return {
        "rows": len(data), "projects": int(data.project.nunique()),
        "flaky": flaky, "non_flaky": len(data) - flaky,
    }


def make_splits(data, n_splits=3, seed=42):
    """Return positional indices; each project validates exactly once."""
    require_both_classes(data, "Outer training data")
    y = (data.category != 5).astype(int)
    for label in (0, 1):
        if data.loc[y == label, "project"].nunique() < n_splits:
            raise ValueError(
                "Each class needs at least n_splits projects to provide "
                "both classes in every validation split."
            )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    splits = list(splitter.split(
        data, y, groups=data.project
    ))
    coverage = np.zeros(len(data), dtype=int)
    for training, validation in splits:
        train, valid = data.iloc[training], data.iloc[validation]
        if set(train.project) & set(valid.project):
            raise ValueError("Inner training and validation projects overlap.")
        require_both_classes(train, "Inner training split")
        require_both_classes(valid, "Inner validation split")
        coverage[validation] += 1
    if not (coverage == 1).all():
        raise ValueError("Every row must validate exactly once.")
    return splits


def score(candidate, training, testing):
    """Fit preprocessing and model on training only; reuse existing metrics."""
    analyzer = candidate.build()
    if analyzer.requires_training:
        analyzer.train(training.copy())
    predictions = analyzer.analyze(testing.copy())
    if predictions is None or not predictions.index.equals(testing.index):
        raise ValueError("Analyzer must preserve scored rows and their order.")
    scores = predictions["flaky_score"].to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Analyzer produced nonfinite scores.")
    metrics = compute_ranking_metrics(
        (testing.category != 5).astype(int), scores
    )
    routing = {
        f"routing_{key}": value
        for key, value in analyzer.get_routing_summary().items()
    }
    if "route" in predictions:
        flaky = testing.category != 5
        for route in ("fast_clean", "ml_deep", "fast_flaky"):
            assigned = predictions.route == route
            routing[f"routing_{route}_flaky_count"] = int(
                (assigned & flaky).sum()
            )
            routing[f"routing_{route}_non_flaky_count"] = int(
                (assigned & ~flaky).sum()
            )
    return predictions, {**metrics, **routing}


def validate(data_dir, output_dir, candidates=None, folds=(1, 2, 3, 4),
             n_splits=3, seed=42):
    """Select each analyzer's settings using only its outer training file."""
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError("Write validation outputs outside the data folder.")
    if output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    candidates = parse_candidates(candidates)
    folds = tuple(folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("Choose nonempty, distinct outer folds.")
    # Prepare and check all inner partitions before creating a run directory.
    prepared = {}
    for fold in folds:
        path = data_dir / f"train_set_{fold}.csv"
        data = read_dataset(path)
        prepared[fold] = (data, make_splits(data, n_splits, seed))
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1, "status": "running", "data_dir": str(data_dir),
        "inner_folds": n_splits, "seed": seed,
        "selection_metric": "pr_auc",
        "selection_metric_name": "Average Precision",
        "aggregation": "unweighted mean over inner folds",
        "tie_break": "first candidate in configuration order",
        "candidates": [candidate.as_dict() for candidate in candidates],
        "implementation": implementation_checksums(),
        "runtime": runtime_versions(), "outer_folds": {},
    }
    write_json(output_dir / "experiment.json", manifest)
    rows, summaries, selections = [], [], {}
    for fold, (data, splits) in prepared.items():
        assignments = pd.DataFrame({
            "row_index": np.arange(len(data)), "project": data.project,
            "inner_fold": np.zeros(len(data), dtype=int),
        })
        counts = []
        for inner, (training, validation) in enumerate(splits, start=1):
            assignments.loc[validation, "inner_fold"] = inner
            train, valid = data.iloc[training], data.iloc[validation]
            counts.append({
                "inner_fold": inner, "training": class_counts(train),
                "validation": class_counts(valid),
            })
            for candidate in candidates:
                _, metrics = score(candidate, train, valid)
                rows.append({
                    "outer_fold": fold, "inner_fold": inner,
                    "analyzer": candidate.analyzer,
                    "candidate_id": candidate.candidate_id, **metrics,
                })
        assignment_path = output_dir / f"fold_{fold}_assignments.csv"
        assignments.to_csv(assignment_path, index=False)
        manifest["outer_folds"][str(fold)] = {
            "training_file": f"train_set_{fold}.csv",
            "training_sha256": checksum(data_dir / f"train_set_{fold}.csv"),
            "assignments_file": assignment_path.name,
            "assignments_sha256": checksum(assignment_path),
            "training_counts": class_counts(data), "splits": counts,
        }
        frame = pd.DataFrame([r for r in rows if r["outer_fold"] == fold])
        chosen = {}
        for candidate in candidates:
            measurements = frame[frame.candidate_id == candidate.candidate_id]
            summary = {
                "outer_fold": fold, "analyzer": candidate.analyzer,
                "candidate_id": candidate.candidate_id,
            }
            for metric in RANKING_METRICS:
                summary[f"{metric}_mean"] = float(measurements[metric].mean())
                summary[f"{metric}_std"] = float(measurements[metric].std())
            summaries.append(summary)
            previous = chosen.get(candidate.analyzer)
            ap = summary["pr_auc_mean"]
            if previous is None or ap > previous["mean_validation_ap"]:
                chosen[candidate.analyzer] = {
                    **candidate.as_dict(), "mean_validation_ap": ap,
                }
        selections[str(fold)] = list(chosen.values())
        print(f"Validated outer fold {fold} ({len(splits)} inner splits).",
              flush=True)
    pd.DataFrame(rows).to_csv(
        output_dir / "validation_metrics.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(
        output_dir / "validation_summary.csv", index=False
    )
    write_json(output_dir / "selected_settings.json", selections)
    manifest["reports_sha256"] = {
        name: checksum(output_dir / name) for name in (
            "validation_metrics.csv", "validation_summary.csv",
            "selected_settings.json",
        )
    }
    manifest["status"] = "complete"
    write_json(output_dir / "experiment.json", manifest)
    return output_dir


def evaluate(run_dir):
    """Explicitly refit selected settings, then score paired outer test files.

    Refuse changed source, runtime, training inputs, or reused output folders.
    Leaves validation reports and existing presentation results unchanged.
    """
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "experiment.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Validation must finish before outer evaluation.")
    if manifest["implementation"] != implementation_checksums():
        raise ValueError("Source changed since validation; create a new run.")
    if manifest["runtime"] != runtime_versions():
        raise ValueError("Runtime changed since validation; create a new run.")
    for name, expected in manifest["reports_sha256"].items():
        if checksum(run_dir / name) != expected:
            raise ValueError("Validation reports or chosen settings changed.")
    selections = json.loads((run_dir / "selected_settings.json").read_text())
    data_dir = Path(manifest["data_dir"])
    prepared, test_checksums = {}, {}
    for fold, info in manifest["outer_folds"].items():
        training_path = data_dir / info["training_file"]
        if checksum(training_path) != info["training_sha256"]:
            raise ValueError("Training data changed since validation.")
        if checksum(run_dir / info["assignments_file"]) != info[
            "assignments_sha256"
        ]:
            raise ValueError("Saved validation assignments changed.")
        training = read_dataset(training_path)
        testing_path = data_dir / f"test_set_{fold}.csv"
        testing = read_dataset(testing_path)
        if set(training.project) & set(testing.project):
            raise ValueError("Outer training and test projects overlap.")
        test_checksums[fold] = checksum(testing_path)
        prepared[fold] = (training, testing)
    output = run_dir / "evaluation"
    output.mkdir(exist_ok=False)
    rows = []
    for fold, (training, testing) in prepared.items():
        for selected in selections[fold]:
            record = {key: selected[key] for key in (
                "candidate_id", "analyzer", "parameters"
            )}
            candidate = parse_candidates([record])[0]
            predictions, metrics = score(candidate, training, testing)
            # Filenames use numeric indices, independent of user-supplied IDs.
            index = len(rows) + 1
            predictions.to_csv(
                output / f"predictions_{index}.csv", index=False
            )
            rows.append({
                "outer_fold": int(fold), "analyzer": candidate.analyzer,
                "candidate_id": candidate.candidate_id,
                "predictions_file": f"predictions_{index}.csv", **metrics,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "outer_metrics.csv", index=False)
    frame.groupby("analyzer", sort=False)[list(RANKING_METRICS)].mean().to_csv(
        output / "outer_summary.csv"
    )
    write_json(output / "evaluation.json", {
        "status": "complete", "test_sha256": test_checksums,
        "selection_metric": "Average Precision",
        "summary_aggregation": "unweighted mean over outer folds",
    })
    return output


def main(argv: Sequence[str] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser(
        "validate", help="Read training folds only"
    )
    validation.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    validation.add_argument("--output-dir", type=Path, required=True)
    validation.add_argument("--config", type=Path)
    validation.add_argument(
        "--folds", type=int, nargs="+", default=[1, 2, 3, 4]
    )
    validation.add_argument("--inner-folds", type=int, default=3)
    validation.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser(
        "evaluate", help="Refit and score outer tests"
    )
    evaluation.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            records = (
                json.loads(args.config.read_text()) if args.config else None
            )
            output = validate(
                args.data_dir, args.output_dir, records, args.folds,
                args.inner_folds, args.seed,
            )
        else:
            output = evaluate(args.run_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved {args.command} outputs to {output}")


if __name__ == "__main__":
    main()
