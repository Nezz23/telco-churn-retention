"""A git-tracked model registry: versioned artifacts, champion-challenger promotion gates, and rollback.

models/registry.json names the champion and keeps an append-only event log. Each version lives in models/<version>/
with its artifact, the artifact's SHA-256 and its full evaluation report. The API refuses to start if the artifact on
disk does not hash to the registered value, so the model that passed CI is byte-for-byte the model that serves.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent / "models"  # independent of the working directory
INDEX = ROOT / "registry.json"

# Promotion gate. Absolute floors protect against a broken model; non-inferiority margins protect against a
# challenger that is merely worse than what is already serving. Measured on the same held-out customers.
FLOORS = {"roc_auc": 0.80, "ece_max": 0.05}
MARGINS = {"log_loss": 0.01, "profit_share": 0.10}  # challenger may be at most this much worse at the CI's edge
SUBGROUP_MIN_N, SUBGROUP_GAP_MAX = 150, 0.10  # no sizeable segment may be miscalibrated by >10 points


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index():
    return json.loads(INDEX.read_text()) if INDEX.exists() else {"champion": None, "events": []}


def _write_index(ix):
    INDEX.write_text(json.dumps(ix, indent=2) + "\n")


def _event(ix, action, version, **extra):
    ix["events"].append({"at": datetime.now(UTC).isoformat(timespec="seconds"), "action": action, "version": version, **extra})


def register(model, report):
    """Store a trained model as a new, not-yet-serving version."""
    version = datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")
    folder = ROOT / version
    folder.mkdir(parents=True)
    joblib.dump(model, folder / "model.joblib", compress=3)
    report = {**report, "version": version, "artifact_sha256": sha256(folder / "model.joblib")}
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    ix = index()
    _event(ix, "registered", version)
    _write_index(ix)
    return version


def load(version=None):
    """Load a version (default: the champion) after verifying its artifact hash."""
    version = version or index()["champion"]
    if version is None:
        raise RuntimeError("No champion model. Train one: python -m churn.train --promote")
    folder = ROOT / version
    report = json.loads((folder / "report.json").read_text())
    if sha256(folder / "model.joblib") != report["artifact_sha256"]:
        raise RuntimeError(f"Artifact for {version} does not match its registered SHA-256; refusing to serve it.")
    if not report.get("gate", {}).get("passed"):
        raise RuntimeError(f"{version} did not pass its promotion gate; refusing to serve it.")
    return joblib.load(folder / "model.joblib"), report


def gate(challenger: dict, champion: dict | None):
    """Return the list of checks and whether all passed. `challenger`/`champion` are head-to-head evaluations
    on the same customers: {"metrics": {...}, "vs_champion": {...}, "subgroups": [...], "everyone_profit": x}."""
    m = challenger["metrics"]
    checks = [
        {
            "check": f"ROC-AUC >= {FLOORS['roc_auc']}",
            "value": m["roc_auc"]["value"],
            "passed": m["roc_auc"]["value"] >= FLOORS["roc_auc"],
        },
        {
            "check": f"Calibration error (ECE) <= {FLOORS['ece_max']}",
            "value": m["ece"]["value"],
            "passed": m["ece"]["value"] <= FLOORS["ece_max"],
        },
        {
            "check": "Profit beats contacting everyone and is positive",
            "value": m["profit"]["value"],
            "passed": m["profit"]["value"] > max(0.0, challenger["everyone_profit"]),
        },
    ]
    sizeable = [s for s in challenger["subgroups"] if s["n"] >= SUBGROUP_MIN_N]
    worst = max(sizeable, key=lambda s: abs(s["calibration_gap"]), default=None)
    checks.append(
        {
            "check": f"Every segment with n >= {SUBGROUP_MIN_N} calibrated within {SUBGROUP_GAP_MAX:.0%}",
            "value": f"{worst['column']}={worst['level']}: {worst['calibration_gap']:+.3f}" if worst else "no segment that large",
            "passed": worst is None or abs(worst["calibration_gap"]) <= SUBGROUP_GAP_MAX,
        }
    )
    if champion is not None:
        d = challenger["vs_champion"]
        checks.append(
            {
                "check": f"Log loss non-inferior to champion (upper 95% bound of difference <= +{MARGINS['log_loss']})",
                "value": d["log_loss"],
                "passed": d["log_loss"]["ci"][1] <= MARGINS["log_loss"],
            }
        )
        floor = -MARGINS["profit_share"] * abs(champion["metrics"]["profit"]["value"])
        checks.append(
            {
                "check": f"Profit non-inferior to champion (lower 95% bound of difference >= {floor:,.0f})",
                "value": d["profit"],
                "passed": d["profit"]["ci"][0] >= floor,
            }
        )
    return {"passed": all(c["passed"] for c in checks), "checks": checks, "has_champion": champion is not None}


def promote(version, gate_report):
    if not gate_report["passed"]:
        raise RuntimeError(f"{version} failed the promotion gate; not promoting.")
    ix = index()
    _event(ix, "promoted", version, previous=ix["champion"])
    ix["champion"] = version
    _write_index(ix)


def rollback():
    """Make the champion that served before the current one the champion again."""
    ix = index()
    # Walk back along promotions, so repeated rollbacks keep stepping to older champions.
    promotions = [e for e in ix["events"] if e["action"] == "promoted" and e["version"] == ix["champion"]]
    previous = promotions[-1].get("previous") if promotions else None
    if previous is None:
        raise RuntimeError("Nothing to roll back to: the current champion has no predecessor.")
    load(previous)  # verify the artifact before switching to it
    _event(ix, "rolled back", previous, previous=ix["champion"])
    ix["champion"] = previous
    _write_index(ix)
    return previous


def main():
    parser = argparse.ArgumentParser(description="Model registry")
    parser.add_argument("command", choices=["status", "rollback", "verify"])
    cmd = parser.parse_args().command
    if cmd == "rollback":
        print(f"champion is now {rollback()}")
    elif cmd == "verify":
        _, report = load()
        print(f"{report['version']} verified ({report['artifact_sha256'][:12]})")
    else:
        print(json.dumps(index(), indent=2))


if __name__ == "__main__":
    main()
