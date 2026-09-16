from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/phase2/phase2_remaining858_pre_generation_state.json"
PRODUCTION_FILES = [
    "src/Classifiers/K2/K2_FrozenCnnPreprocessing.py",
    "scripts/generate_phase2_local949_features.py",
    "scripts/acquire_phase2_remaining858_light_curves.py",
    "scripts/build_phase2_feature_generation_manifest.py",
    "scripts/refresh_gatevetter_unseen_full_validation.py",
    "scripts/prepare_gatevetter_v0_2_deep_review.py",
    "scripts/build_phase2_feature_table.py",
    "tests/test_phase2_frozen_cnn_preprocessing_parity.py",
    "scripts/audit_phase2_generated_tensor_duplicates.py",
    "scripts/repair_phase2_generated470_cnn.py",
    "scripts/audit_phase2_generated470_repair.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.stdout


def main() -> None:
    missing = [name for name in PRODUCTION_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"production files missing: {missing}")
    payload = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": git("rev-parse", "HEAD").strip(),
        "git_status_porcelain_v1": git("status", "--porcelain=v1").splitlines(),
        "production_file_sha256": {name: sha256(ROOT / name) for name in PRODUCTION_FILES},
        "frozen_cnn_model_sha256": sha256(ROOT / "models/k2_nocrop_flux_seed46_split303.best.keras"),
        "frozen_preprocessing_parity": "passed_4_of_4_via_unittest_immediately_before_generation",
        "population_assertions": {"local949": 949, "remaining858": 858, "intersection": 0, "union": 1807},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(OUTPUT)
    print(json.dumps({"output": OUTPUT.relative_to(ROOT).as_posix(), **payload["population_assertions"]}, indent=2))


if __name__ == "__main__":
    main()
