from __future__ import annotations

from pathlib import Path
import pandas as pd


def delete_insane_npz(
    insane_csv: str | Path = "candidates_c5_star_ranking_vetted_insane.csv",
    cache_root: str | Path = "splits/infer_c5_v2_10k/_cache/infer",
    *,
    dry_run: bool = True,
) -> None:
    """
    Delete cached .npz files for stars listed in the INSANE csv.

    - insane_csv must contain a 'star_id' column (e.g. 'EPIC_211327055')
    - cache_root is the folder containing EPIC_*.npz files
    - dry_run=True prints what would be deleted without deleting
    """
    insane_csv = Path(insane_csv)
    cache_root = Path(cache_root)

    df = pd.read_csv(insane_csv, dtype={"star_id": str})
    if "star_id" not in df.columns:
        raise ValueError(f"'star_id' column not found in {insane_csv}")

    star_ids = df["star_id"].astype(str).unique().tolist()

    to_delete: list[Path] = []
    missing: list[str] = []

    for sid in star_ids:
        sid = sid.strip()
        if not sid.startswith("EPIC_"):
            sid = "EPIC_" + sid
        p = cache_root / f"{sid}.npz"
        if p.exists():
            to_delete.append(p)
        else:
            missing.append(sid)

    print(f"[insane-delete] cache_root={cache_root}")
    print(f"[insane-delete] listed={len(star_ids)} exists={len(to_delete)} missing={len(missing)}")
    if missing[:10]:
        print("[insane-delete] first missing:", ", ".join(missing[:10]))

    if dry_run:
        print("[insane-delete] DRY RUN—no files deleted. First 20 that would be deleted:")
        for p in to_delete[:20]:
            print("  ", p)
        return

    deleted = 0
    for p in to_delete:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"[insane-delete] failed to delete {p}: {e}")

    print(f"[insane-delete] deleted={deleted}")


if __name__ == "__main__":
    # 1) Preview
    delete_insane_npz(dry_run=False)

    # 2) Actually delete (flip to False)
    # delete_insane_npz(dry_run=False)
