from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner
from src.Classifiers.K2.Systematics.K2_NoiseHandler import K2_NoiseHandler


class K2FailedDownloader:
    DEFAULT_INPUT_CSV = Path("data/k2_target_lists/K2Campaign5targets.csv")
    DEFAULT_RESULTS_CSV = Path("plots/k2_batch/batch_results.csv")
    DEFAULT_QUERY_COL = "EPIC ID"
    DEFAULT_OUT_DIR = Path("plots/k2_batch")
    DEFAULT_CACHE_DIR = Path.home() / ".lightkurve" / "cache"
    RECOVERY_COLUMNS = [
        "query",
        "epic_id",
        "status",
        "attempts",
        "cache_path",
        "cache_source",
        "error_type",
        "error_message",
        "timestamp_utc",
    ]

    def __init__(
        self,
        input_csv: Path,
        results_csv: Path,
        out_dir: Path,
        query_col: str,
        cache_dir: Path,
        retries: int = 3,
        max_targets: Optional[int] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        limit: int = 50,
        exptime: Optional[str] = None,
    ) -> None:
        self.input_csv = Path(input_csv)
        self.results_csv = Path(results_csv)
        self.out_dir = Path(out_dir)
        self.query_col = str(query_col)
        self.cache_dir = Path(cache_dir)
        self.retries = int(max(1, retries))
        self.max_targets = int(max_targets) if max_targets is not None else None
        self.start_index = int(start_index) if start_index is not None else None
        self.end_index = int(end_index) if end_index is not None else None
        self.limit = int(max(1, limit))
        self.exptime = exptime

        self.epics_dir = self.out_dir / "epics"
        self.progress_path = self.out_dir / "progress_failed.json"
        self.failed_recovery_csv = self.out_dir / "failed_recovery.csv"
        self.handler = K2_NoiseHandler(verbose=False)

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Retry failed K2 EPIC downloads only.")
        p.add_argument(
            "--input_csv",
            "--input-csv",
            dest="input_csv",
            type=Path,
            default=cls.DEFAULT_INPUT_CSV,
            help=f"Input target CSV (default: {cls.DEFAULT_INPUT_CSV})",
        )
        p.add_argument(
            "--results_csv",
            "--results-csv",
            dest="results_csv",
            type=Path,
            default=cls.DEFAULT_RESULTS_CSV,
            help=f"Batch results CSV used to find failures (default: {cls.DEFAULT_RESULTS_CSV})",
        )
        p.add_argument(
            "--query_col",
            "--query-col",
            dest="query_col",
            type=str,
            default=cls.DEFAULT_QUERY_COL,
            help=f"Query column in input CSV (default: {cls.DEFAULT_QUERY_COL})",
        )
        p.add_argument(
            "--out_dir",
            "--out-dir",
            dest="out_dir",
            type=Path,
            default=None,
            help="Output directory for progress/recovery files. Defaults to parent of --results_csv.",
        )
        p.add_argument(
            "--max_targets",
            "--max-targets",
            dest="max_targets",
            type=int,
            default=None,
            help="Maximum failed targets to process after filtering.",
        )
        p.add_argument(
            "--start_index",
            "--start-index",
            dest="start_index",
            type=int,
            default=None,
            help="Start index within failed target list (0-based, inclusive).",
        )
        p.add_argument(
            "--end_index",
            "--end-index",
            dest="end_index",
            type=int,
            default=None,
            help="End index within failed target list (0-based, inclusive).",
        )
        p.add_argument(
            "--cache_dir",
            "--cache-dir",
            dest="cache_dir",
            type=Path,
            default=cls.DEFAULT_CACHE_DIR,
            help=f"Lightkurve cache directory for downloads (default: {cls.DEFAULT_CACHE_DIR})",
        )
        p.add_argument("--retries", type=int, default=3, help="Retries per EPIC (default: 3).")
        p.add_argument("--limit", type=int, default=50, help="Lightkurve search limit (default: 50).")
        p.add_argument(
            "--exptime",
            type=str,
            default=None,
            help="Optional cadence filter passed to fetch_best (e.g. long, short, 1800).",
        )
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.results_csv).parent
        downloader = cls(
            input_csv=Path(args.input_csv),
            results_csv=Path(args.results_csv),
            out_dir=out_dir,
            query_col=args.query_col,
            cache_dir=Path(args.cache_dir),
            retries=int(args.retries),
            max_targets=args.max_targets,
            start_index=args.start_index,
            end_index=args.end_index,
            limit=int(args.limit),
            exptime=args.exptime,
        )
        return downloader.run()

    @staticmethod
    def _extract_epic_number(value: Any) -> Optional[str]:
        if value is None:
            return None
        m = re.search(r"(\d{6,})", str(value))
        return str(m.group(1)) if m is not None else None

    @staticmethod
    def _as_lower_str_series(df: pd.DataFrame, column: str) -> pd.Series:
        return df[column].fillna("").astype(str).str.strip().str.lower()

    def _load_queries(self) -> List[str]:
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {self.input_csv}")
        df = pd.read_csv(self.input_csv)
        if self.query_col not in df.columns:
            raise ValueError(f"Column '{self.query_col}' not found in {self.input_csv}")
        raw_queries = df[self.query_col].dropna().astype(str).tolist()
        return K2BatchRunner._normalize_queries(raw_queries)

    def _failed_query_masks(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        if "triage_status" in df.columns:
            mask = mask | (self._as_lower_str_series(df, "triage_status") == "error")
        if "status" in df.columns:
            mask = mask | (self._as_lower_str_series(df, "status") == "error")
        if "skip_reason" in df.columns:
            skip_lower = self._as_lower_str_series(df, "skip_reason")
            mask = mask | skip_lower.str.contains("triage_status=error", na=False)
        return mask

    def _failed_epic_ids_from_results(self) -> Set[str]:
        if not self.results_csv.exists():
            raise FileNotFoundError(f"Results CSV not found: {self.results_csv}")
        df = pd.read_csv(self.results_csv)
        if len(df) == 0:
            return set()
        failed_mask = self._failed_query_masks(df)
        failed = df.loc[failed_mask].copy()
        if len(failed) == 0:
            return set()

        ids: Set[str] = set()
        candidate_columns = ["query", "epic_id", "EPIC ID"]
        for col in candidate_columns:
            if col not in failed.columns:
                continue
            for v in failed[col].tolist():
                epic = self._extract_epic_number(v)
                if epic is not None:
                    ids.add(epic)
        return ids

    def _select_failed_queries(self, queries: Iterable[str], failed_epic_ids: Set[str]) -> List[str]:
        out: List[str] = []
        for q in queries:
            epic = self._extract_epic_number(q)
            if epic is None:
                continue
            if epic in failed_epic_ids:
                out.append(str(q))
        return out

    def _slice_targets(self, queries: List[str]) -> List[str]:
        if len(queries) == 0:
            return []
        start = max(0, int(self.start_index)) if self.start_index is not None else 0
        end_inclusive = int(self.end_index) if self.end_index is not None else (len(queries) - 1)
        end_inclusive = min(end_inclusive, len(queries) - 1)
        if end_inclusive < start:
            return []
        sliced = queries[start : end_inclusive + 1]
        if self.max_targets is not None:
            sliced = sliced[: max(0, self.max_targets)]
        return sliced

    def _load_progress(self) -> Optional[Dict[str, Any]]:
        if not self.progress_path.exists():
            return None
        try:
            data = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _write_progress(self, last_completed_index: int, last_completed_query: str) -> None:
        payload = {
            "last_completed_index": int(last_completed_index),
            "last_completed_query": str(last_completed_query),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _resolve_resume_start(self, total: int) -> int:
        data = self._load_progress()
        if data is None:
            return 0
        try:
            last_idx = int(data.get("last_completed_index", -1))
        except Exception:
            last_idx = -1
        start_idx = max(0, last_idx + 1)
        if start_idx >= total:
            print("[resume] progress_failed.json indicates all selected failed targets are done")
        else:
            print(f"[resume] resuming at failed target index {start_idx} ({start_idx + 1}/{total})")
        return start_idx

    def _ok_marker_path(self, epic_slug: str) -> Path:
        return self.epics_dir / epic_slug / "download_ok.txt"

    def _write_ok_marker(
        self,
        epic_slug: str,
        query: str,
        attempts: int,
        cache_path: str,
        cache_source: str,
    ) -> None:
        ok_path = self._ok_marker_path(epic_slug)
        ok_path.parent.mkdir(parents=True, exist_ok=True)
        content = [
            f"query={query}",
            f"epic_id={epic_slug}",
            f"status=ok",
            f"attempts={int(attempts)}",
            f"cache_path={cache_path}",
            f"cache_source={cache_source}",
            f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        ]
        ok_path.write_text("\n".join(content) + "\n", encoding="utf-8")

    def _flush_recovery_rows(self, rows: List[Dict[str, Any]]) -> None:
        if len(rows) == 0:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.failed_recovery_csv.exists()
        with self.failed_recovery_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.RECOVERY_COLUMNS)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in self.RECOVERY_COLUMNS})
        rows.clear()

    def run(self) -> Dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.epics_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        all_queries = self._load_queries()
        failed_ids = self._failed_epic_ids_from_results()
        failed_queries = self._select_failed_queries(all_queries, failed_ids)
        targets = self._slice_targets(failed_queries)

        print(f"[K2FailedDownloader] input_queries={len(all_queries)} failed_epics={len(failed_ids)} selected={len(targets)}")
        if len(targets) == 0:
            print("[K2FailedDownloader] no failed EPIC targets matched filters")
            return {
                "out_dir": self.out_dir,
                "failed_recovery_csv": self.failed_recovery_csv,
                "progress_failed_json": self.progress_path,
                "processed": 0,
            }

        start_idx = self._resolve_resume_start(total=len(targets))
        if start_idx >= len(targets):
            return {
                "out_dir": self.out_dir,
                "failed_recovery_csv": self.failed_recovery_csv,
                "progress_failed_json": self.progress_path,
                "processed": 0,
            }

        buffered_rows: List[Dict[str, Any]] = []
        total_processed = 0

        for idx in range(start_idx, len(targets)):
            query = targets[idx]
            epic_slug = K2BatchRunner._epic_slug(query=query, fallback_idx=(idx + 1))
            print(f"[{idx + 1}/{len(targets)}] retry download {query} -> {epic_slug}")

            status = "error"
            error_type = ""
            error_message = ""
            cache_path = ""
            cache_source = ""
            attempts_used = 0

            for attempt in range(1, self.retries + 1):
                attempts_used = attempt
                try:
                    fetched = self.handler.fetch_best(
                        query=query,
                        limit=self.limit,
                        exptime=self.exptime,
                        download_dir=str(self.cache_dir),
                        cache_only=False,
                    )
                    fetch_status = str(fetched.get("status", "ok")).strip().lower()
                    if fetch_status != "ok":
                        raise RuntimeError(f"fetch_status={fetch_status}")

                    status = "ok"
                    cache_path = str(fetched.get("cache_path", "") or "")
                    cache_source = str(fetched.get("cache_source", "") or "")
                    self._write_ok_marker(
                        epic_slug=epic_slug,
                        query=query,
                        attempts=attempts_used,
                        cache_path=cache_path,
                        cache_source=cache_source,
                    )
                    break
                except Exception as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    if attempt < self.retries:
                        print(f"[retry] {query} attempt {attempt}/{self.retries} failed: {error_type}: {error_message}")

            row = {
                "query": str(query),
                "epic_id": str(epic_slug),
                "status": str(status),
                "attempts": int(attempts_used),
                "cache_path": str(cache_path),
                "cache_source": str(cache_source),
                "error_type": str(error_type),
                "error_message": str(error_message),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            buffered_rows.append(row)
            total_processed += 1

            if len(buffered_rows) >= 50:
                self._flush_recovery_rows(buffered_rows)
            self._write_progress(last_completed_index=idx, last_completed_query=query)

        self._flush_recovery_rows(buffered_rows)
        print(f"[K2FailedDownloader] Wrote: {self.failed_recovery_csv}")
        print(f"[K2FailedDownloader] Wrote: {self.progress_path}")
        print(f"[K2FailedDownloader] processed={total_processed}")
        return {
            "out_dir": self.out_dir,
            "failed_recovery_csv": self.failed_recovery_csv,
            "progress_failed_json": self.progress_path,
            "processed": total_processed,
        }
