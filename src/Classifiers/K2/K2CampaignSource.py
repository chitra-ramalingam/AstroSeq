from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

import pandas as pd
import requests


@dataclass
class K2CampaignEpicSource:
    """
    Download EPIC IDs from the official K2 Science Center 'Target list (csv)' for a campaign.
    Campaign 5 target list is linked from the K2 approved targets page. :contentReference[oaicite:2]{index=2}
    """
    campaign: int = 5
    cache_dir: Path = Path("data/k2_target_lists")

    def _targets_url(self) -> str:
        # This is the file linked from the Campaign section on the K2 approved targets page.
        # (Example for C5 shown on the page) :contentReference[oaicite:3]{index=3}
        c = int(self.campaign)
        return f"https://keplergo.github.io/KeplerScienceWebsite/data/campaigns/c{c}/K2Campaign{c}targets.csv"

    def _cache_path(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"K2Campaign{int(self.campaign)}targets.csv"

    @staticmethod
    def _canon_epic(x) -> str:
        # digits only
        return re.sub(r"\D+", "", str(x))

    @staticmethod
    def _find_epic_column(df: pd.DataFrame) -> str:
        # pick the first column whose name contains "epic" (case-insensitive)
        for col in df.columns:
            if "epic" in str(col).lower():
                return col
        # fallback: first column
        return str(df.columns[0])

    def fetch_epic_ids(self, *, prefix: bool = True, use_cache: bool = True) -> List[str]:
        url = self._targets_url()
        cache_path = self._cache_path()

        if use_cache and cache_path.exists():
            df = pd.read_csv(cache_path)
        else:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            cache_path.write_bytes(r.content)
            df = pd.read_csv(cache_path)

        epic_col = self._find_epic_column(df)
        epics = [self._canon_epic(v) for v in df[epic_col].tolist()]
        epics = [e for e in epics if e]  # drop blanks

        # dedupe, keep order
        seen = set()
        out: List[str] = []
        for e in epics:
            if e not in seen:
                seen.add(e)
                out.append(f"EPIC_{e}" if prefix else e)

        return out
