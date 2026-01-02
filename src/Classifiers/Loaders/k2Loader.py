import io
import os
import pandas as pd
from urllib.parse import urlencode, quote_plus
from urllib.request import Request, urlopen
from urllib.error import HTTPError

###############This is a testing class ============
############ Not used
class K2Loader:
    def __init__(self, tap_url: str = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"):
        self.tap_url = tap_url
        self.cols = ["star_id", "pl_orbper", "pl_tranmid", "pl_trandurh", "mission"]

    def loadfile(self, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            self.fetch_k2_ephemerides(filepath)

        df = pd.read_csv(filepath)

        # Back-compat: older cached files won't have mission
        if "mission" not in df.columns:
            df["mission"] = "k2"
            df.to_csv(filepath, index=False)  # repair the cached file

        # Optional: enforce order if you want
        return df[[c for c in self.cols if c in df.columns]]


    def fetch_k2_ephemerides(self, outpath: str) -> pd.DataFrame:
        # Join: EPIC name from k2pandc, ephemerides from pscomppars
        query = """
            SELECT
                k.epic_hostname AS star_id,
                p.pl_orbper     AS pl_orbper,
                p.pl_tranmid    AS pl_tranmid,
                p.pl_trandur    AS pl_trandurh,
                'k2'            AS mission
            FROM pscomppars p
            JOIN k2pandc k
              ON p.pl_name = k.pl_name
            WHERE k.default_flag = 1
              AND k.disposition  = 'CONFIRMED'
              AND p.pl_orbper  IS NOT NULL
              AND p.pl_tranmid IS NOT NULL
              AND p.pl_trandur IS NOT NULL
        """
        # TAP best practice: single-line query :contentReference[oaicite:3]{index=3}
        query = " ".join(query.split())

        # Prefer POST (avoids URL-length/encoding weirdness)
        body = urlencode({"query": query, "format": "csv"}).encode("utf-8")
        req = Request(
            self.tap_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        try:
            with urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except HTTPError as e:
            # Expose the server's actual complaint (super useful for ADQL debugging)
            details = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(
                f"TAP request failed: HTTP {e.code}\n"
                f"URL: {self.tap_url}\n"
                f"Query: {query}\n"
                f"Server response:\n{details[:2000]}"
            ) from e

        df = pd.read_csv(io.StringIO(text))

        # Enforce schema/order
        df = df[self.cols].dropna().drop_duplicates()

        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        df.to_csv(outpath, index=False)
        print(f"Saved: {len(df)} rows -> {outpath}")
        return df
