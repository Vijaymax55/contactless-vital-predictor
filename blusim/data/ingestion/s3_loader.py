"""AWS S3 data loader for piezo mat CSV files.

Replaces the hardcoded ``mat_data_get.py`` in the original GOD dashboard.
All paths and credentials come from environment variables / Hydra config —
no literals in code.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


class S3MatLoader:
    """Downloads hourly piezo mat CSV files from S3 for a given date range.

    Sleep sessions span 21:00 on the previous day to 09:00 on the target
    date (matching the original system's convention).

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix (subject/device ID folder).
        region: AWS region string.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        region: str = "ap-south-1",
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._s3 = boto3.client("s3", region_name=region)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_session(self, date: str | datetime, dest_dir: Path | str) -> list[Path]:
        """Download all CSV files for a sleep session covering date.

        Files downloaded: 21:00–23:00 on the previous day,
        then 00:00–08:00 on the given date.

        Args:
            date: Session date as ``"YYYYMMDD"`` string or datetime.
            dest_dir: Local directory to write downloaded files.

        Returns:
            List of local paths to downloaded files.
        """
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y%m%d")
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        date_str = date.strftime("%Y-%m-%d")
        prev_str = (date - timedelta(days=1)).strftime("%Y-%m-%d")

        target_files: list[str] = []
        for hour in range(21, 24):
            target_files.append(f"sleep_data_{prev_str}_{hour:02d}.csv")
        for hour in range(0, 9):
            target_files.append(f"sleep_data_{date_str}_{hour:02d}.csv")

        downloaded: list[Path] = []
        objects = self._list_objects()
        for key in objects:
            fname = key.split("/")[-1]
            if fname in target_files:
                local_path = dest_dir / fname
                if not local_path.exists():
                    self._s3.download_file(self._bucket, key, str(local_path))
                    logger.info("Downloaded %s → %s", key, local_path)
                downloaded.append(local_path)

        logger.info("Downloaded %d files for session %s", len(downloaded), date_str)
        return downloaded

    def load_session_df(self, date: str | datetime, cache_dir: Optional[Path] = None) -> pd.DataFrame:
        """Download + merge all session CSVs into one sorted DataFrame.

        Args:
            date: Session date.
            cache_dir: If provided, CSVs are cached here to avoid re-download.

        Returns:
            Merged DataFrame sorted by ``Time`` with piezo channel columns.
        """
        dest = cache_dir or Path(f"/tmp/blusim/{date}")
        paths = self.download_session(date, dest)

        frames: list[pd.DataFrame] = []
        for p in sorted(paths):
            try:
                df = pd.read_csv(p)
                frames.append(df)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", p, exc)

        if not frames:
            logger.warning("No data downloaded for session %s", date)
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        if "Time" in merged.columns:
            merged["Time"] = pd.to_datetime(merged["Time"])
            merged = merged.sort_values("Time").reset_index(drop=True)
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _list_objects(self) -> list[str]:
        """List all object keys under the configured prefix."""
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix + "/"):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
