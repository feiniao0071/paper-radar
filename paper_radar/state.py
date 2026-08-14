from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from paper_radar.models import Paper


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "papers": {}}
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("version") != 1 or not isinstance(data.get("papers"), dict):
            raise ValueError(f"Unsupported or malformed state file: {self.path}")
        return data

    def contains(self, paper_id: str) -> bool:
        return paper_id in self.data["papers"]

    def mark(self, paper: Paper, status: str, *, now: datetime | None = None) -> None:
        timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        self.data["papers"][paper.paper_id] = {
            "status": status,
            "seen_at": timestamp,
            "published": paper.published.date().isoformat(),
            "title": paper.title[:240],
        }

    def prune(self, retention_days: int, *, now: datetime | None = None) -> None:
        current_time = (now or datetime.now(UTC)).astimezone(UTC)
        cutoff = current_time - timedelta(days=retention_days)
        retained = {}
        for paper_id, record in self.data["papers"].items():
            try:
                seen_at = datetime.fromisoformat(record["seen_at"])
            except (KeyError, TypeError, ValueError):
                continue
            if seen_at.astimezone(UTC) >= cutoff:
                retained[paper_id] = record
        self.data["papers"] = retained

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.data, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(self.path)

