"""本地教学记录持久化。"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import uuid


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _collection_path(collection: str, data_dir: Path | None = None) -> Path:
    safe_name = "".join(ch for ch in collection if ch.isalnum() or ch in {"_", "-"})
    if not safe_name:
        raise ValueError("记录集合名称无效。")
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{safe_name}_records.json"


def load_records(collection: str, data_dir: Path | None = None) -> list[dict]:
    path = _collection_path(collection, data_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_record(collection: str, record: dict, data_dir: Path | None = None) -> dict:
    path = _collection_path(collection, data_dir)
    records = load_records(collection, data_dir)
    saved = {
        "record_id": uuid.uuid4().hex,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **record,
    }
    records.append(saved)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return saved


def latest_record(collection: str, data_dir: Path | None = None) -> dict | None:
    records = load_records(collection, data_dir)
    return records[-1] if records else None
