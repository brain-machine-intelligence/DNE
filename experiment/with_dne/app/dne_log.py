"""
Structured event logging for DNE experiment.
Table-style one line per event; in-memory status store for live view.
"""
import json
import logging
import sys
import time
from datetime import datetime

_logger = None

# In-memory status: workerId -> { subId, last_event, last_seen_ts, s_count?, progress?, ... }
_status_store = {}
_STORE_MAX = 200
_COL = {"ts": 19, "workerId": 14, "subId": 12, "event": 32}
_PROGRESS_KEYS = ("s_count", "cur_epoch", "progress", "total_epoch", "session", "round", "repetition", "total")


def _get_logger():
    global _logger
    if _logger is not None:
        return _logger
    _logger = logging.getLogger("dne")
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    if not _logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(h)
    return _logger


def _cell(s, width):
    s = str(s)[:width] if s else ""
    return (s + " " * width)[:width]


def _format_line(event, payload):
    """Table-style:  TS | workerId | subId | event | key=val ..."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wid = (payload.get("workerId") or "")[: _COL["workerId"]]
    sid = (payload.get("subId") or "")[: _COL["subId"]]
    ev = (event or "")[: _COL["event"]]
    parts = [
        "[DNE]",
        _cell(ts, _COL["ts"]),
        "|",
        _cell(wid, _COL["workerId"]),
        "|",
        _cell(sid, _COL["subId"]),
        "|",
        _cell(ev, _COL["event"]),
        "|",
    ]
    for k, v in sorted(payload.items()):
        if k in ("event", "workerId", "subId") or v is None:
            continue
        vstr = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        if " " in vstr or "\n" in vstr:
            vstr = json.dumps(vstr, ensure_ascii=False)
        parts.append(f" {k}={vstr}")
    return "".join(parts)


def _update_store(worker_id, sub_id, event, **extra):
    """Update in-memory status for live view; keep progress-related fields."""
    global _status_store
    if not worker_id:
        return
    t = time.time()
    prev = _status_store.get(worker_id, {})
    entry = {
        "subId": sub_id or prev.get("subId", ""),
        "last_event": event,
        "last_seen_ts": t,
    }
    for k in _PROGRESS_KEYS:
        if k in extra and extra[k] is not None:
            entry[k] = extra[k]
        elif k in prev:
            entry[k] = prev[k]
    _status_store[worker_id] = entry
    keys = list(_status_store.keys())
    if len(keys) > _STORE_MAX:
        for k in keys[: len(keys) - _STORE_MAX]:
            del _status_store[k]


def get_status_store():
    """Return copy of status store for /status page."""
    return dict(_status_store)


def update_heartbeat(worker_id, sub_id, **progress_kw):
    """Update last_seen (and optional session/round etc.). Call from /ping."""
    global _status_store
    if not worker_id:
        return
    t = time.time()
    prev = _status_store.get(worker_id, {})
    entry = {
        "subId": sub_id or prev.get("subId", ""),
        "last_event": prev.get("last_event", "heartbeat"),
        "last_seen_ts": t,
    }
    for k in _PROGRESS_KEYS:
        if k in progress_kw and progress_kw[k] is not None:
            entry[k] = progress_kw[k]
        elif k in prev:
            entry[k] = prev[k]
    _status_store[worker_id] = entry


def log_event(event, **kwargs):
    """
    Log one event in table-style one line; update in-memory status.
    Adds workerId/subId from session when in request context.
    """
    payload = {"event": event}
    try:
        from flask import has_request_context, session
        if has_request_context():
            if session.get("workerId") is not None:
                payload["workerId"] = session.get("workerId")
            if session.get("subId") is not None:
                payload["subId"] = session.get("subId")
    except Exception:
        pass
    payload.update(kwargs)
    payload = {k: v for k, v in payload.items() if v is not None}

    extra = {k: payload[k] for k in _PROGRESS_KEYS if k in payload}
    _update_store(payload.get("workerId"), payload.get("subId"), event, **extra)

    line = _format_line(event, payload)
    _get_logger().info(line)

    try:
        from flask import current_app
        if current_app and current_app.logger:
            current_app.logger.info(line)
    except Exception:
        pass
