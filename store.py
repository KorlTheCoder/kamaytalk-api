"""
KamayTalk hybrid store.

  Redis      = HOT layer: live session snapshots, recent events, counters, round setting.
               Fast, shared, good for the admin "live" view.
  Firestore  = DURABLE layer: permanent archive of every event and every feedback form.
  Local file = SAFETY NET: every event/feedback is also appended to data/*.jsonl, so
               nothing is lost even if Redis/Firestore aren't set up or go down.

Both Redis and Firestore are optional. With neither configured the app still works
(in-memory + local files). All network I/O happens on background threads so a slow or
broken database can never slow down the camera frame loop.

Config (environment variables):
  REDIS_URL            e.g. redis://localhost:6379/0   or  rediss://default:PASSWORD@host:port
  FIRESTORE_ENABLED    1 to turn on Firestore
  GOOGLE_APPLICATION_CREDENTIALS   path to the Firebase service-account JSON key
  FIREBASE_PROJECT_ID  optional (read from the key file otherwise)
  KAMAY_DATA_DIR       where the local .jsonl files go (default: api/data)
"""
from __future__ import annotations

import csv
import io
import json
import os
import queue
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

BASE = Path(__file__).resolve().parent
EVENTS_MAX = 5000
FEEDBACK_MAX = 5000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clip(s: Any, n: int = 500) -> str:
    return str(s or "").strip()[:n]


class Store:
    def __init__(self, redis_client: Any = None, fs_client: Any = None, connect: bool = True) -> None:
        self.data_dir = Path(os.environ.get("KAMAY_DATA_DIR", BASE / "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.status = {"redis": "off", "firestore": "off", "local": str(self.data_dir)}
        self.r = redis_client
        self.fs = fs_client
        if redis_client is not None:
            self.status["redis"] = "connected"
        if fs_client is not None:
            self.status["firestore"] = "connected"

        # in-memory mirror (used when Redis is off, and as a cache of what we wrote)
        self._events: deque[dict] = deque(maxlen=EVENTS_MAX)
        self._feedback: list[dict] = []
        self._counts: dict[int, Counter] = {}
        self._words: dict[int, Counter] = {}
        self._sessions: dict[str, tuple[float, dict]] = {}
        self._round = 1
        self._sent_snap: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

        self._load_local()
        if connect:
            if self.r is None:
                self._connect_redis()
            if self.fs is None:
                self._connect_firestore()
        self._sync_round_from_backends()

        self._q: queue.Queue = queue.Queue(maxsize=20000)    # local file + Redis
        self._qf: queue.Queue = queue.Queue(maxsize=20000)   # Firestore (slow, isolated)
        threading.Thread(target=self._worker, args=(self._q, self._do_fast), daemon=True).start()
        threading.Thread(target=self._worker, args=(self._qf, self._do_firestore), daemon=True).start()

    # ------------------------------------------------------------------ connect
    def _connect_redis(self) -> None:
        url = os.environ.get("REDIS_URL", "").strip()
        if not url:
            return
        try:
            import redis

            c = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=3, socket_timeout=3)
            c.ping()
            self.r = c
            self.status["redis"] = "connected"
        except Exception as e:  # noqa: BLE001
            self.r = None
            self.status["redis"] = f"error: {str(e)[:160]}"

    def _connect_firestore(self) -> None:
        if os.environ.get("FIRESTORE_ENABLED", "0").lower() not in ("1", "true", "yes"):
            return
        try:
            from google.cloud import firestore

            c = firestore.Client(project=os.environ.get("FIREBASE_PROJECT_ID") or None)
            next(iter(c.collection("kt_ping").limit(1).stream(timeout=8)), None)  # cheap auth check
            self.fs = c
            self.status["firestore"] = "connected"
        except Exception as e:  # noqa: BLE001
            self.fs = None
            self.status["firestore"] = f"error: {str(e)[:160]}"

    # -------------------------------------------------------------- local files
    def _load_local(self) -> None:
        """Rebuild the in-memory mirror after a restart."""
        ev = self.data_dir / "events.jsonl"
        fb = self.data_dir / "feedback.jsonl"
        rd = self.data_dir / "round.txt"
        try:
            if rd.exists():
                self._round = max(1, int(rd.read_text().strip() or 1))
        except Exception:  # noqa: BLE001
            pass
        for path, sink in ((ev, "event"), (fb, "feedback")):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if sink == "event":
                    self._mem_add_event(d)
                else:
                    self._feedback.append(d)

    def _append_local(self, name: str, doc: dict) -> None:
        with open(self.data_dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ workers
    def _worker(self, q: queue.Queue, fn) -> None:
        while True:
            job = q.get()
            try:
                fn(job)
            except Exception as e:  # noqa: BLE001
                print(f"[store] background write failed: {type(e).__name__}: {e}")
            finally:
                q.task_done()

    def _enqueue(self, q: queue.Queue, job: tuple) -> None:
        try:
            q.put_nowait(job)
        except queue.Full:
            print("[store] queue full, dropping job", job[0])

    def _do_fast(self, job: tuple) -> None:
        kind, doc = job
        if kind == "event":
            self._append_local("events.jsonl", doc)
        elif kind == "feedback":
            self._append_local("feedback.jsonl", doc)
        elif kind == "round":
            (self.data_dir / "round.txt").write_text(str(doc["round"]))

        if self.r is None:
            return
        try:
            rd = int(doc.get("round", 1)) if kind in ("event", "feedback") else 0
            if kind == "event":
                p = self.r.pipeline()
                p.lpush("kt:events", json.dumps(doc, ensure_ascii=False))
                p.ltrim("kt:events", 0, EVENTS_MAX - 1)
                p.hincrby(f"kt:counts:r{rd}", doc["kind"], 1)
                if doc["kind"] == "word":
                    p.hincrby(f"kt:words:r{rd}", doc["text"], 1)
                p.execute()
            elif kind == "feedback":
                p = self.r.pipeline()
                p.lpush("kt:feedback", json.dumps(doc, ensure_ascii=False))
                p.ltrim("kt:feedback", 0, FEEDBACK_MAX - 1)
                p.hincrby(f"kt:counts:r{rd}", "feedback", 1)
                p.execute()
            elif kind == "session":
                p = self.r.pipeline()
                p.set(f"kt:session:{doc['sid']}", json.dumps(doc, ensure_ascii=False), ex=3600)
                p.zadd("kt:sessions", {doc["sid"]: doc["ts"]})
                p.zremrangebyscore("kt:sessions", 0, doc["ts"] - 3600)
                p.execute()
            elif kind == "round":
                self.r.set("kt:round", str(doc["round"]))
            self.status["redis"] = "connected"
        except Exception as e:  # noqa: BLE001
            self.status["redis"] = f"error: {str(e)[:160]}"

    def _do_firestore(self, job: tuple) -> None:
        if self.fs is None:
            return
        kind, doc = job
        try:
            coll = {"event": "events", "feedback": "feedback"}[kind]
            self.fs.collection(coll).document(doc["id"]).set(doc)
            self.status["firestore"] = "connected"
        except Exception as e:  # noqa: BLE001
            self.status["firestore"] = f"error: {str(e)[:160]}"

    def flush(self, timeout: float = 15.0) -> None:
        """Wait until queued writes are done (used by tests and on shutdown)."""
        end = time.time() + timeout
        for q in (self._q, self._qf):
            while q.unfinished_tasks and time.time() < end:
                time.sleep(0.02)

    # ------------------------------------------------------------------ rounds
    def _sync_round_from_backends(self) -> None:
        if self.r is not None:
            try:
                v = self.r.get("kt:round")
                if v:
                    self._round = max(1, int(v))
            except Exception:  # noqa: BLE001
                pass

    def get_round(self) -> int:
        return self._round

    def set_round(self, n: int) -> int:
        n = max(1, min(int(n), 9))
        self._round = n
        self._enqueue(self._q, ("round", {"round": n}))
        return n

    # ------------------------------------------------------------------ writes
    def _mem_add_event(self, d: dict) -> None:
        self._events.appendleft(d)
        rd = int(d.get("round", 1))
        self._counts.setdefault(rd, Counter())[d["kind"]] += 1
        if d["kind"] == "word":
            self._words.setdefault(rd, Counter())[d["text"]] += 1

    def add_event(self, kind: str, text: str, session: str, mode: str, client: str,
                  confidence: Optional[float] = None, extra: Optional[dict] = None) -> dict:
        doc = {
            "id": uuid.uuid4().hex,
            "ts": now_iso(),
            "round": self._round,
            "session": _clip(session, 60),
            "client": _clip(client, 20),
            "mode": mode,
            "kind": kind,            # word | sentence | unsure
            "text": _clip(text, 300),
            "confidence": None if confidence is None else round(float(confidence), 3),
        }
        if extra:
            doc.update(extra)
        with self._lock:
            self._mem_add_event(doc)
        self._enqueue(self._q, ("event", doc))
        self._enqueue(self._qf, ("event", doc))
        return doc

    def add_feedback(self, data: dict) -> dict:
        would = data.get("would_use")
        doc = {
            "id": uuid.uuid4().hex,
            "ts": now_iso(),
            "round": self._round,
            "session": _clip(data.get("session"), 60) or "unknown",
            "client": _clip(data.get("client"), 20) or "web",
            "rating": data["rating"] if type(data.get("rating")) is int and 1 <= data["rating"] <= 5 else None,
            "liked": _clip(data.get("liked")),
            "confused": _clip(data.get("confused")),
            "would_use": would if would in ("yes", "maybe", "no") else None,
            "comment": _clip(data.get("comment")),
        }
        with self._lock:
            self._feedback.append(doc)
            self._counts.setdefault(self._round, Counter())["feedback"] += 1
        self._enqueue(self._q, ("feedback", doc))
        self._enqueue(self._qf, ("feedback", doc))
        return doc

    def touch_session(self, sid: str, snapshot: dict) -> None:
        """Called every frame; only actually written every ~2 s or when the subtitle changes."""
        now = time.time()
        sig = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
        last = self._sent_snap.get(sid)
        if last and last[1] == sig and now - last[0] < 2.0:
            return
        self._sent_snap[sid] = (now, sig)
        doc = {"sid": sid, "ts": now, **snapshot}
        self._sessions[sid] = (now, doc)
        if len(self._sessions) > 200:  # keep memory bounded
            for k in sorted(self._sessions, key=lambda k: self._sessions[k][0])[:50]:
                self._sessions.pop(k, None)
                self._sent_snap.pop(k, None)
        self._enqueue(self._q, ("session", doc))

    # ------------------------------------------------------------------- reads
    def _redis_ok(self) -> bool:
        return self.r is not None and not str(self.status["redis"]).startswith("error")

    def recent_events(self, limit: int = 50, source: str = "hot", round_: Optional[int] = None) -> list[dict]:
        limit = max(1, min(limit, 500))
        if source == "archive" and self.fs is not None:
            try:
                q = self.fs.collection("events").order_by("ts", direction="DESCENDING").limit(limit)
                return [d.to_dict() for d in q.stream()]
            except Exception as e:  # noqa: BLE001
                self.status["firestore"] = f"error: {str(e)[:160]}"
        rows: list[dict] = []
        if self._redis_ok():
            try:
                rows = [json.loads(x) for x in self.r.lrange("kt:events", 0, EVENTS_MAX - 1)]
            except Exception as e:  # noqa: BLE001
                self.status["redis"] = f"error: {str(e)[:160]}"
                rows = []
        if not rows:
            rows = list(self._events)
        if round_:
            rows = [d for d in rows if int(d.get("round", 1)) == round_]
        return rows[:limit]

    def all_events(self, round_: Optional[int] = None) -> list[dict]:
        return self.recent_events(EVENTS_MAX, "hot", round_)

    def all_feedback(self, round_: Optional[int] = None) -> list[dict]:
        rows: list[dict] = []
        if self._redis_ok():
            try:
                rows = [json.loads(x) for x in self.r.lrange("kt:feedback", 0, FEEDBACK_MAX - 1)]
            except Exception as e:  # noqa: BLE001
                self.status["redis"] = f"error: {str(e)[:160]}"
        if not rows:
            rows = list(reversed(self._feedback))
        if round_:
            rows = [d for d in rows if int(d.get("round", 1)) == round_]
        return rows

    def sessions(self, window: int = 90, limit: int = 20) -> list[dict]:
        cutoff = time.time() - window
        out: list[dict] = []
        if self._redis_ok():
            try:
                ids = self.r.zrevrangebyscore("kt:sessions", "+inf", cutoff, start=0, num=limit)
                if ids:
                    vals = self.r.mget([f"kt:session:{i}" for i in ids])
                    out = [json.loads(v) for v in vals if v]
                    return out
                return []
            except Exception as e:  # noqa: BLE001
                self.status["redis"] = f"error: {str(e)[:160]}"
        rows = [d for ts, d in self._sessions.values() if ts >= cutoff]
        return sorted(rows, key=lambda d: -d["ts"])[:limit]

    def summary(self, round_: Optional[int] = None) -> dict:
        rd = self._round if round_ is None else round_      # 0 = all rounds combined
        rounds = list(range(1, 10)) if rd == 0 else [rd]
        counts: Counter = Counter()
        words: Counter = Counter()
        got_redis = False
        if self._redis_ok():
            try:
                for n in rounds:
                    counts.update({k: int(v) for k, v in self.r.hgetall(f"kt:counts:r{n}").items()})
                    words.update({k: int(v) for k, v in self.r.hgetall(f"kt:words:r{n}").items()})
                got_redis = True
            except Exception as e:  # noqa: BLE001
                self.status["redis"] = f"error: {str(e)[:160]}"
        if not got_redis or not counts:
            counts, words = Counter(), Counter()
            for n in rounds:
                counts.update(self._counts.get(n, Counter()))
                words.update(self._words.get(n, Counter()))
        top = words.most_common(15)

        fb = self.all_feedback(rd or None)
        ratings = [d["rating"] for d in fb if isinstance(d.get("rating"), int)]
        would = Counter(d["would_use"] for d in fb if d.get("would_use"))
        return {
            "round": rd,
            "current_round": self._round,
            "status": dict(self.status),
            "counts": {"word": counts.get("word", 0), "sentence": counts.get("sentence", 0),
                       "unsure": counts.get("unsure", 0), "feedback": len(fb)},
            "top_words": [{"word": w, "n": n} for w, n in top],
            "feedback": {
                "n": len(fb),
                "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
                "would_use": {k: would.get(k, 0) for k in ("yes", "maybe", "no")},
                "liked": [d["liked"] for d in fb if d.get("liked")][:30],
                "confused": [d["confused"] for d in fb if d.get("confused")][:30],
            },
            "active_sessions": len(self.sessions()),
        }

    # ------------------------------------------------------------------ export
    FEEDBACK_COLS = ["ts", "round", "session", "client", "rating", "would_use", "liked", "confused", "comment"]
    EVENT_COLS = ["ts", "round", "session", "client", "mode", "kind", "text", "confidence"]

    def export_csv(self, which: str, round_: Optional[int] = None) -> str:
        cols = self.FEEDBACK_COLS if which == "feedback" else self.EVENT_COLS
        rows = self.all_feedback(round_) if which == "feedback" else self.all_events(round_)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(cols)
        def cell(v: Any) -> str:
            # guard against spreadsheet formula injection from free-text fields
            s = "" if v is None else str(v)
            return "'" + s if s and s[0] in "=+-@" else s

        for d in reversed(rows):  # oldest first
            w.writerow([cell(d.get(c)) for c in cols])
        return buf.getvalue()
