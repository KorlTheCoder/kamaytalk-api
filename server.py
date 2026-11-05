"""
KamayTalk API: camera frames in, ASL subtitles out.

Built on the pretrained models from TheRollerBlader/SignBridge (MIT) -
see vendor/UPSTREAM_LICENSE_SignBridge_MIT.txt.

A mobile (or browser) client posts raw camera frames to /v1/frame.
The server runs MediaPipe Holistic (both hands + pose) and then either:
  mode=letters -> TFLite fingerspelling classifier (A-Z, static letters)
  mode=words   -> PyTorch WLASL word classifier on a buffered frame sequence
and returns the running subtitle plus whatever should be spoken now.

Run:  uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import copy
import csv
import atexit
import itertools
import os
import secrets
import sys
import threading
import time
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from store import Store

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "vendor"))


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader so Windows users can put settings in api/.env instead of env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k == "GOOGLE_APPLICATION_CREDENTIALS" and v and not Path(v).is_absolute():
            v = str(BASE / v)  # relative paths are relative to the api/ folder
        os.environ.setdefault(k, v)


_load_dotenv(BASE / ".env")

# ----------------------------------------------------------------------------
# Config (override with environment variables)
# ----------------------------------------------------------------------------
MAX_FRAMES = int(os.environ.get("WLASL_MAX_FRAMES", "50"))       # must match training
MIN_WORD_FRAMES = int(os.environ.get("MIN_WORD_FRAMES", "10"))   # hand frames needed before predicting
WORD_GAP_FRAMES = int(os.environ.get("WORD_GAP_FRAMES", "8"))    # frames without hands = end of sign
WORD_MIN_CONF = float(os.environ.get("WORD_MIN_CONF", "0.50"))   # full-vocabulary probability needed to accept a word
WORD_VOCAB = os.environ.get("WORD_VOCAB", "reliable")            # reliable | all
MIRROR = os.environ.get("MIRROR", "1") == "1"                    # upstream demos flip the webcam image
LETTER_VOTES = int(os.environ.get("LETTER_VOTES", "6"))          # of last 8 frames must agree
LETTER_MIN_CONF = float(os.environ.get("LETTER_MIN_CONF", "0.80"))
SPACE_GAP_FRAMES = int(os.environ.get("SPACE_GAP_FRAMES", "20"))  # hands gone this long = word break
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "12"))
SENTENCE_GAP_SECONDS = float(os.environ.get("SENTENCE_GAP_SECONDS", "2.5"))  # pause that ends a sentence

# Words that scored >=75% on >=4 val+test samples in the upstream training report.
# (Small samples - treat as "best bets", not guarantees.)
RELIABLE_WORDS = [
    "help", "dark", "forget", "wrong", "language", "tall", "full", "table",
    "basketball", "brown", "enjoy", "study", "time", "work", "bird", "paint",
]

WLASL_100_GLOSSES = [
    "accident", "africa", "all", "apple", "basketball", "bed", "before",
    "bird", "birthday", "black", "blue", "book", "bowling", "brown", "but",
    "can", "candy", "chair", "change", "cheat", "city", "clothes", "color",
    "computer", "cook", "cool", "corn", "cousin", "cow", "dance", "dark",
    "deaf", "decide", "doctor", "dog", "drink", "eat", "enjoy", "family",
    "fine", "finish", "fish", "forget", "full", "give", "go", "graduate",
    "hat", "hearing", "help", "hot", "how", "jacket", "kiss", "language",
    "last", "later", "letter", "like", "man", "many", "medicine", "meet",
    "mother", "need", "no", "now", "orange", "paint", "paper", "pink",
    "pizza", "play", "pull", "purple", "right", "same", "school",
    "secretary", "shirt", "short", "son", "study", "table", "tall", "tell",
    "thanksgiving", "thin", "thursday", "time", "walk", "want", "what",
    "white", "who", "woman", "work", "wrong", "year", "yes",
]

# ----------------------------------------------------------------------------
# Fingerspelling model (TFLite)
# ----------------------------------------------------------------------------
try:  # LiteRT (preferred) -> tflite_runtime -> full TensorFlow, whichever your OS has wheels for
    from ai_edge_litert.interpreter import Interpreter  # noqa: E402
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore  # noqa: E402
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore  # noqa: E402


def _load_labels(path: Path) -> list[str]:
    with open(path, encoding="utf-8-sig") as f:
        return [r[0].strip() for r in csv.reader(f) if r]


_letter_interp = Interpreter(model_path=str(BASE / "models/Fingerspelling/keypoint_classifier.tflite"))
_letter_interp.allocate_tensors()
_letter_in = _letter_interp.get_input_details()[0]
_letter_out = _letter_interp.get_output_details()[0]
LETTER_LABELS = _load_labels(BASE / "models/Fingerspelling/keypoint_classifier_label.csv")
_letter_lock = threading.Lock()


def _pre_process_landmark(points: list[list[int]]) -> list[float]:
    """Same maths as upstream fingerspelling_demo.py: wrist-relative, flattened, max-normalised."""
    pts = copy.deepcopy(points)
    bx, by = pts[0][0], pts[0][1]
    for p in pts:
        p[0] -= bx
        p[1] -= by
    flat = list(itertools.chain.from_iterable(pts))
    m = max(abs(v) for v in flat)
    return [v / m if m else 0.0 for v in flat]


def predict_letter(hand_landmarks, w: int, h: int) -> tuple[str, float]:
    pts = [
        [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
        for lm in hand_landmarks.landmark
    ]
    x = np.array([_pre_process_landmark(pts)], dtype=np.float32)
    with _letter_lock:
        _letter_interp.set_tensor(_letter_in["index"], x)
        _letter_interp.invoke()
        probs = np.squeeze(_letter_interp.get_tensor(_letter_out["index"]))
    i = int(np.argmax(probs))
    return LETTER_LABELS[i], float(probs[i])


# ----------------------------------------------------------------------------
# Word model (PyTorch) - optional: letters keep working if torch is missing
# ----------------------------------------------------------------------------
WORDS_ERROR: Optional[str] = None
_word_model = None
_word_mask = None  # additive mask over classes (0 allowed, -inf blocked)
WORD_LABELS: list[str] = []
WORD_INPUT = 258
_torch = None
_word_lock = threading.Lock()

try:
    import torch as _torch  # type: ignore
    import wlasl_demo as wd  # vendored from upstream

    _ck = Path(os.environ.get("WLASL_CHECKPOINT", BASE / "models/wlasl/wlasl_best.pt"))
    _word_model, _arch, _n = wd.load_model_from_checkpoint(
        _ck, _torch.device("cpu"), None, 0.4, False, None
    )
    _word_model.eval()
    WORD_LABELS = list(WLASL_100_GLOSSES) if _n == 100 else [f"class_{i}" for i in range(_n)]
    WORD_INPUT = int(_word_model.input_norm.normalized_shape[0])
    mask = np.full(_n, -np.inf, dtype=np.float32)
    if WORD_VOCAB == "all" or _n != 100:
        mask[:] = 0.0
    else:
        for i, g in enumerate(WORD_LABELS):
            if g in RELIABLE_WORDS:
                mask[i] = 0.0
    _word_mask = mask
except Exception as e:  # noqa: BLE001
    WORDS_ERROR = f"{type(e).__name__}: {e}"


def vocab_words() -> list[str]:
    if _word_model is None:
        return []
    return [g for g, m in zip(WORD_LABELS, _word_mask) if m == 0.0]


def predict_word(seq: list[np.ndarray]) -> tuple[str, float, list[tuple[str, float]]]:
    x = np.stack(seq[:MAX_FRAMES]).astype(np.float32)
    if len(x) < MAX_FRAMES:  # zero-pad the end exactly like training/extraction
        x = np.vstack([x, np.zeros((MAX_FRAMES - len(x), x.shape[1]), np.float32)])
    with _word_lock, _torch.no_grad():
        logits = _word_model(_torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
    # Probability over ALL classes. We rank only the allowed words, but we report each
    # word's *full-vocabulary* probability. If the model thinks the sign is really some
    # other word (or noise), the allowed words get little mass and are rejected by
    # WORD_MIN_CONF, instead of being re-normalised into a confident wrong answer.
    e = np.exp(logits - logits.max())
    full = e / e.sum()
    order = [i for i in np.argsort(-full) if _word_mask[i] == 0.0][:3]
    top = [(WORD_LABELS[i], float(full[i])) for i in order]
    return top[0][0], top[0][1], top


# ----------------------------------------------------------------------------
# Frame decoding + features
# ----------------------------------------------------------------------------
def decode_frame(body: bytes, fmt: str, w: int, h: int, stride: int, rot: int) -> np.ndarray:
    if fmt == "jpeg":
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode JPEG")
    elif fmt == "nv21":
        if w <= 0 or h <= 0 or len(body) < w * h * 3 // 2:
            raise ValueError(f"nv21 buffer too small: got {len(body)} for {w}x{h}")
        yuv = np.frombuffer(body, np.uint8)[: w * h * 3 // 2].reshape(h * 3 // 2, w)
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    elif fmt == "bgra":
        stride = stride or w * 4
        if len(body) < stride * h:
            raise ValueError("bgra buffer too small")
        arr = np.frombuffer(body, np.uint8)[: stride * h].reshape(h, stride)[:, : w * 4].reshape(h, w, 4)
        img = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"unknown fmt {fmt!r}")

    rot %= 360
    if rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def hands_pose_features(res) -> np.ndarray:
    """258 = pose 33x4 + left hand 21x3 + right hand 21x3 (identical to upstream training)."""
    pose = (
        np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks.landmark]).flatten()
        if res.pose_landmarks else np.zeros(33 * 4)
    )
    lh = (
        np.array([[p.x, p.y, p.z] for p in res.left_hand_landmarks.landmark]).flatten()
        if res.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[p.x, p.y, p.z] for p in res.right_hand_landmarks.landmark]).flatten()
        if res.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh]).astype(np.float32)


# ----------------------------------------------------------------------------
# Per-client session: holds the MediaPipe tracker, frame buffer and subtitle
# ----------------------------------------------------------------------------
class Session:
    def __init__(self) -> None:
        import mediapipe as mp

        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.lock = threading.Lock()
        self.touched = time.time()
        self.mode: Optional[str] = None
        self.clear()

    def clear(self) -> None:
        self.tokens: list[str] = []
        self.seq: list[np.ndarray] = []
        self.miss = 0
        self.letter_hist: deque[tuple[str, float]] = deque(maxlen=8)
        self.last_letter: Optional[str] = None
        self.current_word = ""
        self.space_done = True
        self.last_live: Optional[dict] = None   # last finished word guess (stays visible)
        self.last_top3: list = []
        self.last_accepted: Optional[bool] = None
        self.sentences: list[str] = []           # finished sentences (newest last)
        self.last_hand_ts = time.time()

    def close(self) -> None:
        try:
            self.holistic.close()
        except Exception:  # noqa: BLE001
            pass


_sessions: "OrderedDict[str, Session]" = OrderedDict()
_sessions_lock = threading.Lock()


def get_session(sid: str) -> Session:
    with _sessions_lock:
        s = _sessions.get(sid)
        if s is None:
            s = Session()
            _sessions[sid] = s
            while len(_sessions) > MAX_SESSIONS:
                _, old = _sessions.popitem(last=False)
                old.close()
        _sessions.move_to_end(sid)
        s.touched = time.time()
        return s


def subtitle_text(s: Session) -> str:
    if s.mode == "letters":
        return " ".join(s.tokens + ([s.current_word] if s.current_word else [])).strip()
    return " ".join(s.tokens)


def finish_sentence(s: Session) -> Optional[str]:
    """Close the current line of words as a sentence (called after a long pause)."""
    if not s.tokens:
        return None
    text = " ".join(s.tokens).lower().strip()
    text = text[:1].upper() + text[1:] + "."
    s.sentences.append(text)
    s.sentences = s.sentences[-20:]
    s.tokens = []
    return text


def step_words(s: Session, res, has_hand: bool) -> dict:
    out = {"live": None, "committed": None, "speak": None, "top3": [], "finished": None}
    if has_hand:
        s.seq.append(hands_pose_features(res))
        s.miss = 0
    else:
        s.miss += 1

    ended = (not has_hand and s.miss >= WORD_GAP_FRAMES and len(s.seq) > 0) or len(s.seq) >= MAX_FRAMES
    if ended:
        if len(s.seq) >= MIN_WORD_FRAMES:
            word, conf, top = predict_word(s.seq)
            s.last_top3 = [{"word": w, "confidence": round(c, 3)} for w, c in top]
            s.last_live = {"label": word, "confidence": round(conf, 3)}
            s.last_accepted = conf >= WORD_MIN_CONF
            out["finished"] = {"word": word, "conf": conf, "accepted": conf >= WORD_MIN_CONF}
            if conf >= WORD_MIN_CONF:
                s.tokens.append(word)
                s.tokens = s.tokens[-12:]
                out["committed"] = word
                out["speak"] = word
        s.seq = []
        s.miss = 0
    return out


def step_letters(s: Session, res, has_hand: bool, w: int, h: int) -> dict:
    out = {"live": None, "committed": None, "speak": None, "top3": []}
    hand = res.right_hand_landmarks or res.left_hand_landmarks
    if hand is not None:
        s.miss = 0
        s.space_done = False
        letter, conf = predict_letter(hand, w, h)
        out["live"] = {"label": letter, "confidence": round(conf, 3)}
        s.letter_hist.append((letter, conf))
        votes = Counter(l for l, _ in s.letter_hist)
        top, n = votes.most_common(1)[0]
        avg = float(np.mean([c for l, c in s.letter_hist if l == top]))
        if n >= LETTER_VOTES and avg >= LETTER_MIN_CONF and top != s.last_letter:
            s.current_word += top
            s.last_letter = top
            out["committed"] = top
    else:
        s.miss += 1
        s.letter_hist.clear()
        if s.miss >= 5:
            s.last_letter = None  # allow the same letter again (e.g. double letters)
        if s.miss >= SPACE_GAP_FRAMES and not s.space_done:
            if s.current_word:
                s.tokens.append(s.current_word)
                s.tokens = s.tokens[-12:]
                out["speak"] = s.current_word
                s.current_word = ""
            s.space_done = True
    return out


# ----------------------------------------------------------------------------
# HTTP API
# ----------------------------------------------------------------------------
store = Store()
atexit.register(store.flush)

_ADMIN_PW = os.environ.get("ADMIN_PASSWORD") or secrets.token_urlsafe(9)
if not os.environ.get("ADMIN_PASSWORD"):
    print(f"\n[KamayTalk] Admin panel -> /admin   user: admin   password: {_ADMIN_PW}")
    print("[KamayTalk] (set the ADMIN_PASSWORD environment variable to choose your own)\n")
print(f"[KamayTalk] storage: redis={store.status['redis']} | firestore={store.status['firestore']} | local files={store.status['local']}")
_basic = HTTPBasic()


def require_admin(creds: HTTPBasicCredentials = Depends(_basic)) -> None:
    ok_u = secrets.compare_digest(creds.username.encode(), b"admin")
    ok_p = secrets.compare_digest(creds.password.encode(), _ADMIN_PW.encode())
    if not (ok_u and ok_p):
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": 'Basic realm="KamayTalk admin"'})


app = FastAPI(title="KamayTalk API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {
        "ok": True,
        "letters_ready": True,
        "letters": LETTER_LABELS,
        "words_ready": _word_model is not None,
        "words_error": WORDS_ERROR,
        "word_vocab_mode": WORD_VOCAB,
        "word_vocab": vocab_words(),
        "mirror": MIRROR,
        "storage": store.status,
        "round": store.get_round(),
    }


@app.post("/v1/reset")
def reset(session: str = Query("default")):
    s = get_session(session)
    with s.lock:
        s.clear()
    return {"ok": True}


@app.post("/v1/frame")
async def frame(
    request: Request,
    session: str = Query("default"),
    mode: str = Query("words"),
    fmt: str = Query("jpeg"),
    w: int = Query(0),
    h: int = Query(0),
    stride: int = Query(0),
    rot: int = Query(0),
    client: str = Query("web"),
    say: str = Query("word"),
):
    if mode not in ("words", "letters"):
        raise HTTPException(400, "mode must be words or letters")
    if mode == "words" and _word_model is None:
        raise HTTPException(503, f"Word model unavailable ({WORDS_ERROR}). Use mode=letters.")
    body = await request.body()
    try:
        img = decode_frame(body, fmt, w, h, stride, rot)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if MIRROR:
        img = cv2.flip(img, 1)

    s = get_session(session)
    with s.lock:
        if s.mode != mode:  # switching modes starts a fresh subtitle
            s.clear()
            s.mode = mode
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = s.holistic.process(rgb)
        left = res.left_hand_landmarks is not None
        right = res.right_hand_landmarks is not None
        has_hand = left or right
        now = time.time()
        if has_hand:
            s.last_hand_ts = now

        if mode == "words":
            out = step_words(s, res, has_hand)
        else:
            out = step_letters(s, res, has_hand, img.shape[1], img.shape[0])

        new_sentence = None
        if (not has_hand and s.tokens and not s.current_word and not s.seq
                and now - s.last_hand_ts >= SENTENCE_GAP_SECONDS):
            new_sentence = finish_sentence(s)

        # --- log to the hybrid store (non-blocking) ---
        fin = out.get("finished")
        if fin:
            store.add_event("word" if fin["accepted"] else "unsure", fin["word"], session, mode, client, fin["conf"])
        elif mode == "letters" and out["speak"]:
            store.add_event("word", out["speak"], session, mode, client)
        if new_sentence:
            store.add_event("sentence", new_sentence, session, mode, client)
        store.touch_session(session, {"mode": mode, "client": client, "subtitle": subtitle_text(s),
                                      "sentences": s.sentences[-3:]})

        speak = new_sentence if say == "sentence" else out["speak"]
        words = mode == "words"
        return {
            "mode": mode,
            "hands": {"left": left, "right": right},
            "buffered": len(s.seq),
            "live": s.last_live if words else out["live"],
            "top3": s.last_top3 if words else [],
            "accepted": s.last_accepted if words else None,
            "committed": out["committed"],
            "speak": speak,
            "subtitle": subtitle_text(s),
            "sentence": new_sentence,
            "sentences": s.sentences[-3:],
        }


class FeedbackIn(BaseModel):
    session: str = Field("unknown", max_length=60)
    client: str = Field("web", max_length=20)
    rating: Optional[int] = Field(None, ge=1, le=5)
    liked: str = Field("", max_length=500)
    confused: str = Field("", max_length=500)
    would_use: Optional[Literal["yes", "maybe", "no"]] = None
    comment: str = Field("", max_length=500)


_fb_hits: dict[str, list[float]] = {}


@app.post("/v1/feedback")
def feedback(body: FeedbackIn, request: Request):
    ip = (request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (request.client.host if request.client else "?"))
    hits = [t for t in _fb_hits.get(ip, []) if time.time() - t < 3600]
    if len(hits) >= 30:  # basic spam guard (generous: classmates may share one campus IP)
        raise HTTPException(429, "Too much feedback from this network, try again later")
    if not (body.rating or body.liked or body.confused or body.would_use or body.comment):
        raise HTTPException(400, "Please fill in at least one field")
    hits.append(time.time())
    _fb_hits[ip] = hits
    store.add_feedback(body.model_dump())
    return {"ok": True}


# ---------------------------- admin (password protected) ----------------------------
@app.get("/admin", dependencies=[Depends(require_admin)])
def admin_page():
    return FileResponse(BASE / "static" / "admin.html")


@app.get("/admin/api/summary", dependencies=[Depends(require_admin)])
def admin_summary(round: Optional[int] = None):
    return store.summary(round)


@app.get("/admin/api/events", dependencies=[Depends(require_admin)])
def admin_events(limit: int = 50, source: Literal["hot", "archive"] = "hot", round: Optional[int] = None):
    return store.recent_events(limit, source, round)


@app.get("/admin/api/feedback", dependencies=[Depends(require_admin)])
def admin_feedback(round: Optional[int] = None):
    return store.all_feedback(round)


@app.get("/admin/api/sessions", dependencies=[Depends(require_admin)])
def admin_sessions():
    return store.sessions()


class RoundIn(BaseModel):
    round: int = Field(ge=1, le=9)


@app.post("/admin/api/round", dependencies=[Depends(require_admin)])
def admin_set_round(body: RoundIn):
    return {"round": store.set_round(body.round)}


@app.get("/admin/export/{which}.csv", dependencies=[Depends(require_admin)])
def admin_export(which: Literal["events", "feedback"], round: Optional[int] = None):
    return Response(
        store.export_csv(which, round),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="kamaytalk_{which}.csv"'},
    )


@app.get("/demo")
def demo():
    return FileResponse(BASE / "static" / "demo.html")


@app.get("/")
def root():
    return RedirectResponse("/demo")
