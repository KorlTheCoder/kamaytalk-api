"""
Real-time WLASL sign demo: webcam → MediaPipe Holistic → same feature layout as
Training/mediapipe_extraction.py (hands_pose), then the trained model from wlasl_100_train.py.

Model input size, hidden size, layer count, and class count are inferred from the checkpoint
when possible so the demo stays in sync with training defaults (Transformer + hand-gated .npy).
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WLASL_ROOT = SCRIPT_DIR / "Training" / "wlasl_100"
DEFAULT_FEATURE_SUBDIR = "features_hands_pose_hand_gated"


class SignLanguageLSTM(nn.Module):
    """Must match Training/wlasl_100_train.py SignLanguageLSTM."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout=0.4,
        bidirectional=True,
        temporal_pooling="attention",
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.bidirectional = bidirectional
        self.temporal_pooling = temporal_pooling
        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        d = 2 if bidirectional else 1
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * d, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * d, num_classes)

    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        if self.temporal_pooling == "attention":
            scores = self.attn(out).squeeze(-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            h = torch.sum(out * weights, dim=1)
        elif self.temporal_pooling == "mean":
            h = torch.mean(out, dim=1)
        else:
            h = out[:, -1, :]
        h = self.dropout(h)
        return self.fc(h)


class SignLanguageTransformer(nn.Module):
    """Must match Training/wlasl_100_train.py SignLanguageTransformer."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout=0.3,
        nhead=4,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.proj = nn.Linear(input_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, hidden_size) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.proj(x)
        t = x.shape[1]
        x = x + self.pos_emb[:, :t, :]
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        h = self.dropout(x[:, 0, :])
        return self.fc(h)


def unwrap_state_dict(raw):
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def safe_torch_load(path: Path, map_location="cpu"):
    """torch.load with weights_only when supported (PyTorch 2.0+)."""
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def detect_arch(state_dict: dict) -> str:
    if any(k.startswith("encoder.layers.") for k in state_dict):
        return "transformer"
    if any(k.startswith("lstm.") for k in state_dict):
        return "lstm"
    raise ValueError(
        "Could not detect architecture from checkpoint keys "
        f"(expected transformer 'encoder.layers.*' or lstm 'lstm.*'). "
        f"Sample keys: {list(state_dict.keys())[:8]}"
    )


def infer_transformer_config(state_dict: dict) -> dict:
    input_size = int(state_dict["input_norm.weight"].shape[0])
    num_classes = int(state_dict["fc.weight"].shape[0])
    hidden_size = int(state_dict["proj.weight"].shape[0])
    layer_ids = []
    for k in state_dict:
        m = re.match(r"encoder\.layers\.(\d+)\.", k)
        if m:
            layer_ids.append(int(m.group(1)))
    num_layers = max(layer_ids) + 1 if layer_ids else 2
    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
    }


def infer_lstm_config(state_dict: dict) -> dict:
    input_size = int(state_dict["input_norm.weight"].shape[0])
    num_classes = int(state_dict["fc.weight"].shape[0])
    out_dim = int(state_dict["fc.weight"].shape[1])
    bidirectional = any(
        re.match(r"lstm\.weight_ih_l\d+_reverse$", k) for k in state_dict
    )
    layer_indices = []
    for k in state_dict:
        m = re.match(r"lstm\.weight_ih_l(\d+)$", k)
        if m:
            layer_indices.append(int(m.group(1)))
    num_layers = max(layer_indices) + 1 if layer_indices else 1
    d = 2 if bidirectional else 1
    hidden_size = out_dim // d
    temporal_pooling = "attention" if "attn.0.weight" in state_dict else "mean"
    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "bidirectional": bidirectional,
        "temporal_pooling": temporal_pooling,
    }


def build_transformer_with_nhead(input_size, hidden_size, num_layers, num_classes, dropout, nhead):
    return SignLanguageTransformer(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        nhead=nhead,
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    arch_override: str | None,
    dropout: float,
    lstm_no_bidirectional: bool,
    lstm_temporal_pooling: str | None,
) -> tuple[nn.Module, str, int]:
    raw = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = unwrap_state_dict(raw)

    arch = arch_override or detect_arch(state_dict)

    if arch == "transformer":
        cfg = infer_transformer_config(state_dict)
        hidden_size = cfg["hidden_size"]
        # Prefer nhead=4 (Training/wlasl_100_train.py default), then other divisors.
        candidates = [n for n in (4, 8, 2, 1) if hidden_size % n == 0]
        last_err = None
        model = None
        for nhead in candidates:
            try:
                model = build_transformer_with_nhead(
                    cfg["input_size"],
                    cfg["hidden_size"],
                    cfg["num_layers"],
                    cfg["num_classes"],
                    dropout,
                    nhead,
                )
                model.load_state_dict(state_dict, strict=True)
                break
            except RuntimeError as e:
                last_err = e
                model = None
        if model is None:
            raise RuntimeError(
                f"Could not load transformer (tried nhead in {candidates}): {last_err}"
            ) from last_err
        num_classes = cfg["num_classes"]
    else:
        cfg = infer_lstm_config(state_dict)
        bidirectional = cfg["bidirectional"] and not lstm_no_bidirectional
        temporal_pooling = lstm_temporal_pooling or cfg["temporal_pooling"]
        model = SignLanguageLSTM(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_classes=cfg["num_classes"],
            dropout=dropout,
            bidirectional=bidirectional,
            temporal_pooling=temporal_pooling,
        )
        model.load_state_dict(state_dict, strict=True)
        num_classes = cfg["num_classes"]

    model.to(device)
    model.eval()
    return model, arch, num_classes


def collect_class_names_set(
    wlasl_root: Path,
    feature_subdir: str,
    splits: list[str],
) -> tuple[set[str], list[str]]:
    """Returns (class names, per-split status lines for errors)."""
    names: set[str] = set()
    wlasl_root = Path(wlasl_root).resolve()
    details: list[str] = []
    for split in splits:
        root = wlasl_root / split / feature_subdir
        if not root.is_dir():
            details.append(f"  {root}  (missing — fix --wlasl-root or run mediapipe_extraction)")
        else:
            class_dirs = [d for d in root.iterdir() if d.is_dir()]
            details.append(f"  {root}  ({len(class_dirs)} class folders)")
            names.update(d.name for d in class_dirs)
    return names, details


def discover_class_names(
    wlasl_root: Path,
    feature_subdir: str,
    splits: list[str],
    max_classes: int | None,
) -> list[str]:
    """Same ordering idea as Training/wlasl_100_train.discover_class_names (sorted union)."""
    names, details = collect_class_names_set(wlasl_root, feature_subdir, splits)
    if not names:
        raise FileNotFoundError(
            "No class folders found for this layout:\n"
            + "\n".join(details)
            + "\n\nCheck:\n"
            "  • --wlasl-root must be the dataset folder that contains train/, val/, test/ "
            "(not a copy-pasted placeholder like /path/to/...).\n"
            "  • --feature-subdir must match extraction output, e.g. features_hands_pose_hand_gated "
            "or features_hands_pose_hand_gated_nslt1000.\n"
            "  • Run: python Training/mediapipe_extraction.py --dataset-root \"<that same folder>\" ..."
        )
    out = sorted(names)
    if max_classes is not None:
        out = out[:max_classes]
    return out


def list_feature_subdir_candidates(wlasl_root: Path, split: str) -> list[str]:
    """
    Subfolders of <root>/<split>/ named features* that contain at least one class subfolder.
    Used when the default feature_subdir is wrong or missing.
    """
    base = Path(wlasl_root).resolve() / split
    if not base.is_dir():
        return []
    found: list[str] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir() or not p.name.startswith("features"):
            continue
        if any(d.is_dir() for d in p.iterdir()):
            found.append(p.name)
    return found


def resolve_wlasl_paths(wlasl_root: Path, feature_subdir: str, class_splits: list[str]) -> tuple[Path, str]:
    """
    Resolve dataset root and feature subdir; validate root exists; optionally auto-pick one features_* dir.
    """
    root = Path(wlasl_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"WLASL root does not exist:\n  {root}\n"
            "Use the real path to your dataset (the folder containing train/, val/, test/), "
            "e.g.\n  --wlasl-root \"/Users/you/.../Training/WLASL - FULL\""
        )
    if not root.is_dir():
        raise NotADirectoryError(f"WLASL root is not a directory: {root}")

    names, _ = collect_class_names_set(root, feature_subdir, class_splits)
    if names:
        return root, feature_subdir

    primary_split = class_splits[0] if class_splits else "train"
    candidates = list_feature_subdir_candidates(root, primary_split)
    if len(candidates) == 1:
        fs = candidates[0]
        print(f"Note: no classes under {feature_subdir!r}; using detected --feature-subdir {fs!r}.")
        discover_class_names(root, fs, class_splits, None)  # validate all splits / error detail
        return root, fs
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"No classes under {root}/{primary_split}/{feature_subdir!r}. "
            f"Found multiple feature folders: {candidates!r}. "
            f"Pass the one you trained on, e.g. --feature-subdir {candidates[0]!r}"
        )

    raise FileNotFoundError(
        f"No classes under {root}/{primary_split}/{feature_subdir!r} and no features_* folders "
        f"with class subdirs under {root}/{primary_split}/. Run mediapipe_extraction.py first."
    )


def has_hand(results) -> bool:
    return (
        results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
    )


def extract_hands_pose_keypoints(results) -> np.ndarray:
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    lh = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh]).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time WLASL demo (matches wlasl_100_train.py + hand-gated extraction)."
    )
    p.add_argument("--checkpoint", required=True, help="Trained state_dict .pt from wlasl_100_train.py")
    p.add_argument("--wlasl-root", type=Path, default=DEFAULT_WLASL_ROOT)
    p.add_argument(
        "--feature-subdir",
        default=DEFAULT_FEATURE_SUBDIR,
        help="Must match the folder used during mediapipe_extraction / training.",
    )
    p.add_argument(
        "--class-splits",
        nargs="+",
        default=["train", "val"],
        help="Folder names to scan for class labels (union, sorted). "
        "Default train val matches wlasl_100_train.py defaults. "
        "Add test if you trained with --final-eval-split test and need that union.",
    )
    p.add_argument("--max-classes", type=int, default=None)
    p.add_argument(
        "--arch",
        choices=["auto", "transformer", "lstm"],
        default="auto",
        help="Override checkpoint architecture detection (default: auto).",
    )
    p.add_argument("--dropout", type=float, default=0.4, help="Only affects module construction; ignored at eval.")
    p.add_argument("--no-bidirectional", action="store_true", help="LSTM only: override bidirectional=False.")
    p.add_argument(
        "--temporal-pooling",
        choices=["attention", "mean", "last"],
        default=None,
        help="LSTM only: override pooling (default: infer from checkpoint).",
    )
    p.add_argument("--max-frames", type=int, default=50, help="Must match extraction max_frames.")
    p.add_argument(
        "--hand-gated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only append frames where a hand is visible (matches hand_gated .npy). Use --no-hand-gated for all frames.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Below this softmax probability the label is still shown, with a warning style (color / [low conf]).",
    )
    p.add_argument("--camera-id", type=int, default=0)
    return p.parse_args()


def main():
    import cv2
    import mediapipe as mp

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wlasl_root, feature_subdir = resolve_wlasl_paths(
        args.wlasl_root, args.feature_subdir, args.class_splits
    )
    class_names = discover_class_names(
        wlasl_root, feature_subdir, args.class_splits, args.max_classes
    )
    arch_override = None if args.arch == "auto" else args.arch
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    model, arch, ckpt_classes = load_model_from_checkpoint(
        ckpt_path,
        device,
        arch_override=arch_override,
        dropout=args.dropout,
        lstm_no_bidirectional=args.no_bidirectional,
        lstm_temporal_pooling=args.temporal_pooling,
    )

    if ckpt_classes != len(class_names):
        raise SystemExit(
            f"Class count mismatch: checkpoint has {ckpt_classes} outputs but "
            f"discovered {len(class_names)} classes under {feature_subdir}. "
            "Use the same WLASL root / feature-subdir / splits as training, or adjust --max-classes."
        )

    input_size = int(model.input_norm.normalized_shape[0])
    print(f"Loaded {arch} from {ckpt_path} | classes={len(class_names)} | hand_gated={args.hand_gated}")

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    seq: deque[np.ndarray] = deque(maxlen=args.max_frames)
    pred_hist: deque[int] = deque(maxlen=6)

    cap = cv2.VideoCapture(args.camera_id)
    print("WLASL demo — press 'q' quit, 'c' clear buffer.")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            use_frame = (not args.hand_gated) or has_hand(results)
            if use_frame:
                seq.append(extract_hands_pose_keypoints(results))

            text = "Show hands…" if args.hand_gated and len(seq) == 0 else "Collecting hand frames…"
            color = (0, 200, 255)

            n_ready = len(seq)
            if n_ready > 0:
                # Pad end with zeros like Training/mediapipe_extraction (shorter clips).
                chunk = list(seq)
                while len(chunk) < args.max_frames:
                    chunk.append(np.zeros(input_size, dtype=np.float32))
                x = np.stack(chunk[: args.max_frames], axis=0).astype(np.float32)
                with torch.no_grad():
                    logits = model(torch.from_numpy(x).unsqueeze(0).to(device))
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                pred_hist.append(int(np.argmax(probs)))
                stable_idx = Counter(pred_hist).most_common(1)[0][0]
                stable_prob = float(probs[stable_idx])
                gloss = class_names[stable_idx]
                low = stable_prob < args.threshold
                text = f"{gloss} ({stable_prob * 100:.1f}%)" + ("  [low conf]" if low else "")
                color = (0, 255, 0) if not low else (0, 165, 255)

            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(
                frame,
                f"Hand frames: {n_ready}/{args.max_frames}" + (" (gated)" if args.hand_gated else ""),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("WLASL Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                seq.clear()
                pred_hist.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
