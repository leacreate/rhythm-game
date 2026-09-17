"""
Build a rhythm chart from an MP3 using librosa.
BPM is detected and used to scale note density per difficulty level.

Difficulties:
  0 = EASY    ~2 notes/sec, only strong beats
  1 = NORMAL  ~4 notes/sec, beats + strong onsets
  2 = HARD    ~6 notes/sec, beats + all onsets + harmonics
  3 = EXTREME ~8-10 notes/sec, everything including subdivisions
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

try:
    import numpy as np
    import librosa
except ImportError:
    np = None
    librosa = None

CHART_VERSION = 7

DIFFICULTIES = ["EASY", "NORMAL", "HARD", "EXTREME"]

# Per-difficulty tuning
DIFF_PARAMS = {
    0: dict(notes_per_sec=2.0,  min_sep=0.18,  harmonics=False, flux=False, delta_floor=0.10),
    1: dict(notes_per_sec=4.0,  min_sep=0.10,  harmonics=False, flux=True,  delta_floor=0.06),
    2: dict(notes_per_sec=6.5,  min_sep=0.065, harmonics=True,  flux=True,  delta_floor=0.035),
    3: dict(notes_per_sec=10.0, min_sep=0.045, harmonics=True,  flux=True,  delta_floor=0.018),
}


def _chart_cache_path(audio_path: str, cache_dir: Path, difficulty: int) -> Path:
    h = hashlib.sha256(os.path.abspath(audio_path).encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{h}_d{difficulty}.json"


def load_cached_chart(audio_path: str, cache_dir: Path, difficulty: int = 1) -> tuple[list[tuple[int, int]], int, float] | None:
    """Return (pattern, duration_ms, bpm) if cache hit."""
    p = Path(audio_path)
    if not p.is_file():
        return None
    cache = _chart_cache_path(audio_path, cache_dir, difficulty)
    if not cache.is_file():
        return None
    try:
        st = p.stat()
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("mtime") != st.st_mtime_ns:
            return None
        if int(data.get("chart_version", 0)) < CHART_VERSION:
            return None
        pattern = [(int(a), int(b)) for a, b in data["pattern"]]
        duration_ms = int(data["duration_ms"])
        bpm = float(data.get("bpm", 120.0))
        return pattern, duration_ms, bpm
    except (json.JSONDecodeError, OSError, KeyError, ValueError):
        return None


def save_cached_chart(audio_path: str, cache_dir: Path, difficulty: int,
                      pattern: list[tuple[int, int]], duration_ms: int, bpm: float) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = Path(audio_path)
    try:
        st = p.stat()
    except OSError:
        return
    cache = _chart_cache_path(audio_path, cache_dir, difficulty)
    payload = {
        "chart_version": CHART_VERSION,
        "mtime": st.st_mtime_ns,
        "duration_ms": duration_ms,
        "bpm": bpm,
        "pattern": [[a, b] for a, b in pattern],
    }
    try:
        cache.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def _merge_times(*arrays, min_sep: float) -> list[float]:
    all_t: list[float] = []
    for arr in arrays:
        if arr is None or len(arr) == 0:
            continue
        all_t.extend(float(x) for x in np.asarray(arr).flatten())
    all_t.sort()
    out: list[float] = []
    prev = -1e9
    for t in all_t:
        if t - prev >= min_sep:
            out.append(t)
            prev = t
    return out


def _onset_times_from_env(onset_env: np.ndarray, sr: int, hop: int,
                           deltas: tuple[float, ...]) -> np.ndarray:
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop)
    peaks_all: list[np.ndarray] = []
    for delta in deltas:
        peaks = librosa.util.peak_pick(
            onset_env,
            pre_max=3, post_max=5,
            pre_avg=5, post_avg=5,
            delta=delta, wait=2,
        )
        if len(peaks) > 0:
            peaks_all.append(times[peaks])
    if not peaks_all:
        return np.array([], dtype=float)
    return np.unique(np.concatenate(peaks_all)).astype(float)


def _percussive_onset_times(y_perc: np.ndarray, sr: int, hop: int, delta_floor: float) -> np.ndarray:
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop, aggregate=np.median)
    deltas = tuple(d for d in (0.14, 0.09, 0.055, 0.035, 0.02) if d >= delta_floor)
    if not deltas:
        deltas = (delta_floor,)
    return _onset_times_from_env(onset_env, sr, hop, deltas)


def _harmonic_onset_times(y_harm: np.ndarray, sr: int, hop: int, delta_floor: float) -> np.ndarray:
    onset_env = librosa.onset.onset_strength(
        y=y_harm, sr=sr, hop_length=hop, aggregate=np.mean, fmin=80, fmax=4000,
    )
    deltas = tuple(d for d in (0.10, 0.06, 0.035, 0.02) if d >= delta_floor)
    if not deltas:
        deltas = (delta_floor,)
    return _onset_times_from_env(onset_env, sr, hop, deltas)


def _spectral_flux_times(y: np.ndarray, sr: int, hop: int, delta_floor: float) -> np.ndarray:
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop, aggregate=np.mean, fmax=8000,
    )
    deltas = tuple(d for d in (0.12, 0.07, 0.04, 0.025) if d >= delta_floor)
    if not deltas:
        deltas = (delta_floor,)
    return _onset_times_from_env(onset_env, sr, hop, deltas)


def _beat_subdivisions(beat_times: np.ndarray, difficulty: int, bpm: float) -> np.ndarray:
    """Add subdivisions scaled to difficulty and BPM."""
    bt = np.sort(np.asarray(beat_times, dtype=float).flatten())
    if len(bt) < 2:
        return np.array([], dtype=float)
    extra: list[float] = []
    for i in range(len(bt) - 1):
        gap = bt[i + 1] - bt[i]
        if difficulty >= 1 and gap > 0.15:
            extra.append(bt[i] + gap * 0.5)
        if difficulty >= 2 and gap > 0.20 and bpm < 180:
            extra.append(bt[i] + gap * 0.25)
            extra.append(bt[i] + gap * 0.75)
        if difficulty >= 3 and gap > 0.30 and bpm < 140:
            for k in (1, 3, 5, 7):
                extra.append(bt[i] + gap * k / 8)
    return np.array(extra, dtype=float)


def _insert_onset_peaks_in_gaps(
    times_sec: np.ndarray,
    times_on: np.ndarray,
    onset_env: np.ndarray,
    max_gap: float,
    min_sep: float,
    delta_floor: float,
) -> np.ndarray:
    t = np.sort(np.unique(np.asarray(times_sec, dtype=float)))
    if len(t) < 2:
        return t
    extra: list[float] = []
    q_thresh = float(np.quantile(onset_env, max(0.1, 0.35 - delta_floor * 3)))
    for i in range(len(t) - 1):
        t0, t1 = float(t[i]), float(t[i + 1])
        if t1 - t0 <= max_gap:
            continue
        mask = (times_on > t0 + min_sep) & (times_on < t1 - min_sep)
        if not np.any(mask):
            continue
        seg_env = onset_env[mask]
        seg_t = times_on[mask]
        if len(seg_env) < 3:
            continue
        peaks = librosa.util.peak_pick(
            seg_env,
            pre_max=2, post_max=2,
            pre_avg=3, post_avg=3,
            delta=max(q_thresh * 0.4, float(np.median(seg_env)) * 0.25),
            wait=1,
        )
        for p in peaks:
            extra.append(float(seg_t[p]))
    if not extra:
        return t
    return np.array(_merge_times(t, np.array(extra, dtype=float), min_sep=min_sep), dtype=float)


def _downselect_by_onset_strength(
    times_sec: np.ndarray,
    times_on: np.ndarray,
    onset_env: np.ndarray,
    max_notes: int,
    min_sep: float,
) -> np.ndarray:
    times_sec = np.sort(np.unique(np.asarray(times_sec, dtype=float)))
    if len(times_sec) <= max_notes:
        return times_sec
    scores = np.interp(times_sec, times_on, onset_env)
    order = np.argsort(-scores)
    picked: list[float] = []
    for i in order:
        tt = float(times_sec[i])
        if all(abs(tt - p) >= min_sep for p in picked):
            picked.append(tt)
        if len(picked) >= max_notes:
            break
    return np.array(sorted(picked), dtype=float)


def _lanes_from_chroma(times_sec: np.ndarray, y: np.ndarray, sr: int) -> list[int]:
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=12)
    ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    lanes: list[int] = []
    for t in times_sec:
        fi = int(np.clip(np.argmin(np.abs(ct - float(t))), 0, chroma.shape[1] - 1))
        cvec = np.maximum(chroma[:, fi], 1e-8)
        groups = [float(np.sum(cvec[j: j + 3])) for j in (0, 3, 6, 9)]
        lane = int(np.argmax(groups))
        if lanes and lane == lanes[-1]:
            order = np.argsort(groups)
            alt = int(order[-2])
            lane = alt if groups[alt] > 0.35 * groups[int(order[-1])] else (lane + 1) % 4
        lanes.append(lane)
    return lanes




def _norm01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    lo = float(np.percentile(values, 10))
    hi = float(np.percentile(values, 90))
    if hi <= lo + 1e-9:
        return np.ones_like(values) * 0.5
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _song_driven_downselect(
    times_sec: np.ndarray,
    times_on: np.ndarray,
    onset_env: np.ndarray,
    y: np.ndarray,
    sr: int,
    difficulty: int,
    duration_sec: float,
    min_sep: float,
) -> np.ndarray:
    """Pick notes per song section instead of using one global note density.

    Quiet parts naturally get fewer notes while energetic sections get denser
    patterns. The scoring combines onset strength, spectral change and RMS
    energy, so two different songs produce noticeably different charts.
    """
    times_sec = np.sort(np.unique(np.asarray(times_sec, dtype=float)))
    if len(times_sec) == 0:
        return times_sec

    hop = 512
    onset = np.interp(times_sec, times_on, onset_env)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    energy = np.interp(times_sec, rms_t, rms)
    flux_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.mean)
    flux_t = librosa.times_like(flux_env, sr=sr, hop_length=hop)
    flux = np.interp(times_sec, flux_t, flux_env)

    score = (
        0.50 * _norm01(onset)
        + 0.25 * _norm01(flux)
        + 0.25 * _norm01(energy)
    )

    # Work in short musical phrases. This prevents a loud chorus from
    # consuming the whole note budget and leaving verses empty.
    phrase = 4.0
    if len(times_sec) > 1:
        median_gap = float(np.median(np.diff(times_sec)))
        phrase = max(2.5, min(6.0, median_gap * (16 if difficulty < 2 else 12)))

    base_rates = (1.6, 3.2, 5.2, 8.0)
    rate = base_rates[int(np.clip(difficulty, 0, 3))]
    selected: list[float] = []

    start = 0.0
    while start < duration_sec:
        end = min(duration_sec, start + phrase)
        mask = (times_sec >= start) & (times_sec < end)
        idx = np.flatnonzero(mask)
        if len(idx):
            sec_energy = float(np.mean(energy[idx]))
            sec_score = float(np.mean(score[idx]))
            # Energy controls the section's density; strong musical sections
            # can approach the difficulty's normal rate.
            density = 0.55 + 0.85 * max(sec_energy, sec_score)
            target = max(1, int(round(rate * (end - start) * density)))
            target = min(target, len(idx))
            order = idx[np.argsort(score[idx])[::-1]]
            picked: list[float] = []
            for j in order:
                t = float(times_sec[j])
                if all(abs(t - q) >= min_sep for q in picked):
                    picked.append(t)
                if len(picked) >= target:
                    break
            selected.extend(picked)
        start = end

    return np.array(sorted(set(selected)), dtype=float)


def _add_song_patterns(
    times_sec: np.ndarray,
    y: np.ndarray,
    sr: int,
    difficulty: int,
    min_sep: float,
) -> np.ndarray:
    """Add musical subdivisions during high-intensity passages.

    This creates bursts/alternation-like patterns from the actual audio rather
    than using a fixed repeating pattern. The returned values remain timestamps
    so the existing game format stays compatible.
    """
    if difficulty < 2 or len(times_sec) < 2:
        return times_sec
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.mean)
    tt = librosa.times_like(env, sr=sr, hop_length=hop)
    strength = np.interp(times_sec, tt, env)
    threshold = float(np.percentile(strength, 72))
    extra: list[float] = []
    for i in range(len(times_sec) - 1):
        t0, t1 = float(times_sec[i]), float(times_sec[i + 1])
        gap = t1 - t0
        if gap < max(0.16, min_sep * 3) or strength[i] < threshold:
            continue
        # Strong hits can become a short 1/2 or 1/4-beat burst. The audio
        # intensity decides where these happen, so songs get different shapes.
        if difficulty >= 3 and gap > 0.30:
            extra.extend((t0 + gap * 0.25, t0 + gap * 0.50, t0 + gap * 0.75))
        elif gap > 0.24:
            extra.append(t0 + gap * 0.50)
    return np.array(_merge_times(times_sec, np.asarray(extra), min_sep=min_sep), dtype=float)


def detect_bpm(y_perc: np.ndarray, sr: int) -> float:
    """Detect BPM. Normalises double/half-tempo estimates."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        while bpm < 60:
            bpm *= 2
        while bpm > 220:
            bpm /= 2
        return round(bpm, 1)
    except Exception:
        return 120.0


def generate_chart_from_file(
    audio_path: str,
    difficulty: int = 1,
) -> tuple[list[tuple[int, int]], int, float, str | None]:
    """
    Analyze audio → (pattern, duration_ms, bpm, error).
    difficulty: 0=EASY  1=NORMAL  2=HARD  3=EXTREME
    """
    if librosa is None or np is None:
        return [], 0, 0.0, "Install librosa: pip install librosa"

    path = Path(audio_path)
    if not path.is_file():
        return [], 0, 0.0, "File not found"

    difficulty = int(np.clip(difficulty, 0, 3))
    p = DIFF_PARAMS[difficulty]

    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True)
    except Exception as e:
        return [], 0, 0.0, str(e)

    duration_ms = int(len(y) / float(sr) * 1000.0)
    duration_sec = duration_ms / 1000.0
    if duration_ms < 500:
        return [], 0, 0.0, "Audio too short"

    hop = 512

    try:
        y_harm, y_perc = librosa.effects.hpss(y, margin=6.0)
    except Exception:
        y_harm, y_perc = y, y

    bpm = detect_bpm(y_perc, sr)

    try:
        _, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr, trim=False)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    except Exception:
        beat_times = np.array([], dtype=float)

    min_sep     = p["min_sep"]
    delta_floor = p["delta_floor"]

    sources = [beat_times, _percussive_onset_times(y_perc, sr, hop, delta_floor)]
    if p["flux"]:
        sources.append(_spectral_flux_times(y, sr, hop, delta_floor))
    if p["harmonics"]:
        sources.append(_harmonic_onset_times(y_harm, sr, hop, delta_floor))
    subdiv = _beat_subdivisions(beat_times, difficulty, bpm)
    if len(subdiv) > 0:
        sources.append(subdiv)

    times_sec = np.array(_merge_times(*sources, min_sep=min_sep), dtype=float)

    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop, aggregate=np.median)
    times_on  = librosa.times_like(onset_env, sr=sr, hop_length=hop)

    times_sec = _insert_onset_peaks_in_gaps(
        times_sec, times_on, onset_env,
        max_gap=min_sep * 4, min_sep=min_sep, delta_floor=delta_floor,
    )

    times_sec = np.sort(np.unique(times_sec))
    if len(times_sec) == 0:
        return [], duration_ms, bpm, "No rhythm detected"

    onset_env_full = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.mean)
    times_on_full  = librosa.times_like(onset_env_full, sr=sr, hop_length=hop)

    # Make the map follow the song's structure instead of applying the same
    # note rate from the first second to the last.
    times_sec = _song_driven_downselect(
        times_sec, times_on_full, onset_env_full, y, sr,
        difficulty, duration_sec, min_sep,
    )
    times_sec = _add_song_patterns(times_sec, y, sr, difficulty, min_sep)

    max_notes = min(15000, max(200, int(duration_sec * p["notes_per_sec"] * 1.35)))
    if len(times_sec) > max_notes:
        times_sec = _downselect_by_onset_strength(
            times_sec, times_on_full, onset_env_full, max_notes, min_sep=min_sep
        )

    lanes = _lanes_from_chroma(times_sec, y, sr)
    pattern = [(int(round(float(t) * 1000.0)), int(lanes[i])) for i, t in enumerate(times_sec)]
    pattern.sort(key=lambda x: x[0])
    return pattern, duration_ms, bpm, None
