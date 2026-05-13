import json
import os
import sys
import time
from pathlib import Path

import pygame

from audio_chart import (
    DIFFICULTIES,
    generate_chart_from_file,
    load_cached_chart,
    save_cached_chart,
)

pygame.init()
pygame.mixer.init(frequency=44100, channels=2, buffer=512)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
SETTINGS_PATH = SCRIPT_DIR / "settings.json"
MUSIC_DIR     = SCRIPT_DIR / "music"
CHART_CACHE_DIR = SCRIPT_DIR / "charts_cache"

# ── Constants ─────────────────────────────────────────────────────────────────
FPS        = 60
LANE_COUNT = 4
KEYS       = [pygame.K_d, pygame.K_f, pygame.K_j, pygame.K_k]
KEY_LABELS = ["D", "F", "J", "K"]

BG        = (10, 10, 15)
GRID      = (20, 20, 30)
HIT_LINE  = (60, 60, 80)
WHITE     = (255, 255, 255)
GREY      = (120, 120, 140)
DARK_GREY = (40, 40, 55)

LANE_COLOURS = [
    (232, 255, 110),
    (255, 110, 176),
    (110, 236, 255),
    (188, 110, 255),
]

DIFF_COLOURS = [
    (110, 236, 255),   # EASY    – cyan
    (110, 255, 158),   # NORMAL  – green
    (255, 180, 50),    # HARD    – orange
    (255, 80,  80),    # EXTREME – red
]

FALLBACK_DURATION_MS = 8000
FALLBACK_PATTERN = [
    (0, 0), (400, 1), (800, 2), (1200, 3),
    (1600, 0), (1600, 2), (2000, 1), (2400, 3),
    (2800, 0), (3000, 1), (3200, 2), (3400, 3),
    (3600, 0), (3600, 3), (4000, 1), (4000, 2),
    (4400, 0), (4600, 1), (4800, 2), (5000, 3),
    (5200, 0), (5200, 2), (5600, 1), (5800, 0),
    (6000, 3), (6200, 1), (6400, 2), (6600, 0),
    (6800, 1), (6800, 3), (7000, 2), (7200, 0),
    (7400, 1), (7600, 3), (7800, 2), (7800, 0),
]

DEFAULT_SETTINGS = {
    "music_enabled": True,
    "music_volume": 0.6,
    "music_path": "",
    "fullscreen": True,
    "difficulty": 1,
}


# ── Settings helpers ──────────────────────────────────────────────────────────
def load_settings():
    if not SETTINGS_PATH.is_file():
        return dict(DEFAULT_SETTINGS)
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_SETTINGS)
    out = dict(DEFAULT_SETTINGS)
    out.update({k: data[k] for k in DEFAULT_SETTINGS if k in data})
    return out


def save_settings(settings):
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError:
        pass


def list_mp3_files():
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(MUSIC_DIR.glob("*.mp3"), key=lambda p: p.name.lower())
    return [str(p.resolve()) for p in files]


# ── Note ──────────────────────────────────────────────────────────────────────
class Note:
    def __init__(self, game, lane, hit_ms):
        self.g      = game
        self.lane   = lane
        self.hit_ms = hit_ms
        self.hit    = False
        self.missed = False
        self.alpha  = 255

    def _y_for_time(self, music_ms):
        ms_until_hit = self.hit_ms - music_ms
        px_per_ms    = self.g.note_speed / (1000.0 / FPS)
        return self.g.hit_y - ms_until_hit * px_per_ms

    @property
    def y(self):
        return self._y_for_time(self.g.music_ms)

    def update(self):
        if self.hit or self.missed:
            self.alpha = max(0, self.alpha - 25)

    def draw(self, surf):
        if self.alpha <= 0:
            return
        g  = self.g
        y  = self.y
        if y + g.note_h < -10:
            return
        col = LANE_COLOURS[self.lane]
        lw  = g.lane_w
        x   = self.lane * lw + 6
        w   = lw - 12
        s   = pygame.Surface((w, g.note_h), pygame.SRCALPHA)
        s.fill((*col, int(self.alpha)))
        surf.blit(s, (x, int(y)))
        lbl = g.font_small.render(KEY_LABELS[self.lane], True, (0, 0, 0))
        surf.blit(lbl, (x + w // 2 - lbl.get_width() // 2,
                        int(y) + g.note_h // 2 - lbl.get_height() // 2))

    @property
    def dead(self):
        return self.alpha <= 0 and (self.hit or self.missed)

    @property
    def missed_check(self):
        return not self.hit and not self.missed and self.y > self.g.hit_y + self.g.good_win


# ── Popup ─────────────────────────────────────────────────────────────────────
class Popup:
    def __init__(self, game, text, lane, colour):
        self.g      = game
        self.text   = text
        lw          = game.lane_w
        self.x      = lane * lw + lw // 2
        self.y      = game.hit_y - 20
        self.colour = colour
        self.life   = 40

    def update(self):
        self.y    -= 1
        self.life -= 1

    def draw(self, surf):
        alpha = int(255 * self.life / 40)
        lbl   = self.g.font_med.render(self.text, True, self.colour)
        s     = pygame.Surface(lbl.get_size(), pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))
        s.blit(lbl, (0, 0))
        s.set_alpha(alpha)
        surf.blit(s, (self.x - lbl.get_width() // 2, int(self.y)))


# ── DifficultyFlash ───────────────────────────────────────────────────────────
class DiffFlash:
    """Big centred flash when difficulty changes."""
    def __init__(self, game):
        self.g    = game
        self.life = 0

    def trigger(self):
        self.life = 90   # frames

    def update(self):
        if self.life > 0:
            self.life -= 1

    def draw(self, surf):
        if self.life <= 0:
            return
        diff   = self.g.settings.get("difficulty", 1)
        label  = DIFFICULTIES[diff]
        colour = DIFF_COLOURS[diff]
        alpha  = int(255 * min(self.life, 30) / 30)   # fade in then hold then fade
        if self.life < 30:
            alpha = int(255 * self.life / 30)
        txt = self.g.font_big.render(label, True, colour)
        s   = pygame.Surface(txt.get_size(), pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))
        s.blit(txt, (0, 0))
        s.set_alpha(alpha)
        W, H = self.g.w, self.g.h
        surf.blit(s, (W // 2 - txt.get_width() // 2, H // 2 - txt.get_height() // 2))


# ── Game ──────────────────────────────────────────────────────────────────────
class RhythmGame:
    def __init__(self):
        self.settings       = load_settings()
        self.clock          = pygame.time.Clock()
        self.diff_flash     = None   # created after display init
        self.reset()
        self._recreate_display()
        self.diff_flash     = DiffFlash(self)
        pygame.display.set_caption("Rhythm Game")
        self.settings_cursor    = 0
        self.music_list         = list_mp3_files()
        self.music_pick_index   = 0
        self.music_pick_scroll  = 0

    # ── Layout ────────────────────────────────────────────────────────────────
    def _layout_from_size(self, w, h):
        self.w, self.h   = w, h
        self.lane_w      = w // LANE_COUNT
        self.note_h      = max(16, int(h * 0.037))
        self.hit_y       = h - max(70, int(h * 0.15))
        self.note_speed  = max(3, int(h * 0.0083))
        self.perfect_win = max(16, int(h * 0.05))
        self.good_win    = max(28, int(h * 0.10))
        self.font_big    = pygame.font.SysFont("Courier New", max(22, h // 20), bold=True)
        self.font_med    = pygame.font.SysFont("Courier New", max(16, h // 33), bold=True)
        self.font_small  = pygame.font.SysFont("Courier New", max(12, h // 46))
        px_travel        = self.hit_y + self.note_h
        self.lead_ms     = (px_travel / self.note_speed) * (1000.0 / FPS)

    def _recreate_display(self):
        was_playing = getattr(self, "state", None) == "playing"
        pygame.mixer.music.pause()
        fs = self.settings.get("fullscreen", True)
        if fs:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
        self._layout_from_size(*self.screen.get_size())
        if was_playing:
            pygame.mixer.music.unpause()

    def _hotkey_toggle_fullscreen(self):
        self.settings["fullscreen"] = not self.settings.get("fullscreen", True)
        save_settings(self.settings)
        self._recreate_display()

    # ── Music helpers ─────────────────────────────────────────────────────────
    def _apply_music_volume(self):
        pygame.mixer.music.set_volume(self.settings.get("music_volume", 0.6))

    def _start_bg_music(self):
        pygame.mixer.music.stop()
        if not self.settings.get("music_enabled", True):
            return
        path = self.settings.get("music_path") or ""
        if not path or not os.path.isfile(path):
            return
        try:
            pygame.mixer.music.load(path)
            self._apply_music_volume()
            pygame.mixer.music.play(0)
            self._music_start_wall  = time.perf_counter()
            self._using_music_clock = True
        except pygame.error:
            self._using_music_clock = False

    def _stop_bg_music(self):
        pygame.mixer.music.fadeout(600)

    # ── Round state ───────────────────────────────────────────────────────────
    def _clear_round(self):
        self.notes      = []
        self.popups     = []
        self.score      = 0
        self.combo      = 0
        self.max_combo  = 0
        self.multi      = 1
        self.music_ms   = 0
        self.spawned    = set()
        self.key_flash  = [0] * LANE_COUNT
        self._using_music_clock = False
        self._music_start_wall  = None
        self._fallback_ms       = 0
        self.song_bpm           = 0.0

    def reset(self):
        self._clear_round()
        self.running = True
        self.state   = "menu"

    # ── Difficulty helpers ────────────────────────────────────────────────────
    def _difficulty(self):
        return int(self.settings.get("difficulty", 1))

    def _change_difficulty(self, delta):
        d = max(0, min(3, self._difficulty() + delta))
        self.settings["difficulty"] = d
        save_settings(self.settings)
        if self.diff_flash:
            self.diff_flash.trigger()

    # ── Loading screen ────────────────────────────────────────────────────────
    def _show_loading_screen(self):
        self.screen.fill(BG)
        diff   = self._difficulty()
        t1     = self.font_big.render("GENERATING NOTES", True, DIFF_COLOURS[diff])
        self.screen.blit(t1, (self.w // 2 - t1.get_width() // 2, self.h // 2 - 70))
        path   = (self.settings.get("music_path") or "").strip()
        name   = Path(path).name if path else "demo beatmap"
        t2     = self.font_med.render(name[:52], True, GREY)
        self.screen.blit(t2, (self.w // 2 - t2.get_width() // 2, self.h // 2 - 14))
        dlbl   = self.font_med.render(f"Difficulty: {DIFFICULTIES[diff]}", True, DIFF_COLOURS[diff])
        self.screen.blit(dlbl, (self.w // 2 - dlbl.get_width() // 2, self.h // 2 + 20))
        hint   = self.font_small.render(
            "Hits follow percussive transients, beats, and pitch — not a generic grid",
            True, DARK_GREY,
        )
        self.screen.blit(hint, (self.w // 2 - hint.get_width() // 2, self.h // 2 + 50))
        pygame.display.flip()

    # ── Chart preparation ─────────────────────────────────────────────────────
    def _prepare_round_chart(self):
        diff = self._difficulty()
        self.pattern          = list(FALLBACK_PATTERN)
        self.chart_duration_ms = FALLBACK_DURATION_MS
        self.play_end_ms      = self.chart_duration_ms + 3000
        self.chart_status     = "Demo chart (no MP3)"
        self.now_playing      = ""
        self.song_bpm         = 0.0

        path = (self.settings.get("music_path") or "").strip()
        if not path or not os.path.isfile(path):
            return

        cached = load_cached_chart(path, CHART_CACHE_DIR, difficulty=diff)
        if cached:
            raw_pattern, duration_ms, bpm = cached
            self.chart_status = (
                f"Full song beatmap — {Path(path).name} "
                f"[{DIFFICULTIES[diff]}] (cached)"
            )
        else:
            result = generate_chart_from_file(path, difficulty=diff)
            raw_pattern, duration_ms, bpm, err = result
            if err or not raw_pattern:
                self.chart_status = err or "Analysis failed — demo chart"
                return
            save_cached_chart(path, CHART_CACHE_DIR, diff, raw_pattern, duration_ms, bpm)
            self.chart_status = (
                f"Full song beatmap — {Path(path).name} [{DIFFICULTIES[diff]}]"
            )

        if len(raw_pattern) < 12:
            self.chart_status = "Too few notes — demo chart"
            return

        self.pattern           = raw_pattern
        self.now_playing       = Path(path).name
        self.song_bpm          = bpm
        last_hit               = max(h for h, _ in self.pattern)
        self.chart_duration_ms = int(duration_ms)
        self.play_end_ms       = max(self.chart_duration_ms, last_hit) + 3500

    # ── Music clock ───────────────────────────────────────────────────────────
    def _update_music_ms(self, dt_ms):
        if self._using_music_clock:
            pos = pygame.mixer.music.get_pos()
            if pos >= 0:
                self.music_ms = pos
                return
            if self._music_start_wall is not None:
                self.music_ms = int((time.perf_counter() - self._music_start_wall) * 1000)
                return
        self._fallback_ms += dt_ms
        self.music_ms = self._fallback_ms

    # ── Note spawning & hitting ───────────────────────────────────────────────
    def spawn_notes(self):
        for hit_ms, lane in self.pattern:
            key = (hit_ms, lane)
            if key in self.spawned:
                continue
            if self.music_ms >= hit_ms - self.lead_ms - 50:
                self.spawned.add(key)
                self.notes.append(Note(self, lane, hit_ms))

    def hit_lane(self, lane):
        self.key_flash[lane] = 8
        candidates = [n for n in self.notes if not n.hit and not n.missed and n.lane == lane]
        if not candidates:
            self.combo = 0
            self.multi = 1
            return
        n    = min(candidates, key=lambda x: abs(x.y - self.hit_y))
        dist = abs(n.y - self.hit_y)
        if dist <= self.perfect_win:
            n.hit       = True
            self.combo += 1
            self.multi  = 4 if self.combo >= 16 else 3 if self.combo >= 8 else 2 if self.combo >= 4 else 1
            self.score += 100 * self.multi
            self.max_combo = max(self.max_combo, self.combo)
            self.popups.append(Popup(self, "PERFECT", lane, (255, 251, 110)))
        elif dist <= self.good_win:
            n.hit       = True
            self.combo += 1
            self.multi  = 4 if self.combo >= 16 else 3 if self.combo >= 8 else 2 if self.combo >= 4 else 1
            self.score += 50 * self.multi
            self.max_combo = max(self.max_combo, self.combo)
            self.popups.append(Popup(self, "GOOD", lane, (110, 255, 158)))

    # ── Draw helpers ──────────────────────────────────────────────────────────
    def _lane_radius(self):
        return min(6, max(2, self.lane_w // 8))

    def draw_grid(self):
        self.screen.fill(BG)
        for i in range(1, LANE_COUNT):
            pygame.draw.line(self.screen, GRID, (i * self.lane_w, 0), (i * self.lane_w, self.h))
        pygame.draw.rect(self.screen, HIT_LINE, (0, self.hit_y - 1, self.w, 3))

    def draw_keys(self):
        r = self._lane_radius()
        for i in range(LANE_COUNT):
            col   = LANE_COLOURS[i]
            lw    = self.lane_w
            alpha = min(255, 60 + self.key_flash[i] * 22)
            rect  = (i * lw + 6, self.hit_y + 14, lw - 12, max(28, int(self.h * 0.055)))
            pygame.draw.rect(self.screen, tuple(int(c * alpha / 255) for c in col), rect, border_radius=r)
            pygame.draw.rect(self.screen, col, rect, width=2, border_radius=r)
            lbl = self.font_med.render(KEY_LABELS[i], True, col)
            self.screen.blit(lbl, (
                i * lw + lw // 2 - lbl.get_width() // 2,
                self.hit_y + 14 + rect[3] // 2 - lbl.get_height() // 2,
            ))
            if self.key_flash[i] > 0:
                self.key_flash[i] -= 1

    def draw_hud(self):
        W, H   = self.w, self.h
        diff   = self._difficulty()
        dcol   = DIFF_COLOURS[diff]

        # Score
        sc = self.font_big.render(f"{self.score:07d}", True, (232, 255, 110))
        self.screen.blit(sc, (W // 2 - sc.get_width() // 2, 10))

        # Combo
        if self.combo > 1:
            cb = self.font_med.render(f"×{self.combo}", True, (255, 110, 176))
            self.screen.blit(cb, (W // 2 - cb.get_width() // 2, 10 + sc.get_height() + 4))

        # Multi badge
        if self.multi > 1:
            mb    = self.font_small.render(f" ×{self.multi} MULTI ", True, (10, 10, 15))
            badge = pygame.Surface((mb.get_width() + 4, mb.get_height() + 4))
            badge.fill((110, 236, 255))
            badge.blit(mb, (2, 2))
            self.screen.blit(badge, (W - badge.get_width() - 8, 10))

        # Difficulty badge (top-left)
        dlbl  = self.font_small.render(f" {DIFFICULTIES[diff]} ", True, (10, 10, 15))
        dbadge = pygame.Surface((dlbl.get_width() + 4, dlbl.get_height() + 4))
        dbadge.fill(dcol)
        dbadge.blit(dlbl, (2, 2))
        self.screen.blit(dbadge, (8, 10))

        # BPM (top-left, below diff badge)
        if self.song_bpm > 0:
            bpm_lbl = self.font_small.render(f"{self.song_bpm:.0f} BPM", True, GREY)
            self.screen.blit(bpm_lbl, (8, 10 + dbadge.get_height() + 4))

        # Progress bar
        song_ms = max(1, self.chart_duration_ms)
        prog    = min(1.0, self.music_ms / song_ms)
        pygame.draw.rect(self.screen, DARK_GREY, (0, H - 5, W, 5))
        pygame.draw.rect(self.screen, dcol, (0, H - 5, int(W * prog), 5))

        # Track name & hint
        np_name   = getattr(self, "now_playing", "") or "Demo"
        track_line = self.font_small.render(f"Playing: {np_name[:46]}", True, (110, 236, 255))
        self.screen.blit(track_line, (W // 2 - track_line.get_width() // 2, H - 72))
        hint_txt  = "D F J K — hit notes  |  ← A / D → — change difficulty"
        hint      = self.font_small.render(hint_txt, True, GREY)
        self.screen.blit(hint, (W // 2 - hint.get_width() // 2, H - 48))

    # ── Screen draw methods ───────────────────────────────────────────────────
    def draw_menu(self):
        W, H = self.w, self.h
        self.screen.fill(BG)
        diff  = self._difficulty()
        dcol  = DIFF_COLOURS[diff]

        title = self.font_big.render("RHYTHM", True, (232, 255, 110))
        self.screen.blit(title, (W // 2 - title.get_width() // 2, int(H * 0.12)))

        # Difficulty selector
        dlbl = self.font_med.render(
            f"◄  {DIFFICULTIES[diff]}  ►   (A / D to change)",
            True, dcol,
        )
        self.screen.blit(dlbl, (W // 2 - dlbl.get_width() // 2, int(H * 0.24)))

        lines = [
            "D F J K — press each lane when its note hits the line",
            "Pick an MP3 in Settings: notes are generated for the whole track",
            "PERFECT = 100 pts   GOOD = 50 pts",
            "",
            "SPACE — start playing",
            "S — settings (track, volume, fullscreen)",
            "A / D — easier / harder difficulty",
            "F11 — toggle fullscreen",
            "ESC — quit",
        ]
        y0 = int(H * 0.36)
        for i, line in enumerate(lines):
            s = self.font_small.render(line, True, GREY)
            self.screen.blit(s, (W // 2 - s.get_width() // 2, y0 + i * int(H * 0.038)))

        if self.diff_flash:
            self.diff_flash.draw(self.screen)

    def draw_results(self):
        W, H = self.w, self.h
        self.screen.fill(BG)
        diff = self._difficulty()
        t    = self.font_big.render("RESULTS", True, (232, 255, 110))
        self.screen.blit(t, (W // 2 - t.get_width() // 2, int(H * 0.18)))
        lines = [
            f"Score      {self.score:,}",
            f"Max Combo  {self.max_combo}",
        ]
        cs = getattr(self, "chart_status", "")
        if cs:
            lines.append(cs[:56])
        if self.song_bpm > 0:
            lines.append(f"BPM: {self.song_bpm:.0f}  |  Difficulty: {DIFFICULTIES[diff]}")
        lines += ["", "SPACE — retry    M — menu    A/D — difficulty    ESC — quit"]
        y0 = int(H * 0.30)
        for i, line in enumerate(lines):
            col = WHITE if i < 2 else (DIFF_COLOURS[diff] if i == 3 and self.song_bpm > 0 else GREY)
            s   = self.font_med.render(line, True, col)
            self.screen.blit(s, (W // 2 - s.get_width() // 2, y0 + i * int(H * 0.048)))

    def _settings_labels(self):
        on    = "ON" if self.settings.get("music_enabled", True) else "OFF"
        vol   = int(self.settings.get("music_volume", 0.6) * 100)
        fs    = "ON" if self.settings.get("fullscreen", True) else "OFF"
        track = self.settings.get("music_path") or ""
        short = Path(track).name if track else "(none — put .mp3 in music folder)"
        if len(short) > 44:
            short = short[:41] + "..."
        return [
            f"Music: {on}     ←/→  Space",
            f"Volume: {vol}%     ←/→",
            f"Track: {short}     ENTER to pick",
            f"Fullscreen: {fs}     ←/→  Space",
        ]

    def draw_settings(self):
        W, H = self.w, self.h
        self.screen.fill(BG)
        title = self.font_big.render("SETTINGS", True, (232, 255, 110))
        self.screen.blit(title, (W // 2 - title.get_width() // 2, int(H * 0.08)))
        lines  = self._settings_labels()
        y0     = int(H * 0.2)
        line_h = int(H * 0.072)
        for i, line in enumerate(lines):
            sel = i == self.settings_cursor
            col = (110, 236, 255) if sel and i == 2 else ((232, 255, 110) if sel else GREY)
            s   = self.font_med.render(("› " if sel else "  ") + line, True, col)
            self.screen.blit(s, (max(24, W // 2 - s.get_width() // 2), y0 + i * line_h))
        hint = self.font_small.render(
            "UP/DOWN — move   ENTER — pick track   ESC — save & back", True, GREY,
        )
        self.screen.blit(hint, (W // 2 - hint.get_width() // 2, H - 48))

    def draw_music_pick(self):
        W, H = self.w, self.h
        self.screen.fill(BG)
        title = self.font_big.render("CHOOSE MP3", True, (232, 255, 110))
        self.screen.blit(title, (W // 2 - title.get_width() // 2, int(H * 0.06)))
        folder_hint = self.font_small.render(str(MUSIC_DIR), True, GREY)
        self.screen.blit(folder_hint, (W // 2 - folder_hint.get_width() // 2, int(H * 0.12)))
        self.music_list = list_mp3_files()
        if not self.music_list:
            msg = self.font_med.render("No .mp3 files found. Add some and press R.", True, WHITE)
            self.screen.blit(msg, (W // 2 - msg.get_width() // 2, H // 2))
            return
        max_vis = max(4, (H - int(H * 0.22)) // int(H * 0.045))
        if self.music_pick_index >= self.music_pick_scroll + max_vis:
            self.music_pick_scroll = self.music_pick_index - max_vis + 1
        if self.music_pick_index < self.music_pick_scroll:
            self.music_pick_scroll = self.music_pick_index
        y = int(H * 0.2)
        for j in range(self.music_pick_scroll, min(len(self.music_list), self.music_pick_scroll + max_vis)):
            name = Path(self.music_list[j]).name
            sel  = j == self.music_pick_index
            col  = (232, 255, 110) if sel else WHITE
            s    = self.font_med.render(("› " if sel else "  ") + name, True, col)
            self.screen.blit(s, (40, y))
            y += int(H * 0.045)
        foot = self.font_small.render("UP/DOWN — move   ENTER — select   ESC — back   R — refresh", True, GREY)
        self.screen.blit(foot, (W // 2 - foot.get_width() // 2, H - 40))

    # ── Key handlers ──────────────────────────────────────────────────────────
    def _settings_handle_key(self, key):
        if key == pygame.K_ESCAPE:
            save_settings(self.settings)
            self.state = "menu"
            return
        if key == pygame.K_UP:
            self.settings_cursor = (self.settings_cursor - 1) % 4
            return
        if key == pygame.K_DOWN:
            self.settings_cursor = (self.settings_cursor + 1) % 4
            return
        if self.settings_cursor == 0:
            if key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_RETURN, pygame.K_SPACE):
                self.settings["music_enabled"] = not self.settings.get("music_enabled", True)
                save_settings(self.settings)
        elif self.settings_cursor == 1:
            v = self.settings.get("music_volume", 0.6)
            if key == pygame.K_LEFT:
                self.settings["music_volume"] = max(0.0, round(v - 0.05, 2))
            elif key == pygame.K_RIGHT:
                self.settings["music_volume"] = min(1.0, round(v + 0.05, 2))
            self._apply_music_volume()
            save_settings(self.settings)
        elif self.settings_cursor == 2:
            if key == pygame.K_RETURN:
                self.music_list = list_mp3_files()
                self.music_pick_index  = 0
                self.music_pick_scroll = 0
                if self.settings.get("music_path") in self.music_list:
                    self.music_pick_index = self.music_list.index(self.settings["music_path"])
                self.state = "music_pick"
        elif self.settings_cursor == 3:
            if key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_RETURN, pygame.K_SPACE):
                self.settings["fullscreen"] = not self.settings.get("fullscreen", True)
                save_settings(self.settings)
                self._recreate_display()

    def _music_pick_handle_key(self, key):
        if key == pygame.K_ESCAPE:
            self.state = "settings"
            return
        if key == pygame.K_r:
            self.music_list = list_mp3_files()
            self.music_pick_index = min(self.music_pick_index, max(0, len(self.music_list) - 1))
            return
        if not self.music_list:
            return
        if key == pygame.K_UP:
            self.music_pick_index = (self.music_pick_index - 1) % len(self.music_list)
        elif key == pygame.K_DOWN:
            self.music_pick_index = (self.music_pick_index + 1) % len(self.music_list)
        elif key == pygame.K_RETURN:
            self.settings["music_path"] = self.music_list[self.music_pick_index]
            save_settings(self.settings)
            self.state = "settings"

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        while self.running:
            dt = self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.VIDEORESIZE:
                    if not self.settings.get("fullscreen", True):
                        self.screen = pygame.display.set_mode(
                            (max(400, event.w), max(300, event.h)), pygame.RESIZABLE,
                        )
                        self._layout_from_size(*self.screen.get_size())

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self._hotkey_toggle_fullscreen()
                        continue

                    if self.state == "settings":
                        self._settings_handle_key(event.key)
                        continue
                    if self.state == "music_pick":
                        self._music_pick_handle_key(event.key)
                        continue

                    # Global A/D difficulty change (menu + results)
                    if self.state in ("menu", "results"):
                        if event.key == pygame.K_a:
                            self._change_difficulty(-1)
                            continue
                        if event.key == pygame.K_d and self.state != "playing":
                            self._change_difficulty(1)
                            continue

                    if event.key == pygame.K_ESCAPE:
                        if self.state == "playing":
                            self._stop_bg_music()
                            self.state = "menu"
                        else:
                            self.running = False

                    elif event.key == pygame.K_s and self.state == "menu":
                        self.settings_cursor = 0
                        self.state = "settings"

                    elif event.key == pygame.K_SPACE:
                        if self.state in ("menu", "results"):
                            self._clear_round()
                            self._show_loading_screen()
                            self._prepare_round_chart()
                            self._start_bg_music()
                            self.state = "playing"

                    elif event.key == pygame.K_m and self.state == "results":
                        self._stop_bg_music()
                        self.reset()
                        self.state = "menu"

                    elif self.state == "playing":
                        for i, k in enumerate(KEYS):
                            if event.key == k:
                                self.hit_lane(i)

            # ── Playing update ────────────────────────────────────────────────
            if self.state == "playing":
                self._update_music_ms(dt)
                self.spawn_notes()

                for n in self.notes:
                    n.update()
                    if n.missed_check:
                        n.missed = True
                        self.combo = 0
                        self.multi = 1
                        self.popups.append(Popup(self, "MISS", n.lane, (255, 68, 68)))

                self.notes  = [n for n in self.notes if not n.dead and n.y < self.h + 40]
                self.popups = [p for p in self.popups if p.life > 0]
                for p in self.popups:
                    p.update()

                if self.music_ms >= self.play_end_ms and not self.notes:
                    self.state = "results"
                    self._stop_bg_music()

                self.draw_grid()
                self.draw_keys()
                for n in self.notes:
                    n.draw(self.screen)
                for p in self.popups:
                    p.draw(self.screen)
                self.draw_hud()

            elif self.state == "menu":
                self.draw_menu()
                if self.diff_flash:
                    self.diff_flash.update()

            elif self.state == "results":
                self.draw_results()
                if self.diff_flash:
                    self.diff_flash.update()

            elif self.state == "settings":
                self.draw_settings()

            elif self.state == "music_pick":
                self.draw_music_pick()

            pygame.display.flip()

        save_settings(self.settings)
        pygame.mixer.music.stop()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    RhythmGame().run()
