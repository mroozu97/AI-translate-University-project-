# ai_controller.py
"""
Autorski moduł AI-lite do sterowania pipeline'em audio→ASR→tłumaczenie.

Co robi:
- wyciąga cechy (feature engineering) z segmentu audio (RMS, ZCR, widmowy tilt, centroid, długość)
- klasyfikuje segment na: SILENCE / NOISE / SPEECH / COMMAND (model punktowy + progi adaptacyjne)
- wykrywa komendy głosowe z transkrypcji (STOP, zmiana języka) – logika "intent detection"
- posiada adaptację online (uczenie w locie): kalibracja poziomu tła i automatyczne dostrajanie progów

Nie używa gotowych "AI denoiserów" ani zewnętrznych modeli ML – to własna, inżynierska AI
oparta o cechy i funkcję decyzyjną.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import re

import numpy as np


# -----------------------------
# Konfiguracja i typy
# -----------------------------

SegmentClass = str  # "SILENCE" | "NOISE" | "SPEECH" | "COMMAND"


@dataclass
class AIControllerConfig:
    # Zakładamy wejście w float32 mono [-1, 1]
    samplerate: int = 48000

    # Minimalna długość segmentu rozważana jako "sensowna"
    min_segment_ms: int = 250

    # Parametry adaptacji tła (online noise profiling)
    noise_ema_alpha: float = 0.05   # jak szybko aktualizujemy poziom tła
    init_noise_floor: float = 0.004 # startowy szacowany RMS tła (zależnie od mikrofonu)

    # Progi bazowe (będą korygowane adaptacyjnie)
    speech_rms_factor: float = 3.0      # mowa jeśli RMS > noise_floor * factor
    command_rms_factor: float = 2.2     # komenda (często krótsza) - niższy próg
    silence_rms_factor: float = 1.4     # cisza jeśli RMS < noise_floor * factor

    # Cechy widmowe: używamy ich do rozróżnienia NOISE vs SPEECH
    # (mowa ma zwykle centroid + tilt w typowych zakresach)
    speech_centroid_hz_range: Tuple[float, float] = (200.0, 3500.0)
    speech_tilt_db_range: Tuple[float, float] = (-35.0, -5.0)  # uśredniony spadek energii z freq

    # Model punktowy (wagi cech) – „AI” w sensie funkcji decyzyjnej
    w_rms: float = 0.55
    w_zcr: float = 0.10
    w_centroid: float = 0.20
    w_tilt: float = 0.15

    # Granice score dla klas
    speech_score_threshold: float = 0.55
    noise_score_threshold: float = 0.35

    # Komendy/aliasy języków (dla intent detection)
    stop_phrases: List[str] = field(default_factory=lambda: [
        "zegnaj gulu widzimy sie w piekle",
        "żegnaj gulu widzimy się w piekle",
        "żegnaj gólu widzimy się w piekle",
    ])

    lang_aliases: Dict[str, str] = field(default_factory=lambda: {
        # English
        "angielski": "EN-GB",
        "angielski brytyjski": "EN-GB",
        "english": "EN-GB",
        "english uk": "EN-GB",
        "angielski amerykanski": "EN-US",
        "angielski amerykański": "EN-US",
        "english us": "EN-US",
        "american english": "EN-US",
        # German
        "niemiecki": "DE",
        "german": "DE",
        # French
        "francuski": "FR",
        "french": "FR",
        # Spanish
        "hiszpanski": "ES",
        "hiszpański": "ES",
        "spanish": "ES",
        # Italian
        "wloski": "IT",
        "włoski": "IT",
        "italian": "IT",
        # Polish
        "polski": "PL",
        "polish": "PL",
    })

    change_lang_patterns: List[str] = field(default_factory=lambda: [
        r"\bzmie[nń]\s+jezyk\s+na\s+(.+)$",
        r"\bzmie[nń]\s+język\s+na\s+(.+)$",
        r"\bt[łl]umacz\s+na\s+(.+)$",
        r"\bustaw\s+jezyk\s+(?:t[łl]umaczenia\s+)?na\s+(.+)$",
        r"\bustaw\s+język\s+(?:t[łl]umaczenia\s+)?na\s+(.+)$",
        r"\btranslate\s+to\s+(.+)$",
    ])


@dataclass
class SegmentFeatures:
    duration_s: float
    rms: float
    zcr: float
    centroid_hz: float
    tilt_db: float
    score: float


@dataclass
class Intent:
    """Zwracany z analizy transkrypcji (tekst -> intencja)."""
    type: str  # "NONE" | "STOP" | "CHANGE_LANG"
    payload: Optional[str] = None  # np. kod języka DeepL


# -----------------------------
# Kontroler AI
# -----------------------------

class AIController:
    def __init__(self, config: AIControllerConfig):
        self.cfg = config
        self.noise_floor_rms = float(config.init_noise_floor)

    # --------- Tekst: normalizacja + intent detection ---------

    @staticmethod
    def _normalize_text(txt: str) -> str:
        txt = txt.lower()
        txt = re.sub(r"[^\w\sąćęłńóśżź]", "", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def detect_intent(self, transcript_text: str) -> Intent:
        """
        Analiza intencji (intent detection) na podstawie transkrypcji.
        To część 'AI sterującego' – pozwala obsłużyć komendy.
        """
        norm = self._normalize_text(transcript_text)

        # STOP
        for p in self.cfg.stop_phrases:
            if p in norm:
                return Intent(type="STOP")

        # ZMIANA JĘZYKA
        for pat in self.cfg.change_lang_patterns:
            m = re.search(pat, norm)
            if not m:
                continue
            raw = m.group(1).strip()
            raw = re.sub(r"\b(prosze|proszę|teraz|dziekuje|dziękuję)\b$", "", raw).strip()

            if raw in self.cfg.lang_aliases:
                return Intent(type="CHANGE_LANG", payload=self.cfg.lang_aliases[raw])

            for k, v in self.cfg.lang_aliases.items():
                if raw.startswith(k):
                    return Intent(type="CHANGE_LANG", payload=v)

            return Intent(type="NONE")

        return Intent(type="NONE")

    # --------- Audio: feature engineering + klasyfikacja ---------

    def extract_features(self, audio: np.ndarray) -> SegmentFeatures:
        """
        Wyciąga cechy segmentu audio.
        audio: float32 mono [-1,1], samplerate = cfg.samplerate
        """
        x = np.asarray(audio, dtype=np.float32)
        n = int(x.size)
        sr = self.cfg.samplerate
        dur = n / float(sr) if sr > 0 else 0.0

        # RMS
        rms = float(np.sqrt(np.mean(x * x)) + 1e-12)

        # ZCR (zero crossing rate)
        s = np.sign(x)
        s[s == 0] = 1
        zc = float(np.mean(s[1:] != s[:-1])) if n > 1 else 0.0

        # Spektralne cechy (FFT na oknie Hann)
        if n < 64:
            centroid = 0.0
            tilt_db = -80.0
        else:
            w = np.hanning(n).astype(np.float32)
            X = np.fft.rfft(x * w)
            mag = np.abs(X).astype(np.float32) + 1e-12
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)

            # centroid
            centroid = float(np.sum(freqs * mag) / np.sum(mag))

            # tilt: regresja liniowa log-mag vs log-freq (przybliżenie spadku)
            valid = freqs > 50.0
            f = freqs[valid]
            m2 = mag[valid]
            lf = np.log10(f + 1e-6)
            lm = 20.0 * np.log10(m2 + 1e-12)

            if lf.size >= 5:
                A = np.vstack([lf, np.ones_like(lf)]).T
                slope, _intercept = np.linalg.lstsq(A, lm, rcond=None)[0]
                tilt_db = float(slope)  # ujemne = spadek energii z częstotliwością
            else:
                tilt_db = -80.0

        # Score (model punktowy): mapujemy cechy do [0,1] i liczymy ważoną sumę
        score = float(self._score(rms, zc, centroid, tilt_db, dur))

        return SegmentFeatures(
            duration_s=dur,
            rms=rms,
            zcr=zc,
            centroid_hz=centroid,
            tilt_db=tilt_db,
            score=score,
        )

    def classify_segment(self, feats: SegmentFeatures) -> SegmentClass:
        """
        Klasyfikacja segmentu:
        - SILENCE: bardzo cicho (blisko noise floor)
        - NOISE: jest energia, ale cechy widmowe nie wyglądają na mowę
        - SPEECH: prawdopodobna mowa
        - COMMAND: krótka, wyraźna mowa (używane jako heurystyka, właściwy intent z tekstu)
        """
        # Odrzuć za krótkie segmenty
        if feats.duration_s * 1000.0 < self.cfg.min_segment_ms:
            # Zbyt krótkie – najczęściej klik/artefakt/szum
            self._update_noise_floor(feats, is_speech_like=False)
            return "NOISE"

        # Progi adaptacyjne zależne od noise_floor
        nf = self.noise_floor_rms
        if feats.rms < nf * self.cfg.silence_rms_factor:
            self._update_noise_floor(feats, is_speech_like=False)
            return "SILENCE"

        # Gdy głośniej niż tło – użyj score
        if feats.score >= self.cfg.speech_score_threshold and feats.rms > nf * self.cfg.speech_rms_factor:
            # Krótkie, wyraźne segmenty traktujemy jako potencjalne komendy
            if feats.duration_s < 1.3 and feats.rms > nf * self.cfg.command_rms_factor:
                self._update_noise_floor(feats, is_speech_like=True)
                return "COMMAND"
            self._update_noise_floor(feats, is_speech_like=True)
            return "SPEECH"

        # Jeśli score nie dobił – zakładamy NOISE i aktualizujemy tło
        self._update_noise_floor(feats, is_speech_like=False)
        return "NOISE"

    # --------- Wewnętrzne: scoring + adaptacja ---------

    def _score(self, rms: float, zcr: float, centroid_hz: float, tilt_db: float, dur_s: float) -> float:
        """
        Model punktowy: skaluje cechy do [0,1] i liczy ważoną sumę.
        """
        # RMS względny do noise floor
        nf = max(self.noise_floor_rms, 1e-6)
        rms_rel = np.clip((rms / (nf * 5.0)), 0.0, 1.0)  # mowa zwykle kilka x tło

        # ZCR: mowa ma typowo średnie ZCR (zależy od głoski), szum bywa ekstremalny
        zcr_n = float(np.clip(1.0 - abs(zcr - 0.08) / 0.08, 0.0, 1.0))

        # Centroid: mowa mieści się w dość wąskim przedziale
        cmin, cmax = self.cfg.speech_centroid_hz_range
        if centroid_hz <= 0:
            cent_n = 0.0
        else:
            # 1 w środku przedziału, spada poza
            if centroid_hz < cmin:
                cent_n = float(np.clip(centroid_hz / cmin, 0.0, 1.0))
            elif centroid_hz > cmax:
                cent_n = float(np.clip(cmax / centroid_hz, 0.0, 1.0))
            else:
                cent_n = 1.0

        # Tilt: mowa ma typowo ujemny slope w pewnym zakresie
        tmin, tmax = self.cfg.speech_tilt_db_range
        if tilt_db < tmin:
            tilt_n = 0.0
        elif tilt_db > tmax:
            tilt_n = 0.0
        else:
            # w środku zakresu = 1
            mid = (tmin + tmax) / 2.0
            half = (tmax - tmin) / 2.0
            tilt_n = float(np.clip(1.0 - abs(tilt_db - mid) / (half + 1e-9), 0.0, 1.0))

        score = (
            self.cfg.w_rms * rms_rel
            + self.cfg.w_zcr * zcr_n
            + self.cfg.w_centroid * cent_n
            + self.cfg.w_tilt * tilt_n
        )

        # Delikatna kara za ekstremalnie długie segmenty (często hałas / muzyka)
        if dur_s > 10.0:
            score *= 0.85

        return float(np.clip(score, 0.0, 1.0))

    def _update_noise_floor(self, feats: SegmentFeatures, is_speech_like: bool) -> None:
        """
        Adaptacja online: aktualizujemy noise_floor tylko gdy segment wygląda jak cisza/szum.
        Dzięki temu progi dopasowują się do mikrofonu i środowiska.
        """
        if is_speech_like:
            return

        alpha = self.cfg.noise_ema_alpha
        self.noise_floor_rms = float((1.0 - alpha) * self.noise_floor_rms + alpha * feats.rms)


# -----------------------------
# Pomocnicze API dla integracji
# -----------------------------

def decide_send_to_asr(controller: AIController, audio_segment: np.ndarray) -> Tuple[bool, SegmentFeatures, SegmentClass]:
    """
    Funkcja pomocnicza: czy wysłać segment do Whispera?
    Zwraca: (send_to_asr, features, class)
    """
    feats = controller.extract_features(audio_segment)
    cls = controller.classify_segment(feats)
    send = cls in ("SPEECH", "COMMAND")
    return send, feats, cls
