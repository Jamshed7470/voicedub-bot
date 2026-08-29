"""Модель проекта: единственный источник правды для пайплайна и студии.

Всё состояние задачи живёт в data/jobs/<job_id>/project.json. Пайплайн
пишет туда результаты стадий, студия — правки пользователя, и оба читают
одно и то же. Поле version растёт при каждой записи и служит для
оптимистичной блокировки: два одновременных редактора не затрут друг друга
молча, второй получит 409 и увидит свежие данные.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

# фиксированная палитра: цвет спикера должен быть узнаваем и на таймлайне,
# и в таблице, и не меняться между открытиями проекта
PALETTE = [
    "#4C8DFF", "#F2585B", "#33B679", "#FFB020", "#A855F7", "#0EA5E9",
    "#EC4899", "#84CC16", "#F97316", "#14B8A6", "#8B5CF6", "#EF4444",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def free_color(used: list[str] | set[str]) -> str:
    """Первый цвет палитры, который ещё никем не занят.

    Выбирать цвет по счётчику спикеров нельзя: после удалений и добавлений
    счётчик совпадёт с чужим индексом, и два человека станут одного цвета —
    а на таймлайне их только по цвету и различают.
    """
    taken = set(used)
    for i in range(len(PALETTE) * 8):
        color = color_for(i)
        if color not in taken:
            return color
    return color_for(len(taken))


def color_for(index: int) -> str:
    """Больше 12 спикеров — палитра циклится с затемнением оттенка."""
    base = PALETTE[index % len(PALETTE)]
    ring = index // len(PALETTE)
    if not ring:
        return base
    r, g, b = (int(base[i:i + 2], 16) for i in (1, 3, 5))
    k = max(0.45, 1.0 - 0.22 * ring)
    return "#%02X%02X%02X" % (int(r * k), int(g * k), int(b * k))


class Stage(str, Enum):
    UPLOADED = "uploaded"
    SEPARATED = "separated"
    TRANSCRIBED = "transcribed"
    DIARIZED = "diarized"
    PROFILED = "profiled"
    TRANSLATED = "translated"
    REVIEW = "review"
    APPROVED = "approved"
    SYNTHESIZING = "synthesizing"
    MIXING = "mixing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


STAGE_TITLES_RU = {
    Stage.UPLOADED: "загружено",
    Stage.SEPARATED: "дорожки разделены",
    Stage.TRANSCRIBED: "речь распознана",
    Stage.DIARIZED: "спикеры определены",
    Stage.PROFILED: "голоса зафиксированы",
    Stage.TRANSLATED: "переведено",
    Stage.REVIEW: "ждёт проверки",
    Stage.APPROVED: "утверждено",
    Stage.SYNTHESIZING: "озвучивается",
    Stage.MIXING: "собирается",
    Stage.DONE: "готово",
    Stage.FAILED: "ошибка",
    Stage.CANCELLED: "отменено",
}


class Source(BaseModel):
    file_name: str = ""
    duration_sec: float = 0.0
    video_path: str | None = None
    vocals_path: str | None = None
    background_path: str | None = None
    title: str = ""


class Settings(BaseModel):
    voice_mode: str = "auto"              # clone | preset | auto
    unique_voices: bool = True
    auto_approve: bool = False
    emotion_subprofiles: bool = False
    num_speakers_hint: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    keep_background: bool = True
    original_as_second_track: bool = False
    translation_style: str = "normal"


class SpeakerStats(BaseModel):
    segments_count: int = 0
    total_speech_sec: float = 0.0
    share_pct: float = 0.0
    first_appearance_sec: float = 0.0


class Reference(BaseModel):
    path: str | None = None
    clean_sec: float = 0.0
    snr_db: float = 0.0
    score: float = 0.0
    clone_allowed: bool = False
    best_samples: list[int] = Field(default_factory=list)


class CastingCandidate(BaseModel):
    preset_id: str
    display_name: str = ""
    score: float = 0.0


class Voice(BaseModel):
    mode: str = "clone"                   # clone | preset
    preset_id: str | None = None
    preset_name: str | None = None
    profile_path: str | None = None
    identity_path: str | None = None
    locked: bool = False
    casting_candidates: list[CastingCandidate] = Field(default_factory=list)
    edited_by_user: bool = False


class Speaker(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    label: str = ""
    name: str | None = None
    role: str | None = None
    gender: str = "unknown"               # male | female | unknown
    gender_confidence: float = 0.0
    gender_edited_by_user: bool = False
    age: str = "adult"
    color: str = PALETTE[0]
    stats: SpeakerStats = Field(default_factory=SpeakerStats)
    centroid_path: str | None = None
    reference: Reference = Field(default_factory=Reference)
    voice: Voice = Field(default_factory=Voice)
    merged_from: list[str] = Field(default_factory=list)
    merged_into: str | None = None
    notes: str = ""

    @property
    def display(self) -> str:
        return self.name or self.label or self.id


class SynthInfo(BaseModel):
    path: str | None = None
    seed: int | None = None
    attempts: int = 0
    identity_sim: float = 0.0
    duration_ratio: float = 0.0
    backcheck_cer: float | None = None
    status: str = "pending"               # pending | ok | qc_failed


class VoiceOverride(BaseModel):
    mode: str | None = None
    preset_id: str | None = None


class EditedByUser(BaseModel):
    fields: list[str] = Field(default_factory=list)
    ts: str | None = None

    def touch(self, field: str) -> None:
        if field not in self.fields:
            self.fields.append(field)
        self.ts = now_iso()


class Segment(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    start: float
    end: float
    speaker_id: str
    speaker_confidence: float = 0.0
    speaker_margin: float = 0.0
    overlap: bool = False
    overlap_with: list[str] = Field(default_factory=list)
    text_src: str = ""
    text_tgt: str = ""
    text_tts: str = ""
    emotion: str = "neutral"
    events: list[str] = Field(default_factory=list)
    asr_confidence: float = 0.0
    budget_chars: int = 0
    over_budget: bool = False
    voice_override: VoiceOverride | None = None
    synth: SynthInfo = Field(default_factory=SynthInfo)
    flags: list[str] = Field(default_factory=list)
    edited_by_user: EditedByUser = Field(default_factory=EditedByUser)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class QCWarning(BaseModel):
    code: str
    severity: str = "warn"                # info | warn | error
    message_ru: str = ""
    speaker_id: str | None = None
    segment_ids: list[int] = Field(default_factory=list)


class IdentityReport(BaseModel):
    voice: str = ""
    segments: int = 0
    passed: int = 0
    mean_pairwise_identity: float = 0.0


class QC(BaseModel):
    warnings: list[QCWarning] = Field(default_factory=list)
    identity_report: dict[str, IdentityReport] = Field(default_factory=dict)
    overall_identity: float = 0.0


class HistoryOp(BaseModel):
    ts: str = Field(default_factory=now_iso)
    op: str
    before: Any = None
    after: Any = None


class Project(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = SCHEMA_VERSION
    job_id: str
    owner_telegram_id: int = 0
    chat_id: int = 0
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    version: int = 1
    source: Source = Field(default_factory=Source)
    lang_src: str = ""
    lang_tgt: str = "ru"
    stage: Stage = Stage.UPLOADED
    settings: Settings = Field(default_factory=Settings)
    speakers: list[Speaker] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    qc: QC = Field(default_factory=QC)
    history: list[HistoryOp] = Field(default_factory=list)
    result_path: str | None = None
    error: str | None = None

    # ---------- доступ ----------

    def speaker(self, sid: str) -> Speaker | None:
        return next((s for s in self.speakers if s.id == sid), None)

    def segment(self, seg_id: int) -> Segment | None:
        return next((s for s in self.segments if s.id == seg_id), None)

    def active_speakers(self) -> list[Speaker]:
        """Спикеры, кроме объединённых в других."""
        return [s for s in self.speakers if not s.merged_into]

    def recompute_stats(self) -> None:
        """Пересчитывает статистику спикеров по текущей раскладке сегментов.

        Вызывается после любой правки, меняющей принадлежность сегмента:
        цифры в карточке спикера обязаны совпадать с таблицей, иначе
        интерфейс врёт пользователю.
        """
        total = sum(seg.duration for seg in self.segments) or 1.0
        by_speaker: dict[str, list[Segment]] = {}
        for seg in self.segments:
            by_speaker.setdefault(seg.speaker_id, []).append(seg)

        for sp in self.speakers:
            segs = by_speaker.get(sp.id, [])
            speech = sum(s.duration for s in segs)
            sp.stats = SpeakerStats(
                segments_count=len(segs),
                total_speech_sec=round(speech, 2),
                share_pct=round(100 * speech / total, 1),
                first_appearance_sec=round(min((s.start for s in segs), default=0.0), 2),
            )

    def warnings_count(self) -> int:
        return sum(1 for s in self.segments if s.flags)

    def push_history(self, op: str, before: Any = None, after: Any = None,
                     limit: int = 50) -> None:
        self.history.append(HistoryOp(op=op, before=before, after=after))
        if len(self.history) > limit:
            del self.history[:-limit]
