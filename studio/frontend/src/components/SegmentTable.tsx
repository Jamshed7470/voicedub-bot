// Таблица реплик.
//
// Строк бывает больше полутора тысяч, поэтому рисуется только видимое окно:
// без этого таблица на длинном фильме прокручивается рывками, и проверять
// в ней что-либо невозможно.

import { useEffect, useMemo, useRef, useState } from "react";
import { mediaUrl } from "../lib/api";
import { nowPlaying, onPlaybackChange, play as playAudio } from "../lib/player";
import { useProject, visibleSegments } from "../store/useProject";
import { FLAG_LABELS } from "../lib/types";
import type { Segment, Speaker } from "../lib/types";

const ROW_HEIGHT = 46;
const OVERSCAN = 8;

export function SegmentTable() {
  const project = useProject((s) => s.project);
  const filter = useProject((s) => s.filter);
  const setFilter = useProject((s) => s.setFilter);
  const current = useProject((s) => s.currentSegment);
  const setCurrent = useProject((s) => s.setCurrent);
  const selected = useProject((s) => s.selected);
  const toggleSelected = useProject((s) => s.toggleSelected);
  const assignSpeaker = useProject((s) => s.assignSpeaker);
  const segments = useProject(visibleSegments);

  const box = useRef<HTMLDivElement>(null);
  const [scroll, setScroll] = useState(0);
  const [height, setHeight] = useState(600);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setHeight(el.clientHeight));
    ro.observe(el);
    setHeight(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  // прокрутка к выбранной реплике: её выбирают и с таймлайна, и клавишами
  useEffect(() => {
    const el = box.current;
    if (!el || current == null) return;
    const i = segments.findIndex((s) => s.id === current);
    if (i < 0) return;
    const top = i * ROW_HEIGHT;
    if (top < el.scrollTop || top > el.scrollTop + el.clientHeight - ROW_HEIGHT * 2) {
      el.scrollTop = Math.max(0, top - el.clientHeight / 2);
    }
  }, [current, segments]);

  const first = Math.max(0, Math.floor(scroll / ROW_HEIGHT) - OVERSCAN);
  const last = Math.min(segments.length,
                        Math.ceil((scroll + height) / ROW_HEIGHT) + OVERSCAN);
  const window_ = segments.slice(first, last);

  const speakers = useMemo(
    () => new Map((project?.speakers ?? []).map((s) => [s.id, s])),
    [project?.speakers]);

  if (!project) return null;

  const counts = {
    low: project.segments.filter((s) => s.flags.includes("low_speaker_conf")).length,
    overlap: project.segments.filter((s) => s.overlap).length,
    budget: project.segments.filter((s) => s.over_budget).length,
    qc: project.segments.filter((s) => s.synth.status === "qc_failed").length,
    edited: project.segments.filter((s) => s.edited_by_user.fields.length).length,
  };

  return (
    <section className="flex-1 min-h-0 flex flex-col">
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-line
                      overflow-x-auto text-xs">
        <Chip active={filter.kind === "all"} onClick={() => setFilter({ kind: "all" })}>
          Все {project.segments.length}
        </Chip>
        <Chip active={filter.kind === "flag" && filter.flag === "low_speaker_conf"}
              disabled={!counts.low} tone="warn"
              onClick={() => setFilter({ kind: "flag", flag: "low_speaker_conf" })}>
          ⚠ Низкая уверенность {counts.low}
        </Chip>
        <Chip active={filter.kind === "flag" && filter.flag === "overlap"}
              disabled={!counts.overlap}
              onClick={() => setFilter({ kind: "flag", flag: "overlap" })}>
          Наложения {counts.overlap}
        </Chip>
        <Chip active={filter.kind === "flag" && filter.flag === "over_budget"}
              disabled={!counts.budget}
              onClick={() => setFilter({ kind: "flag", flag: "over_budget" })}>
          Длиннее слота {counts.budget}
        </Chip>
        <Chip active={filter.kind === "flag" && filter.flag === "identity_qc_failed"}
              disabled={!counts.qc} tone="danger"
              onClick={() => setFilter({ kind: "flag", flag: "identity_qc_failed" })}>
          Тембр не совпал {counts.qc}
        </Chip>
        <Chip active={filter.kind === "edited"} disabled={!counts.edited}
              onClick={() => setFilter({ kind: "edited" })}>
          Изменённые мной {counts.edited}
        </Chip>
      </div>

      {selected.size > 0 && (
        <div className="flex items-center gap-2 px-3 py-2 bg-accent/10
                        border-b border-accent/30 text-xs">
          <span>Выбрано реплик: {selected.size}</span>
          <select
            className="bg-ink-700 border border-line rounded px-2 py-1"
            defaultValue=""
            onChange={(e) => {
              if (e.target.value) {
                void assignSpeaker([...selected], e.target.value);
                e.target.value = "";
              }
            }}>
            <option value="">Назначить спикера…</option>
            {project.speakers.filter((s) => !s.merged_into).map((s) => (
              <option key={s.id} value={s.id}>{s.id} {s.name ?? ""}</option>
            ))}
          </select>
          <button onClick={() => useProject.getState().clearSelection()}
                  className="ml-auto text-muted hover:text-white">снять выделение</button>
        </div>
      )}

      <div className="flex gap-2 px-3 py-1.5 text-[11px] text-muted
                      border-b border-line font-medium">
        <span className="w-12">#</span>
        <span className="w-28">время</span>
        <span className="w-24">спикер</span>
        <span className="flex-1">перевод</span>
        <span className="w-20 text-center">увер.</span>
        <span className="w-14 text-center">QC</span>
        <span className="w-16 text-center">звук</span>
      </div>

      <div ref={box} onScroll={(e) => setScroll(e.currentTarget.scrollTop)}
           className="flex-1 overflow-y-auto">
        <div style={{ height: segments.length * ROW_HEIGHT, position: "relative" }}>
          {window_.map((seg, i) => (
            <Row
              key={seg.id} seg={seg}
              speaker={speakers.get(seg.speaker_id)}
              top={(first + i) * ROW_HEIGHT}
              active={seg.id === current}
              checked={selected.has(seg.id)}
              speakers={project.speakers.filter((s) => !s.merged_into)}
              onSelect={(e) => {
                if (e.shiftKey || e.ctrlKey || e.metaKey) toggleSelected(seg.id, true);
                else setCurrent(seg.id);
              }}
            />
          ))}
        </div>
        {segments.length === 0 && (
          <p className="p-6 text-center text-sm text-muted">
            По этому фильтру реплик нет.
          </p>
        )}
      </div>
    </section>
  );
}

function Row({ seg, speaker, top, active, checked, speakers, onSelect }: {
  seg: Segment; speaker?: Speaker; top: number; active: boolean;
  checked: boolean; speakers: Speaker[];
  onSelect: (e: React.MouseEvent) => void;
}) {
  const patchSegment = useProject((s) => s.patchSegment);
  const preview = useProject((s) => s.preview);
  const previewing = useProject((s) => s.previewing.has(seg.id));
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(seg.text_tgt);
  const playing = usePlayingKey();

  useEffect(() => setText(seg.text_tgt), [seg.text_tgt]);

  const save = () => {
    setEditing(false);
    if (text !== seg.text_tgt) void patchSegment(seg.id, { text_tgt: text });
  };

  const budget = seg.budget_chars || 0;
  const tooLong = budget > 0 && text.length > budget;

  return (
    <div
      onClick={onSelect}
      style={{ top, height: ROW_HEIGHT }}
      className={`absolute inset-x-0 flex items-center gap-2 px-3 text-xs
        border-b border-ink-700/50 cursor-pointer
        ${active ? "bg-accent/15" : checked ? "bg-accent/8" : "hover:bg-ink-700/40"}`}
    >
      <span className="w-12 text-muted tabular-nums">{seg.id}</span>
      <span className="w-28 text-muted tabular-nums">
        {fmt(seg.start)}–{fmt(seg.end)}
      </span>

      <select
        value={seg.speaker_id}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => void patchSegment(seg.id, { speaker_id: e.target.value })}
        className="w-24 bg-transparent border border-transparent hover:border-line
                   rounded px-1 py-0.5 outline-none"
        style={{ color: speaker?.color }}>
        {speakers.map((s) => (
          <option key={s.id} value={s.id} className="bg-ink-700 text-white">
            {s.id}{s.name ? ` ${s.name}` : ""}
          </option>
        ))}
      </select>

      <div className="flex-1 min-w-0" onClick={(e) => e.stopPropagation()}>
        {editing ? (
          <input
            autoFocus value={text} onChange={(e) => setText(e.target.value)}
            onBlur={save}
            onKeyDown={(e) => {
              if (e.key === "Enter") save();
              if (e.key === "Escape") { setText(seg.text_tgt); setEditing(false); }
            }}
            className={`w-full bg-ink-900 border rounded px-1.5 py-1 outline-none
                        ${tooLong ? "border-danger" : "border-accent"}`} />
        ) : (
          <button onClick={() => setEditing(true)}
                  className="w-full text-left truncate hover:text-accent">
            {seg.text_tgt || <span className="text-muted">— пусто —</span>}
            {seg.edited_by_user.fields.includes("text_tgt") && (
              <span className="ml-1 text-accent" title="изменено вами">•</span>
            )}
          </button>
        )}
      </div>

      <span className="w-20 flex items-center gap-1" title={
        seg.flags.map((f) => FLAG_LABELS[f] ?? f).join(", ")}>
        <span className="flex-1 h-1 rounded bg-ink-600 overflow-hidden">
          <span className="block h-full rounded"
                style={{
                  width: `${Math.max(4, seg.speaker_confidence * 100)}%`,
                  background: seg.speaker_confidence >= 0.6 ? "#33b679"
                    : seg.speaker_confidence >= 0.4 ? "#ffb020" : "#f2585b",
                }} />
        </span>
        <span className="tabular-nums text-[10px] text-muted">
          {seg.speaker_confidence.toFixed(2)}
        </span>
      </span>

      <span className="w-14 text-center">
        {seg.synth.status === "ok" && <span className="text-ok" title="проверку прошёл">●</span>}
        {seg.synth.status === "qc_failed" && (
          <span className="text-danger"
                title={`тембр ${seg.synth.identity_sim.toFixed(2)}`}>⚠</span>
        )}
        {seg.synth.status === "pending" && <span className="text-muted">—</span>}
      </span>

      <span className="w-16 flex items-center gap-1 justify-center"
            onClick={(e) => e.stopPropagation()}>
        <button title="Оригинал"
                onClick={() => playAudio(
                  mediaUrl(`/segments/${seg.id}/original.wav`),
                  `orig:${seg.id}`)}
                className={`w-6 h-6 rounded hover:bg-ink-600 ${
                  playing === `orig:${seg.id}` ? "bg-ink-500 text-white" : "text-muted"}`}>
          {playing === `orig:${seg.id}` ? "■" : "▶"}
        </button>
        <button title="Озвучка"
                onClick={() => {
                  if (seg.synth.status === "pending") void preview(seg.id);
                  else playAudio(mediaUrl(`/segments/${seg.id}/synth.wav`),
                                 `synth:${seg.id}`);
                }}
                className={`w-6 h-6 rounded hover:bg-ink-600 text-accent ${
                  playing === `synth:${seg.id}` ? "bg-accent/25" : ""}`}>
          {previewing ? "…" : playing === `synth:${seg.id}` ? "■" : "▶"}
        </button>
      </span>
    </div>
  );
}

function Chip({ children, active, disabled, tone, onClick }: {
  children: React.ReactNode; active?: boolean; disabled?: boolean;
  tone?: "warn" | "danger"; onClick: () => void;
}) {
  const color = tone === "danger" ? "text-danger" : tone === "warn" ? "text-warn" : "";
  return (
    <button disabled={disabled} onClick={onClick}
      className={`shrink-0 px-2.5 py-1 rounded-full border transition-colors
        ${active ? "bg-accent border-accent text-white"
                 : `border-line hover:border-ink-500 ${color}`}
        ${disabled ? "opacity-35 cursor-default" : ""}`}>
      {children}
    </button>
  );
}

function usePlayingKey(): string | null {
  const [key, setKey] = useState<string | null>(nowPlaying());
  useEffect(() => onPlaybackChange(setKey), []);
  return key;
}

function fmt(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}
