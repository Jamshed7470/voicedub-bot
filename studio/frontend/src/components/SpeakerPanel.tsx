// Левая колонка: карточки спикеров.
//
// Карточка отвечает на три вопроса разом: кто это, сколько говорит и каким
// голосом будет озвучен. Всё остальное (кастинг, объединение) — действия,
// которые человек делает, уже поняв ответы.

import { useState } from "react";
import { mediaUrl } from "../lib/api";
import { useProject } from "../store/useProject";
import type { Speaker } from "../lib/types";

const GENDER_ICON: Record<string, string> = {
  male: "♂", female: "♀", unknown: "?",
};

function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function play(url: string): void {
  const audio = new Audio(url);
  void audio.play().catch(() => undefined);
}

export function SpeakerPanel() {
  const project = useProject((s) => s.project);
  const filter = useProject((s) => s.filter);
  const setFilter = useProject((s) => s.setFilter);
  const openDrawer = useProject((s) => s.openDrawer);
  const mergeSpeakers = useProject((s) => s.mergeSpeakers);
  const [dragging, setDragging] = useState<string | null>(null);

  if (!project) return null;
  const speakers = project.speakers.filter((s) => !s.merged_into);

  const onDrop = (target: Speaker) => {
    if (!dragging || dragging === target.id) return;
    const source = speakers.find((s) => s.id === dragging);
    if (!source) return;
    const n = source.stats.segments_count;
    const ok = window.confirm(
      `Объединить ${source.id} в ${target.id}?\n\n` +
      `${n} реплик будут переназначены на ${target.display ?? target.id}. ` +
      "Профиль голоса потребует пересборки.");
    if (ok) void mergeSpeakers(dragging, target.id);
    setDragging(null);
  };

  return (
    <aside className="w-[320px] shrink-0 border-r border-line bg-ink-800
                      overflow-y-auto">
      <div className="sticky top-0 z-10 bg-ink-800/95 backdrop-blur px-3 py-2
                      border-b border-line flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-wide">
          СПИКЕРЫ <span className="text-muted font-normal">{speakers.length}</span>
        </h2>
        <button
          onClick={() => setFilter({ kind: "all" })}
          className={`text-xs px-2 py-1 rounded transition-colors ${
            filter.kind === "all" ? "bg-accent text-white" : "text-muted hover:text-white"
          }`}>
          все
        </button>
      </div>

      <div className="p-2 space-y-2">
        {speakers.map((sp) => (
          <SpeakerCard
            key={sp.id} speaker={sp}
            active={filter.kind === "speaker" && filter.id === sp.id}
            dragging={dragging === sp.id}
            onSelect={() => setFilter(
              filter.kind === "speaker" && filter.id === sp.id
                ? { kind: "all" } : { kind: "speaker", id: sp.id })}
            onCasting={() => openDrawer({ kind: "casting", speakerId: sp.id })}
            onDragStart={() => setDragging(sp.id)}
            onDragEnd={() => setDragging(null)}
            onDrop={() => onDrop(sp)}
          />
        ))}
      </div>
    </aside>
  );
}

function SpeakerCard(props: {
  speaker: Speaker; active: boolean; dragging: boolean;
  onSelect: () => void; onCasting: () => void;
  onDragStart: () => void; onDragEnd: () => void; onDrop: () => void;
}) {
  const { speaker: sp } = props;
  const patchSpeaker = useProject((s) => s.patchSpeaker);
  const rebuildProfile = useProject((s) => s.rebuildProfile);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(sp.name ?? "");

  const voiceLabel = sp.voice.mode === "clone"
    ? "клон оригинала"
    : sp.voice.preset_name ?? sp.voice.preset_id ?? "не назначен";

  const saveName = () => {
    setEditing(false);
    if ((sp.name ?? "") !== name) void patchSpeaker(sp.id, { name });
  };

  const cycleGender = () => {
    const next = { male: "female", female: "unknown", unknown: "male" }[sp.gender];
    void patchSpeaker(sp.id, { gender: next });
  };

  return (
    <div
      draggable
      onDragStart={props.onDragStart}
      onDragEnd={props.onDragEnd}
      onDragOver={(e) => e.preventDefault()}
      onDrop={props.onDrop}
      onClick={props.onSelect}
      className={`rounded-lg border p-2.5 cursor-pointer transition-all
        ${props.active ? "border-accent bg-ink-700" : "border-line bg-ink-700/40 hover:border-ink-500"}
        ${props.dragging ? "opacity-40" : ""}`}
    >
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full shrink-0"
              style={{ background: sp.color }} />
        {editing ? (
          <input
            autoFocus value={name} onChange={(e) => setName(e.target.value)}
            onBlur={saveName}
            onKeyDown={(e) => { if (e.key === "Enter") saveName(); }}
            onClick={(e) => e.stopPropagation()}
            className="flex-1 bg-ink-900 border border-accent rounded px-1.5 py-0.5
                       text-sm outline-none" />
        ) : (
          <button
            onClick={(e) => { e.stopPropagation(); setEditing(true); }}
            className="flex-1 text-left text-sm font-medium truncate hover:text-accent">
            {sp.id} {sp.name ?? sp.label.replace(/^Спикер /, "· ")}
          </button>
        )}
        <button
          onClick={(e) => { e.stopPropagation(); cycleGender(); }}
          title="Сменить пол"
          className={`w-6 h-6 rounded text-sm shrink-0 hover:bg-ink-600
            ${sp.gender_edited_by_user ? "text-accent" : "text-muted"}`}>
          {GENDER_ICON[sp.gender]}
        </button>
      </div>

      <div className="mt-1.5 flex items-center gap-2 text-xs text-muted">
        <span>{sp.stats.segments_count} реплик</span>
        <span>·</span>
        <span>{fmtTime(sp.stats.total_speech_sec)}</span>
        <span>·</span>
        <span>{sp.stats.share_pct}%</span>
        {sp.role === "narrator" && (
          <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded bg-ink-600">
            основной
          </span>
        )}
      </div>

      <div className="mt-2 flex items-center gap-1.5">
        <span className="text-xs text-muted shrink-0">голос:</span>
        <span className="text-xs truncate flex-1"
              title={voiceLabel}>{voiceLabel}</span>
        {sp.reference.path && (
          <button
            title="Прослушать референс"
            onClick={(e) => {
              e.stopPropagation();
              play(mediaUrl(`/speakers/${sp.id}/reference.wav`));
            }}
            className="w-6 h-6 rounded hover:bg-ink-600 text-accent shrink-0">▶</button>
        )}
      </div>

      <div className="mt-2 flex flex-wrap gap-1">
        <button
          onClick={(e) => { e.stopPropagation(); props.onCasting(); }}
          className="text-xs px-2 py-1 rounded bg-ink-600 hover:bg-ink-500">
          Голос ▾
        </button>
        {sp.reference.best_samples.slice(0, 3).map((_, i) => (
          <button key={i}
            title={`Образец ${i + 1} оригинального голоса`}
            onClick={(e) => {
              e.stopPropagation();
              play(mediaUrl(`/speakers/${sp.id}/samples/${i}.wav`));
            }}
            className="text-xs px-2 py-1 rounded bg-ink-600 hover:bg-ink-500">
            ▶{i + 1}
          </button>
        ))}
        {!sp.voice.locked && (
          <button
            onClick={(e) => { e.stopPropagation(); void rebuildProfile(sp.id); }}
            className="text-xs px-2 py-1 rounded bg-warn/20 text-warn
                       hover:bg-warn/30">
            Пересобрать
          </button>
        )}
      </div>

      {!sp.reference.clone_allowed && sp.voice.mode === "preset" && (
        <p className="mt-2 text-[11px] leading-snug text-warn/90">
          ⚠ чистой речи {sp.reference.clean_sec.toFixed(1)} с — клонировать
          нечего, назначен голос из банка
        </p>
      )}
      {sp.merged_from.length > 0 && (
        <p className="mt-1.5 text-[11px] text-muted">
          собран из {sp.merged_from.length + 1} кластеров
        </p>
      )}
    </div>
  );
}
