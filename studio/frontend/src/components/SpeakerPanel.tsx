// Левая колонка: карточки спикеров.
//
// Карточка отвечает на три вопроса разом: кто это, сколько говорит и каким
// голосом будет озвучен. Всё остальное (кастинг, объединение) — действия,
// которые человек делает, уже поняв ответы.

import { useEffect, useState } from "react";
import { mediaUrl } from "../lib/api";
import { nowPlaying, onPlaybackChange, play } from "../lib/player";
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

/** Подсветка кнопки, которая сейчас звучит. */
function usePlayingKey(): string | null {
  const [key, setKey] = useState<string | null>(nowPlaying());
  useEffect(() => onPlaybackChange(setKey), []);
  return key;
}

export function SpeakerPanel() {
  const project = useProject((s) => s.project);
  const filter = useProject((s) => s.filter);
  const setFilter = useProject((s) => s.setFilter);
  const openDrawer = useProject((s) => s.openDrawer);
  const mergeSpeakers = useProject((s) => s.mergeSpeakers);
  const createSpeaker = useProject((s) => s.createSpeaker);
  const deleteSpeaker = useProject((s) => s.deleteSpeaker);
  const [dragging, setDragging] = useState<string | null>(null);
  const [removing, setRemoving] = useState<Speaker | null>(null);

  if (!project) return null;
  const speakers = project.speakers.filter((s) => !s.merged_into);

  const onDrop = (target: Speaker) => {
    if (!dragging || dragging === target.id) return;
    const source = speakers.find((s) => s.id === dragging);
    if (!source) return;
    const n = source.stats.segments_count;
    const ok = window.confirm(
      `Объединить ${source.id} в ${target.id}?\n\n` +
      `${n} реплик будут переназначены на ${target.name ?? target.id}. ` +
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
        <span className="flex items-center gap-1">
          <button
            onClick={() => setFilter({ kind: "all" })}
            className={`text-xs px-2 py-1 rounded transition-colors ${
              filter.kind === "all" ? "bg-accent text-white" : "text-muted hover:text-white"
            }`}>
            все
          </button>
          <button
            onClick={() => void createSpeaker()}
            title="Добавить спикера: система могла свести двух людей в одного"
            className="text-xs px-2 py-1 rounded bg-ink-600 hover:bg-ink-500">
            + спикер
          </button>
        </span>
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
            onDelete={() => setRemoving(sp)}
          />
        ))}
      </div>

      {removing && (
        <DeleteDialog
          speaker={removing}
          others={speakers.filter((s) => s.id !== removing.id)}
          onClose={() => setRemoving(null)}
          onConfirm={(moveTo) => {
            void deleteSpeaker(removing.id, moveTo);
            setRemoving(null);
          }}
        />
      )}
    </aside>
  );
}

function DeleteDialog({ speaker, others, onClose, onConfirm }: {
  speaker: Speaker; others: Speaker[];
  onClose: () => void; onConfirm: (moveTo: string | null) => void;
}) {
  const [target, setTarget] = useState(others[0]?.id ?? "");
  const count = speaker.stats.segments_count;

  return (
    <div className="fixed inset-0 z-50 bg-black/60 grid place-items-center p-6"
         onClick={onClose}>
      <div className="bg-ink-800 border border-line rounded-xl w-full max-w-md p-5"
           onClick={(e) => e.stopPropagation()}>
        <h2 className="text-base font-semibold">
          Удалить {speaker.id} {speaker.name ?? ""}?
        </h2>

        {count > 0 ? (
          <>
            <p className="mt-2 text-xs text-muted leading-relaxed">
              У него {count} реплик. Без спикера они останутся без голоса, и
              озвучка остановится — выберите, кому их передать.
            </p>
            <select value={target} onChange={(e) => setTarget(e.target.value)}
                    className="mt-3 w-full bg-ink-900 border border-line rounded
                               px-2 py-1.5 text-sm">
              {others.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id} {s.name ?? s.label} — {s.stats.segments_count} реплик
                </option>
              ))}
            </select>
          </>
        ) : (
          <p className="mt-2 text-xs text-muted">
            Реплик у него нет — удаление ничего не затронет.
          </p>
        )}

        <div className="mt-5 flex gap-2 justify-end">
          <button onClick={onClose}
                  className="px-3 py-1.5 rounded bg-ink-600 hover:bg-ink-500 text-sm">
            Отмена
          </button>
          <button
            disabled={count > 0 && !target}
            onClick={() => onConfirm(count > 0 ? target : null)}
            className="px-3 py-1.5 rounded bg-danger hover:brightness-110
                       text-sm font-medium disabled:opacity-40">
            Удалить
          </button>
        </div>
      </div>
    </div>
  );
}

function SpeakerCard(props: {
  speaker: Speaker; active: boolean; dragging: boolean;
  onSelect: () => void; onCasting: () => void; onDelete: () => void;
  onDragStart: () => void; onDragEnd: () => void; onDrop: () => void;
}) {
  const { speaker: sp } = props;
  const playing = usePlayingKey();
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
        <button
          onClick={(e) => { e.stopPropagation(); props.onDelete(); }}
          title="Удалить спикера"
          className="w-6 h-6 rounded text-sm shrink-0 text-muted
                     hover:bg-danger/20 hover:text-danger">
          ✕
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

      {/* Выбор голоса — главное действие на карточке, поэтому это широкий
          селектор с явной подписью, а не мелкая кнопка в ряду прочих:
          раньше «Голос ▾» стояла рядом с ▶1 ▶2 ▶3 и читалась как ещё одна
          кнопка проигрывания. */}
      <div className="mt-2">
        <span className="text-[11px] text-muted">Голос для озвучки</span>
        <div className="mt-1 flex items-stretch gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); props.onCasting(); }}
            title="Выбрать голос: клон оригинала или голос из банка"
            className="flex-1 min-w-0 flex items-center gap-1.5 px-2 py-1.5
                       rounded border border-ink-500 bg-ink-800
                       hover:border-accent hover:bg-ink-700 transition-colors">
            <span className="flex-1 text-left text-xs truncate"
                  title={voiceLabel}>{voiceLabel}</span>
            <span className="text-muted text-[10px] shrink-0">выбрать ▾</span>
          </button>
          {sp.reference.path && (
            <button
              title="Прослушать голос, которым будет озвучен спикер"
              onClick={(e) => {
                e.stopPropagation();
                play(mediaUrl(`/speakers/${sp.id}/reference.wav`), `ref:${sp.id}`);
              }}
              className={`w-8 rounded border shrink-0 ${
                playing === `ref:${sp.id}`
                  ? "border-accent bg-accent/20 text-accent"
                  : "border-ink-500 bg-ink-800 hover:border-accent text-accent"}`}>
              {playing === `ref:${sp.id}` ? "■" : "▶"}</button>
          )}
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-1">
        <span className="text-[11px] text-muted mr-0.5">оригинал:</span>
        {sp.reference.best_samples.slice(0, 3).map((_, i) => (
          <button key={i}
            title={`Как этот человек звучит в оригинале — образец ${i + 1}`}
            onClick={(e) => {
              e.stopPropagation();
              play(mediaUrl(`/speakers/${sp.id}/samples/${i}.wav`),
                   `sample:${sp.id}:${i}`);
            }}
            className={`text-xs px-2 py-1 rounded ${
              playing === `sample:${sp.id}:${i}`
                ? "bg-accent text-white" : "bg-ink-600 hover:bg-ink-500"}`}>
            {playing === `sample:${sp.id}:${i}` ? "■" : "▶"}{i + 1}
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
