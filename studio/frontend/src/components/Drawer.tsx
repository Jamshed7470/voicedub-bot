// Правая панель: кастинг голоса и список замечаний.
//
// Кастинг показывает не только назначенный голос, но и то, из чего система
// выбирала: человеку нужно понимать, был ли выбор очевидным.

import { mediaUrl } from "../lib/api";
import { useProject } from "../store/useProject";
import { FLAG_LABELS } from "../lib/types";

export function Drawer() {
  const drawer = useProject((s) => s.drawer);
  const openDrawer = useProject((s) => s.openDrawer);
  if (!drawer) return null;

  return (
    <aside className="w-[340px] shrink-0 border-l border-line bg-ink-800
                      overflow-y-auto">
      <div className="sticky top-0 bg-ink-800/95 backdrop-blur px-3 py-2
                      border-b border-line flex items-center justify-between">
        <h2 className="text-sm font-semibold">
          {drawer.kind === "casting" ? "КАСТИНГ" : "ЗАМЕЧАНИЯ"}
        </h2>
        <button onClick={() => openDrawer(null)}
                className="w-6 h-6 rounded hover:bg-ink-600 text-muted">✕</button>
      </div>
      {drawer.kind === "casting"
        ? <Casting speakerId={drawer.speakerId} />
        : <QCList />}
    </aside>
  );
}

function Casting({ speakerId }: { speakerId: string }) {
  const project = useProject((s) => s.project);
  const voices = useProject((s) => s.voices);
  const patchSpeaker = useProject((s) => s.patchSpeaker);
  const sp = project?.speakers.find((s) => s.id === speakerId);
  if (!sp) return null;

  const candidates = sp.voice.casting_candidates ?? [];
  const rest = voices.filter(
    (v) => !candidates.some((c) => c.preset_id === v.id) &&
           (sp.gender === "unknown" || v.gender === sp.gender));

  return (
    <div className="p-3 space-y-4">
      <div>
        <p className="text-xs text-muted">Спикер</p>
        <p className="text-sm font-medium">{sp.id} {sp.name ?? sp.label}</p>
        <p className="text-xs text-muted mt-1">
          чистой речи {sp.reference.clean_sec.toFixed(1)} с ·
          SNR {sp.reference.snr_db.toFixed(0)} дБ
        </p>
      </div>

      <div>
        <p className="text-xs text-muted mb-1.5">Свой голос</p>
        <button
          disabled={!sp.reference.clone_allowed && sp.voice.mode !== "clone"}
          onClick={() => void patchSpeaker(sp.id, { voice: { mode: "clone" } })}
          className={`w-full text-left px-2.5 py-2 rounded border text-sm
            ${sp.voice.mode === "clone"
              ? "border-accent bg-accent/10" : "border-line hover:border-ink-500"}
            ${!sp.reference.clone_allowed ? "opacity-60" : ""}`}>
          Клон оригинала
          {!sp.reference.clone_allowed && (
            <span className="block text-[11px] text-warn mt-0.5">
              речи меньше 8 с — звучание будет нестабильным
            </span>
          )}
        </button>
      </div>

      {candidates.length > 0 && (
        <div>
          <p className="text-xs text-muted mb-1.5">
            Лучшие по тембру ({candidates.length})
          </p>
          <div className="space-y-1.5">
            {candidates.map((c) => (
              <VoiceRow key={c.preset_id} id={c.preset_id} name={c.display_name}
                        score={c.score} speakerId={sp.id}
                        active={sp.voice.preset_id === c.preset_id} />
            ))}
          </div>
        </div>
      )}

      {rest.length > 0 && (
        <div>
          <p className="text-xs text-muted mb-1.5">Весь банк ({rest.length})</p>
          <div className="space-y-1 max-h-[40vh] overflow-y-auto pr-1">
            {rest.map((v) => (
              <VoiceRow key={v.id} id={v.id} name={v.display_name}
                        speakerId={sp.id} active={sp.voice.preset_id === v.id} />
            ))}
          </div>
        </div>
      )}

      {voices.length === 0 && (
        <p className="text-xs text-muted leading-relaxed">
          Банк голосов пуст. Соберите его командой:
          <code className="block mt-1 p-1.5 bg-ink-900 rounded font-mono text-[11px]">
            python -m scripts.build_voice_bank --from-xtts
          </code>
        </p>
      )}
    </div>
  );
}

function VoiceRow({ id, name, score, speakerId, active }: {
  id: string; name: string; score?: number; speakerId: string; active: boolean;
}) {
  const patchSpeaker = useProject((s) => s.patchSpeaker);
  return (
    <div className={`flex items-center gap-1.5 px-2 py-1.5 rounded border text-sm
      ${active ? "border-accent bg-accent/10" : "border-line"}`}>
      <button
        onClick={() => {
          const a = new Audio(`/api/voices/${id}/sample.wav`);
          void a.play().catch(() => undefined);
        }}
        className="w-6 h-6 rounded hover:bg-ink-600 text-accent shrink-0">▶</button>
      <span className="flex-1 truncate" title={name}>{name}</span>
      {score !== undefined && (
        <span className="text-[10px] text-muted tabular-nums">
          {score.toFixed(2)}
        </span>
      )}
      {!active && (
        <button
          onClick={() => void patchSpeaker(speakerId,
            { voice: { mode: "preset", preset_id: id } })}
          className="text-[11px] px-1.5 py-0.5 rounded bg-ink-600 hover:bg-ink-500">
          назначить
        </button>
      )}
    </div>
  );
}

function QCList() {
  const project = useProject((s) => s.project);
  const setFilter = useProject((s) => s.setFilter);
  const setCurrent = useProject((s) => s.setCurrent);
  if (!project) return null;

  const groups: { flag: string; ids: number[] }[] = [];
  for (const flag of Object.keys(FLAG_LABELS)) {
    const ids = project.segments.filter((s) => s.flags.includes(flag))
                                .map((s) => s.id);
    if (ids.length) groups.push({ flag, ids });
  }

  const noRef = project.speakers.filter(
    (s) => !s.merged_into && !s.reference.clone_allowed);

  return (
    <div className="p-3 space-y-3 text-sm">
      {groups.length === 0 && noRef.length === 0 && (
        <p className="text-muted">Замечаний нет — можно утверждать.</p>
      )}

      {noRef.map((sp) => (
        <div key={sp.id} className="p-2 rounded border border-warn/40 bg-warn/5">
          <p className="text-warn text-xs">
            {sp.id}: чистой речи {sp.reference.clean_sec.toFixed(1)} с →
            назначен голос из банка
          </p>
        </div>
      ))}

      {groups.map((g) => (
        <button key={g.flag}
          onClick={() => {
            setFilter({ kind: "flag", flag: g.flag });
            if (g.ids[0]) setCurrent(g.ids[0]);
          }}
          className="w-full text-left p-2 rounded border border-line
                     hover:border-accent">
          <span className="text-xs">
            {FLAG_LABELS[g.flag]}: <b>{g.ids.length}</b> реплик
          </span>
        </button>
      ))}

      {project.qc.overall_identity > 0 && (
        <div className="mt-4 p-2.5 rounded bg-ink-700">
          <p className="text-xs text-muted">Стабильность голосов</p>
          <p className="text-2xl tabular-nums">
            {project.qc.overall_identity.toFixed(2)}
          </p>
          <p className="text-[11px] text-muted leading-snug mt-1">
            средняя схожесть реплик одного спикера между собой;
            1.00 — голос не меняется вовсе
          </p>
        </div>
      )}
    </div>
  );
}
