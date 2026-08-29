// Сборка экрана студии и горячие клавиши.
//
// Клавиши здесь не украшение: проверить полторы тысячи реплик мышью
// нереально. J/K ведут по списку, цифра назначает спикера текущей реплике,
// пробел играет — руки не уходят с клавиатуры.

import { useEffect, useState } from "react";
import { Drawer } from "./components/Drawer";
import { Player } from "./components/Player";
import { SegmentTable } from "./components/SegmentTable";
import { SpeakerPanel } from "./components/SpeakerPanel";
import { openEvents } from "./lib/api";
import { STAGE_LABELS } from "./lib/types";
import { useProject, visibleSegments } from "./store/useProject";

export default function App() {
  const { project, loading, error } = useProject();
  const load = useProject((s) => s.load);
  const applyEvent = useProject((s) => s.applyEvent);
  const [wsAlive, setWsAlive] = useState(true);

  useEffect(() => { void load(); }, [load]);

  // живая связь: прогресс рендера и готовность превью. Обрыв не смертелен —
  // всё состояние всё равно перечитывается из проекта
  useEffect(() => {
    let socket: WebSocket | null = null;
    let timer: number | undefined;
    const connect = () => {
      socket = openEvents(applyEvent);
      socket.onopen = () => setWsAlive(true);
      socket.onclose = () => {
        setWsAlive(false);
        timer = window.setTimeout(connect, 4000);
      };
    };
    connect();
    return () => { socket?.close(); if (timer) clearTimeout(timer); };
  }, [applyEvent]);

  useHotkeys();

  if (loading) return <Splash>Загружаю проект…</Splash>;
  if (error) return <Splash error>{error}</Splash>;
  if (!project) return <Splash error>Проект недоступен</Splash>;

  return (
    <div className="h-screen flex flex-col bg-ink-900 text-white">
      <TopBar wsAlive={wsAlive} />
      <div className="flex-1 min-h-0 flex">
        <SpeakerPanel />
        <main className="flex-1 min-w-0 flex flex-col">
          <Player />
          <SegmentTable />
        </main>
        <Drawer />
      </div>
      <Notice />
    </div>
  );
}

function TopBar({ wsAlive }: { wsAlive: boolean }) {
  const project = useProject((s) => s.project)!;
  const saving = useProject((s) => s.saving);
  const approve = useProject((s) => s.approve);
  const openDrawer = useProject((s) => s.openDrawer);
  const [confirming, setConfirming] = useState(false);

  const canApprove = ["review", "translated", "profiled"].includes(project.stage);
  const speakers = project.speakers.filter((s) => !s.merged_into);

  return (
    <>
      <header className="flex items-center gap-3 px-4 h-12 border-b border-line
                         bg-ink-800 shrink-0">
        <h1 className="text-sm font-semibold truncate max-w-[24rem]"
            title={project.source.title || project.source.file_name}>
          {project.source.title || project.source.file_name || "Проект"}
        </h1>
        <span className={`text-[11px] px-2 py-0.5 rounded-full border
          ${project.stage === "review" ? "border-warn text-warn"
            : project.stage === "done" ? "border-ok text-ok"
            : "border-line text-muted"}`}>
          {STAGE_LABELS[project.stage]}
        </span>
        <span className="text-xs text-muted">
          {project.lang_src || "?"} → {project.lang_tgt}
        </span>
        <span className="text-xs text-muted">· {speakers.length} спикеров</span>

        <RangePreviewButton />

        <button onClick={() => openDrawer({ kind: "qc" })}
                className={`text-xs px-2 py-1 rounded ${
                  project.warnings_count ? "text-warn hover:bg-ink-600"
                                         : "text-muted hover:bg-ink-600"}`}>
          ⚠ {project.warnings_count} замечаний
        </button>

        <span className="ml-auto flex items-center gap-2 text-xs">
          {!wsAlive && <span className="text-warn">связь потеряна…</span>}
          <span className="text-muted">
            {saving ? "сохраняю…" : `сохранено · v${project.version}`}
          </span>
          <button
            disabled={!canApprove}
            onClick={() => setConfirming(true)}
            className={`px-3 py-1.5 rounded font-medium ${
              canApprove ? "bg-accent hover:brightness-110"
                         : "bg-ink-600 text-muted cursor-default"}`}>
            Утвердить и рендерить
          </button>
        </span>
      </header>

      {confirming && (
        <ApproveDialog onClose={() => setConfirming(false)}
                       onConfirm={() => { setConfirming(false); void approve(); }} />
      )}
      <Progress />
    </>
  );
}

function RangePreviewButton() {
  const preview = useProject((s) => s.rangePreview);
  const previewRange = useProject((s) => s.previewRange);
  const playhead = useProject((s) => s.playhead);

  // Главный вопрос человека на этом экране — «правильно ли он озвучивает».
  // Отдельные реплики отвечают на него плохо: нужно услышать дубляж
  // поверх картинки. До полного рендера это единственный способ.
  const play = () => {
    const video = (window as unknown as { __player?: HTMLVideoElement }).__player;
    const audio = document.getElementById("range-audio") as HTMLAudioElement | null;
    if (!audio || !preview.url) return;
    audio.currentTime = 0;
    if (video) {
      video.pause();
      video.currentTime = preview.start;
      video.muted = true;          // оригинал заглушается: слушаем дубляж
      void video.play().catch(() => undefined);
    }
    void audio.play().catch(() => undefined);
  };

  const stop = () => {
    const video = (window as unknown as { __player?: HTMLVideoElement }).__player;
    const audio = document.getElementById("range-audio") as HTMLAudioElement | null;
    audio?.pause();
    if (video) { video.pause(); video.muted = false; }
  };

  if (preview.status === "working") {
    return (
      <span className="text-xs text-warn px-2 py-1">
        озвучиваю отрывок… ≈1 мин
      </span>
    );
  }

  if (preview.status === "ready" && preview.url) {
    return (
      <span className="flex items-center gap-1.5">
        <audio id="range-audio" src={preview.url} onEnded={stop} />
        <button onClick={play}
                className="text-xs px-2.5 py-1 rounded bg-ok/20 text-ok
                           hover:bg-ok/30 font-medium">
          ▶ Дубляж {fmtClock(preview.start)}–{fmtClock(preview.end)}
        </button>
        <button onClick={stop} title="Остановить и вернуть оригинал"
                className="text-xs px-2 py-1 rounded hover:bg-ink-600 text-muted">
          ■
        </button>
        <button onClick={() => void previewRange(Math.floor(playhead))}
                title="Озвучить отрывок с текущего места"
                className="text-xs px-2 py-1 rounded hover:bg-ink-600 text-muted">
          ↻
        </button>
      </span>
    );
  }

  return (
    <button
      onClick={() => void previewRange(Math.floor(playhead))}
      title="Синтезировать минуту дубляжа с текущего места и послушать её поверх видео"
      className="text-xs px-2.5 py-1 rounded bg-ink-600 hover:bg-ink-500">
      🔊 Послушать озвучку
    </button>
  );
}

function fmtClock(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function ApproveDialog({ onClose, onConfirm }: {
  onClose: () => void; onConfirm: () => void;
}) {
  const project = useProject((s) => s.project)!;
  const speakers = project.speakers.filter((s) => !s.merged_into);

  return (
    <div className="fixed inset-0 z-50 bg-black/60 grid place-items-center p-6"
         onClick={onClose}>
      <div className="bg-ink-800 border border-line rounded-xl w-full max-w-lg
                      p-5" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-base font-semibold">Утвердить и запустить рендер</h2>
        <p className="text-xs text-muted mt-1">
          После утверждения задача уйдёт в озвучку. Голоса менять будет
          нельзя до окончания рендера.
        </p>

        <div className="mt-4 max-h-64 overflow-y-auto space-y-1 text-sm">
          {speakers.map((sp) => (
            <div key={sp.id} className="flex items-center gap-2 py-1">
              <span className="w-2 h-2 rounded-full" style={{ background: sp.color }} />
              <span className="w-16">{sp.id}</span>
              <span className="flex-1 truncate text-muted">
                {sp.voice.mode === "clone" ? "клон оригинала"
                  : sp.voice.preset_name ?? sp.voice.preset_id ?? "— не назначен —"}
              </span>
              <span className="text-xs text-muted">{sp.stats.segments_count}</span>
            </div>
          ))}
        </div>

        {project.warnings_count > 0 && (
          <p className="mt-3 text-xs text-warn">
            Осталось {project.warnings_count} непроверенных замечаний.
            Утвердить можно и так — они попадут в отчёт.
          </p>
        )}

        <div className="mt-5 flex gap-2 justify-end">
          <button onClick={onClose}
                  className="px-3 py-1.5 rounded bg-ink-600 hover:bg-ink-500 text-sm">
            Отмена
          </button>
          <button onClick={onConfirm}
                  className="px-3 py-1.5 rounded bg-accent hover:brightness-110
                             text-sm font-medium">
            Утвердить
          </button>
        </div>
      </div>
    </div>
  );
}

function Progress() {
  const progress = useProject((s) => s.progress);
  if (!progress) return null;
  return (
    <div className="px-4 py-1.5 bg-ink-700 border-b border-line flex items-center gap-3">
      <span className="text-xs">{progress.stage}/10 · {progress.label}</span>
      <span className="flex-1 h-1 rounded bg-ink-900 overflow-hidden">
        <span className="block h-full bg-accent transition-all"
              style={{ width: `${progress.pct}%` }} />
      </span>
      <span className="text-xs tabular-nums text-muted">{progress.pct}%</span>
    </div>
  );
}

function Notice() {
  const notice = useProject((s) => s.notice);
  const setNotice = useProject((s) => s.setNotice);
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 6000);
    return () => clearTimeout(t);
  }, [notice, setNotice]);
  if (!notice) return null;
  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 px-4 py-2
                    rounded-lg bg-ink-700 border border-line shadow-xl text-sm">
      {notice}
      <button onClick={() => setNotice(null)}
              className="ml-3 text-muted hover:text-white">✕</button>
    </div>
  );
}

function Splash({ children, error }: { children: React.ReactNode; error?: boolean }) {
  return (
    <div className="h-screen grid place-items-center bg-ink-900 text-white p-6">
      <div className="max-w-md text-center">
        <p className={error ? "text-danger" : "text-muted"}>{children}</p>
        {error && (
          <p className="mt-3 text-xs text-muted leading-relaxed">
            Если ссылка устарела — пришлите боту команду <code>/review</code>,
            он выдаст новую.
          </p>
        )}
      </div>
    </div>
  );
}

function useHotkeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;

      const st = useProject.getState();
      const list = visibleSegments(st);
      const i = list.findIndex((s) => s.id === st.currentSegment);
      const player = (window as unknown as { __player?: HTMLVideoElement }).__player;

      if (e.code === "Space") {
        e.preventDefault();
        if (player) player.paused ? void player.play() : player.pause();
      } else if (e.key === "j" || e.key === "о") {
        e.preventDefault();
        if (i < list.length - 1) st.setCurrent(list[i + 1].id);
      } else if (e.key === "k" || e.key === "л") {
        e.preventDefault();
        if (i > 0) st.setCurrent(list[i - 1].id);
      } else if (/^[1-9]$/.test(e.key)) {
        const speakers = st.project?.speakers.filter((s) => !s.merged_into) ?? [];
        const sp = speakers[Number(e.key) - 1];
        if (sp && st.currentSegment != null) {
          e.preventDefault();
          void st.assignSpeaker([st.currentSegment], sp.id);
        }
      } else if (e.key === "s" || e.key === "ы") {
        if (st.currentSegment != null && player) {
          e.preventDefault();
          void st.splitSegment(st.currentSegment, player.currentTime);
        }
      } else if (e.key === "m" || e.key === "ь") {
        if (st.currentSegment != null) {
          e.preventDefault();
          void st.mergeWithNext(st.currentSegment);
        }
      } else if (e.key === "z" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        void st.undo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
