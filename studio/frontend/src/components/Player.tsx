// Видеоплеер и таймлайн.
//
// Проверить «тот ли это человек» можно только глядя на видео — поэтому
// текущая реплика подсвечена рамкой цвета своего спикера, а под видео
// идут дорожки: по одной на спикера, чтобы разговор был виден целиком.

import { useEffect, useRef, useState } from "react";
import { mediaUrl } from "../lib/api";
import { stopAll } from "../lib/player";
import { useProject, visibleSegments } from "../store/useProject";
import type { Segment, Speaker } from "../lib/types";

export function Player() {
  const project = useProject((s) => s.project);
  const current = useProject((s) => s.currentSegment);
  const setCurrent = useProject((s) => s.setCurrent);
  const setPlayhead = useProject((s) => s.setPlayhead);
  const video = useRef<HTMLVideoElement>(null);
  const [loop, setLoop] = useState(false);
  const [time, setTime] = useState(0);

  const seg = project?.segments.find((s) => s.id === current) ?? null;
  const speaker = project?.speakers.find((s) => s.id === seg?.speaker_id);

  // переход к реплике: клик по строке или по региону таймлайна
  useEffect(() => {
    if (!seg || !video.current) return;
    if (Math.abs(video.current.currentTime - seg.start) > 0.35) {
      video.current.currentTime = seg.start;
    }
  }, [current]);

  // зацикливание одной реплики — так удобнее сравнивать с синтезом
  useEffect(() => {
    const el = video.current;
    if (!el) return;
    const onTime = () => {
      setTime(el.currentTime);
      setPlayhead(el.currentTime);
      if (loop && seg && el.currentTime >= seg.end) el.currentTime = seg.start;
    };
    const onPlay = () => stopAll();   // видео и образец голоса разом — каша
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("play", onPlay);
    return () => {
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("play", onPlay);
    };
  }, [loop, seg, setPlayhead]);

  useEffect(() => {
    const el = video.current;
    if (!el) return;
    (window as unknown as { __player: HTMLVideoElement }).__player = el;
  }, []);

  if (!project) return null;

  return (
    <div className="flex flex-col border-b border-line">
      <div className="flex gap-3 p-3">
        <div className="relative shrink-0 rounded-lg overflow-hidden"
             style={{ boxShadow: speaker ? `0 0 0 2px ${speaker.color}` : undefined }}>
          {project.media_available ? (
            <video ref={video} controls preload="metadata"
                   className="w-[420px] max-h-[240px] bg-black"
                   src={mediaUrl("/media/video")} />
          ) : (
            <div className="w-[420px] h-[236px] bg-ink-900 grid place-items-center
                            text-center text-sm text-muted px-6">
              Медиафайлы этой задачи удалены (срок хранения 7 дней).<br />
              Правки и отчёт доступны, воспроизведение — нет.
            </div>
          )}
        </div>

        <div className="flex-1 min-w-0 flex flex-col">
          <div className="text-xs text-muted">
            {fmtTime(time)} / {fmtTime(project.source.duration_sec)}
          </div>
          {seg && (
            <>
              <p className="mt-2 text-sm text-muted leading-snug line-clamp-2">
                {seg.text_src || "— оригинал не сохранён —"}
              </p>
              <p className="mt-1.5 text-base leading-snug">{seg.text_tgt}</p>
              <div className="mt-auto flex items-center gap-2 pt-2">
                <button
                  onClick={() => setLoop((v) => !v)}
                  className={`text-xs px-2 py-1 rounded ${
                    loop ? "bg-accent text-white" : "bg-ink-600 hover:bg-ink-500"}`}>
                  зациклить реплику
                </button>
                <span className="text-xs text-muted">
                  {speaker?.id} · {speaker?.voice.mode === "clone"
                    ? "клон" : speaker?.voice.preset_name ?? "пресет"}
                </span>
              </div>
            </>
          )}
        </div>
      </div>

      <Timeline onSeek={(t) => { if (video.current) video.current.currentTime = t; }} />
    </div>
  );
}

function Timeline({ onSeek }: { onSeek: (t: number) => void }) {
  const project = useProject((s) => s.project);
  const setCurrent = useProject((s) => s.setCurrent);
  const current = useProject((s) => s.currentSegment);
  const playhead = useProject((s) => s.playhead);
  const [zoom, setZoom] = useState(1);
  const box = useRef<HTMLDivElement>(null);

  if (!project) return null;
  const total = Math.max(1, project.source.duration_sec);
  const speakers = project.speakers.filter((s) => !s.merged_into).slice(0, 12);
  const width = 100 * zoom;

  return (
    <div className="border-t border-line bg-ink-900/60">
      <div className="flex items-center gap-2 px-3 py-1.5 text-xs text-muted">
        <span>ТАЙМЛАЙН</span>
        <button onClick={() => setZoom((z) => Math.max(1, z / 1.6))}
                className="px-1.5 rounded hover:bg-ink-600">−</button>
        <span>{zoom.toFixed(1)}×</span>
        <button onClick={() => setZoom((z) => Math.min(24, z * 1.6))}
                className="px-1.5 rounded hover:bg-ink-600">+</button>
        <span className="ml-auto">клик по реплике — перемотка</span>
      </div>

      <div ref={box} className="overflow-x-auto pb-1"
           onWheel={(e) => {
             if (!e.ctrlKey) return;
             e.preventDefault();
             setZoom((z) => Math.min(24, Math.max(1, z * (e.deltaY < 0 ? 1.2 : 0.83))));
           }}>
        <div className="relative" style={{ width: `${width}%`, minWidth: "100%" }}>
          {speakers.map((sp) => (
            <div key={sp.id} className="relative h-6 border-b border-ink-700/60">
              <span className="absolute left-1 top-0.5 text-[10px] text-muted
                               pointer-events-none z-10 bg-ink-900/80 px-1 rounded">
                {sp.id}
              </span>
              {project.segments
                .filter((s) => s.speaker_id === sp.id)
                .map((s) => (
                  <Region key={s.id} seg={s} speaker={sp} total={total}
                          active={s.id === current}
                          onClick={() => { setCurrent(s.id); onSeek(s.start); }} />
                ))}
            </div>
          ))}
          <div className="absolute top-0 bottom-0 w-px bg-accent pointer-events-none"
               style={{ left: `${(playhead / total) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}

function Region({ seg, speaker, total, active, onClick }: {
  seg: Segment; speaker: Speaker; total: number; active: boolean;
  onClick: () => void;
}) {
  const left = (seg.start / total) * 100;
  const width = Math.max(0.08, ((seg.end - seg.start) / total) * 100);
  const flagged = seg.flags.length > 0;

  return (
    <button
      onClick={onClick}
      title={`#${seg.id} ${seg.text_tgt.slice(0, 60)}`}
      className={`absolute top-0.5 bottom-0.5 rounded-sm transition-all
                  ${active ? "ring-2 ring-white z-10" : "hover:brightness-125"}`}
      style={{
        left: `${left}%`, width: `${width}%`,
        background: speaker.color,
        opacity: seg.overlap ? 0.55 : 0.85,
        backgroundImage: seg.overlap
          ? "repeating-linear-gradient(45deg,transparent,transparent 3px,rgba(0,0,0,.35) 3px,rgba(0,0,0,.35) 6px)"
          : undefined,
      }}
    >
      {flagged && (
        <span className="absolute -top-1 left-0 text-[9px] text-warn leading-none">▲</span>
      )}
    </button>
  );
}

function fmtTime(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}
