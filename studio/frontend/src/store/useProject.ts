// Состояние студии.
//
// Правки идут «оптимистично»: интерфейс меняется сразу, запрос уходит
// следом. Если сервер ответил 409 (проект изменили в другом окне),
// проект перечитывается целиком — гадать, какие правки чьи, бессмысленно,
// а тихо затирать чужую работу нельзя.

import { create } from "zustand";
import { ApiError, api } from "../lib/api";
import type { BankVoice, Project, Segment, Speaker } from "../lib/types";

type Filter =
  | { kind: "all" }
  | { kind: "speaker"; id: string }
  | { kind: "flag"; flag: string }
  | { kind: "edited" };

interface State {
  project: Project | null;
  voices: BankVoice[];
  loading: boolean;
  error: string | null;
  notice: string | null;
  saving: boolean;
  currentSegment: number | null;
  selected: Set<number>;
  filter: Filter;
  drawer: null | { kind: "casting"; speakerId: string } | { kind: "qc" };
  playhead: number;
  previewing: Set<number>;
  progress: { stage: number; label: string; pct: number } | null;

  load: () => Promise<void>;
  refresh: () => Promise<void>;
  setCurrent: (id: number | null) => void;
  setPlayhead: (t: number) => void;
  setFilter: (f: Filter) => void;
  toggleSelected: (id: number, additive: boolean) => void;
  clearSelection: () => void;
  openDrawer: (d: State["drawer"]) => void;
  setNotice: (text: string | null) => void;

  patchSegment: (id: number, patch: Record<string, unknown>) => Promise<void>;
  assignSpeaker: (segmentIds: number[], speakerId: string) => Promise<void>;
  patchSpeaker: (id: string, patch: Record<string, unknown>) => Promise<void>;
  mergeSpeakers: (from: string, into: string) => Promise<void>;
  rebuildProfile: (id: string) => Promise<void>;
  splitSegment: (id: number, at: number) => Promise<void>;
  mergeWithNext: (id: number) => Promise<void>;
  preview: (id: number) => Promise<void>;
  approve: () => Promise<void>;
  undo: () => Promise<void>;
  applyEvent: (e: Record<string, unknown>) => void;
}

export const useProject = create<State>((set, get) => ({
  project: null,
  voices: [],
  loading: true,
  error: null,
  notice: null,
  saving: false,
  currentSegment: null,
  selected: new Set(),
  filter: { kind: "all" },
  drawer: null,
  playhead: 0,
  previewing: new Set(),
  progress: null,

  load: async () => {
    set({ loading: true, error: null });
    try {
      const [project, voices] = await Promise.all([
        api.project(),
        api.voices().catch(() => ({ voices: [] })),
      ]);
      set({
        project, voices: voices.voices, loading: false,
        currentSegment: project.segments[0]?.id ?? null,
      });
    } catch (e) {
      set({ loading: false, error: (e as Error).message });
    }
  },

  refresh: async () => {
    try {
      set({ project: await api.project() });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  setCurrent: (id) => set({ currentSegment: id }),
  setPlayhead: (t) => set({ playhead: t }),
  setFilter: (filter) => set({ filter, selected: new Set() }),
  openDrawer: (drawer) => set({ drawer }),
  setNotice: (notice) => set({ notice }),

  toggleSelected: (id, additive) => {
    const selected = new Set(additive ? get().selected : []);
    selected.has(id) ? selected.delete(id) : selected.add(id);
    set({ selected });
  },
  clearSelection: () => set({ selected: new Set() }),

  // -------------------------------------------------------------- правки

  patchSegment: async (id, patch) => {
    const project = get().project;
    if (!project) return;
    const previous = project.segments.find((s) => s.id === id);
    if (!previous) return;

    // показываем правку сразу: ожидание ответа на каждое нажатие делает
    // работу с таблицей на полторы тысячи строк невыносимой
    set({
      saving: true,
      project: {
        ...project,
        segments: project.segments.map((s) =>
          s.id === id ? { ...s, ...patch } as Segment : s),
      },
    });
    try {
      const updated = await api.patchSegment(id, patch as never, project.version);
      set((st) => ({
        saving: false,
        project: st.project && {
          ...st.project,
          version: st.project.version + 1,
          segments: st.project.segments.map((s) => (s.id === id ? updated : s)),
        },
      }));
      await get().refresh();
    } catch (e) {
      await handleError(e, set, get);
    }
  },

  assignSpeaker: async (segmentIds, speakerId) => {
    const project = get().project;
    if (!project) return;
    set({ saving: true });
    try {
      if (segmentIds.length === 1) {
        await api.patchSegment(segmentIds[0], { speaker_id: speakerId },
                               project.version);
      } else {
        await api.bulk(segmentIds, { speaker_id: speakerId }, project.version);
      }
      await get().refresh();
      set({ saving: false, selected: new Set() });
    } catch (e) {
      await handleError(e, set, get);
    }
  },

  patchSpeaker: async (id, patch) => {
    const project = get().project;
    if (!project) return;
    set({ saving: true });
    try {
      await api.patchSpeaker(id, patch, project.version);
      await get().refresh();
      set({ saving: false });
    } catch (e) {
      await handleError(e, set, get);
    }
  },

  mergeSpeakers: async (from, into) => {
    const project = get().project;
    if (!project) return;
    set({ saving: true });
    try {
      const res = await api.mergeSpeakers(from, into, project.version);
      await get().refresh();
      set({
        saving: false,
        notice: `Переназначено реплик: ${res.moved_segments}. ` +
          "Профиль голоса нужно пересобрать.",
      });
    } catch (e) {
      await handleError(e, set, get);
    }
  },

  rebuildProfile: async (id) => {
    const project = get().project;
    if (!project) return;
    try {
      await api.rebuildProfile(id, project.version);
      set({ notice: `Профиль ${id} пересобирается…` });
    } catch (e) {
      await handleError(e, set, get);
    }
  },

  splitSegment: async (id, at) => {
    const project = get().project;
    if (!project) return;
    set({ saving: true });
    try {
      await api.splitSegment(id, at, project.version);
      await get().refresh();
      set({ saving: false });
    } catch (e) {
      await handleError(e, set, get);
    }
  },

  mergeWithNext: async (id) => {
    const project = get().project;
    if (!project) return;
    const ordered = [...project.segments].sort((a, b) => a.start - b.start);
    const i = ordered.findIndex((s) => s.id === id);
    const next = ordered[i + 1];
    if (!next) return;
    if (next.speaker_id !== ordered[i].speaker_id) {
      set({ notice: "Склеивать можно только реплики одного спикера" });
      return;
    }
    set({ saving: true });
    try {
      await api.mergeSegments([id, next.id], project.version);
      await get().refresh();
      set({ saving: false });
    } catch (e) {
      await handleError(e, set, get);
    }
  },

  preview: async (id) => {
    set((st) => ({ previewing: new Set(st.previewing).add(id) }));
    try {
      await api.previewSegment(id);
    } catch (e) {
      set((st) => {
        const p = new Set(st.previewing);
        p.delete(id);
        return { previewing: p, notice: (e as Error).message };
      });
    }
  },

  approve: async () => {
    const project = get().project;
    if (!project) return;
    set({ saving: true });
    try {
      await api.approve(project.version);
      await get().refresh();
      set({ saving: false, notice: "Утверждено. Рендер запущен." });
    } catch (e) {
      await handleError(e, set, get);
    }
  },

  undo: async () => {
    const project = get().project;
    if (!project) return;
    try {
      const res = await api.undo(project.version);
      await get().refresh();
      set({ notice: `Отменено: ${res.undone}` });
    } catch (e) {
      await handleError(e, set, get);
    }
  },

  applyEvent: (e) => {
    const type = e.type as string;
    if (type === "progress") {
      set({ progress: { stage: Number(e.stage), label: String(e.label),
                        pct: Number(e.pct) } });
    } else if (type === "preview_ready") {
      const id = Number(e.ref);
      set((st) => {
        const p = new Set(st.previewing);
        p.delete(id);
        return { previewing: p };
      });
      void get().refresh();
      const audio = new Audio(String(e.url) + `?t=${encodeURIComponent(
        new URLSearchParams(window.location.search).get("t") ?? "")}`);
      void audio.play().catch(() => undefined);
    } else if (type === "preview_failed") {
      set((st) => {
        const p = new Set(st.previewing);
        p.delete(Number(e.ref));
        return { previewing: p, notice: `Превью не удалось: ${e.error}` };
      });
    } else if (type === "stage_changed" || type === "rediarized" ||
               type === "profile_rebuilt") {
      void get().refresh();
      if (type === "profile_rebuilt") set({ notice: "Профиль пересобран" });
      if (type === "rediarized") set({ notice: `Спикеров: ${e.speakers}` });
    } else if (type === "error") {
      set({ notice: String(e.message) });
    }
  },
}));

async function handleError(e: unknown, set: (p: Partial<State>) => void,
                           get: () => State): Promise<void> {
  if (e instanceof ApiError && e.status === 409) {
    set({ saving: false,
          notice: "Проект изменили в другом окне — обновляю данные" });
    await get().refresh();
    return;
  }
  set({ saving: false, notice: (e as Error).message });
}

// ------------------------------------------------------------- выборки

export const visibleSegments = (st: State): Segment[] => {
  const segs = st.project?.segments ?? [];
  const ordered = [...segs].sort((a, b) => a.start - b.start);
  switch (st.filter.kind) {
    case "speaker":
      return ordered.filter((s) => s.speaker_id === (st.filter as { id: string }).id);
    case "flag":
      return ordered.filter((s) =>
        s.flags.includes((st.filter as { flag: string }).flag));
    case "edited":
      return ordered.filter((s) => s.edited_by_user.fields.length > 0);
    default:
      return ordered;
  }
};

export const speakerById = (st: State, id: string): Speaker | undefined =>
  st.project?.speakers.find((s) => s.id === id);
