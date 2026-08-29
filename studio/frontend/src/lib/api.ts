// Обращения к API студии.
//
// Две вещи, которые здесь нельзя забывать:
// 1) токен из адресной строки уходит в каждый запрос — без него 401;
// 2) любая правка несёт If-Match с версией проекта, а ответ 409 означает,
//    что проект изменили в другом окне, и надо перечитать, а не повторять.

import type { BankVoice, Project, Segment } from "./types";

export class ApiError extends Error {
  status: number;
  currentVersion?: number;
  constructor(status: number, message: string, currentVersion?: number) {
    super(message);
    this.status = status;
    this.currentVersion = currentVersion;
  }
}

export const jobId = (): string => {
  const m = window.location.pathname.match(/\/studio\/([^/?#]+)/);
  return m ? m[1] : "";
};

export const token = (): string =>
  new URLSearchParams(window.location.search).get("t") ?? "";

const base = () => `/api/projects/${jobId()}`;

async function request<T>(url: string, init: RequestInit = {},
                          version?: number): Promise<T> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token()}`,
    ...(init.body ? { "Content-Type": "application/json" } : {}),
    ...(version !== undefined ? { "If-Match": String(version) } : {}),
    ...((init.headers as Record<string, string>) ?? {}),
  };
  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    let message = `Ошибка ${res.status}`;
    let current: number | undefined;
    try {
      const body = await res.json();
      message = body.error ?? body.detail ?? message;
      current = body.current_version;
    } catch {
      /* тело может быть не JSON — оставляем код */
    }
    throw new ApiError(res.status, message, current);
  }
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("json") ? res.json() : res.text()) as Promise<T>;
}

export const api = {
  project: () => request<Project>(base()),

  voices: () => request<{ voices: BankVoice[] }>("/api/voices"),

  patchSegment: (id: number, patch: Partial<{
    speaker_id: string; text_tgt: string;
    voice_override: { mode: string | null; preset_id: string | null } | null;
  }>, version: number) =>
    request<Segment>(`${base()}/segments/${id}`, {
      method: "PATCH", body: JSON.stringify(patch),
    }, version),

  bulk: (segment_ids: number[], patch: Record<string, unknown>, version: number) =>
    request<{ updated: number; version: number }>(`${base()}/segments/bulk`, {
      method: "POST", body: JSON.stringify({ segment_ids, ...patch }),
    }, version),

  splitSegment: (id: number, at_sec: number, version: number) =>
    request<Segment[]>(`${base()}/segments/${id}/split`, {
      method: "POST", body: JSON.stringify({ at_sec }),
    }, version),

  mergeSegments: (segment_ids: number[], version: number) =>
    request<Segment>(`${base()}/segments/merge`, {
      method: "POST", body: JSON.stringify({ segment_ids }),
    }, version),

  patchSpeaker: (id: string, patch: Record<string, unknown>, version: number) =>
    request(`${base()}/speakers/${id}`, {
      method: "PATCH", body: JSON.stringify(patch),
    }, version),

  mergeSpeakers: (from_id: string, into_id: string, version: number) =>
    request<{ moved_segments: number }>(`${base()}/speakers/merge`, {
      method: "POST", body: JSON.stringify({ from_id, into_id }),
    }, version),

  rebuildProfile: (id: string, version: number) =>
    request(`${base()}/speakers/${id}/rebuild-profile`, { method: "POST" }, version),

  previewSegment: (id: number) =>
    request(`${base()}/preview/segment/${id}`, { method: "POST" }),

  previewRange: (start_sec: number, end_sec: number) =>
    request(`${base()}/preview/range`, {
      method: "POST", body: JSON.stringify({ start_sec, end_sec }),
    }),

  rediarize: (body: Record<string, number | null>, version: number) =>
    request(`${base()}/rediarize`, {
      method: "POST", body: JSON.stringify(body),
    }, version),

  approve: (version: number) =>
    request<{ stage: string; version: number }>(`${base()}/approve`,
      { method: "POST" }, version),

  undo: (version: number) =>
    request<{ undone: string; version: number }>(`${base()}/undo`,
      { method: "POST" }, version),
};

// URL медиа: в <audio>/<video> заголовки не подставить, поэтому токен
// уходит параметром — сервер принимает оба способа
export const mediaUrl = (path: string): string =>
  `${base()}${path}${path.includes("?") ? "&" : "?"}t=${encodeURIComponent(token())}`;

export const openEvents = (onEvent: (e: Record<string, unknown>) => void): WebSocket => {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(
    `${proto}://${window.location.host}${base()}/events?t=${encodeURIComponent(token())}`);
  ws.onmessage = (m) => {
    try { onEvent(JSON.parse(m.data)); } catch { /* мусор игнорируем */ }
  };
  return ws;
};
