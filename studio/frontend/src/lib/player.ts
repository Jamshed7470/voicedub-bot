// Один звук за раз.
//
// Каждое прослушивание раньше создавало собственный Audio, и нажатие на
// второй голос не останавливало первый — они звучали одновременно и
// сравнивать их было невозможно. Это ровно та задача, ради которой в
// студию и заходят, поэтому проигрыватель здесь один на всё приложение.

let current: HTMLAudioElement | null = null;
let currentKey: string | null = null;
const listeners = new Set<(key: string | null) => void>();

function notify(): void {
  for (const fn of listeners) fn(currentKey);
}

/** Подписка на «что сейчас играет» — для подсветки активной кнопки. */
export function onPlaybackChange(fn: (key: string | null) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function nowPlaying(): string | null {
  return currentKey;
}

/** Останавливает всё: и точечное прослушивание, и превью отрывка с видео. */
export function stopAll(): void {
  if (current) {
    current.pause();
    current.src = "";
    current = null;
  }
  const range = document.getElementById("range-audio") as HTMLAudioElement | null;
  if (range && !range.paused) {
    range.pause();
    const video = (window as unknown as { __player?: HTMLVideoElement }).__player;
    if (video) { video.pause(); video.muted = false; }
  }
  currentKey = null;
  notify();
}

/**
 * Играет url, остановив предыдущее. Повторное нажатие на тот же
 * источник останавливает его — так кнопка работает как переключатель.
 */
export function play(url: string, key = url): void {
  if (currentKey === key && current && !current.paused) {
    stopAll();
    return;
  }
  stopAll();
  const audio = new Audio(url);
  current = audio;
  currentKey = key;
  notify();
  audio.onended = () => {
    if (current === audio) { current = null; currentKey = null; notify(); }
  };
  audio.onerror = () => {
    if (current === audio) { current = null; currentKey = null; notify(); }
  };
  void audio.play().catch(() => {
    if (current === audio) { current = null; currentKey = null; notify(); }
  });
}
