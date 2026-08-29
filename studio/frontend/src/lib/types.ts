// Типы проекта — зеркало project/schema.py.
// Держатся вручную в синхроне: генератор ради одного файла усложнил бы
// сборку, а расхождение сразу видно по ошибке типов при обращении к полю.

export type Stage =
  | "uploaded" | "separated" | "transcribed" | "diarized" | "profiled"
  | "translated" | "review" | "approved" | "synthesizing" | "mixing"
  | "done" | "failed" | "cancelled";

export type Gender = "male" | "female" | "unknown";
export type VoiceMode = "clone" | "preset";

export interface SpeakerStats {
  segments_count: number;
  total_speech_sec: number;
  share_pct: number;
  first_appearance_sec: number;
}

export interface Reference {
  path: string | null;
  clean_sec: number;
  snr_db: number;
  score: number;
  clone_allowed: boolean;
  best_samples: number[];
}

export interface CastingCandidate {
  preset_id: string;
  display_name: string;
  score: number;
}

export interface Voice {
  mode: VoiceMode;
  preset_id: string | null;
  preset_name: string | null;
  profile_path: string | null;
  identity_path: string | null;
  locked: boolean;
  casting_candidates: CastingCandidate[];
  edited_by_user: boolean;
}

export interface Speaker {
  id: string;
  label: string;
  name: string | null;
  role: string | null;
  gender: Gender;
  gender_confidence: number;
  gender_edited_by_user: boolean;
  age: string;
  color: string;
  stats: SpeakerStats;
  reference: Reference;
  voice: Voice;
  merged_from: string[];
  merged_into: string | null;
  notes: string;
}

export interface SynthInfo {
  path: string | null;
  seed: number | null;
  attempts: number;
  identity_sim: number;
  duration_ratio: number;
  backcheck_cer: number | null;
  status: "pending" | "ok" | "qc_failed";
}

export interface Segment {
  id: number;
  start: number;
  end: number;
  speaker_id: string;
  speaker_confidence: number;
  speaker_margin: number;
  overlap: boolean;
  overlap_with: string[];
  text_src: string;
  text_tgt: string;
  text_tts: string;
  emotion: string;
  events: string[];
  budget_chars: number;
  over_budget: boolean;
  voice_override: { mode: VoiceMode | null; preset_id: string | null } | null;
  synth: SynthInfo;
  flags: string[];
  edited_by_user: { fields: string[]; ts: string | null };
}

export interface QCWarning {
  code: string;
  severity: "info" | "warn" | "error";
  message_ru: string;
  speaker_id: string | null;
  segment_ids: number[];
}

export interface Project {
  job_id: string;
  version: number;
  lang_src: string;
  lang_tgt: string;
  stage: Stage;
  source: {
    file_name: string;
    duration_sec: number;
    title: string;
    video_path: string | null;
    vocals_path: string | null;
    background_path: string | null;
  };
  settings: {
    voice_mode: string;
    auto_approve: boolean;
    keep_background: boolean;
    unique_voices: boolean;
  };
  speakers: Speaker[];
  segments: Segment[];
  qc: {
    warnings: QCWarning[];
    identity_report: Record<string, {
      voice: string; segments: number; passed: number;
      mean_pairwise_identity: number;
    }>;
    overall_identity: number;
  };
  warnings_count: number;
  media_available: boolean;
  result_path: string | null;
}

export interface BankVoice {
  id: string;
  display_name: string;
  gender: Gender;
  age_group: string;
  timbre_tags: string[];
  languages: string[];
  source: string;
  has_profile: boolean;
  f0_hz: number | null;
}

export const FLAG_LABELS: Record<string, string> = {
  low_speaker_conf: "низкая уверенность",
  suspicious_isolated: "одиночная реплика",
  overlap: "наложение речи",
  too_short_for_embedding: "слишком короткая",
  over_budget: "перевод длиннее слота",
  identity_qc_failed: "тембр не совпал",
};

export const STAGE_LABELS: Record<Stage, string> = {
  uploaded: "загружено",
  separated: "дорожки разделены",
  transcribed: "речь распознана",
  diarized: "спикеры определены",
  profiled: "голоса зафиксированы",
  translated: "переведено",
  review: "ПРОВЕРКА",
  approved: "утверждено",
  synthesizing: "озвучивается",
  mixing: "собирается",
  done: "готово",
  failed: "ошибка",
  cancelled: "отменено",
};
