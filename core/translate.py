"""Перевод диалога: Claude API (если задан ключ) или локальный NLLB.

Перевод идёт с полным контекстом: JSON-массив сегментов окнами до 80 штук
с перекрытием и кратким резюме предыдущего контекста. Каждому сегменту
передаётся бюджет max_chars (length-aware).
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

# Whisper коды → FLORES-200 (для NLLB)
FLORES = {
    "ru": "rus_Cyrl", "en": "eng_Latn", "tr": "tur_Latn", "ar": "arb_Arab",
    "zh": "zho_Hans", "zh-cn": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "hi": "hin_Deva", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "pl": "pol_Latn", "nl": "nld_Latn",
    "cs": "ces_Latn", "hu": "hun_Latn", "uk": "ukr_Cyrl", "tg": "tgk_Cyrl",
    "fa": "pes_Arab", "kk": "kaz_Cyrl", "uz": "uzn_Latn", "az": "azj_Latn",
    "id": "ind_Latn", "vi": "vie_Latn", "th": "tha_Thai", "he": "heb_Hebr",
    "ro": "ron_Latn", "sv": "swe_Latn", "da": "dan_Latn", "no": "nob_Latn",
    "fi": "fin_Latn", "el": "ell_Grek", "bg": "bul_Cyrl", "sr": "srp_Cyrl",
    "hr": "hrv_Latn", "sk": "slk_Latn",
}

LANG_NAMES_RU = {
    "ru": "русский", "en": "английский", "tr": "турецкий", "ar": "арабский",
    "zh-cn": "китайский", "ja": "японский", "ko": "корейский", "hi": "хинди",
    "es": "испанский", "fr": "французский", "de": "немецкий", "it": "итальянский",
    "pt": "португальский", "pl": "польский", "nl": "нидерландский",
    "cs": "чешский", "hu": "венгерский",
}


def compute_max_chars(seg: dict, tgt_lang: str, cfg) -> int:
    dur = max(0.5, seg["end"] - seg["start"])
    return max(10, int(dur * cfg.speech_rate(tgt_lang)))


def get_translator(cfg, style: str = "normal"):
    if style == "street":
        if cfg.xai_api_key:
            return GrokTranslator(cfg)
        log.warning("Выбран уличный стиль, но XAI_API_KEY не задан — обычный перевод")
    if cfg.anthropic_api_key:
        return ClaudeTranslator(cfg)
    log.info("ANTHROPIC_API_KEY не задан — использую локальный переводчик NLLB")
    return NLLBTranslator(cfg)


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — профессиональный переводчик для дубляжа видео.
Переводи с языка «{src}» на язык «{tgt}».

Правила:
1. Перевод для ДУБЛЯЖА, а не дословный подстрочник: сохраняй смысл, стиль,
   регистр (ты/вы), междометия, юмор и эмоциональную окраску.
2. Учитывай пол говорящего (поле gender) для правильных форм глаголов
   и прилагательных.
3. ВСЕ числа, даты, годы, порядковые, суммы, проценты, единицы измерения
   и номера пиши СЛОВАМИ на целевом языке в правильной грамматической форме:
   «9 мая» → «девятого мая», «в 2026 году» → «в две тысячи двадцать шестом
   году», «5%» → «пять процентов», «10 км» → «десять километров».
   В итоговом тексте НЕ ДОЛЖНО остаться ни одной цифры и символов %, №, $, €.
4. Держись бюджета длины: поле max_chars — максимум символов для сегмента
   (чтобы дубляж успевал за оригиналом). Превышение до 30% допустимо,
   но стремись уложиться.
5. Ответ — СТРОГО тот же JSON-массив, что на входе, с теми же id и полями,
   где переведено ТОЛЬКО поле text. Никакого текста вне JSON."""


class ClaudeTranslator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.y("translation", "claude_model", default="claude-sonnet-4-6")
        self.temperature = float(cfg.y("translation", "temperature", default=0.2))
        import anthropic
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    def _call(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    @staticmethod
    def _extract_json(text: str):
        text = text.strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        return json.loads(text)

    def translate_segments(self, segments: list[dict], src_lang: str,
                           tgt_lang: str, speakers: dict, progress=None) -> None:
        """Переводит segments in-place (поле text → перевод)."""
        window = int(self.cfg.y("translation", "window_segments", default=80))
        overlap = int(self.cfg.y("translation", "overlap_segments", default=5))
        system = SYSTEM_PROMPT.format(
            src=LANG_NAMES_RU.get(src_lang, src_lang),
            tgt=LANG_NAMES_RU.get(tgt_lang, tgt_lang),
        )

        done = 0
        prev_context = ""
        i = 0
        while i < len(segments):
            batch = segments[i:i + window]
            payload = [
                {
                    "id": s["id"],
                    "speaker": s["speaker"],
                    "gender": speakers.get(s["speaker"], {}).get("gender", "unknown"),
                    "emotion": s.get("emotion", "neutral"),
                    "max_chars": compute_max_chars(s, tgt_lang, self.cfg),
                    "text": s["text"],
                }
                for s in batch
            ]
            user = ""
            if prev_context:
                user += f"Краткий контекст предыдущей части диалога:\n{prev_context}\n\n"
            user += "Переведи сегменты:\n" + json.dumps(payload, ensure_ascii=False)

            translated = None
            for attempt in (1, 2):
                try:
                    translated = self._extract_json(self._call(system, user))
                    break
                except Exception:  # noqa: BLE001
                    log.exception("Claude: ошибка перевода окна (попытка %d)", attempt)
            if translated:
                by_id = {t.get("id"): t.get("text", "") for t in translated
                         if isinstance(t, dict)}
                for s in batch:
                    new = (by_id.get(s["id"]) or "").strip()
                    if new:
                        s["text"] = new
            else:
                log.error("Claude: окно не переведено, оставляю оригинальный текст")

            tail = batch[-overlap:] if len(batch) > overlap else batch
            prev_context = " / ".join(f"{s['speaker']}: {s['text'][:80]}" for s in tail)
            done += len(batch)
            if progress:
                progress(min(100, int(100 * done / len(segments))))
            i += window

    def compress_segment(self, text: str, max_chars: int, tgt_lang: str) -> str:
        """Сжатый вариант перевода сегмента без потери смысла (length-aware)."""
        try:
            out = self._call(
                f"Ты сокращаешь реплики для дубляжа на языке "
                f"«{LANG_NAMES_RU.get(tgt_lang, tgt_lang)}». Сохраняй смысл. "
                f"Все числа — словами, цифры запрещены. "
                f"Ответ — только сокращённый текст, без пояснений.",
                f"Сократи до {max_chars} символов:\n{text}",
            ).strip()
            return out or text
        except Exception:  # noqa: BLE001
            log.exception("Claude: не удалось сжать сегмент")
            return text


# ---------------------------------------------------------------------------
# Локальный NLLB
# ---------------------------------------------------------------------------

class NLLBTranslator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.nllb_model
        self._model = None
        self._tokenizer = None

    def _load(self, src_flores: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        if self._model is None:
            log.info("NLLB: загружаю %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, src_lang=src_flores)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            if self.cfg.device == "cuda":
                self._model = self._model.to("cuda")
        else:
            self._tokenizer.src_lang = src_flores

    def translate_segments(self, segments: list[dict], src_lang: str,
                           tgt_lang: str, speakers: dict, progress=None) -> None:
        import torch

        src_flores = FLORES.get(src_lang)
        tgt_flores = FLORES.get(tgt_lang)
        if not src_flores or not tgt_flores:
            from core.errors import UserError
            raise UserError(
                f"Локальный переводчик NLLB не поддерживает пару "
                f"{src_lang} → {tgt_lang}. Укажите ANTHROPIC_API_KEY в .env, "
                f"чтобы переводить через Claude."
            )
        self._load(src_flores)
        tok, model = self._tokenizer, self._model
        bos = tok.convert_tokens_to_ids(tgt_flores)

        batch_size = 8
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            texts = [s["text"] for s in batch]
            inputs = tok(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=512)
            if self.cfg.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, forced_bos_token_id=bos,
                                     max_length=512, num_beams=4)
            for s, ids in zip(batch, out):
                s["text"] = tok.decode(ids, skip_special_tokens=True).strip()
            if progress:
                progress(min(100, int(100 * (i + len(batch)) / len(segments))))

    def compress_segment(self, text: str, max_chars: int, tgt_lang: str) -> str:
        # NLLB не умеет сжимать по инструкции — оставляем как есть,
        # ужмётся atempo (или обрежется мягко по границе предложения)
        if len(text) <= max_chars * 1.6:
            return text
        cut = text[: int(max_chars * 1.5)]
        for sep in (". ", "! ", "? ", ", "):
            pos = cut.rfind(sep)
            if pos > max_chars // 2:
                return cut[: pos + 1].strip()
        return cut.strip()


# ---------------------------------------------------------------------------
# Grok (xAI) — «уличный» стиль перевода
# ---------------------------------------------------------------------------

STREET_SYSTEM_PROMPT = """Ты — переводчик дубляжа в ДЕРЗКОМ УЛИЧНОМ стиле.
Переводи с языка «{src}» на язык «{tgt}» так, будто озвучку делает острый
на язык уличный пацан: сленг, стёб, подколы, грубые словечки и оскорбления
в адрес происходящего — приветствуются. Крепкие выражения разрешены.

Правила:
1. Смысл сцены сохраняется: зритель должен понять то же самое, но подача —
   наглая, смешная, уличная. Подкалывай персонажей, утрируй.
2. Учитывай пол говорящего (поле gender) для форм глаголов.
3. ВСЕ числа, даты, проценты, суммы пиши СЛОВАМИ на целевом языке.
   В тексте не должно остаться ни одной цифры и символов %, №, $, €.
4. КРИТИЧНО — длина: у каждого сегмента есть поле len_target (число символов
   оригинала). Твой перевод должен быть длиной от девяноста до ста десяти
   процентов len_target. Не короче и не длиннее.
5. Ответ — СТРОГО тот же JSON-массив, что на входе, с теми же id,
   где переведено ТОЛЬКО поле text. Никакого текста вне JSON."""


class GrokTranslator:
    """Перевод через xAI Grok API (OpenAI-совместимый chat/completions)."""

    API_URL = "https://api.x.ai/v1/chat/completions"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.y("translation", "xai_model", default="grok-4")
        self.len_min = float(cfg.y("translation", "street_len_min", default=0.9))
        self.len_max = float(cfg.y("translation", "street_len_max", default=1.1))

    def _call(self, system: str, user: str, temperature: float = 0.7) -> str:
        import requests
        resp = requests.post(
            self.API_URL,
            headers={"Authorization": f"Bearer {self.cfg.xai_api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _fit_length(self, text: str, len_target: int, tgt_lang: str) -> str:
        """Дожимает/растягивает сегмент до 90–110% длины оригинала (одна попытка)."""
        try:
            out = self._call(
                f"Ты редактируешь реплику дубляжа в уличном стиле на языке "
                f"«{LANG_NAMES_RU.get(tgt_lang, tgt_lang)}». Сохрани смысл и дерзость. "
                f"Все числа — словами, цифры запрещены. Ответ — только текст реплики.",
                f"Перепиши эту реплику так, чтобы её длина была от "
                f"{int(len_target * self.len_min)} до {int(len_target * self.len_max)} "
                f"символов:\n{text}",
            ).strip()
            return out or text
        except Exception:  # noqa: BLE001
            log.exception("Grok: не удалось подогнать длину сегмента")
            return text

    def translate_segments(self, segments: list[dict], src_lang: str,
                           tgt_lang: str, speakers: dict, progress=None) -> None:
        window = int(self.cfg.y("translation", "window_segments", default=80))
        overlap = int(self.cfg.y("translation", "overlap_segments", default=5))
        system = STREET_SYSTEM_PROMPT.format(
            src=LANG_NAMES_RU.get(src_lang, src_lang),
            tgt=LANG_NAMES_RU.get(tgt_lang, tgt_lang),
        )

        done = 0
        prev_context = ""
        i = 0
        while i < len(segments):
            batch = segments[i:i + window]
            originals = {s["id"]: s["text"] for s in batch}
            payload = [
                {
                    "id": s["id"],
                    "speaker": s["speaker"],
                    "gender": speakers.get(s["speaker"], {}).get("gender", "unknown"),
                    "len_target": len(s["text"]),
                    "text": s["text"],
                }
                for s in batch
            ]
            user = ""
            if prev_context:
                user += f"Контекст предыдущей части:\n{prev_context}\n\n"
            user += "Переведи сегменты:\n" + json.dumps(payload, ensure_ascii=False)

            translated = None
            for attempt in (1, 2):
                try:
                    translated = ClaudeTranslator._extract_json(
                        self._call(system, user))
                    break
                except Exception:  # noqa: BLE001
                    log.exception("Grok: ошибка перевода окна (попытка %d)", attempt)
            if translated:
                by_id = {t.get("id"): t.get("text", "") for t in translated
                         if isinstance(t, dict)}
                for s in batch:
                    new = (by_id.get(s["id"]) or "").strip()
                    if not new:
                        continue
                    # контроль длины: 90–110% оригинала
                    target = max(1, len(originals[s["id"]]))
                    ratio = len(new) / target
                    if ratio < self.len_min - 0.05 or ratio > self.len_max + 0.05:
                        new = self._fit_length(new, target, tgt_lang)
                    s["text"] = new
            else:
                log.error("Grok: окно не переведено, оставляю оригинальный текст")

            tail = batch[-overlap:] if len(batch) > overlap else batch
            prev_context = " / ".join(f"{s['speaker']}: {s['text'][:80]}" for s in tail)
            done += len(batch)
            if progress:
                progress(min(100, int(100 * done / len(segments))))
            i += window

    def compress_segment(self, text: str, max_chars: int, tgt_lang: str) -> str:
        return self._fit_length(text, max_chars, tgt_lang)
