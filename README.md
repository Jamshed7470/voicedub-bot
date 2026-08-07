# 🎬 VoiceDub Bot — Telegram-бот автоматического дубляжа видео и аудио

[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![aiogram 3](https://img.shields.io/badge/aiogram-3.x-2CA5E0.svg)](https://aiogram.dev)
[![TTS: XTTS-v2](https://img.shields.io/badge/TTS-XTTS--v2%20(local)-orange.svg)](https://github.com/idiap/coqui-ai-TTS)
[![ASR: WhisperX](https://img.shields.io/badge/ASR-WhisperX-green.svg)](https://github.com/m-bain/whisperX)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> **EN:** AI video dubbing Telegram bot: speech recognition with word-level timings
> (WhisperX), speaker diarization (pyannote 3.1), per-speaker voice cloning
> (XTTS-v2, fully local), emotion-aware synthesis, background music preservation
> (Demucs) and context-aware translation (Claude API or local NLLB).
> Documentation below is in Russian.

Отправьте боту видео, аудио или ссылку (YouTube, TikTok, Instagram…) — он:

1. сам определит формат и извлечёт аудио;
2. распознает речь с точными таймингами (WhisperX);
3. определит спикеров — у каждого будет постоянный ID и **его собственный
   уникальный голос** на всём протяжении (pyannote + клонирование голоса);
4. определит пол каждого спикера;
5. переведёт текст на выбранный язык с сохранением смысла и стиля
   (Claude API или локальный NLLB);
6. озвучит **локальной** мультиязычной моделью XTTS-v2 с сохранением эмоций
   и исходных таймингов;
7. сохранит фоновую музыку и невербальные звуки (смех, плач — копируются
   из оригинала);
8. соберёт финальное видео и пришлёт его вам вместе с субтитрами (.srt).

Все сообщения бота — на русском. Поддерживается 17 языков озвучки: русский,
английский, турецкий, арабский, китайский, японский, корейский, хинди,
испанский, французский, немецкий, итальянский, португальский, польский,
нидерландский, чешский, венгерский.

---

## Что понадобится

| Что | Зачем | Обязательно? |
|---|---|---|
| Компьютер с NVIDIA GPU от 8 ГБ VRAM (лучше 12+), 16 ГБ RAM, ~30 ГБ на диске | веса моделей и быстрая обработка | рекомендуется (без GPU работает, но медленно) |
| Python 3.10 или 3.11 | язык, на котором написан бот | да |
| ffmpeg | вся работа с аудио и видео | да |
| Токен Telegram-бота | сам бот | да (уже вписан в `.env`) |
| Токен HuggingFace (бесплатный) | определение спикеров | да |
| Ключ Anthropic API | перевод через Claude (макс. качество) | нет — без него включится локальный NLLB |

> ⚠️ **Важно про Python:** нужна именно версия **3.10 или 3.11**.
> На Python 3.12+ часть ML-библиотек не установится.

---

## Шаг 1. Установить Python 3.11

**Windows:**
1. Откройте https://www.python.org/downloads/release/python-3119/
2. Внизу страницы скачайте «Windows installer (64-bit)».
3. Запустите установщик и **обязательно поставьте галочку
   «Add python.exe to PATH»**, затем «Install Now».
4. Проверка: откройте PowerShell (Пуск → введите «PowerShell») и выполните:
   ```powershell
   py -3.11 --version
   ```
   Должно показать `Python 3.11.x`.

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv
python3.11 --version
```

## Шаг 2. Установить ffmpeg

**Windows (самый простой способ — winget):**
```powershell
winget install --id Gyan.FFmpeg -e
```
Закройте и заново откройте PowerShell, затем проверьте:
```powershell
ffmpeg -version
```
Если winget недоступен: скачайте архив «ffmpeg-release-full» с
https://www.gyan.dev/ffmpeg/builds/ , распакуйте, и добавьте папку `bin`
в PATH (Пуск → «Изменение переменных среды» → Path → Создать → путь к `bin`).

**Linux:**
```bash
sudo apt install -y ffmpeg
```

## Шаг 3. Скачать проект и установить зависимости

Откройте PowerShell (или терминал) **в папке проекта `voicedub`** и выполните:

**Windows:**
```powershell
# 1) создать виртуальное окружение
py -3.11 -m venv .venv

# 2) активировать его (строчка (.venv) появится слева в приглашении)
.\.venv\Scripts\Activate.ps1
#   если PowerShell ругается на запрет скриптов, выполните один раз:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3) PyTorch с поддержкой GPU (NVIDIA) — ставится ПЕРВЫМ, отдельной командой:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
#   если GPU нет — просто: pip install torch torchaudio

# 4) все остальные зависимости
pip install -r requirements.txt
```

**Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Установка займёт 10–20 минут (несколько гигабайт). Это нормально.

## Шаг 4. Токен HuggingFace (обязательно!)

Без него не будет работать определение спикеров. Всё бесплатно, займёт 5 минут:

1. Зарегистрируйтесь на https://huggingface.co (кнопка Sign Up).
2. Подтвердите e-mail.
3. Зайдите в **Settings → Access Tokens** (https://huggingface.co/settings/tokens),
   нажмите **New token**, тип — **Read**, любое имя. Скопируйте токен
   (начинается с `hf_...`).
4. Теперь примите условия двух моделей (иначе они не скачаются). Откройте
   каждую страницу и нажмите **«Agree and access repository»**:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
5. Откройте файл `.env` в папке проекта (обычным Блокнотом) и вставьте токен:
   ```
   HF_TOKEN=hf_ваш_токен
   ```

## Шаг 5. (Необязательно) Ключ Anthropic для перевода через Claude

Если хотите максимальное качество перевода:
1. Получите API-ключ на https://console.anthropic.com
2. Впишите его в `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

Если оставить пустым — бот автоматически использует **локальный** переводчик
NLLB (бесплатно, без интернета, качество ниже).

## Шаг 6. Проверить .env

Откройте `.env` и убедитесь:

```
BOT_TOKEN=уже вписан
HF_TOKEN=hf_...          ← ваш токен из шага 4
ANTHROPIC_API_KEY=       ← можно оставить пустым
DEVICE=auto              ← оставить как есть
MODEL_PROFILE=full       ← для слабых ПК/без GPU поставьте light
```

> 🔒 **Безопасность:** токен бота даёт полный контроль над ботом. Файл `.env`
> нельзя никому отправлять и публиковать (GitHub, чаты). Если токен утёк —
> напишите @BotFather команду `/revoke`, он выдаст новый.

## Шаг 7. Запуск

Из папки `voicedub` с активированным окружением:

```powershell
python -m bot.main
```

При первом запуске бот скачает веса моделей (10–20 ГБ) — это происходит
один раз, дальше всё работает локально (интернет нужен только Telegram
и, если включён, Claude API).

Когда в консоли появится «VoiceDub Bot запущен» — откройте своего бота
в Telegram, нажмите **Start**, выберите язык командой `/lang` и отправьте
видео или ссылку.

Остановить бота: `Ctrl+C` в консоли.

---

## Команды бота

| Команда | Что делает |
|---|---|
| `/start` | приветствие |
| `/lang` | выбор языка озвучки (запоминается) |
| `/settings` | сохранять фоновую музыку (вкл/выкл), оригинальная дорожка вторым треком (вкл/выкл) |
| `/status` | текущий этап или позиция в очереди |
| `/cancel` | отменить свою задачу |
| `/help` | справка, лимиты, форматы |

## Лимиты

- Файлы из Telegram — до **20 МБ** (ограничение Telegram Bot API).
  Файл больше? Просто пришлите **ссылку** на видео.
- Длительность — до **60 минут** (меняется в `configs/config.yaml`).
- Результат — до **50 МБ** (лимит отправки ботом); при превышении бот сам
  пережмёт видео.

### Как снять лимит 20 МБ (продвинутое, необязательно)

Запустите свой локальный сервер Bot API
(https://github.com/tdlib/telegram-bot-api), получите `api_id`/`api_hash`
на https://my.telegram.org и пропишите в `.env`:
```
TELEGRAM_LOCAL_API_URL=http://127.0.0.1:8081
```

---

## Тесты

```powershell
pytest tests/test_normalize.py tests/test_timing.py -v
```

Полный smoke-тест пайплайна (нужны веса моделей и HF_TOKEN):
```powershell
$env:VOICEDUB_SMOKE = "1"
pytest tests/test_smoke.py -s
```

## Частые проблемы

| Проблема | Решение |
|---|---|
| `ffmpeg не является внутренней или внешней командой` | ffmpeg не в PATH — повторите Шаг 2, перезапустите PowerShell |
| Ошибка при установке whisperx / coqui-tts | у вас Python 3.12+ — нужен 3.10/3.11 (Шаг 1, затем пересоздайте .venv) |
| «Не удалось определить спикеров» | не принято соглашение моделей pyannote (Шаг 4, п. 4) или неверный HF_TOKEN |
| Всё очень медленно | нет GPU: поставьте `MODEL_PROFILE=light` в `.env`; обработка на CPU в 10–30 раз медленнее |
| `CUDA out of memory` | закройте другие программы, использующие GPU, или включите `MODEL_PROFILE=light` |
| Бот молчит на файл > 20 МБ | это лимит Telegram — пришлите ссылку на видео |
| Первый запуск «завис» | скачиваются веса моделей (10–20 ГБ) — дождитесь |

## Структура проекта

```
voicedub/
├── bot/            # Telegram-бот (aiogram 3): хендлеры, очередь, прогресс
├── core/           # пайплайн: ffmpeg, ASR, диаризация, перевод, TTS, микс
├── configs/        # config.yaml — все настройки пайплайна
├── data/jobs/      # временные файлы задач (автоочистка)
├── tests/          # тесты
├── .env            # секреты (НЕ публиковать!)
└── requirements.txt
```

Приятного пользования! 🎧
