# Локальный Telegram Bot API сервер

Снимает лимиты облачного Bot API: скачивание 20 МБ → 2 ГБ, отправка 50 МБ → 2 ГБ.
Длинные ролики перестанут резаться на части и ужиматься до низкого разрешения.

## Что нужно один раз

**1. Docker Desktop** — https://www.docker.com/products/docker-desktop/
Установка требует прав администратора и, возможно, перезагрузки.
После установки запусти Docker Desktop и дождись статуса «Engine running».

**2. api_id и api_hash** — https://my.telegram.org/apps
Вход по номеру телефона твоего Telegram (это личный аккаунт, не бот).
Раздел «API development tools» → создать приложение (название любое,
например `voicedub`) → скопировать `api_id` (число) и `api_hash` (строка).

Это ключи серверного приложения, а не бота: они позволяют своему серверу
общаться с Telegram. Храни их как пароль.

**3. Заполнить `telegram-api/.env`** (создать рядом с docker-compose.yml):

```
TELEGRAM_API_ID=1234567
TELEGRAM_API_HASH=0123456789abcdef0123456789abcdef
```

## Запуск

```
cd telegram-api
docker compose up -d          # поднять сервер
docker compose logs -f        # смотреть логи
docker compose down           # остановить
```

Проверка, что сервер жив: `curl http://127.0.0.1:8081/` — ответит ошибкой
метода, это нормально, главное что отвечает.

## Переезд бота на свой сервер

Telegram не пускает бота на локальный сервер, пока он «залогинен» в облаке.
Выйти нужно один раз:

```
python telegram-api/migrate_bot.py to-local
```

Затем дописать в основной `.env` проекта:

```
TELEGRAM_LOCAL_API_URL=http://127.0.0.1:8081
TELEGRAM_LOCAL_FILES_DIR=C:\Users\jamsh\Desktop\bot telegram\voicedub\telegram-api\data
```

и перезапустить бота (`start_bot.bat`). В логе появится строка
«Использую локальный Bot API … (лимит отправки 1900 МБ)».

## Вернуться в облако

```
python telegram-api/migrate_bot.py to-cloud
```

и очистить обе строки `TELEGRAM_LOCAL_*` в `.env`.

## Как это работает

- `TELEGRAM_LOCAL=1` (флаг `--local`) — сервер отдаёт боту **пути к файлам
  на диске**, а не ссылки для скачивания. Файл не перекачивается по сети.
- Папка `telegram-api/data` — общая для контейнера и бота. Внутри контейнера
  она видна как `/var/lib/telegram-bot-api`; бот сопоставляет пути через
  `TELEGRAM_LOCAL_FILES_DIR`, иначе не найдёт файл.
- Порт открыт только на `127.0.0.1` — снаружи сервер недоступен.

## Если что-то не так

| Симптом | Причина |
|---|---|
| `Logged in as bot ... in another server` | не выполнен `migrate_bot.py to-local` |
| Бот не находит скачанный файл | не задан или неверен `TELEGRAM_LOCAL_FILES_DIR` |
| `connection refused` на 8081 | контейнер не запущен: `docker compose ps` |
| Сервер стартует и падает | неверные `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` |
