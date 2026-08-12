@echo off
chcp 866 >nul
title Telegram Bot API (локальный)
cd /d "%~dp0telegram-api"

echo ==================================================
echo    Локальный Telegram Bot API сервер
echo ==================================================
echo.

where docker >nul 2>&1
if errorlevel 1 goto no_docker
if not exist ".env" goto no_env

docker info >nul 2>&1
if errorlevel 1 goto no_engine

echo Поднимаю сервер...
docker compose up -d
echo.
echo Состояние:
docker compose ps
echo.
echo Логи: docker compose logs -f    Остановить: docker compose down
echo.
pause
exit /b 0

:no_docker
echo [ОШИБКА] Docker не найден.
echo Установи Docker Desktop: https://www.docker.com/products/docker-desktop/
echo.
pause
exit /b 1

:no_engine
echo [ОШИБКА] Docker установлен, но движок не запущен.
echo Открой Docker Desktop и дождись статуса "Engine running".
echo.
pause
exit /b 1

:no_env
echo [ОШИБКА] Нет файла telegram-api\.env с ключами.
echo Возьми api_id и api_hash на https://my.telegram.org/apps и создай файл:
echo    TELEGRAM_API_ID=1234567
echo    TELEGRAM_API_HASH=0123456789abcdef0123456789abcdef
echo.
pause
exit /b 1
