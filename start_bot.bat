@echo off
chcp 866 >nul
title VoiceDub Bot
cd /d "%~dp0"

echo ==================================================
echo    VoiceDub Bot
echo ==================================================
echo.

if not exist ".venv\Scripts\python.exe" goto no_venv
if not exist ".env" goto no_env

echo [1/2] Проверяю, не запущен ли бот уже...
powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*bot.main*' }; if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host ('      stopped PID ' + $_.ProcessId) } } else { Write-Host '      no old processes' }"

echo [2/2] Запускаю. Остановить - Ctrl+C или закрыть окно.
echo.
".venv\Scripts\python.exe" -m bot.main

echo.
echo ==================================================
echo    Бот остановлен. Логи: logs\voicedub.log
echo ==================================================
pause
exit /b 0

:no_venv
echo [ОШИБКА] Не найдено виртуальное окружение .venv
echo Ожидался файл: %cd%\.venv\Scripts\python.exe
echo.
pause
exit /b 1

:no_env
echo [ОШИБКА] Нет файла .env с токеном бота
echo Скопируйте .env.example в .env и заполните BOT_TOKEN
echo.
pause
exit /b 1
