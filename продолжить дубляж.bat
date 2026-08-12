@echo off
title VoiceDub - продолжение дубляжа
cd /d "%~dp0"

echo ============================================================
echo  Продолжаю дубляж с того места, где остановились.
echo  Готово: 712 реплик из 2057. Заново они озвучиваться не будут.
echo  Результат появится в папке "готовые видео".
echo ============================================================
echo.

.venv\Scripts\python.exe -m scripts.run_url "C:\Users\jamsh\AppData\Local\Temp\claude\C--Users-jamsh-Desktop-bot-telegram\b8bff554-fb4b-4771-b004-041dd1626b16\scratchpad\full.mp4" --lang ru --voice bank --speakers auto --job-id 6c92d6e34f76

echo.
echo ============================================================
echo  Работа завершена. Смотрите папку "готовые видео".
echo ============================================================
pause
