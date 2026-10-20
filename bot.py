import asyncio
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime

import aiohttp
import gspread
from aiogram import Bot, Dispatcher, F, types
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from dotenv import load_dotenv
from google.oauth2 import service_account
from openai import AsyncOpenAI

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USERNAME = os.getenv("ALLOWED_USERNAME", "").lstrip("@").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

# ---------- Google Sheets ----------
def get_sheet():
    if GOOGLE_SERVICE_ACCOUNT_JSON:
        # Ключ передан как переменная окружения (base64 или чистый JSON)
        raw = GOOGLE_SERVICE_ACCOUNT_JSON
        try:
            raw = base64.b64decode(raw).decode()
        except Exception:
            pass
        info = json.loads(raw)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
    else:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
    client = gspread.authorize(creds)
    return client.open_by_key(GOOGLE_SHEET_ID)

# ---------- OpenAI ----------
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ---------- Telegram ----------
# Долгие таймауты: контейнер хостинга может медленно ходить к api.telegram.org
telegram_session = AiohttpSession(
    timeout=aiohttp.ClientTimeout(total=180, connect=60)
)
bot = Bot(token=TELEGRAM_BOT_TOKEN, session=telegram_session)
dp = Dispatcher()


def is_allowed(message: types.Message) -> bool:
    if not ALLOWED_USERNAME:
        return True
    username = (message.from_user.username or "").lower()
    return username == ALLOWED_USERNAME


def sheet_columns(sheet_name: str) -> list[str]:
    """Возвращает список колонок листа (из первой строки)."""
    sheet = get_sheet().worksheet(sheet_name)
    return sheet.row_values(1)


def write_row(sheet_name: str, values: dict):
    """Дописывает строку в конец листа, сопоставляя значения с колонками."""
    sheet = get_sheet().worksheet(sheet_name)
    headers = sheet.row_values(1)
    row = [""] * len(headers)
    for key, val in values.items():
        if key in headers:
            row[headers.index(key)] = val
    sheet.append_row(row, value_input_option="USER_ENTERED")


SYSTEM_PROMPT = """Ты — ассистент по учёту здоровья. Пользователь присылает
голосовые, фото и текстовые сообщения о своём питании, тренировках, весе,
сне, воде и самочувствии. Твоя задача — понять, о чём речь, и вернуть JSON
для записи в Google Таблицу.

Доступные листы и их колонки:
1. "Питание": Дата, Приём пищи, Блюдо, Количество (г), Калории (ккал),
   Белки (г), Жиры (г), Углеводы (г), Вода (мл), Заметки
2. "Тренировки": Дата, Тип тренировки, Упражнение, Подходы, Повторы,
   Вес (кг), Длительность (мин), Интенсивность (1-10), Заметки
3. "Показатели тела": Дата, Вес (кг), % жира, Талия (см), Грудь (см),
   Бёдра (см), Плечи (см), Рука (см), Бедро (см), Заметки
4. "Сон и вода": Дата, Лёг спать, Проснулся, Сон (часы),
   Качество сна (1-10), Вода (л), Кофеин (мг), Заметки
5. "Самочувствие": Дата, Настроение (1-10), Энергия (1-10),
   Стресс (1-10), Мотивация (1-10), Боли/травмы, Заметки

Правила:
- Определи, к какому листу относится сообщение. Если несколько тем —
  верни несколько объектов в массиве "entries".
- Дата — сегодняшняя в формате ГГГГ-ММ-ДД.
- Числа — числами (без единиц измерения).
- Если данных нет — оставь поле пустым или пропусти.
- Верни ТОЛЬКО валидный JSON, без пояснений.

Формат ответа:
{"entries":[{"sheet":"Питание","values":{"Блюдо":"овсянка","Калории (ккал)":350}}]}
"""


async def parse_with_gpt(text: str) -> list[dict]:
    resp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    # Убираем возможные ```json ... ``` обёртки
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    data = json.loads(content)
    return data.get("entries", [])


async def transcribe_voice(file_id: str) -> str:
    file = await bot.get_file(file_id)
    file_bytes = await bot.download_file(file.file_path)
    data = io.BytesIO(file_bytes.read())
    data.name = "voice.ogg"
    resp = await openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=data,
    )
    return resp.text


async def analyze_photo(file_id: str) -> str:
    file = await bot.get_file(file_id)
    file_bytes = await bot.download_file(file.file_path)
    b64 = __import__("base64").b64encode(file_bytes.read()).decode()
    resp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Опиши блюдо на фото и оцени КБЖУ. "
                                "Верни JSON: {\"Блюдо\":\"...\","
                                "\"Калории (ккал)\":N,\"Белки (г)\":N,"
                                "\"Жиры (г)\":N,\"Углеводы (г)\":N}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    return content


async def process_and_write(message: types.Message, text: str, source: str):
    try:
        entries = await parse_with_gpt(text)
    except Exception as e:
        await message.answer(f"Не удалось разобрать сообщение: {e}")
        return

    if not entries:
        await message.answer("Не понял, к какому разделу отнести запись. "
                             "Попробуй сказать: «вес 82.5», «съел овсянку», "
                             "«спал 7 часов».")
        return

    written = []
    for entry in entries:
        sheet_name = entry.get("sheet")
        values = entry.get("values", {})
        values.setdefault("Дата", datetime.now().strftime("%Y-%m-%d"))
        try:
            write_row(sheet_name, values)
            written.append(f"{sheet_name}: {values}")
        except Exception as e:
            await message.answer(f"Ошибка записи в «{sheet_name}»: {e}")

    if written:
        await message.answer("✅ Записал:\n" + "\n".join(written))


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    if not is_allowed(message):
        await message.answer("Доступ запрещён.")
        return
    await message.answer(
        "Привет! Я бот «Здоровье ИИ». Просто говори голосом или пиши:\n"
        "• «вес 82.5»\n"
        "• «позавтракал овсянкой»\n"
        "• «спал 7 часов»\n"
        "• «тренировка: жим лёжа 4х8 по 60 кг»\n"
        "• пришли фото еды — я распознаю блюдо и КБЖУ"
    )


@dp.message(F.voice)
async def on_voice(message: types.Message):
    if not is_allowed(message):
        await message.answer("Доступ запрещён.")
        return
    await message.answer("🎙 Слушаю и распознаю...")
    try:
        text = await transcribe_voice(message.voice.file_id)
    except Exception as e:
        await message.answer(f"Не удалось распознать голос: {e}")
        return
    await process_and_write(message, text, "voice")


@dp.message(F.photo)
async def on_photo(message: types.Message):
    if not is_allowed(message):
        await message.answer("Доступ запрещён.")
        return
    await message.answer("📷 Анализирую фото еды...")
    file_id = message.photo[-1].file_id
    try:
        photo_json = await analyze_photo(file_id)
        values = json.loads(photo_json)
        values.setdefault("Дата", datetime.now().strftime("%Y-%m-%d"))
        write_row("Питание", values)
        await message.answer("✅ Записал в «Питание»:\n" + str(values))
    except Exception as e:
        await message.answer(f"Не удалось распознать фото: {e}")


@dp.message(F.text)
async def on_text(message: types.Message):
    if not is_allowed(message):
        await message.answer("Доступ запрещён.")
        return
    await process_and_write(message, message.text, "text")


async def main():
    print("Бот запущен. Ctrl+C для остановки.", flush=True)
    while True:
        try:
            await dp.start_polling(
                bot,
                allowed_updates=dp.resolve_used_update_types(),
            )
        except Exception as e:
            # Сбой сети/таймаут не должен убивать бота — перезапускаем опрос
            print(f"[polling] ошибка: {e}; повтор через 5с", flush=True)
            await asyncio.sleep(5)


def start_health_server():
    """Минимальный HTTP-сервер для проверки здоровья на Render."""
    port = int(os.getenv("PORT", "3000"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=start_health_server, daemon=True).start()
    asyncio.run(main())
