import os  # Работа с операционной системой (директории, файлы).
import cv2  # OpenCV для чтения и обработки изображений.
import torch  # PyTorch для работы с нейросетевой моделью.
import numpy as np  # NumPy для операций с массивами.
import logging  # Модуль логирования.
import asyncio  # Асинхронная работа (необходимо для Telegram-бота).
from datetime import datetime  # Работа с датой и временем.

# Flask (веб-сервер), YOLO-модель (ultralytics), Telegram-бот.
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from threading import Thread  # Для запуска Flask-сервера в отдельном потоке.
from werkzeug.serving import make_server  # Внутренний WSGI-сервер для Flask.

# === Конфигурация ===
MODEL_PATH = "weights/yolo11n_master_200_tile_aug_unfreez_best.pt"  # Путь к файлу весов обученной модели YOLO.
TILE_SIZE = 1024  # Размер одного тайла (1024×1024 пикселей).
OVERLAP = 256  # Количество пикселей перекрытия между тайлами.
BOT_TOKEN = os.getenv("TG_TOKEN", "your_token_here")  # Токен Telegram-бота из переменных окружения или заглушка.
UPLOAD_DIR = "uploads"  # Директория для сохранения загруженных изображений.
RESULTS_DIR = "results"  # Директория для сохранения изображений с результатом.
LOG_FILE = "server.log"  # Файл журнала (логов).

# Создаем директории, если их нет.
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Логирование ===
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)  # Указываем уровень логирования и формат сообщений.

# Определяем устройство: CUDA (GPU) если доступен, иначе CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Загружаем модель YOLO и переносим на выбранное устройство.
model = YOLO(MODEL_PATH).to(device)

# === Инференс ===
def run_inference_on_image(img_path):
    # Логируем начало обработки.
    logging.info(f"🔍 Inference: {img_path}")
    # Считываем изображение с диска.
    original = cv2.imread(img_path)
    # Получаем высоту (h) и ширину (w) изображения.
    h, w = original.shape[:2]
    # Размер шага (stride) с учетом перекрытия.
    stride = TILE_SIZE - OVERLAP

    tiles = []   # Список для тайлов (кусочков изображения).
    coords = []  # Список для хранения (x, y) верхнего левого угла каждого тайла.

    # Разбиваем изображение на перекрывающиеся тайлы.
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Вырезаем фрагмент (тайл) из исходного изображения.
            tile = original[y:y+TILE_SIZE, x:x+TILE_SIZE]
            # Если тайл выходит за пределы изображения (на краю), добавляем черное заполнение.
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                # Создаем пустую картинку нужного размера.
                pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                # Копируем тайл в левый верхний угол новой картинки.
                pad[:tile.shape[0], :tile.shape[1]] = tile
                tile = pad
            tiles.append(tile)
            # Запоминаем координаты тайла (сдвиг относительно исходного изображения).
            coords.append((x, y))

    # Подаем список тайлов в модель для пакетной инференции.
    results = model(tiles, verbose=False)

    # Обходим каждый результат и соответствующие ему смещения.
    for r, (x0, y0) in zip(results, coords):
        # Результат r содержит рамки в формате [x1, y1, x2, y2] относительно тайла.
        for box in r.boxes.xyxy.cpu().numpy():
            # Переносим координаты рамки от тайла к оригинальному изображению.
            x1, y1, x2, y2 = map(int, box[:4] + np.array([x0, y0, x0, y0]))
            # Рисуем прямоугольник на исходном изображении.
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Формируем имя файла для сохранения результата с текущим временем.
    out_path = os.path.join(
        RESULTS_DIR,
        f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    )
    # Сохраняем итоговое изображение с рамками.
    cv2.imwrite(out_path, original)
    return out_path  # Возвращаем путь к результату.

# === Flask API ===
flask_app = Flask(__name__)  # Создаем Flask-приложение.

@flask_app.route("/", methods=["GET"])
def index():
    # Простая проверка сервера: возвращает текст, если приложение запущено.
    return "🚀 LizaAlert bot is running!"

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Endpoint для получения изображения и выполнения инференса.
    if "image" not in request.files:
        # Если нет файла в запросе, возвращаем ошибку.
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    # Проверяем расширение файла на допустимые (jpg/png).
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({"error": "Unsupported format"}), 400

    # Сохраняем полученное изображение в директорию uploads.
    img_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(img_path)
    # Запускаем инференс на сохраненном файле.
    result_path = run_inference_on_image(img_path)
    # Отправляем файл с результатом клиенту.
    return send_file(result_path, mimetype="image/jpeg")

class FlaskServer(Thread):
    def run(self):
        # Запускаем Flask-сервер в отдельном потоке на порту 7860.
        make_server("0.0.0.0", 7860, flask_app).serve_forever()

# === Telegram Bot ===
async def start(update: Update, context):
    # Обработчик команды /start: приветственное сообщение.
    logging.info(f"/start by {update.effective_user.id}")
    await update.message.reply_text("👋 Send an image for pedestrian detection.")

async def handle_image(update: Update, context):
    # Обработчик, если пользователь прислал фотографию.
    try:
        # Берем самую высокоразрешенную версию фотографии.
        photo = update.message.photo[-1]
        file = await photo.get_file()
        image_bytes = await file.download_as_bytearray()

        # Генерируем имя файла по ID пользователя и текущему времени.
        uid = update.effective_user.id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(UPLOAD_DIR, f"{uid}_{timestamp}.jpg")

        # Сохраняем фотографию локально.
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        # Запускаем инференс и отправляем результат обратно пользователю.
        result_path = run_inference_on_image(img_path)
        await update.message.reply_photo(photo=open(result_path, "rb"))

    except Exception as e:
        # В случае ошибки логируем и уведомляем пользователя.
        logging.exception("Telegram photo error")
        await update.message.reply_text("⚠️ Error processing image.")

async def handle_document(update: Update, context):
    # Обработчик, если пользователь прислал файл (document), поддерживаем картинки.
    doc = update.message.document
    if not doc.mime_type.startswith("image/"):
        # Если прислан не image-файл, просим прислать корректный.
        await update.message.reply_text("❗ Please send an image file (jpg/png).")
        return

    try:
        file = await doc.get_file()
        image_bytes = await file.download_as_bytearray()

        uid = update.effective_user.id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{uid}_{timestamp}.jpg"
        img_path = os.path.join(UPLOAD_DIR, filename)

        with open(img_path, "wb") as f:
            f.write(image_bytes)

        result_path = run_inference_on_image(img_path)
        await update.message.reply_photo(photo=open(result_path, "rb"))

    except Exception as e:
        logging.exception("Telegram document error")
        await update.message.reply_text("⚠️ Error processing document image.")

async def handle_other(update: Update, context):
    # Обработчик для любых других сообщений: просим отправить изображение.
    await update.message.reply_text("📸 Please send an image.")

# === Запуск приложений ===
async def main():
    # Запускаем Flask-сервер в отдельном потоке.
    flask_thread = FlaskServer()
    flask_thread.start()

    # Настраиваем Telegram-бота с токеном и хэндлерами.
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    app.add_handler(MessageHandler(~(filters.PHOTO | filters.Document.IMAGE), handle_other))

    # Запускаем бесконечный polling Telegram-бота.
    await app.run_polling()

if __name__ == "__main__":
    # Чтобы asyncio мог работать во встроенных средах типа Jupyter, применяем nest_asyncio.
    import nest_asyncio
    nest_asyncio.apply()

    import asyncio
    asyncio.run(main())
