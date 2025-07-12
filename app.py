import os
import cv2
import torch
import numpy as np
import logging
import asyncio
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from threading import Thread
from werkzeug.serving import make_server

# === Конфигурация ===
MODEL_PATH = "weights/yolo11n_master_200_tile_aug_unfreez_best.pt"
TILE_SIZE = 1024
OVERLAP = 256
BOT_TOKEN = os.getenv("TG_TOKEN", "your_token_here")
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
LOG_FILE = "server.log"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Логирование ===
logging.basicConfig(level=logging.INFO, filename=LOG_FILE, filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(MODEL_PATH).to(device)

# === Инференс ===
def run_inference_on_image(img_path):
    logging.info(f"🔍 Inference: {img_path}")
    original = cv2.imread(img_path)
    h, w = original.shape[:2]
    stride = TILE_SIZE - OVERLAP

    tiles, coords = [], []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = original[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                pad[:tile.shape[0], :tile.shape[1]] = tile
                tile = pad
            tiles.append(tile)
            coords.append((x, y))

    results = model(tiles, verbose=False)

    for r, (x0, y0) in zip(results, coords):
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4] + np.array([x0, y0, x0, y0]))
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 0, 255), 2)

    out_path = os.path.join(RESULTS_DIR, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(out_path, original)
    return out_path

# === Flask ===
flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
def index():
    return "🚀 LizaAlert bot is running!"

@flask_app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({"error": "Unsupported format"}), 400

    img_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(img_path)
    result_path = run_inference_on_image(img_path)
    return send_file(result_path, mimetype="image/jpeg")

class FlaskServer(Thread):
    def run(self):
        make_server("0.0.0.0", 7860, flask_app).serve_forever()

# === Telegram ===
async def start(update: Update, context):
    logging.info(f"/start by {update.effective_user.id}")
    await update.message.reply_text("👋 Send an image for pedestrian detection.")

async def handle_image(update: Update, context):
    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        image_bytes = await file.download_as_bytearray()

        uid = update.effective_user.id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(UPLOAD_DIR, f"{uid}_{timestamp}.jpg")

        with open(img_path, "wb") as f:
            f.write(image_bytes)

        result_path = run_inference_on_image(img_path)
        await update.message.reply_photo(photo=open(result_path, "rb"))

    except Exception as e:
        logging.exception("Telegram photo error")
        await update.message.reply_text("⚠️ Error processing image.")

async def handle_document(update: Update, context):
    doc = update.message.document
    if not doc.mime_type.startswith("image/"):
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
    await update.message.reply_text("📸 Please send an image.")

# === Запуск ===
async def main():
    flask_thread = FlaskServer()
    flask_thread.start()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    app.add_handler(MessageHandler(~(filters.PHOTO | filters.Document.IMAGE), handle_other))
    await app.run_polling()

  

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    import asyncio
    asyncio.run(main())
