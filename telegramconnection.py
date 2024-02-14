import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode

from ultralytics import YOLO
import cv2
import time

TOKEN = "6633049165:AAH0RwkrnAmmEhE4hKF4Ujmnl-21VxwFMJA"
model = YOLO('yolov8n.pt')
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)
VIDEO_URL = "https://camera.lipetsk.ru/ms-27.camera.lipetsk.ru/live/6c9ed6fe-7ed1-11ee-9ee9-0050568c9a93/playlist.m3u8"


dp = Dispatcher()


async def main() -> None:

    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
    cap = cv2.VideoCapture(VIDEO_URL)

    while True:
        ret, img = cap.read()

        results = model.predict(img)
        for i in results[0].boxes.data:
            if i[5] == 0:
                cv2.imwrite('C:\\Users\\artem\\PycharmProjects\\yolov8test\\photo.png', img)
                await bot.send_photo(chat_id='-1002033456616', photo=types.FSInputFile(path="C:\\Users\\artem\\PycharmProjects\\yolov8test\\photo.png"), caption="caption" )
                time.sleep(3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
