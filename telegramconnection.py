import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode

from ultralytics import YOLO
import cv2
import time

model = YOLO('yolov8x-seg.pt')
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
VIDEO_URL = "https://camera.lipetsk.ru/ms-27.camera.lipetsk.ru/live/6c9ed6fe-7ed1-11ee-9ee9-0050568c9a93/playlist.m3u8"
COUNT = 0

dp = Dispatcher()


async def main() -> None:

    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
    cap = cv2.VideoCapture(VIDEO_URL)

    while True:
        ret, img = cap.read()
        COUNT = 0
        results = model.predict(img, show=True,)
        for i in results[0].boxes.data:
            if i[5] == 0:
                COUNT += 1
                cv2.drawMarker(img, (int(i[0]+(i[2]-i[0])/2), int((i[1]-30))), (255, 0, 0), 0, 10, 10)
                cv2.putText(img, "person", (int(i[0]), int(i[1])), 2, 1, (255, 0, 0))

        if COUNT != 0:
            cv2.imwrite('C:\\Users\\artem\\PycharmProjects\\yolov8test\\photo.png', img)
            await bot.send_photo(chat_id='-1002033456616',
                                 photo=types.FSInputFile(
                                     path="C:\\Users\\artem\\PycharmProjects\\yolov8test\\photo.png"),
                                 caption="caption")
            time.sleep(2)
            cap = cv2.VideoCapture(VIDEO_URL)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    TOKEN = input("Токен: ")
    asyncio.run(main())
