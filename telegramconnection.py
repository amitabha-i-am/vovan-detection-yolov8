import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode


from ultralytics import YOLO
import cv2
import mss.tools
import time

TOKEN = "6633049165:AAH0RwkrnAmmEhE4hKF4Ujmnl-21VxwFMJA"
model = YOLO('yolov8n.pt')
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)

dp = Dispatcher()


async def main() -> None:

    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)

    with mss.mss() as sct:
        while True:
            sct_img = sct.grab(monitor)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

            img = cv2.imread(output, 1)

            results = model.predict(img)
            for i in results[0].boxes.data:
                if i[5] == 0:
                    cv2.imwrite('C:\\Users\\artem\\PycharmProjects\\yolov8test\\photo.png', img)
                    await bot.send_photo(chat_id='-1002033456616', photo=types.FSInputFile(path="C:\\Users\\artem\\PycharmProjects\\yolov8test\\photo.png"), caption="caption" )
                    time.sleep(3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
