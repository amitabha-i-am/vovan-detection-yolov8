from ultralytics import YOLO
import cv2
import mss.tools
import numpy as np

model = YOLO('yolov8n.pt')

monitor = {"top": 0, "left": 0, "width": 960, "height": 1080}
output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)

with mss.mss() as sct:

    while True:
        sct_img = sct.grab(monitor)
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

        img = cv2.imread(output, 1)

        results = model.predict(source=img, show=True)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
