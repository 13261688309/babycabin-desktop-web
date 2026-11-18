"""Optional YuNet candidate for local face-box detection, not identification."""

import hashlib
import math
from pathlib import Path


MODEL_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
INPUT_SIZE = (320, 320)
SCORE_THRESHOLD = .8


def unpad_boxes(rows, width, height, scaled_width, scaled_height, left, top):
    result = []
    if rows is None:
        return result
    sx, sy = width / scaled_width, height / scaled_height
    for row in rows:
        x, y, w, h = map(float, row[:4])
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            continue
        x0, y0 = max(0., (x-left)*sx), max(0., (y-top)*sy)
        x1, y1 = min(width, (x+w-left)*sx), min(height, (y+h-top)*sy)
        if x0 < x1 and y0 < y1:
            result.append((x0, y0, float(x1), float(y1)))
    return result


class YuNetFaceDetector:
    def __init__(self, weights):
        import cv2
        import numpy as np

        with Path(weights).open('rb') as model:
            payload = model.read(256 * 1024)
        if hashlib.sha256(payload).hexdigest() != MODEL_SHA256:
            raise ValueError('Unexpected YuNet weights')
        self.cv2, self.np = cv2, np
        # Load a buffer so OpenCV does not have to open a Unicode Windows path.
        self.detector = cv2.FaceDetectorYN.create('onnx', payload, b'', INPUT_SIZE,
                                                   SCORE_THRESHOLD, .3, 5000)

    def detect(self, frame):
        height, width = frame.shape[:2]
        target_w, target_h = INPUT_SIZE
        scale = min(target_w / width, target_h / height)
        w, h = max(1, round(width*scale)), max(1, round(height*scale))
        left, top = (target_w-w)//2, (target_h-h)//2
        if (w, h) == (width, height):
            resized = frame
        else:
            resized = self.cv2.resize(frame, (w, h), interpolation=(
                self.cv2.INTER_AREA if scale < 1 else self.cv2.INTER_LINEAR))
        canvas = self.np.zeros((target_h, target_w, 3), dtype=self.np.uint8)
        canvas[top:top+h, left:left+w] = resized
        _, rows = self.detector.detect(canvas)
        return unpad_boxes(rows, width, height, w, h, left, top)
