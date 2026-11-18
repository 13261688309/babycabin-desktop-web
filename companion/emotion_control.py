"""Trusted local visual classifier. Cry is an image label, not a diagnosis or mood measurement."""
import hashlib
from pathlib import Path

from .state import Detection,LABELS

MODEL_SHA256='2adf7e59db4a3263b209fb43f955192cd2fddae08539af04c179c9f4e44085e7'


class EmotionModel:
    def __init__(self,weights):
        path=Path(weights)
        if not path.is_file() or path.suffix!='.pt':
            raise ValueError('Trusted local emotion weights required')
        with path.open('rb') as source:
            digest=hashlib.file_digest(source,'sha256').hexdigest()
        if digest!=MODEL_SHA256:
            raise ValueError('Emotion model hash mismatch')
        from ultralytics import YOLO
        self.model=YOLO(str(path.resolve()))
        if self.model.task!='detect' or set(self.model.names.values())!=LABELS:
            raise ValueError('Expected Cry/Happy/Neutral/Sleep detection model')

    def detect(self,frame):
        result=self.model.predict(frame,conf=.05,iou=.55,imgsz=640,device='cpu',verbose=False)[0]
        if result.boxes is None:return []
        return [Detection(result.names[int(label)],float(confidence),tuple(box))
                for box,confidence,label in zip(result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]
