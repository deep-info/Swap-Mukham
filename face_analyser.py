import os
import cv2
import torch
import threading
import insightface
import numpy as np
from tqdm import tqdm
import multiprocessing
import concurrent.futures

single_face_detect_conditions = [
    "best detection",
    "left most",
    "right most",
    "top most",
    "bottom most",
    "middle",
    "biggest",
    "smallest",
]

multi_face_detect_conditions = [
    "all face",
    "specific face",
    "age less than",
    "age greater than",
    "all male",
    "all female"
]

face_detect_conditions =  multi_face_detect_conditions + single_face_detect_conditions


def get_single_face(faces, method="best detection"):
    total_faces = len(faces)
    if total_faces == 1:
        return faces[0]

    if method == "best detection":
        return sorted(faces, key=lambda face: face["det_score"])[-1]
    elif method == "left most":
        return sorted(faces, key=lambda face: face["bbox"][0])[0]
    elif method == "right most":
        return sorted(faces, key=lambda face: face["bbox"][0])[-1]
    elif method == "top most":
        return sorted(faces, key=lambda face: face["bbox"][1])[0]
    elif method == "bottom most":
        return sorted(faces, key=lambda face: face["bbox"][1])[-1]
    elif method == "middle":
        return sorted(faces, key=lambda face: (
                (face["bbox"][0] + face["bbox"][2]) / 2 - 0.5) ** 2 +
                ((face["bbox"][1] + face["bbox"][3]) / 2 - 0.5) ** 2)[len(faces) // 2]
    elif method == "biggest":
        return sorted(faces, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]))[-1]
    elif method == "smallest":
        return sorted(faces, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]))[0]

def filter_face_by_age(faces, age, method="age less than"):
    if method == "age less than":
        return [face for face in faces if face["age"] < age]
    elif method == "age greater than":
        return [face for face in faces if face["age"] > age]
    elif method == "age equals to":
        return [face for face in faces if face["age"] == age]

def cosine_distance(a, b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return 1 - np.dot(a, b)

def is_similar_face(face1, face2, threshold=0.6):
    distance = cosine_distance(face1["embedding"], face2["embedding"])
    return distance < threshold


class AnalyseFace:
    def __init__(self, name='buffalo_l', provider=["CPUExecutionProvider"]):
        self.analyser = insightface.app.FaceAnalysis(
            name=name,
            allowed_modules=['detection', 'recognition', 'genderage'],
            providers=provider
        )

    def prepare(self, detection_size=640, detection_threshold=0.6, detect_condition="best detection"):
        self.detection_size = int(detection_size)
        self.detection_threshold = float(detection_threshold)
        self.detect_condition = detect_condition
        self.analyser.prepare(ctx_id=0, det_size=(self.detection_size , self.detection_size ), det_thresh=self.detection_threshold)

    def get_faces(self, image, scale=1.):
        if isinstance(image, str):
            image = cv2.imread(image)

        faces = self.analyser.get(image)

        if scale != 1: # landmark-scale
            for i, face in enumerate(faces):
                landmark = face['kps']
                center = np.mean(landmark, axis=0)
                landmark = center + (landmark - center) * scale
                faces[i]['kps'] = landmark

        return faces

    def get_face(self, image, scale=1.):
        faces = self.get_faces(image, scale=scale)
        return get_single_face(faces, method=self.detect_condition)