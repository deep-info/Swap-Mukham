import cv2
import onnx
import onnxruntime
import numpy as np
from tqdm import tqdm

# https://github.com/yahoo/open_nsfw

def prepare_image(img):
    img = cv2.resize(img, (224,224)).astype('float32')
    img -= np.array([104, 117, 123], dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return img

class NSFWChecker:
    def __init__(self, model_path=None, provider=["CPUExecutionProvider"], session_options=None):
        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name
        self.session_options = session_options
        if self.session_options == None:
            self.session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(model_path, sess_options=self.session_options, providers=provider)

    def check_image(self, image, threshold=0.9):
        if isinstance(image, str):
            image = cv2.imread(image)
        img = prepare_image(image)
        score = self.session.run(None, {self.input_name:img})[0][0][1]
        if score >= threshold:
            return True
        return False

    def check_video(self, video_path, threshold=0.9, max_frames=100):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames = min(total_frames, max_frames)
        indexes = np.arange(total_frames, dtype=int)
        shuffled_indexes = np.random.permutation(indexes)[:max_frames]

        for idx in tqdm(shuffled_indexes, desc="Checking"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            valid_frame, frame = cap.read()
            if valid_frame:
                img = prepare_image(frame)
                score = self.session.run(None, {self.input_name:img})[0][0][1]
                if score >= threshold:
                    cap.release()
                    return True
        cap.release()
        return False

    def check_image_paths(self, image_paths, threshold=0.9, max_frames=100):
        total_frames = len(image_paths)
        max_frames = min(total_frames, max_frames)
        indexes = np.arange(total_frames, dtype=int)
        shuffled_indexes = np.random.permutation(indexes)[:max_frames]

        for idx in tqdm(shuffled_indexes, desc="Checking"):
            frame = cv2.imread(image_paths[idx])
            img = prepare_image(frame)
            score = self.session.run(None, {self.input_name:img})[0][0][1]
            if score >= threshold:
                return True
        return False