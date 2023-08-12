import time
import onnx
import cv2
import onnxruntime
import numpy as np
from onnx import numpy_helper
from numpy.linalg import norm as l2norm
from utils.face_alignment import norm_crop2


class Inswapper():
    def __init__(self, model_file=None, provider=['CPUExecutionProvider'], session_options=None):
        self.model_file = model_file
        model = onnx.load(self.model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])

        self.session_options = session_options
        if self.session_options is None:
            self.session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(self.model_file, sess_options=self.session_options, providers=provider)

    def forward(self, frame, target, source, n_pass=1):
        trg, matrix = norm_crop2(frame, target['kps'], 128)

        latent = (source['embedding'] / l2norm(source['embedding'])).reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)

        blob = cv2.dnn.blobFromImage(trg, 1.0 / 255, (128, 128), (0., 0., 0.), swapRB=True)

        for _ in range(max(int(n_pass),1)):
            blob = self.session.run(['output'], {'target': blob, 'source': latent})[0]

        out = blob.transpose((0, 2, 3, 1))[0]
        out = (out * 255).clip(0,255)
        out = out.astype('uint8')[:, :, ::-1]

        del blob, latent

        return trg, out, matrix
