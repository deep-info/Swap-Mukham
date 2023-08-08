import cv2
import torch
import numpy as np

from utils.device import get_device_and_provider
from utils.face_alignment import get_cropped_head
from utils.image import paste_to_whole, mix_two_image

from face_swapper import Inswapper
from face_parsing import FaceParser
from face_upscaler import get_available_upscalers_names, cv2_upscalers, load_face_upscaler
from face_analyser import AnalyseFace, single_face_detect_conditions, face_detect_conditions, get_single_face, is_similar_face

model_paths = {
    "inswapper":"assets/pretrained_models/inswapper_128.onnx",
    "faceparser":"assets/pretrained_models/faceparser.onnx"
}

class SwapMukham:
    def __init__(self, device='cpu'):
        self.load_face_swapper(device=device)
        self.load_face_analyser(device=device)
        # self.load_face_parser(device=device)
        # self.load_face_upscaler(device=device)

        self.face_parser = None
        self.face_upscaler = None
        self.face_upscaler_name = ""

    def set_values(self, args):
        self.age = args.get('age', 0)
        self.detect_condition = args.get('detect_condition', "left most")
        self.similarity = args.get('similarity', 0.6)
        self.swap_condition = args.get('swap_condition', 'left most')
        self.face_scale = args.get('face_scale', 1.0)
        self.num_of_pass = args.get('num_of_pass', 1)
        self.mask_crop_values = args.get('mask_crop_values', (0,0,0,0))
        self.mask_erode_amount = args.get('mask_erode_amount', 0.1)
        self.mask_blur_amount = args.get('mask_blur_amount', 0.1)
        self.use_laplacian_blending = args.get('use_laplacian_blending', False)
        self.use_face_parsing = args.get('use_face_parsing', False)
        self.face_parse_regions = args.get('face_parse_regions', [1,2,3,4,5,10,11,12,13])
        self.face_upscaler_opacity = args.get('face_upscaler_opacity', 1.)

    def load_face_swapper(self, device='cpu'):
        device, provider = get_device_and_provider(device=device)
        self.swapper = Inswapper(model_file=model_paths['inswapper'], provider=provider)
        print(f"[{device}] Face swapper model loaded.")

    def load_face_analyser(self, device='cpu'):
        device, provider = get_device_and_provider(device=device)
        self.analyser = AnalyseFace(name='buffalo_l', provider=provider)
        self.analyser.prepare()
        print(f"[{device}] Face detection & recognition model loaded.")

    def load_face_parser(self, device='cpu'):
        device, provider = get_device_and_provider(device=device)
        self.face_parser = FaceParser(model_path=model_paths['faceparser'], provider=provider)
        print(f"[{device}] Face parsing model loaded.")

    def load_face_upscaler(self, name, device='cpu'):
        device, provider = get_device_and_provider(device=device)
        if name in get_available_upscalers_names():
            self.face_upscaler = load_face_upscaler(name=name, provider=provider)
            self.face_upscaler_name = name
            print(f"[{device}] Face upscaler model loaded.")
        else:
            self.face_upscaler_name = ""
            self.face_upscaler = None

    def collect_heads(self, frame):
        faces = self.analyser.get_faces(frame)
        return [get_cropped_head(frame, face.kps) for face in faces if face["det_score"] > 0.5]

    def analyse_source_faces(self, source_specific):
        analysed_source_specific = []
        for i, (source, specific) in enumerate(source_specific):
            if source is not None:
                analysed_source = self.analyser.get_face(source)
                if specific is not None:
                    analysed_specific = self.analyser.get_face(specific)
                else:
                    analysed_specific = None
                analysed_source_specific.append((analysed_source, analysed_specific))
        self.analysed_source_specific = analysed_source_specific

    def process_frame(self, data):
        frame, custom_mask = data
        overwrite = False
        frame_path = None
        if isinstance(frame, str):
            overwrite = True
            frame_path = frame
            frame = cv2.imread(frame_path)
        if isinstance(custom_mask, str):
            custom_mask = cv2.imread(custom_mask)
        analysed_target_faces = self.analyser.get_faces(frame, scale=self.face_scale)
        _frame = frame.copy()
        condition = self.swap_condition

        for analysed_target in analysed_target_faces:
            if (condition == "all face" or
                (condition == "age less than" and analysed_target["age"] < args['age']) or
                (condition == "age greater than" and analysed_target["age"] > args['age']) or
                (condition == "all Male" and analysed_target["gender"] == 1) or
                (condition == "all Female" and analysed_target["gender"] == 0)):

                trg_face = analysed_target
                src_face = self.analysed_source_specific[0][0]
                _frame = self.swap_face(_frame, trg_face, src_face)

            elif condition == "specific face":
                for analysed_source, analysed_specific in self.analysed_source_specific:
                    if is_similar_face(analysed_specific, analysed_target, threshold=self.similarity):
                        trg_face = analysed_target
                        src_face = analysed_source
                        _frame = self.swap_face(_frame, trg_face, src_face)

        if condition in single_face_detect_conditions and len(analysed_target_faces) > 0:
            analysed_target = get_single_face(analysed_target_faces, method=condition)
            trg_face = analysed_target
            src_face = self.analysed_source_specific[0][0]
            _frame = self.swap_face(_frame, trg_face, src_face)

        if custom_mask is not None:
            _mask = custom_mask
            _frame = _mask * frame.astype('float32') + (1 - _mask) * _frame.astype('float32')
            _frame = _frame.clip(0,255).astype('uint8')
        if overwrite:
            cv2.imwrite(frame, _frame)
        else:
            return _frame

    def swap_face(self, frame, trg_face, src_face):
        generated_face, matrix = self.swapper.forward(frame, trg_face, src_face, n_pass=self.num_of_pass)
        upscaled_face, matrix = self.upscale_face(generated_face, matrix)
        mask = self.face_parsed_mask(upscaled_face)
        result = paste_to_whole(
            upscaled_face,
            frame,
            matrix,
            mask=mask,
            crop_mask=self.mask_crop_values,
            blur_amount=self.mask_blur_amount,
            erode_amount = self.mask_erode_amount
        )
        return result

    def upscale_face(self, face, matrix):
        _face = cv2.resize(face, (512,512))
        if self.face_upscaler is not None:
            model, runner = self.face_upscaler
            face = runner(face, model)
        upscaled_face = cv2.resize(face, (512,512))
        upscaled_face = mix_two_image(_face, upscaled_face, self.face_upscaler_opacity)
        return upscaled_face, matrix / 0.25

    def face_parsed_mask(self, face):
        if self.face_parser is not None and self.use_face_parsing:
            mask = self.face_parser.parse(face, regions=self.face_parse_regions)
        else:
            mask = None
        return mask

if __name__ == "__main__":
    SM = SwapMukham(device='cuda')
    args = {'swap_condition': "specific face"}
    SM.set_values(args)
    SM.load_face_upscaler()
    source_faces = [("src3.png", "vlcsnap-2023-07-01-22h39m24s796.png")]
    SM.analyse_source_faces(source_faces)
    SM.process_frame("vlcsnap-2023-07-01-22h33m48s181.png")