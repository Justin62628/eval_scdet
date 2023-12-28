import glob
import os
import threading
from queue import Queue

import cv2
import numpy as np
import pytest
from tqdm import tqdm

# from Utils.scdet import TransitionDetectionBase, TransitionDetectionHSV


# class TestTransitionDetection(TransitionDetectionBase):
#     def save_scene(self, title):
#         pass


# class TestTransitionDetectionHSV(TransitionDetectionHSV):
#     def save_scene(self, title):
#         pass


class TestPyScdet:
    def __init__(self, method='threshold', threshold=20, downscale=2, min_scene_len=5):
        import scenedetect

        self.scdet = \
            {'threshold': scenedetect.detectors.ThresholdDetector(threshold=threshold, min_scene_len=min_scene_len),
             'content': scenedetect.detectors.ContentDetector(threshold=threshold, min_scene_len=min_scene_len),
             'adaptive': scenedetect.detectors.AdaptiveDetector(adaptive_threshold=threshold,
                                                                min_scene_len=min_scene_len)}[method]
        self.downscale = downscale
        self.cnt = 0

    def get_downscaled(self, img):
        img = cv2.resize(
            img, (round(img.shape[1] / self.downscale),
                  round(img.shape[0] / self.downscale)))
        return img

    def check_scene(self, _img1, _img2) -> bool:
        cuts = self.scdet.process_frame(self.cnt, self.get_downscaled(_img2))
        self.cnt += 1
        return len(cuts) > 0


class TestStyler:

    infer_lock = threading.Lock()

    def __init__(self, onnx_path, threshold=20, resize=224):
        import onnxruntime
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        # self.ses = onnxruntime.InferenceSession(onnx_path,
        #                                         providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
        self.ses = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.resize = resize
        self.threshold = threshold
        print(self.ses.get_inputs())

    def get_downscaled(self, img):
        img = cv2.resize(
            img, (self.resize, self.resize)
        )
        img = img.astype(np.float16) / 255.

        return img

    def check_scene(self, _img1, _img2) -> bool:

        _img1, _img2 = self.get_downscaled(_img1), self.get_downscaled(_img2)
        _input = np.concatenate((_img1, _img2), axis=2).transpose((2, 0, 1))  # (6, H, W)
        _ori_input = {self.ses.get_inputs()[0].name: _input}
        with TestStyler.infer_lock:
            _out = self.ses.run(None, _ori_input)
        return _out[0][0][0] > self.threshold


def read_img(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    if img is None:
        return
    img = img[:, :, ::-1].copy()
    return img


def validate_single_dir(pics, scene, scdet):
    im0 = read_img(pics[0])
    is_TP, is_FP, is_TN, is_FN = False, False, False, False
    is_detected = False
    prev_pic = pics[0]
    for pic in pics[1:]:
        im1 = read_img(pic)
        result = scdet.check_scene(im0, im1)
        if is_detected and result:  # 2nd time detected
            is_FP = True
            is_TP = False
            break
        if result:
            is_TP = scene is not None and os.path.basename(prev_pic) == os.path.basename(scene)
            is_FP = not is_TP
            is_detected = True
        im0 = im1
        prev_pic = pic

    if not is_detected:
        is_TN = scene is None
        is_FN = not is_TN
    return is_TP, is_FP, is_TN, is_FN


class ValCaseThread(threading.Thread):
    """
    """

    def __init__(self, scene_queue: Queue, scdet_method: str, *args, **kwargs):
        super(ValCaseThread, self).__init__()
        self.scene_queue = scene_queue
        self.scdet_method = scdet_method
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0
        self.args = args
        self.kwargs = kwargs

        self.styler_inst = None

    def get_scdet(self):

        if self.scdet_method == 'styler':
            if self.styler_inst is None:
                self.styler_inst = TestStyler(*self.args, **self.kwargs)
            return self.styler_inst

        # return {
        #         'svfi': TestTransitionDetection,
        #         'svfi_hsv': TestTransitionDetectionHSV,
        #         'pyscdet': TestPyScdet}[self.scdet_method](*self.args, **self.kwargs)

    def run(self) -> None:
        while True:
            scene = self.scene_queue.get()
            if scene is None:
                # print(f"{self.name} break")
                break
            scene = scene.strip()
            scene_path = os.path.join(root, scene)
            scene_dir = os.path.join(root, os.path.dirname(scene_path))
            # scdet = TestTransitionDetection(int(24 * 0.3), 12)
            # scdet = TestTransitionDetection(int(24 * 0.3), 12, fixed_max_scdet=30, use_fixed_scdet=True)
            # scdet = TestPyScdet('threshold')
            scdet = self.get_scdet()
            pics = glob.glob(os.path.join(scene_dir, "*.jpg"))
            if not os.path.isfile(scene_path):
                scene = None
            isTP, isFP, isTN, isFN = validate_single_dir(pics, scene, scdet)
            self.TP += 1 if isTP else 0
            self.FP += 1 if isFP else 0
            self.TN += 1 if isTN else 0
            self.FN += 1 if isFN else 0


root = r"examples"  # !!! change this line
with open(os.path.join(root, "train.txt"), 'r', encoding='utf-8') as f:
    scene_list = f.readlines()[:1000]


def run(scdet_method: str, *args, **kwargs):
    TP, FP, TN, FN = 0, 0, 0, 0

    case_queue = Queue(50)

    thread_cnt = 6
    thread_list = list()
    for i in range(thread_cnt):
        t = ValCaseThread(case_queue, scdet_method, *args, **kwargs)
        t.start()
        thread_list.append(t)

    print(f"Len: {len(scene_list)}")
    for case in tqdm(scene_list):
        case_queue.put(case)

    for case in range(thread_cnt):
        case_queue.put(None)

    for t in thread_list:
        t.join()
        TP += t.TP
        FP += t.FP
        TN += t.TN
        FN += t.FN

    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    precision = TP / max((TP + FP), 1)
    recall = TP / max((TP + FN), 1)
    accuracy = (TP + TN) / max((TP + FP + TN + FN), 1)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")


class TestScdet:
    # @pytest.mark.skip()
    @pytest.mark.parametrize('threshold', [11])
    @pytest.mark.parametrize('scdet_len', [8])
    def test_svfi_scdet(self, scdet_len, threshold):
        print(f"threshold: {threshold}, scdet_len: {scdet_len}")  # (12, 12), (8, 12)
        run('svfi', scdet_len, threshold)

    @pytest.mark.parametrize('threshold', [16])
    @pytest.mark.parametrize('scdet_len', [4])
    def test_svfi_scdet_hsv(self, scdet_len, threshold):
        print(f"threshold: {threshold}, scdet_len: {scdet_len}")  # (16, 4)
        run('svfi_hsv', scdet_len, threshold, fixed_max_scdet=60)

    @pytest.mark.parametrize('threshold', [20, 30, 50])
    @pytest.mark.skip()
    def test_fixed_scdet(self, threshold):
        run('svfi', 12, fixed_max_scdet=threshold, use_fixed_scdet=True)

    @pytest.mark.parametrize('downscale', [2])
    @pytest.mark.parametrize('min_scene_len', [5, 15])
    @pytest.mark.parametrize('threshold', [70, 120])
    @pytest.mark.skip()
    def test_pyscdet_threshold(self, threshold, min_scene_len, downscale):
        run('pyscdet', method='threshold', threshold=threshold, downscale=downscale, min_scene_len=min_scene_len)

    # @pytest.mark.skip()
    @pytest.mark.parametrize('threshold', [31])
    @pytest.mark.parametrize('downscale', [2])
    @pytest.mark.parametrize('min_scene_len', [2])
    def test_pyscdet_content(self, threshold, min_scene_len, downscale):
        print(f"threshold = {threshold}, min_scene_len = {min_scene_len}, downscale = {downscale}")  # (31, 2, 2)
        run('pyscdet', method='content', threshold=threshold, downscale=downscale, min_scene_len=min_scene_len)

    @pytest.mark.parametrize('threshold', [50, 100, 150])
    @pytest.mark.parametrize('downscale', [2])
    @pytest.mark.parametrize('min_scene_len', [5, 10])
    # @pytest.mark.skip()
    def test_pyscdet_adaptive(self, threshold, min_scene_len, downscale):
        run('pyscdet', method='adaptive', threshold=threshold, downscale=downscale, min_scene_len=min_scene_len)

    @pytest.mark.parametrize('threshold', [0.85, 0.9, 0.98])
    # @pytest.mark.parametrize('threshold', [0.85])
    @pytest.mark.parametrize('params', [
                                        ("sc_efficientformerv2_s0_12263_224_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx", 224),
                                        ("sc_efficientformerv2_s0 + rife46_flow_84119_224_CHW_6ch_clamp_softmax_op17_fp16.onnx", 224),
                                        ("sc_efficientnetv2b0_17957_256_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx", 256),
                                        ("sc_efficientnetv2b0+rife46_flow_1362_256_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx", 256),
                                        ]
                             )
    def test_styler(self, threshold, params):
        onnx_path, resize = params
        onnx_path = os.path.join(r"D:\60-fps-Project\Projects\RIFE GUI\models\scdet", onnx_path)  # !!! change this line
        run('styler',
            onnx_path=onnx_path,
            threshold=threshold, resize=resize)


if __name__ == '__main__':
    # run('svfi_hsv', 8, 20, fixed_max_scdet=200)
    # run('pyscdet', method='content', threshold=31, downscale=2, min_scene_len=2)
    # !!! change this line
    run('styler', onnx_path=r"D:\60-fps-Project\Projects\RIFE GUI\models\scdet\sc_efficientformerv2_s0_12263_224_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx", threshold=0.9, resize=224)
    pass
