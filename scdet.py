import os
import threading
import traceback
from collections import deque
from queue import Queue
from typing import Dict, Any, List, Tuple, Union

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import onnxruntime
import onnx

from Utils.StaticParameters import CV2_INTER
from Utils.utils import Tools, get_global_settings_from_local_jsons
from Utils.StaticParameters import appDir
GAS = get_global_settings_from_local_jsons()


class TransitionDetectionBase:
    """
    转场检测的基础类，提供了核心的差值计算、检测逻辑以及转场场景保存等功能。

    Attributes:
        scene_stack_len (int): 用于转场识别的帧队列长度
        absdiff_queue (collections.deque): 存储连续帧差值的队列
        black_scene_queue (collections.deque): 用于检测开头黑场的队列
        scene_checked_queue (collections.deque): 已经判断过的帧差值队列
        scdet_threshold (float): 非固定阈值模式下的转场检测阈值
        pure_scene_threshold (float): 纯黑场识别阈值
        no_scdet (bool): 是否关闭转场检测
        use_fixed_scdet (bool): 是否启用固定阈值的转场检测
        dead_thres (float): 当帧差值大于此值直接判定为转场
        born_thres (float): 判定为转场的最小阈值
        scdet_cnt (int): 已检测到的转场次数计数
        img1 (np.ndarray): 当前帧
        img2 (np.ndarray): 下一帧
        scedet_info (dict): 用于记录转场信息的数据结构
        norm_resize (Tuple[int, int]): 在默认差值计算中，所采用的图像标准resize大小
    """

    def __init__(
        self,
        scene_queue_length: int,
        scdet_threshold: float = 12,
        pure_scene_threshold: float = 10,
        no_scdet: bool = False,
        use_fixed_scdet: bool = False,
        fixed_max_scdet: float = 80
    ) -> None:
        """
        初始化转场检测基础类。

        Args:
            scene_queue_length (int): 用于转场识别的帧队列长度
            scdet_threshold (float): 非固定转场识别模式下的阈值
            pure_scene_threshold (float): 纯黑场识别阈值
            no_scdet (bool): 是否禁用转场检测
            use_fixed_scdet (bool): 是否使用固定阈值转场识别
            fixed_max_scdet (float): 在所有转场识别模式下的死值，当use_fixed_scdet为True时，即固定转场阈值
        """
        self.scene_stack_len: int = scene_queue_length
        self.absdiff_queue: deque = deque(maxlen=self.scene_stack_len)
        self.black_scene_queue: deque = deque(maxlen=self.scene_stack_len)
        self.scene_checked_queue: deque = deque(maxlen=self.scene_stack_len // 2)

        self.scdet_threshold: float = scdet_threshold
        self.pure_scene_threshold: float = pure_scene_threshold

        self.no_scdet: bool = no_scdet
        self.use_fixed_scdet: bool = use_fixed_scdet

        # 死值（对于极大的帧差值可以直接判定为转场）
        self.dead_thres: float = fixed_max_scdet
        # 最小判定阈值
        self.born_thres: float = 2.0

        self.scdet_cnt: int = 0  # 记录已检测到的转场数

        self.img1: Union[np.ndarray, None] = None
        self.img2: Union[np.ndarray, None] = None

        # 记录转场信息的字典
        self.scedet_info: Dict[str, Union[int, List[int]]] = {
            "scene": 0,
            "normal": 0,
            "dup": 0,
            "recent_scene": -1,
            "scene_list": []
        }

        # 默认归一化后用于差分比较的分辨率
        self.norm_resize: Tuple[int, int] = (300, 300)
        self.debug_scdet = GAS.get("debug_scdet", False)

    def get_diff(self, img0: np.ndarray, img1: np.ndarray) -> float:
        """
        获取 img0 与 img1 之间的差异度，可以被子类重写（例如使用HSV或其他方法）。

        Args:
            img0 (np.ndarray): 帧0图像数据
            img1 (np.ndarray): 帧1图像数据

        Returns:
            float: 差异值
        """
        return self._get_norm_img_diff(img0, img1)

    @staticmethod
    def _get_u1_from_u2_img(img: np.ndarray) -> np.ndarray:
        """
        将可能的浮点图像转换为uint8范围 [0, 255]。默认使用cv2.normalize。

        Args:
            img (np.ndarray): 输入图像

        Returns:
            np.ndarray: 转换或归一化后的图像
        """
        if img.dtype != np.uint8:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return img

    def _get_norm_img(self, img: np.ndarray) -> np.ndarray:
        """
        对图像进行resize和直方图均衡化处理。

        Args:
            img (np.ndarray): 输入图像

        Returns:
            np.ndarray: 归一化及直方图均衡化后的图像
        """
        img = self._get_u1_from_u2_img(img)
        # 这里是原始代码的简易策略：分辨率很大时，做更粗糙的降采样；反之降采样一半
        # 可根据实际情况修改
        if img.shape[0] > 1000:
            img = img[::4, ::4, 0]
        else:
            img = img[::2, ::2, 0]

        img = cv2.resize(img, self.norm_resize)
        return cv2.equalizeHist(img)

    def _check_pure_img(self, img: np.ndarray) -> bool:
        """
        判断图像是否是纯黑场（或很低亮度变化）。

        Args:
            img (np.ndarray): 输入图像

        Returns:
            bool: 如果方差小于pure_scene_threshold则认为是黑场
        """
        try:
            # 仅取通道0的1/16像素用于方差判断
            return np.var(img[::4, ::4, 0]) < self.pure_scene_threshold
        except Exception:
            return False

    def _get_norm_img_diff(self, img0: np.ndarray, img1: np.ndarray) -> float:
        """
        默认的帧差分策略：对输入帧做一定降采样、归一化、直方图均衡后求绝对差平均值。

        Args:
            img0 (np.ndarray): 帧0图像数据
            img1 (np.ndarray): 帧1图像数据

        Returns:
            float: 差异值
        """
        # 先简单判断是否几乎完全相同
        if np.allclose(img0[::4, ::4, 0], img1[::4, ::4, 0]):
            return 0.0

        norm0 = self._get_norm_img(img0)
        norm1 = self._get_norm_img(img1)
        diff_value: float = float(cv2.absdiff(norm0, norm1).mean())
        return diff_value

    def _check_coef(self) -> Tuple[float, float]:
        """
        使用线性回归来拟合 absdiff_queue，得到斜率coef与截距intercept。

        Returns:
            Tuple[float, float]: (coef, intercept)
        """
        x_vals = np.array(range(len(self.absdiff_queue))).reshape(-1, 1)
        y_vals = np.array(self.absdiff_queue).reshape(-1, 1)

        reg = LinearRegression()
        reg.fit(x_vals, y_vals)

        # reg.coef_ 是一个二维数组，如 [[coef]]
        coef_val = float(reg.coef_[0][0])
        intercept_val = float(reg.intercept_[0])
        return coef_val, intercept_val

    def _check_var(self) -> float:
        """
        计算absdiff_queue与其线性回归值之差的方差的0.65次方。

        Returns:
            float: 差值方差^(0.65)
        """
        coef, intercept = self._check_coef()
        coef_array = coef * np.array(range(len(self.absdiff_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.absdiff_queue, dtype=np.float32)
        residual_array = diff_array - coef_array
        return float((residual_array.var()) ** 0.65)

    def _judge_mean(self, diff: float) -> bool:
        """
        判断当前diff是否构成转场。核心依据是方差变化量及死值判断等。

        Args:
            diff (float): 当前帧差值

        Returns:
            bool: 是否检测到转场
        """
        var_before = self._check_var()
        self.absdiff_queue.append(diff)
        var_after = self._check_var()

        # 如果方差差值大于scdet_threshold，且当前diff大于born_thres => 认为检测到转场
        if (var_after - var_before > self.scdet_threshold) and (diff > self.born_thres):
            self.scdet_cnt += 1
            self.save_scene(
                f"diff: {diff:.3f}, var_a: {var_before:.3f}, var_b: {var_after:.3f}, cnt: {self.scdet_cnt}"
            )
            self.absdiff_queue.clear()
            self.scene_checked_queue.append(diff)
            return True
        else:
            # 如果当前diff大于dead_thres，直接判定为转场（硬切）
            if diff > self.dead_thres:
                self.absdiff_queue.clear()
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
                self.scene_checked_queue.append(diff)
                return True
            return False

    def save_scene(self, title: str) -> None:
        """
        保存转场相关的帧对比信息（如拼接图、差值大小等）。需在子类中实现。

        Args:
            title (str): 在图像上显示的文本
        """
        raise NotImplementedError("请在子类中实现具体的保存逻辑。")

    def check_scene(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        add_diff: bool = False,
        no_diff: bool = False,
        use_diff: float = -1.0,
        **kwargs: Any
    ) -> bool:
        """
        主要接口：检查img1与img2之间是否构成转场。
        当 use_diff != -1 时优先使用外部给定diff值，否则内部计算。

        Args:
            img1 (np.ndarray): 当前帧图像
            img2 (np.ndarray): 下一帧图像
            add_diff (bool): 是否仅在队列还没塞满时先加入diff（适合初始阶段）
            no_diff (bool): 是否丢弃队列最旧的diff
            use_diff (float): 如果不为-1，则直接使用此值作为差值
            **kwargs (Any): 额外的可选参数，可根据需要扩展

        Returns:
            bool: 是否检测到转场
        """
        if self.no_scdet:
            return False

        self.img1 = img1.copy()
        self.img2 = img2.copy()

        # 计算diff，或使用外部给定的diff
        diff = use_diff if use_diff != -1 else self.get_diff(self.img1, self.img2)

        if self.debug_scdet:
            # DEBUG
            debug_str = " ".join([f"{i:.3f}" for i in self.absdiff_queue])
            
            mean_before = np.mean(self.absdiff_queue) if len(self.absdiff_queue) else 0.1
            diff_ratio = diff - mean_before
            debug_str = f"{diff:.3f} {diff_ratio:.3f} [{debug_str}]"
            
            cv2.putText(
                img1,
                debug_str,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        # 如果使用固定阈值的转场检测模式
        if self.use_fixed_scdet:
            if diff < self.dead_thres:
                return False
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Fix Scdet, cnt: {self.scdet_cnt}")
            return True

        # 检测是否是黑场： diff近似为0，且图像方差非常小
        if diff < 0.001:
            if self._check_pure_img(self.img1):
                self.black_scene_queue.append(0)
            return False
        elif len(self.black_scene_queue) and np.mean(self.black_scene_queue) == 0:
            # 如果前面若干帧全是黑场，此时出现非黑场，则识别为场切
            self.black_scene_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Pure Scene, cnt: {self.scdet_cnt}")
            return True

        # 检测硬切场景
        if diff > self.dead_thres:
            # self.absdiff_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
            self.scene_checked_queue.append(diff)
            return True

        # 若队列还没达到满长度，或者add_diff为True，则将当前差值加入队列
        if len(self.absdiff_queue) < self.scene_stack_len or add_diff:
            if diff not in self.absdiff_queue:
                self.absdiff_queue.append(diff)
            return False

        # 当 no_diff=True 时，尝试弹出一个最旧元素，避免把队列塞满后过早做判断
        if no_diff and len(self.absdiff_queue):
            self.absdiff_queue.pop()
            if not len(self.absdiff_queue):
                return False

        # 到这里才进行基于方差变化量的转场判断
        result = self._judge_mean(diff)

        if self.debug_scdet:
            # DEBUG
            if result:
                cv2.putText(
                    img1,
                    "scene",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
        return result

    def update_scene_status(self, recent_scene: int, scene_type: str) -> None:
        """
        更新转场检测的计数信息。

        Args:
            recent_scene (int): 最近一次转场的索引/ID
            scene_type (str): 场景类型标记，比如'scene'/'normal'/'dup'
        """
        self.scedet_info[scene_type] += 1
        if scene_type == "scene":
            self.scedet_info["recent_scene"] = recent_scene
            # 记录到列表中
            if isinstance(self.scedet_info["scene_list"], list):
                self.scedet_info["scene_list"].append(recent_scene)

    def get_scene_status(self) -> Dict[str, Union[int, List[int]]]:
        """
        获取当前转场检测的统计信息。

        Returns:
            dict: 包含scene/normal/dup等计数和最近场景信息
        """
        return self.scedet_info


class TransitionDetectionSudo(TransitionDetectionBase):
    
    infer_lock = threading.Lock()

    def __init__(self,
        # base
        scene_queue_length: int,
        scdet_threshold: float = 12,
        pure_scene_threshold: float = 10,
        no_scdet: bool = False,
        use_fixed_scdet: bool = False,
        fixed_max_scdet: float = 80,
        ):
        super().__init__(
            scene_queue_length=scene_queue_length,
            scdet_threshold=scdet_threshold,
            pure_scene_threshold=pure_scene_threshold,
            no_scdet=no_scdet,
            use_fixed_scdet=use_fixed_scdet,
            fixed_max_scdet=fixed_max_scdet * 1.125 # 80 -> 90
        )
        onnx_path = GAS.get("scdet_onnx_path", r"models/scdet/sc_efficientformerv2_s0_12263_224_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx")
        onnx_path = os.path.join(appDir, onnx_path)
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        self.ses = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.resize = GAS.get("scdet_onnx_resize", 224)
        self.dom_ratio = GAS.get("scdet_onnx_dom_ratio", 10)
        self.born_thres: float = 1.0

    def get_downscaled(self, img):
        img = self._get_u1_from_u2_img(img)
        img = cv2.resize(
            img, (self.resize, self.resize)
        )
        img = img.astype(np.float16) / 255.
        return img
    
    def get_diff(self, img0: np.ndarray, img1: np.ndarray) -> float:
        with self.infer_lock:
            _img1, _img2 = self.get_downscaled(img0), self.get_downscaled(img1)
            _input = np.concatenate((_img1, _img2), axis=2).transpose((2, 0, 1))  # (6, H, W)
            _ori_input = {self.ses.get_inputs()[0].name: _input}
            _out = self.ses.run(None, _ori_input)
            return float(abs(_out[0][0][0])) * 100
    
    def _judge_mean(self, diff: float) -> bool:
        """
        判断当前diff是否构成转场。核心依据是方差变化量及死值判断等。

        Args:
            diff (float): 当前帧差值

        Returns:
            bool: 是否检测到转场
        """
        var_before = self._check_var()
        mean_before = np.mean(self.absdiff_queue)
        self.absdiff_queue.append(diff)
        var_after = self._check_var()

        # 如果方差差值大于scdet_threshold，且当前diff大于born_thres => 认为检测到转场
        if (((var_after - var_before) > self.scdet_threshold)) or (
                diff > mean_before * self.dom_ratio and diff > self.born_thres
            ):
            self.scdet_cnt += 1
            self.save_scene(
                f"diff: {diff:.3f}, mb: {mean_before:.3f}, var_a: {var_before:.3f}, var_b: {var_after:.3f}, cnt: {self.scdet_cnt}"
            )
            self.absdiff_queue.pop()
            self.scene_checked_queue.append(diff)
            return True
        else:
            # 如果当前diff大于dead_thres，直接判定为转场（硬切）
            if diff > self.dead_thres:
                self.absdiff_queue.pop()
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
                self.scene_checked_queue.append(diff)
                return True
            return False


class TransitionDetectionHSV(TransitionDetectionBase):
    """
    基于HSV色彩空间进行平均像素差值的转场检测器。
    与父类的主要差异在于 get_diff() 函数会先转换图像到HSV，再计算三通道差。
    """
    hsl_weight: List[float] = [1.0, 1.0, 1.0]

    @staticmethod
    def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
        """
        计算 left 与 right 在像素值层面的平均绝对误差。
        要求二者形状一致并且是二维8位图像（单通道）或已拆分为通道的三维特定slice。

        Args:
            left (np.ndarray): 图像或图像通道数据
            right (np.ndarray): 另一图像或图像通道数据

        Returns:
            float: 像素平均误差
        """
        # left, right: shape(h, w)
        num_pixels = float(left.shape[0] * left.shape[1])
        return float(np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)

    def get_img_hsv(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取图像的HSV三通道。

        Args:
            img (np.ndarray): 输入图像，要求至少有3通道(BGR)

        Returns:
            (hue, sat, lum): 三个通道的numpy数组
        """
        img = self._get_u1_from_u2_img(img)

        # 与父类类似的降采样处理
        if img.shape[0] > 1000:
            img = img[::4, ::4, :]
        else:
            img = img[::2, ::2, :]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat, lum = cv2.split(hsv)
        return hue, sat, lum

    def get_diff(self, img0: np.ndarray, img1: np.ndarray) -> float:
        """
        通过HSV三通道的平均距离得到转场差值。

        Args:
            img0 (np.ndarray): 帧0图像数据
            img1 (np.ndarray): 帧1图像数据

        Returns:
            float: 差异值
        """
        hsl0 = self.get_img_hsv(img0)
        hsl1 = self.get_img_hsv(img1)

        # 分别对H/S/V三通道做mean_pixel_distance
        score_components = [
            self.mean_pixel_distance(hsl0[i], hsl1[i]) for i in range(3)
        ]
        # 根据hsl_weight加权求和
        total_weight = sum(abs(w) for w in self.hsl_weight)
        weighted_sum = sum(
            c * w for c, w in zip(score_components, self.hsl_weight)
        )
        return float(weighted_sum / total_weight)

STDB = TransitionDetectionSudo if GAS.get("use_scdet_onnx", True) else TransitionDetectionHSV
class SvfiTransitionDetection(STDB):
    """
    SVFI场景下的转场检测器，可在检测到转场后将关键帧或对比图保存至指定目录。
    """

    def __init__(
        self,
        project_dir: str,
        scene_queue_length: int,
        scdet_threshold: float = 16,
        pure_scene_threshold: float = 10,
        no_scdet: bool = False,
        use_fixed_scdet: bool = False,
        fixed_max_scdet: float = 120,
        scdet_output: bool = False
    ) -> None:
        """
        Args:
            project_dir (str): 项目主目录，用于存储转场截图
            scene_queue_length (int): 用于转场识别的帧队列长度
            scdet_threshold (float): 非固定转场识别模式下的阈值
            pure_scene_threshold (float): 纯黑场识别阈值
            no_scdet (bool): 是否禁用转场检测
            use_fixed_scdet (bool): 是否使用固定转场识别模式
            fixed_max_scdet (float): 当use_fixed_scdet=True时的死值阈值
            scdet_output (bool): 是否将检测结果以图像形式输出
        """
        super().__init__(
            scene_queue_length=scene_queue_length,
            scdet_threshold=scdet_threshold,
            pure_scene_threshold=pure_scene_threshold,
            no_scdet=no_scdet,
            use_fixed_scdet=use_fixed_scdet,
            fixed_max_scdet=fixed_max_scdet
        )

        self.scene_dir: str = os.path.join(project_dir, "scene")
        os.makedirs(self.scene_dir, exist_ok=True)

        self.scene_stack: Queue = Queue(maxsize=scene_queue_length)
        self.scdet_output: bool = scdet_output

        # 假设某些项目内的常量
        try:
            from Utils.StaticParameters import RGB_TYPE
            self.rgb_size = RGB_TYPE.SIZE
        except ImportError:
            # 默认用255
            self.rgb_size = 255

    def save_scene(self, title: str) -> None:
        """
        将当前帧与下一帧拼接并保存到scene目录下，用于调试或可视化。

        Args:
            title (str): 标题信息，可写入图像中。
        """
        if not self.scdet_output:
            return

        try:
            comp_stack = np.hstack((self.img1, self.img2))
            new_h = int(960 * comp_stack.shape[0] / comp_stack.shape[1])
            comp_stack = cv2.resize(
                comp_stack, (960, new_h), interpolation=CV2_INTER
            )

            # 显示标题
            cv2.putText(
                comp_stack,
                title,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (int(self.rgb_size), 0, 0),
                2
            )

            # 根据场景类型命名文件
            title_lower = title.lower()
            if "pure" in title_lower:
                filename = f"{self.scdet_cnt:08d}_pure.png"
            elif "band" in title_lower:
                filename = f"{self.scdet_cnt:08d}_band.png"
            else:
                filename = f"{self.scdet_cnt:08d}.png"

            path = os.path.join(self.scene_dir, filename)
            if os.path.exists(path):
                os.remove(path)

            return Tools.write_image(comp_stack, path, ".png")
        except Exception:
            traceback.print_exc()


class VsTransitionDetection(TransitionDetectionHSV):
    """
    VS场景的转场检测器，与SVFI类似，只是输出方式稍有不同。
    """

    def __init__(
        self,
        project_dir: str,
        scene_queue_length: int,
        scdet_threshold: float = 16,
        no_scdet: bool = False,
        use_fixed_scdet: bool = False,
        fixed_max_scdet: float = 60,
        scdet_output: bool = False
    ) -> None:
        super().__init__(
            scene_queue_length=scene_queue_length,
            scdet_threshold=scdet_threshold,
            no_scdet=no_scdet,
            use_fixed_scdet=use_fixed_scdet,
            fixed_max_scdet=fixed_max_scdet
        )
        self.scene_dir: str = os.path.join(project_dir, "scene")
        os.makedirs(self.scene_dir, exist_ok=True)

        self.scdet_output: bool = scdet_output

    def save_scene(self, title: str) -> None:
        """
        将当前帧与下一帧拼接后并以PNG格式保存到scene目录下，用于调试或可视化。

        Args:
            title (str): 标题信息，用于在图像上做标注。
        """
        if not self.scdet_output:
            return

        try:
            comp_stack = np.hstack((self.img1, self.img2))
            new_h = int(960 * comp_stack.shape[0] / comp_stack.shape[1])
            comp_stack = cv2.resize(comp_stack, (960, new_h))

            cv2.putText(
                comp_stack,
                title,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            title_lower = title.lower()
            if "pure" in title_lower:
                filename = f"{self.scdet_cnt:08d}_pure.png"
            elif "band" in title_lower:
                filename = f"{self.scdet_cnt:08d}_band.png"
            else:
                filename = f"{self.scdet_cnt:08d}.png"

            path = os.path.join(self.scene_dir, filename)
            if os.path.exists(path):
                os.remove(path)

            # 部分Windows环境可能需要 tofile 才不会被编码为乱码
            enc_img = cv2.imencode('.png', cv2.cvtColor(comp_stack, cv2.COLOR_RGB2BGR))[1]
            enc_img.tofile(path)

        except Exception:
            traceback.print_exc()
