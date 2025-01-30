import glob
from multiprocessing import Pool
import os

from tqdm import tqdm
from tests.test_scdet import validate_single_dir
# from Utils.scdet import TransitionDetectionBase, TransitionDetectionHSV, TransitionDetectionYUV, TransitionDetectionSudo  # type: ignore
from Utils.scdet import TransitionDetectionBase, TransitionDetectionSudo  # type: ignore

class TestTransitionDetection(TransitionDetectionBase):
    def save_scene(self, title):
        pass

# ============= 示例的全局变量区（子进程需访问） =============
scdet = None  # 全局 scdet 对象引用；子进程初始化时会赋值
def init_scdet(_root, scene_queue_length=4, scdet_threshold=12, pure_scene_threshold=10,
               no_scdet=False, use_fixed_scdet=False, fixed_max_scdet=80):
    """
    Pool initializer：
    当 Pool 中的每个子进程启动时，会调用本函数一次。
    这里创建并保存好全局 scdet 对象 (包含 onnxruntime.Session)。
    """
    global scdet, root
    root = _root
    scdet = TestTransitionDetectionSudo(scene_queue_length=scene_queue_length,
                                    scdet_threshold=scdet_threshold,
                                    pure_scene_threshold=pure_scene_threshold,
                                    no_scdet=no_scdet,
                                    use_fixed_scdet=use_fixed_scdet,
                                    fixed_max_scdet=fixed_max_scdet)

class TestTransitionDetectionSudo(TransitionDetectionSudo):
    def save_scene(self, title):
        pass


# -------------------- 多进程 worker 函数 -------------------- #
def process_case(scene):
    """
    子进程中执行的函数（Pool worker），返回 (TP, FP, TN, FN)。
    这里用到全局的 scdet 对象（已在 init_scdet 中初始化）。
    """
    global scdet, root
    scene_path = os.path.join(root, scene)
    scene_dir = os.path.dirname(scene_path)
    
    # 收集同一文件夹下所有 .jpg
    pics = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
    if not os.path.isfile(scene_path):
        # 如果场景文件不存在，就当做 None
        scene = None

    return validate_single_dir(pics, scene, scdet)


# -------------------- 核心多进程 + tqdm 入口函数 -------------------- #
def run(scene_list, num_workers=2, scene_queue_length=4,
        scdet_threshold=12,
        pure_scene_threshold=10,
        no_scdet=False,
        use_fixed_scdet=False,
        fixed_max_scdet=80):
    """
    scdet_class: 传入一个转场检测类，如 TransitionDetectionSudo
    scene_list:  待处理场景名称列表
    num_workers: 进程数
    *args, **kwargs: 传入到 scdet_class 的初始化参数
    """
    # 全局计数
    TP = FP = TN = FN = 0

    # 用 multiprocessing.Pool 并行处理
    # 并在主进程中用 tqdm 显示总进度
    total_cases = len(scene_list)
    with Pool(
        processes=num_workers,
        initializer=init_scdet,
        initargs=(root, scene_queue_length, scdet_threshold,
                  pure_scene_threshold, no_scdet,
                  use_fixed_scdet, fixed_max_scdet)
    ) as pool:
        # 这里用 imap 或 imap_unordered 都可以
        # total=len(case_list) 用来让 tqdm 知道总任务数
        for (is_TP, is_FP, is_TN, is_FN) in tqdm(
            pool.imap(process_case, scene_list),
            total=total_cases,
            desc="Multiprocessing Scenes"
        ):
            TP += 1 if is_TP else 0
            FP += 1 if is_FP else 0
            TN += 1 if is_TN else 0
            FN += 1 if is_FN else 0

    # 输出结果统计
    precision = TP / max((TP + FP), 1)
    recall = TP / max((TP + FN), 1)
    accuracy = (TP + TN) / max((TP + FP + TN + FN), 1)

    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    root = r"F:\Datasets\scenes"
    with open(os.path.join(root, "train.txt"), 'r', encoding='utf-8') as f:
        scene_list = f.readlines()[:500]
    
    # 运行多进程 + tqdm
    # 示例中使用 TransitionDetectionSudo 作为 scdet_class
    run(                                                                   
        scene_list=scene_list,
        num_workers=1,                         # 可根据CPU核心数设置
        scene_queue_length=4,                  # 给 scdet_class 的示例参数
        fixed_max_scdet=80                     # ...
    )

