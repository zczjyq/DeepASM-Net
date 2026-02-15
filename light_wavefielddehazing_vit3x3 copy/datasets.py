import os
import re
import random
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF


#HAZE_PATTERN = re.compile(r"hazy_clear_(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
#CLEAN_PATTERN = re.compile(r"clear_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

HAZE_PATTERN = re.compile(r"hazy_clear_(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
CLEAN_PATTERN = re.compile(r"clear_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

@dataclass
class PairItem:
    hazy_path: str
    clean_path: str
    clean_id: int
    haze_level: int


def _list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]


def build_pairs_hazy_clear(root: str, hazy_dir: str, clean_dir: str) -> List[PairItem]:
    hazy_root = os.path.join(root, hazy_dir)
    clean_root = os.path.join(root, clean_dir)

    clean_map: Dict[int, str] = {}
    for path in _list_images(clean_root):
        name = os.path.basename(path)
        m = CLEAN_PATTERN.match(name)
        if m:
            clean_id = int(m.group(1))
            clean_map[clean_id] = path

    pairs: List[PairItem] = []
    for path in _list_images(hazy_root):
        name = os.path.basename(path)
        m = HAZE_PATTERN.match(name)
        if not m:
            continue
        clean_id = int(m.group(1))
        haze_level = int(m.group(2))
        clean_path = clean_map.get(clean_id)
        if clean_path is None:
            continue
        pairs.append(PairItem(path, clean_path, clean_id, haze_level))

    pairs.sort(key=lambda x: (x.clean_id, x.haze_level))
    return pairs


def build_pairs_same_name(root: str, hazy_dir: str, clean_dir: str) -> List[PairItem]:
    """按文件名完全匹配配对，适用于 test 等 hazy/GT 同名场景"""
    hazy_root = os.path.join(root, hazy_dir)
    clean_root = os.path.join(root, clean_dir)
    clean_map: Dict[str, str] = {}
    for path in _list_images(clean_root):
        name = os.path.basename(path)
        clean_map[name.lower()] = path
    pairs: List[PairItem] = []
    for path in _list_images(hazy_root):
        name = os.path.basename(path)
        clean_path = clean_map.get(name.lower())
        if clean_path is None:
            continue
        clean_id = int(hashlib.md5(name.lower().encode("utf-8")).hexdigest()[:8], 16)
        pairs.append(PairItem(path, clean_path, clean_id, 0))
    pairs.sort(key=lambda x: (x.clean_id, x.haze_level))
    return pairs


def build_pairs_hazy_gt_suffix(root: str, hazy_dir: str, clean_dir: str) -> List[PairItem]:
    """
    按前缀配对：hazy 目录下 XX_hazy.png，GT 目录下 XX_GT.png（或 XX_gt.png），
    通过去掉 _hazy / _GT 得到相同前缀 XX 进行配对。
    如 01_hazy.png <-> 01_GT.png
    """
    hazy_root = os.path.join(root, hazy_dir)
    clean_root = os.path.join(root, clean_dir)
    clean_map: Dict[str, str] = {}
    for path in _list_images(clean_root):
        name = os.path.basename(path)
        base, _ = os.path.splitext(name)
        base_lower = base.lower()
        if base_lower.endswith("_gt"):
            prefix = base_lower[:-3]
        else:
            continue
        clean_map[prefix] = path
    pairs: List[PairItem] = []
    for path in _list_images(hazy_root):
        name = os.path.basename(path)
        base, _ = os.path.splitext(name)
        base_lower = base.lower()
        if base_lower.endswith("_hazy"):
            prefix = base_lower[:-5]
        else:
            continue
        clean_path = clean_map.get(prefix)
        if clean_path is None:
            continue
        clean_id = int(hashlib.md5(prefix.encode("utf-8")).hexdigest()[:8], 16)
        pairs.append(PairItem(path, clean_path, clean_id, 0))
    pairs.sort(key=lambda x: (x.clean_id, x.haze_level))
    return pairs


def build_pairs_reside(root: str, hazy_dir: str, clean_dir: str) -> List[PairItem]:
    """
    RESIDE train 配对：hazy 为 scene_level.png，GT 为 scene_level_beta.png
    如 hazy 100_6.png -> GT 100_6_0.71799.png
    """
    hazy_root = os.path.join(root, hazy_dir)
    clean_root = os.path.join(root, clean_dir)
    clean_by_prefix: Dict[str, str] = {}
    for path in _list_images(clean_root):
        name = os.path.basename(path)
        base, _ = os.path.splitext(name)
        parts = base.split("_")
        if len(parts) >= 2:
            prefix = f"{parts[0]}_{parts[1]}"
            clean_by_prefix[prefix.lower()] = path
    pairs: List[PairItem] = []
    for path in _list_images(hazy_root):
        name = os.path.basename(path)
        base, _ = os.path.splitext(name)
        clean_path = clean_by_prefix.get(base.lower())
        if clean_path is None:
            continue
        clean_id = int(hashlib.md5(base.lower().encode("utf-8")).hexdigest()[:8], 16)
        pairs.append(PairItem(path, clean_path, clean_id, 0))
    pairs.sort(key=lambda x: (x.clean_id, x.haze_level))
    return pairs


def _discover_in_gt_pairs(root: str) -> List[Tuple[str, str]]:
    # Discover folders where IN and GT are siblings.
    pairs = []
    for dirpath, dirnames, _ in os.walk(root):
        if "IN" in dirnames and "GT" in dirnames:
            rel = os.path.relpath(dirpath, root)
            hazy_dir = os.path.join(rel, "IN")
            clean_dir = os.path.join(rel, "GT")
            pairs.append((hazy_dir, clean_dir))
    return pairs


def split_by_clean_id(pairs: List[PairItem], train_ratio: float, seed: int, shuffle: bool = True) -> Tuple[List[int], List[int]]:
    clean_ids = sorted({p.clean_id for p in pairs})
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(clean_ids)
    n_train = max(1, int(len(clean_ids) * train_ratio))
    train_ids = clean_ids[:n_train]
    val_ids = clean_ids[n_train:]
    if len(val_ids) == 0:
        val_ids = train_ids[-1:]
    return train_ids, val_ids


def _ensure_min_size(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w >= size and h >= size:
        return img
    return TF.resize(img, [max(size, h), max(size, w)], interpolation=Image.BILINEAR)


def _random_crop_pair(img_a: Image.Image, img_b: Image.Image, size: int) -> Tuple[Image.Image, Image.Image]:
    img_a = _ensure_min_size(img_a, size)
    img_b = _ensure_min_size(img_b, size)
    i, j, h, w = T.RandomCrop.get_params(img_a, output_size=(size, size))
    return TF.crop(img_a, i, j, h, w), TF.crop(img_b, i, j, h, w)


def _center_crop_pair(img_a: Image.Image, img_b: Image.Image, size: int) -> Tuple[Image.Image, Image.Image]:
    img_a = _ensure_min_size(img_a, size)
    img_b = _ensure_min_size(img_b, size)
    return TF.center_crop(img_a, [size, size]), TF.center_crop(img_b, [size, size])


def _load_image(path: str, use_cv2: bool = False) -> Image.Image:
    """加载图像，use_cv2 时用 cv2（通常更快，且支持中文路径）"""
    if use_cv2:
        try:
            import cv2
            buf = np.fromfile(path, dtype=np.uint8)
            arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if arr is None:
                raise ValueError(f"cv2.imdecode failed: {path}")
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(arr)
        except Exception:
            pass
    return Image.open(path).convert("RGB")


class HazyClearDataset(Dataset):
    def __init__(self, pairs: List[PairItem], image_size: int = 256, train: bool = True, random_flip: bool = True,
                 use_cv2_load: bool = False):
        self.pairs = pairs
        self.image_size = image_size
        self.train = train
        self.random_flip = random_flip
        self.use_cv2_load = use_cv2_load

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        item = self.pairs[idx]
        hazy = _load_image(item.hazy_path, self.use_cv2_load)
        clean = _load_image(item.clean_path, self.use_cv2_load)

        if self.image_size is not None:
            if self.train:
                hazy, clean = _random_crop_pair(hazy, clean, self.image_size)
            else:
                hazy, clean = _center_crop_pair(hazy, clean, self.image_size)

        if self.train and self.random_flip and random.random() < 0.5:
            hazy = TF.hflip(hazy)
            clean = TF.hflip(clean)

        hazy_t = TF.to_tensor(hazy)
        clean_t = TF.to_tensor(clean)

        return hazy_t, clean_t, item.clean_id, item.haze_level


def _debug_dataset_info(root: str, hazy_dir: str, clean_dir: str, pair_type: str):
    """输出数据集路径与文件数量的调试信息"""
    hazy_root = os.path.join(root, hazy_dir)
    clean_root = os.path.join(root, clean_dir)
    hazy_exists = os.path.isdir(hazy_root)
    clean_exists = os.path.isdir(clean_root)
    hazy_files = _list_images(hazy_root) if hazy_exists else []
    clean_files = _list_images(clean_root) if clean_exists else []
    print(f"  [debug] type={pair_type}")
    print(f"  [debug] root={os.path.abspath(root)} (exists={os.path.isdir(root)})")
    print(f"  [debug] hazy_full={os.path.abspath(hazy_root)} (exists={hazy_exists}) count={len(hazy_files)}")
    print(f"  [debug] clean_full={os.path.abspath(clean_root)} (exists={clean_exists}) count={len(clean_files)}")
    if hazy_files:
        print(f"  [debug] hazy_samples={[os.path.basename(p) for p in hazy_files[:3]]}")
    if clean_files:
        print(f"  [debug] clean_samples={[os.path.basename(p) for p in clean_files[:3]]}")


def _collect_pairs_from_entries(root: str, entries: list, debug: bool = True) -> List[PairItem]:
    """从 datasets/test_datasets 配置中收集配对"""
    pairs: List[PairItem] = []
    for i, entry in enumerate(entries):
        hd, cd = entry.get("hazy_dir"), entry.get("clean_dir")
        if not hd or not cd:
            if debug:
                print(f"  [debug] entry[{i}] 缺少 hazy_dir 或 clean_dir，跳过")
            continue
        t = entry.get("type", "same_name")
        if debug:
            _debug_dataset_info(root, hd, cd, t)
        n_before = len(pairs)
        if t == "same_name":
            pairs.extend(build_pairs_same_name(root, hd, cd))
        elif t == "reside":
            pairs.extend(build_pairs_reside(root, hd, cd))
        elif t == "hazy_gt":
            pairs.extend(build_pairs_hazy_gt_suffix(root, hd, cd))
        else:
            pairs.extend(build_pairs_hazy_clear(root, hd, cd))
        if debug:
            print(f"  [debug] entry[{i}] 新增 pairs={len(pairs) - n_before}")
    uniq = {}
    for p in pairs:
        uniq[p.hazy_path] = p
    pairs = list(uniq.values())
    pairs.sort(key=lambda x: (x.clean_id, x.haze_level))
    if debug:
        print(f"  [debug] 去重后总 pairs={len(pairs)}")
    return pairs


def get_dataloaders(cfg: dict, debug: bool = None):
    root = cfg["paths"]["root"]
    use_native = cfg["split"].get("use_native_split", False)
    if debug is None:
        debug = cfg["paths"].get("debug_datasets", True)
    if debug:
        print(f"[debug] paths.root = {root}")
        print(f"[debug] split.use_native_split = {use_native}")

    if use_native:
        # 使用数据集自带的 train/test：训练集来自 datasets，验证集来自 test_datasets
        train_entries = cfg["paths"].get("datasets", [])
        test_entries = cfg["paths"].get("test_datasets", [])
        if not train_entries:
            raise ValueError("use_native_split 时需配置 paths.datasets 作为训练集")
        if not test_entries:
            raise ValueError("use_native_split 时需配置 paths.test_datasets 作为验证/测试集")
        if debug:
            print("[debug] 正在收集训练集 pairs...")
        train_pairs = _collect_pairs_from_entries(root, train_entries, debug=debug)
        if debug:
            print("[debug] 正在收集测试集 pairs...")
        val_pairs = _collect_pairs_from_entries(root, test_entries, debug=debug)
        if len(train_pairs) == 0:
            raise ValueError(
                "训练集为空，请检查 paths.datasets 配置。"
                "确保 paths.root 指向包含 RESIDE-IN 的目录，"
                "且 RESIDE-IN/train/hazy 与 RESIDE-IN/train/GT 存在且配对正确。"
            )
        if len(val_pairs) == 0:
            raise ValueError("测试集为空，请检查 paths.test_datasets 配置")
        print(f"==> 使用数据集自带划分: train={len(train_pairs)}, test={len(val_pairs)}")
    else:
        # 原逻辑：从 datasets 收集全部数据，按 train_ratio 自己划分
        pairs: List[PairItem] = []
        datasets_cfg = cfg["paths"].get("datasets", [])
        for entry in datasets_cfg:
            t = entry.get("type", "same_name")
            if t == "same_name":
                pairs.extend(build_pairs_same_name(root, entry["hazy_dir"], entry["clean_dir"]))
            elif t == "reside":
                pairs.extend(build_pairs_reside(root, entry["hazy_dir"], entry["clean_dir"]))
            elif t == "hazy_gt":
                pairs.extend(build_pairs_hazy_gt_suffix(root, entry["hazy_dir"], entry["clean_dir"]))
            else:
                pairs.extend(build_pairs_hazy_clear(root, entry["hazy_dir"], entry["clean_dir"]))
        if cfg["paths"].get("auto_discover_in_gt", True):
            for hazy_dir, clean_dir in _discover_in_gt_pairs(root):
                pairs.extend(build_pairs_same_name(root, hazy_dir, clean_dir))
        if cfg["paths"].get("use_legacy_hazy_clear", True):
            pairs.extend(build_pairs_hazy_clear(root, cfg["paths"]["hazy_dir"], cfg["paths"]["clean_dir"]))
        uniq = {}
        for p in pairs:
            uniq[p.hazy_path] = p
        pairs = list(uniq.values())
        pairs.sort(key=lambda x: (x.clean_id, x.haze_level))
        if len(pairs) == 0:
            raise ValueError(
                f"未找到任何图像对。请检查 paths.root='{root}' 下 datasets 配置，"
                f"或设置 split.use_native_split: true 并配置 test_datasets。"
            )
        train_ids, val_ids = split_by_clean_id(
            pairs, cfg["split"]["train_ratio"], cfg.get("seed", 123), cfg["split"]["shuffle"]
        )
        train_pairs = [p for p in pairs if p.clean_id in train_ids]
        val_pairs = [p for p in pairs if p.clean_id in val_ids]
        if len(train_pairs) == 0:
            raise ValueError("划分后训练集为空，请检查 split.train_ratio 或数据量。")

    use_cv2 = cfg["transforms"].get("use_cv2_load", False)
    train_set = HazyClearDataset(
        train_pairs,
        image_size=cfg["transforms"]["image_size"],
        train=True,
        random_flip=cfg["transforms"]["random_flip"],
        use_cv2_load=use_cv2,
    )
    val_set = HazyClearDataset(
        val_pairs,
        image_size=cfg["transforms"]["image_size"],
        train=False,
        random_flip=False,
        use_cv2_load=use_cv2,
    )

    load_cfg = cfg["loader"]
    nw = load_cfg["num_workers"]
    loader_kw = dict(
        num_workers=nw,
        pin_memory=load_cfg["pin_memory"],
    )
    if nw > 0:
        loader_kw["prefetch_factor"] = load_cfg.get("prefetch_factor", 2)
        loader_kw["persistent_workers"] = load_cfg.get("persistent_workers", False)
    train_loader = DataLoader(
        train_set,
        batch_size=load_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=load_cfg["val_batch_size"],
        shuffle=False,
        drop_last=False,
        **loader_kw,
    )

    return train_loader, val_loader, train_set, val_set
