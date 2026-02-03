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
    return [os.path.join(folder, f) for f in os.listdir(folder)]


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
        # Stable id based on filename (consistent across runs)
        clean_id = int(hashlib.md5(name.lower().encode("utf-8")).hexdigest()[:8], 16)
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


def get_dataloaders(cfg: dict):
    pairs: List[PairItem] = []
    root = cfg["paths"]["root"]

    # 1) Explicit datasets list
    datasets_cfg = cfg["paths"].get("datasets", [])
    for entry in datasets_cfg:
        if entry.get("type") == "same_name":
            pairs.extend(build_pairs_same_name(root, entry["hazy_dir"], entry["clean_dir"]))
        else:
            pairs.extend(build_pairs_hazy_clear(root, entry["hazy_dir"], entry["clean_dir"]))

    # 2) Auto-discover IN/GT datasets under root
    if cfg["paths"].get("auto_discover_in_gt", True):
        for hazy_dir, clean_dir in _discover_in_gt_pairs(root):
            pairs.extend(build_pairs_same_name(root, hazy_dir, clean_dir))

    # 3) Legacy hazy/clear dataset if configured
    if cfg["paths"].get("use_legacy_hazy_clear", True):
        pairs.extend(build_pairs_hazy_clear(root, cfg["paths"]["hazy_dir"], cfg["paths"]["clean_dir"]))

    # Deduplicate by hazy path
    uniq = {}
    for p in pairs:
        uniq[p.hazy_path] = p
    pairs = list(uniq.values())
    pairs.sort(key=lambda x: (x.clean_id, x.haze_level))

    if len(pairs) == 0:
        raise ValueError(
            f"未找到任何图像对。请检查 paths.root='{root}' 下是否存在 IN/GT 目录（或 hazy/clear），"
            f"且文件名能正确配对。可通过 --root 指定数据路径。"
        )

    train_ids, val_ids = split_by_clean_id(pairs, cfg["split"]["train_ratio"], cfg.get("seed", 123), cfg["split"]["shuffle"])
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
