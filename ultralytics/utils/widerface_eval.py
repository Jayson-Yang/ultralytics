# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""WIDER FACE official validation AP (Easy / Medium / Hard).

Logic follows the official MATLAB evaluation toolkit (IoU=0.5, 1000-point PR sampling).
Reference: http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ultralytics.utils import LOGGER

WIDER_SETTINGS = {
    "setting_name_list": ["easy_val", "medium_val", "hard_val"],
    "IoU_thresh": 0.5,
    "thresh_num": 1000,
}

REQUIRED_GT_FILES = (
    "wider_face_val.mat",
    "wider_easy_val.mat",
    "wider_medium_val.mat",
    "wider_hard_val.mat",
)


def xyxy2xywh_tl(boxes: np.ndarray) -> np.ndarray:
    """Convert xyxy boxes to top-left xywh format required by the official WIDER FACE toolkit."""
    xywh = np.empty_like(boxes)
    xywh[:, 0] = boxes[:, 0]
    xywh[:, 1] = boxes[:, 1]
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh


def _load_gt_mat_to_lists(gt_dir: Path) -> tuple[list[str], list[list[str]], list[list[np.ndarray]], list[list[list]]]:
    """Load official WIDER FACE ground-truth MAT files."""
    from scipy.io import loadmat

    gt_mat = loadmat(gt_dir / "wider_face_val.mat")
    easy_mat = loadmat(gt_dir / "wider_easy_val.mat")
    medium_mat = loadmat(gt_dir / "wider_medium_val.mat")
    hard_mat = loadmat(gt_dir / "wider_hard_val.mat")

    event_list = [_[0][0] for _ in gt_mat["event_list"]]
    file_list, facebox_list, easy_list, medium_list, hard_list = [], [], [], [], []
    for file_list_per_event, box_list_per_event, easy_list_per_event, medium_list_per_event, hard_list_per_event in zip(
        gt_mat["file_list"],
        gt_mat["face_bbx_list"],
        easy_mat["gt_list"],
        medium_mat["gt_list"],
        hard_mat["gt_list"],
    ):
        file_list.append([_[0][0] for _ in file_list_per_event[0]])
        facebox_list.append([_[0] for _ in box_list_per_event[0]])
        easy_list.append(
            [
                _[0].tolist() if not _[0].tolist() else np.concatenate(_[0]).tolist()
                for _ in easy_list_per_event[0]
            ]
        )
        medium_list.append(
            [
                _[0].tolist() if not _[0].tolist() else np.concatenate(_[0]).tolist()
                for _ in medium_list_per_event[0]
            ]
        )
        hard_list.append(
            [
                _[0].tolist() if not _[0].tolist() else np.concatenate(_[0]).tolist()
                for _ in hard_list_per_event[0]
            ]
        )
    set_gt_lists = [easy_list, medium_list, hard_list]
    return event_list, file_list, facebox_list, set_gt_lists


def build_stem_to_event_map(gt_dir: str | Path) -> dict[str, str]:
    """Map image stem -> official WIDER event name from GT MAT files."""
    event_list, file_list, _, _ = _load_gt_mat_to_lists(Path(gt_dir))
    return {stem: event for event, stems in zip(event_list, file_list) for stem in stems}


def resolve_wider_event_stem(im_file: Path, stem_to_event: dict[str, str]) -> tuple[str, str] | None:
    """Resolve official (event, stem) from an image path.

    Supports both official nested layout ``.../21--Festival/21_Festival_....jpg`` and flat YOLO layout
    ``.../val/21_Festival_....jpg`` (looks up event via *stem_to_event*).
    """
    stem = im_file.stem
    parent = im_file.parent.name
    if parent not in {"val", "train", "test", "images"}:
        return parent, stem
    event = stem_to_event.get(stem)
    return (event, stem) if event else None


def count_matched_preds(
    event_list: list[str],
    file_list: list[list[str]],
    pred_dict: dict[tuple[str, str], np.ndarray],
) -> tuple[int, int, int]:
    """Return (matched_images, images_with_dets, total_gt_images)."""
    total = sum(len(stems) for stems in file_list)
    matched = images_with_dets = 0
    for event, stems in zip(event_list, file_list):
        for stem in stems:
            boxes = pred_dict.get((event, stem))
            if boxes is None:
                continue
            matched += 1
            if boxes.size:
                images_with_dets += 1
    return matched, images_with_dets, total


def build_pred_list(
    event_list: list[str],
    file_list: list[list[str]],
    pred_dict: dict[tuple[str, str], np.ndarray],
) -> list[list[np.ndarray]]:
    """Align in-memory predictions with official GT event/file ordering."""
    pred_list = []
    for event, stems in zip(event_list, file_list):
        box_list = []
        for stem in stems:
            boxes = pred_dict.get((event, stem), np.zeros((0, 5), dtype=np.float32))
            if boxes.size:
                boxes = boxes[np.argsort(-boxes[:, 4])]
            box_list.append(boxes)
        pred_list.append(box_list)
    return pred_list


def _normalize_scores(pred_list: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    """Min-max normalize confidence scores to [0, 1] across all predictions."""
    scores = [box[-1] for event in pred_list for img in event for box in img if img.size]
    if not scores:
        return pred_list
    min_score, max_score = min(scores), max(scores)
    denom = max_score - min_score
    if denom <= 0:
        return pred_list
    for event in pred_list:
        for img in event:
            if img.size:
                img[:, 4] = (img[:, 4] - min_score) / denom
    return pred_list


def _box_overlap(boxlist: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Compute IoU between one box (xyxy) and a list of boxes (xyxy)."""
    x1 = np.maximum(boxlist[:, 0], box[0])
    y1 = np.maximum(boxlist[:, 1], box[1])
    x2 = np.minimum(boxlist[:, 2], box[2])
    y2 = np.minimum(boxlist[:, 3], box[3])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    overlap = np.zeros(boxlist.shape[0], dtype=np.float32)
    valid = (w >= 0) & (h >= 0)
    inter = w[valid] * h[valid]
    aarea = (boxlist[valid, 2] - boxlist[valid, 0] + 1) * (boxlist[valid, 3] - boxlist[valid, 1] + 1)
    barea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    overlap[valid] = inter / (aarea + barea - inter)
    return overlap


def _image_evaluation(
    pred_info: np.ndarray, gt_bbx: np.ndarray, ignore: list[int], iou_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-image TP/FP assignment following the official toolkit."""
    pred_num = pred_info.shape[0]
    gt_num = gt_bbx.shape[0]
    pred_info = pred_info.copy()
    gt_bbx = gt_bbx.copy()
    pred_info[:, 2] = pred_info[:, 0] + pred_info[:, 2]
    pred_info[:, 3] = pred_info[:, 1] + pred_info[:, 3]
    gt_bbx[:, 2] = gt_bbx[:, 0] + gt_bbx[:, 2]
    gt_bbx[:, 3] = gt_bbx[:, 1] + gt_bbx[:, 3]

    pred_recall = np.zeros(pred_num, dtype=np.float32)
    recall_list = np.zeros(gt_num, dtype=np.float32)
    proposal_list = np.ones(pred_num, dtype=np.float32)
    cnt = 0
    for h in range(pred_num):
        overlap_list = _box_overlap(gt_bbx, pred_info[h, :4])
        idx = int(np.argmax(overlap_list))
        if overlap_list[idx] >= iou_thresh:
            if ignore[idx] == 0:
                recall_list[idx] = -1
                proposal_list[h] = -1
            elif recall_list[idx] == 0:
                recall_list[idx] = 1
                cnt += 1
        pred_recall[h] = cnt
    return pred_recall, proposal_list


def _image_pr_info(
    thresh_num: int, pred_info: np.ndarray, proposal_list: np.ndarray, pred_recall: np.ndarray
) -> np.ndarray:
    """Sample PR curve points for one image."""
    img_pr_info = np.zeros((thresh_num, 2), dtype=np.float32)
    thresholds = np.linspace(1 - 1.0 / thresh_num, 0.0, thresh_num)
    num = 0
    for t in range(thresh_num):
        thresh = thresholds[t]
        indexes = np.where(pred_info[:, 4] >= thresh)[0]
        if indexes.size > num:
            r_index = int(np.max(indexes))
            p_index_sum = float(np.sum(proposal_list[: r_index + 1] == 1))
            img_pr_info[t, 0] = p_index_sum
            img_pr_info[t, 1] = pred_recall[r_index]
            num = indexes.size
        elif num > 0:
            img_pr_info[t] = img_pr_info[t - 1]
    return img_pr_info


def _dataset_pr_info(thresh_num: int, org_pr_curve: np.ndarray, count_face: int) -> np.ndarray:
    """Convert accumulated PR counts to precision/recall."""
    pr_curve = np.zeros((thresh_num, 2), dtype=np.float32)
    pr_curve[:, 0] = org_pr_curve[:, 1] / np.maximum(org_pr_curve[:, 0], 1e-8)
    pr_curve[:, 1] = org_pr_curve[:, 1] / max(count_face, 1)
    return pr_curve


def _calc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC-style AP from precision-recall curve."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def _evaluate_subset(
    norm_pred_list: list[list[np.ndarray]],
    facebox_list: list[list[np.ndarray]],
    set_gt_list: list[list[list]],
    set_name: str,
    settings: dict[str, Any],
) -> tuple[float, np.ndarray]:
    """Evaluate one difficulty subset (easy / medium / hard)."""
    iou_thresh = settings["IoU_thresh"]
    thresh_num = settings["thresh_num"]
    count_face = 0
    org_pr_curve = np.zeros((thresh_num, 2), dtype=np.float32)

    for i, (gt_bbx_list, pred_list, sub_gt_list) in enumerate(zip(facebox_list, norm_pred_list, set_gt_list)):
        for gt_bbx, pred_info, keep_index in zip(gt_bbx_list, pred_list, sub_gt_list):
            gt_bbx = np.reshape(gt_bbx.copy(), (-1, 4))
            pred_info = np.reshape(pred_info.copy(), (-1, 5))
            count_face += len(keep_index)
            if gt_bbx.shape[0] == 0 or pred_info.shape[0] == 0:
                continue
            keep_index_py = [_ - 1 for _ in keep_index]
            ignore = [1 if idx in keep_index_py else 0 for idx in range(len(gt_bbx))]
            pred_recall, proposal_list = _image_evaluation(pred_info, gt_bbx, ignore, iou_thresh)
            img_pr_info = _image_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
            org_pr_curve += img_pr_info

    pr_curve = _dataset_pr_info(thresh_num, org_pr_curve, count_face)
    ap = _calc_ap(pr_curve[:, 1], pr_curve[:, 0])
    LOGGER.info(f"WIDER FACE {set_name}: AP={ap:.4f} (faces={count_face})")
    return ap, pr_curve


def evaluate_widerface(
    pred_dict: dict[tuple[str, str], np.ndarray],
    gt_dir: str | Path,
    settings: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Run official WIDER FACE validation AP on in-memory predictions.

    Args:
        pred_dict: Mapping ``(event_name, image_stem) -> (N, 5)`` array in ``xywh + score`` format.
        gt_dir: Directory containing ``wider_face_val.mat`` and easy/medium/hard MAT files.
        settings: Optional evaluation settings override.

    Returns:
        Dict with ``metrics/wider_easy_ap``, ``metrics/wider_medium_ap``, ``metrics/wider_hard_ap``.
    """
    settings = {**WIDER_SETTINGS, **(settings or {})}
    gt_dir = Path(gt_dir)
    missing = [f for f in REQUIRED_GT_FILES if not (gt_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"WIDER GT missing in {gt_dir}: {missing}")

    event_list, file_list, facebox_list, set_gt_lists = _load_gt_mat_to_lists(gt_dir)
    matched, with_dets, total = count_matched_preds(event_list, file_list, pred_dict)
    LOGGER.info(
        f"WIDER pred alignment: {matched}/{total} images matched, {with_dets} with detections "
        f"(collected keys={len(pred_dict)})"
    )
    if matched == 0:
        LOGGER.warning(
            "No WIDER predictions aligned with official GT. "
            "Check image stems and event folders, or flat-layout stem_to_event mapping."
        )
    pred_list = build_pred_list(event_list, file_list, pred_dict)
    norm_pred_list = _normalize_scores(pred_list)

    aps = []
    for set_name, set_gt_list in zip(settings["setting_name_list"], set_gt_lists):
        ap, _ = _evaluate_subset(norm_pred_list, facebox_list, set_gt_list, set_name, settings)
        aps.append(ap)

    return {
        "metrics/wider_easy_ap": round(aps[0], 5),
        "metrics/wider_medium_ap": round(aps[1], 5),
        "metrics/wider_hard_ap": round(aps[2], 5),
    }


def resolve_wider_gt_dir(data: dict[str, Any]) -> Path | None:
    """Resolve WIDER ground-truth directory from dataset YAML."""
    root = Path(data["path"])
    raw = data.get("wider_gt_dir")
    if raw:
        p = Path(raw)
        return p.resolve() if p.is_absolute() else (root / raw).resolve()
    for candidate in ("WIDER_FACE/wider_face_split", "wider_face_split", "WIDER_FACE"):
        p = (root / candidate).resolve()
        if all((p / f).is_file() for f in REQUIRED_GT_FILES):
            return p
    return None
