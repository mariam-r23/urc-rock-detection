"""Pure OpenCV/NumPy rock & crater detector (no ROS imports, fully unit-testable).

Pipeline
--------
1. Convert to 8-bit gray, crop away sky (above horizon) and rover body (bottom strip).
2. CLAHE + light blur to normalise illumination.
3. Subtract a heavily blurred copy (local background) -> signed local contrast.
4. Threshold dark (shadows / dark rocks) and bright (lit faces / rims) regions.
5. Morphological open/close, then contour extraction (colour + contour segmentation).
6. Filter by area / solidity / aspect, score confidence, classify rock vs crater.
7. Non-maximum suppression (+ optional temporal confirmation in SimpleTracker).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROCK = "rock"
CRATER = "crater"


@dataclass
class DetectorParams:
    process_width: int = 640          # images wider than this are down-scaled first
    horizon_frac: float = 0.30        # top fraction of image treated as sky / far field
    ignore_bottom_frac: float = 0.04  # bottom strip (rover body / bumper) ignored
    clahe_clip: float = 1.5
    clahe_grid: int = 8
    blur_ksize: int = 5
    bg_sigma_frac: float = 0.04       # local-background sigma as fraction of width
    dark_thresh: float = 14.0         # gray levels below local background (floor if adaptive_k > 0)
    bright_thresh: float = 18.0       # gray levels above local background (floor if adaptive_k > 0)
    adaptive_k: float = 0.0           # >0: thresholds = max(floor, k * robust sigma of local contrast)
    open_ksize: int = 3
    close_ksize: int = 15
    min_area_frac: float = 0.0006     # of ROI area
    max_area_frac: float = 0.30
    min_solidity: float = 0.50
    max_aspect: float = 5.0           # max(w/h, h/w) of bounding box
    crater_min_area_frac: float = 0.02
    light_from: Optional[Tuple[float, float]] = None  # (dx, dy) image-space direction toward the light (optional)
    min_confidence: float = 0.35
    nms_iou: float = 0.4
    containment_thresh: float = 0.8
    min_edge_frac: float = 0.12       # fraction of boundary pixels with a real intensity edge
    edge_grad_thresh: float = 4.0     # gray levels / pixel (Sobel/8)
    crater_close_frac: float = 0.07   # large closing kernel (fraction of width) to join crater walls
    crater_min_wall_frac: float = 0.10  # dark AND bright wall pixels must each cover this share of the blob
    max_detections: int = 40


@dataclass
class Detection:
    label: str
    confidence: float
    x: int            # bbox in ORIGINAL image pixels
    y: int
    w: int
    h: int
    area_px: float
    u_ground: float   # pixel of ground contact point (bottom-centre of bbox)
    v_ground: float
    u_center: float
    v_center: float

    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- helpers
def to_gray_u8(image: np.ndarray) -> Optional[np.ndarray]:
    """Return a contiguous uint8 gray image or None if the input is unusable."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    img = image
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
        elif img.shape[2] in (3, 4):
            if img.dtype != np.uint8:
                img = _normalise_to_u8(img)
            code = cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            return np.ascontiguousarray(cv2.cvtColor(img, code))
        else:
            return None
    elif img.ndim != 2:
        return None
    if img.dtype != np.uint8:
        img = _normalise_to_u8(img)
    return np.ascontiguousarray(img)


def _normalise_to_u8(img: np.ndarray) -> np.ndarray:
    f = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(f, 1), np.percentile(f, 99)
    if hi - lo < 1e-6:
        return np.zeros(f.shape, np.uint8)
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _containment(small: Sequence[float], big: Sequence[float]) -> float:
    sx, sy, sw, sh = small
    bx, by, bw, bh = big
    x1, y1 = max(sx, bx), max(sy, by)
    x2, y2 = min(sx + sw, bx + bw), min(sy + sh, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (sw * sh) if sw * sh > 0 else 0.0


def _odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


# ----------------------------------------------------------------------------- detector
class RockCraterDetector:
    def __init__(self, params: Optional[DetectorParams] = None):
        self.p = params or DetectorParams()

    def detect(self, image: np.ndarray, return_debug: bool = False):
        """Detect hazards. Returns list[Detection] (and a full-size mask if return_debug)."""
        p = self.p
        empty = ([], None) if return_debug else []
        gray = to_gray_u8(image)
        if gray is None:
            return empty
        H0, W0 = gray.shape
        if H0 < 32 or W0 < 32:
            return empty

        scale = 1.0
        if W0 > p.process_width:
            scale = p.process_width / float(W0)
            gray = cv2.resize(gray, (p.process_width, max(1, int(round(H0 * scale)))),
                              interpolation=cv2.INTER_AREA)
        H, W = gray.shape

        y0 = int(np.clip(p.horizon_frac, 0.0, 0.9) * H)
        y1 = int((1.0 - np.clip(p.ignore_bottom_frac, 0.0, 0.5)) * H)
        if y1 - y0 < 32:
            return empty
        roi = gray[y0:y1, :]

        # a (near-)constant image carries no information -> no detections
        if float(roi.max()) - float(roi.min()) < 8.0:
            return empty

        if p.clahe_clip > 0:
            g = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(p.clahe_grid, p.clahe_grid)).apply(roi)
        else:
            g = roi
        k = _odd(p.blur_ksize)
        g = cv2.GaussianBlur(g, (k, k), 0)

        diff = g.astype(np.float32) - self._local_background(g, p.bg_sigma_frac * W)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        grad = cv2.magnitude(gx, gy)
        t_dark, t_bright = p.dark_thresh, p.bright_thresh
        if p.adaptive_k > 0:
            sigma = 1.4826 * float(np.median(np.abs(diff - np.median(diff))))
            t_dark = max(t_dark, p.adaptive_k * sigma)
            t_bright = max(t_bright, p.adaptive_k * sigma)
        mask = ((diff < -t_dark) | (diff > t_bright)).astype(np.uint8) * 255

        raw_dark = (diff < -t_dark)
        raw_bright = (diff > t_bright)
        if p.open_ksize > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_ksize,) * 2))
        if p.close_ksize > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_ksize,) * 2))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float((y1 - y0) * W)
        min_area = p.min_area_frac * roi_area
        max_area = p.max_area_frac * roi_area

        cands: List[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 4 or bh < 4:
                continue
            if max(bw / bh, bh / bw) > p.max_aspect:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < p.min_solidity:
                continue
            filled = np.zeros((bh, bw), np.uint8)
            cv2.drawContours(filled, [c - np.array([[bx, by]])], -1, 255, -1)
            sel = filled > 0
            d_roi = diff[by:by + bh, bx:bx + bw]
            # edge support: share of the blob's inner border band that sits on a real intensity edge
            # (rejects soft terrain blotches, accepts rocks even when the mask has a thin halo)
            band = sel & ~(cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0)
            if not band.any():
                band = sel
            g_roi = grad[by:by + bh, bx:bx + bw]
            edge_frac = float(np.mean(g_roi[band] >= p.edge_grad_thresh))
            if edge_frac < p.min_edge_frac:
                continue
            contrast = float(np.mean(np.abs(d_roi[sel]))) if sel.any() else 0.0
            contrast_score = float(np.clip(contrast / (3.0 * min(t_dark, t_bright)), 0, 1))
            size_score = float(np.clip(area / (4.0 * max(min_area, 1.0)), 0, 1))
            conf = 0.45 * contrast_score + 0.30 * min(1.0, solidity) + 0.25 * size_score
            if conf < p.min_confidence:
                continue
            label = self._classify(area, roi_area, d_roi, sel, p, t_dark, t_bright)
            ox, oy = bx / scale, (by + y0) / scale
            ow, oh = bw / scale, bh / scale
            cands.append(Detection(
                label=label, confidence=round(conf, 4),
                x=int(round(ox)), y=int(round(oy)), w=int(round(ow)), h=int(round(oh)),
                area_px=float(area / (scale * scale)),
                u_ground=ox + ow / 2.0, v_ground=oy + oh,
                u_center=ox + ow / 2.0, v_center=oy + oh / 2.0))

        cands.extend(self._crater_pass(mask, raw_dark, raw_bright, diff, W, y0, scale, roi_area, p, t_dark, t_bright))
        dets = self._nms(cands)[: p.max_detections]
        if return_debug:
            m = np.zeros((H, W), np.uint8)
            m[y0:y1, :] = mask
            full = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else m
            return dets, full
        return dets

    # internals ------------------------------------------------------------
    @staticmethod
    def _local_background(g: np.ndarray, sigma: float) -> np.ndarray:
        """Large-sigma Gaussian computed on a down-sampled copy for speed."""
        f = 4 if min(g.shape) >= 128 else 1
        small = cv2.resize(g, (max(1, g.shape[1] // f), max(1, g.shape[0] // f)),
                           interpolation=cv2.INTER_AREA) if f > 1 else g
        s = max(1.0, sigma / f)
        bg = cv2.GaussianBlur(small.astype(np.float32), (0, 0), s, borderType=cv2.BORDER_REFLECT)
        if f > 1:
            bg = cv2.resize(bg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
        return bg

    @staticmethod
    def _crater_pass(mask, raw_dark, raw_bright, diff, W, y0, scale, roi_area, p: DetectorParams,
                     t_dark, t_bright) -> List[Detection]:
        """Join the dark and bright inner walls of large depressions into one hazard blob."""
        k = _odd(int(p.crater_close_frac * W))
        if k < 5:
            return []
        big = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for c in contours:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area < 0.6 * p.crater_min_area_frac * roi_area or hull_area > p.max_area_frac * roi_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(hull)
            if max(bw / bh, bh / bw) > p.max_aspect:
                continue
            if cv2.contourArea(c) / hull_area < 0.8:
                continue
            m = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(m, [hull], -1, 255, -1)
            sel = m > 0
            n = float(sel.sum())
            fd = float(np.count_nonzero(raw_dark & sel)) / n
            fb = float(np.count_nonzero(raw_bright & sel)) / n
            if fd < p.crater_min_wall_frac or fb < p.crater_min_wall_frac:
                continue
            ys, xs = np.nonzero(sel & raw_dark)
            yb, xb = np.nonzero(sel & raw_bright)
            sep = np.hypot(xs.mean() - xb.mean(), ys.mean() - yb.mean()) / np.hypot(bw, bh)
            if sep < 0.12:
                continue
            conf = float(np.clip(0.5 + 2.0 * min(fd, fb), 0.0, 0.9))
            ox, oy, ow, oh = bx / scale, (by + y0) / scale, bw / scale, bh / scale
            out.append(Detection(
                label=CRATER, confidence=round(conf, 4),
                x=int(round(ox)), y=int(round(oy)), w=int(round(ow)), h=int(round(oh)),
                area_px=float(hull_area / (scale * scale)),
                u_ground=ox + ow / 2.0, v_ground=oy + oh / 2.0,   # a crater lies flat: centre is on the ground
                u_center=ox + ow / 2.0, v_center=oy + oh / 2.0))
        return out

    @staticmethod
    def _classify(area, roi_area, d_roi, sel, p: DetectorParams, t_dark: float, t_bright: float) -> str:
        if area >= p.crater_min_area_frac * roi_area:
            return CRATER
        if p.light_from is not None and area >= 0.3 * p.crater_min_area_frac * roi_area:
            # In a depression the wall facing the light is shadowed (dark toward the light);
            # for a rock the lit face points toward the light. Compare dark/bright centroids.
            ys, xs = np.nonzero(sel & (d_roi < -t_dark))
            yb, xb = np.nonzero(sel & (d_roi > t_bright))
            if len(xs) > 0 and len(xb) > 0:
                dv = np.array([xs.mean() - xb.mean(), ys.mean() - yb.mean()])
                lv = np.array(p.light_from, dtype=float)
                n = np.linalg.norm(dv) * np.linalg.norm(lv)
                if n > 1e-6 and float(dv @ lv) / n > 0.5:
                    return CRATER
        return ROCK

    def _nms(self, dets: List[Detection]) -> List[Detection]:
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        keep: List[Detection] = []
        for d in dets:
            ok = True
            for k in keep:
                if iou(d.bbox(), k.bbox()) > self.p.nms_iou or \
                        _containment(d.bbox(), k.bbox()) > self.p.containment_thresh:
                    ok = False
                    break
            if ok:
                keep.append(d)
        return keep


# ----------------------------------------------------------------------------- tracker
class SimpleTracker:
    """Greedy IoU tracker: a hazard is published only after `min_hits` matches.

    Suppresses single-frame false positives (glare, noise) and keeps hazards stable. Tracks
    missed for more than `max_age` frames are deleted. Set min_hits=1 to disable filtering.
    """

    def __init__(self, min_hits: int = 2, max_age: int = 3, iou_thresh: float = 0.25):
        self.min_hits, self.max_age, self.iou_thresh = int(min_hits), int(max_age), float(iou_thresh)
        self._tracks: List[dict] = []
        self._next_id = 1

    def reset(self):
        self._tracks.clear()

    def update(self, dets: List[Detection]) -> List[Detection]:
        for t in self._tracks:
            t["matched"] = False
        pairs = []
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(dets):
                v = iou(t["det"].bbox(), d.bbox())
                if v >= self.iou_thresh:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self._tracks[ti]
            t["det"], t["hits"], t["miss"], t["matched"] = dets[di], t["hits"] + 1, 0, True
        for di, d in enumerate(dets):
            if di not in used_d:
                self._tracks.append({"id": self._next_id, "det": d, "hits": 1, "miss": 0, "matched": True})
                self._next_id += 1
        for t in self._tracks:
            if not t["matched"]:
                t["miss"] += 1
        self._tracks = [t for t in self._tracks if t["miss"] <= self.max_age]
        return [t["det"] for t in self._tracks if t["matched"] and t["hits"] >= self.min_hits]
