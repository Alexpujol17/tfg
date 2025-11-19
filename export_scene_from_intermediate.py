#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Export walls (exterior/interior) and openings (front/int doors) to JSON
# a partir de la imagen "Intermediate" (paredes negras + puerta cian).

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import thin
from skimage.transform import probabilistic_hough_line
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree

# =========================
# Configuración (ajústala)
# =========================
# Paleta de la intermedia (muros y puertas):
RGB_EXT_WALL = (0, 0, 0)        # muro exterior (negro)
RGB_INT_WALL = (215, 165, 159)  # tabique interior (si no sale en la intermedia, quedará vacío)

# Puerta exterior en INTERMEDIA suele ser CIAN; mejor usar rango en vez de un solo valor:
# Máscara por rangos (G y B altos, R bajo)
FRONT_DOOR_R_MAX = 80
FRONT_DOOR_G_MIN = 200
FRONT_DOOR_B_MIN = 200

# Si en tus intermedias hubiera puerta interior con otro color, puedes definir otra máscara.

# Tolerancia de color estricta para muros (no para puerta cian, que va por rango)
COLOR_TOL = 2  # 1–4

# Parámetros de líneas (Hough) para muros
HOUGH_LINE_LENGTH = 10
HOUGH_LINE_GAP    = 2
HOUGH_THRESHOLD   = 10

# Filtros de aperturas (componentes tipo “puerta”)
MIN_OPEN_AREA      = 20     # píxeles
MAX_OPEN_AREA      = 1200   # píxeles
MAX_DIST_TO_WALL   = 2      # píxeles a pared
MIN_OPEN_ASPECT    = 2.0    # max(h,w)/min(h,w) >= MIN_OPEN_ASPECT
DEDUP_RADIUS_PX    = 8      # agrupar centros cercanos

# Fallback geométrico (si no hay puerta por color): buscar “gaps” entre segmentos exteriores
GAP_COLINEAR_TOL_PX = 2     # tolerancia de colinealidad (misma fila/col) en píxeles
GAP_MAX_LEN_PX      = 20    # ancho máximo de hueco considerado puerta (ajústalo a tu escala ~ 8–18 px para 0.7–0.9 m)
GAP_MIN_LEN_PX      = 6     # ancho mínimo de hueco


# =========================
# Utilidades
# =========================
def mask_rgb_tol(img: np.ndarray, rgb: tuple[int, int, int], tol: int = COLOR_TOL) -> np.ndarray:
    r, g, b = rgb
    return (
        (np.abs(img[:, :, 0] - r) <= tol) &
        (np.abs(img[:, :, 1] - g) <= tol) &
        (np.abs(img[:, :, 2] - b) <= tol)
    )

def mask_cyan_range(img: np.ndarray) -> np.ndarray:
    """Máscara de cian robusta para puerta: G y B altos, R bajo."""
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return (R <= FRONT_DOOR_R_MAX) & (G >= FRONT_DOOR_G_MIN) & (B >= FRONT_DOOR_B_MIN)

def lines_from_mask(mask: np.ndarray,
                    line_length: int = HOUGH_LINE_LENGTH,
                    line_gap: int = HOUGH_LINE_GAP,
                    threshold: int = HOUGH_THRESHOLD):
    """Devuelve segmentos a partir del esqueleto de la máscara."""
    if not mask.any():
        return []
    skel = thin(mask)
    lines = probabilistic_hough_line(
        skel, threshold=threshold, line_length=line_length, line_gap=line_gap
    )
    # lines: list of ((x0,y0), (x1,y1))
    return [{"p1": [int(x0), int(y0)], "p2": [int(x1), int(y1)]} for (x0, y0), (x1, y1) in lines]

def opening_centers(mask: np.ndarray,
                    wall_mask: np.ndarray,
                    min_area: int = MIN_OPEN_AREA,
                    max_area: int = MAX_OPEN_AREA,
                    max_dist_to_wall: int = MAX_DIST_TO_WALL,
                    min_aspect: float = MIN_OPEN_ASPECT):
    """
    Devuelve centros [cx, cy] de componentes conectados del 'mask' que:
      - área dentro [min_area, max_area]
      - están a <= max_dist_to_wall píxeles de la pared
      - rectángulo delgado: max(h,w)/min(h,w) >= min_aspect
    """
    if not mask.any():
        return []

    # elimina posibles píxeles solapados con pared (ruido)
    mask = mask & (~wall_mask)

    lbl = label(mask.astype(np.uint8), connectivity=1)
    ys, xs = np.nonzero(wall_mask)
    if len(xs) == 0:
        return []

    tree = cKDTree(np.c_[xs, ys])
    keep = []
    for reg in regionprops(lbl):
        if reg.area < min_area or reg.area > max_area:
            continue
        minr, minc, maxr, maxc = reg.bbox
        h, w = (maxr - minr), (maxc - minc)
        if h == 0 or w == 0:
            continue
        aspect = max(h, w) / max(1, min(h, w))
        if aspect < min_aspect:
            continue
        cy, cx = reg.centroid  # (fila=y, col=x)
        dist, _ = tree.query([cx, cy], k=1)
        if dist <= max_dist_to_wall:
            keep.append([int(round(cx)), int(round(cy))])
    return keep

def dedup_points(points, radius=DEDUP_RADIUS_PX):
    """Agrupa puntos muy cercanos en un único centro (media)."""
    if not points:
        return []
    pts = np.array(points, dtype=float)
    used = np.zeros(len(pts), dtype=bool)
    out = []
    for i in range(len(pts)):
        if used[i]:
            continue
        d = np.linalg.norm(pts - pts[i], axis=1)
        idx = np.where(d <= radius)[0]
        used[idx] = True
        out.append(np.round(pts[idx].mean(axis=0)).astype(int).tolist())
    return out

def is_horizontal(seg, tol=GAP_COLINEAR_TOL_PX):
    (x1,y1),(x2,y2) = (seg["p1"], seg["p2"])
    return abs(y1 - y2) <= tol and abs(x1 - x2) > tol

def is_vertical(seg, tol=GAP_COLINEAR_TOL_PX):
    (x1,y1),(x2,y2) = (seg["p1"], seg["p2"])
    return abs(x1 - x2) <= tol and abs(y1 - y2) > tol

def segment_length(seg):
    (x1,y1),(x2,y2) = (seg["p1"], seg["p2"])
    return int(round(np.hypot(x2-x1, y2-y1)))

def midpoint(xa, xb):
    return int(round((xa + xb)/2))

def detect_gaps_on_exterior(exterior_segments):
    """
    Fallback geométrico: si no hay puertas por color, detecta "huecos"
    entre segmentos exteriores colineales con un gap pequeño.
    Devuelve lista de centros [cx, cy] de los gaps.
    """
    gaps = []

    # --- Horizontales: agrupar por y aproximado ---
    horiz = [s for s in exterior_segments if is_horizontal(s)]
    # agrupa por y (redondeado a GAP_COLINEAR_TOL_PX)
    buckets = {}
    for s in horiz:
        y = int(round((s["p1"][1] + s["p2"][1]) / 2))
        key = int(round(y / max(1, GAP_COLINEAR_TOL_PX)))
        buckets.setdefault(key, []).append(s)

    for key, segs in buckets.items():
        # normaliza cada segmento como (x_min, x_max, y)
        norm = []
        for s in segs:
            x1, x2 = sorted([s["p1"][0], s["p2"][0]])
            y = int(round((s["p1"][1] + s["p2"][1]) / 2))
            norm.append((x1, x2, y))
        # ordena por x
        norm.sort(key=lambda t: t[0])
        # busca gaps entre consecutivos
        for (x1a, x2a, ya), (x1b, x2b, yb) in zip(norm, norm[1:]):
            # gap entre x2a y x1b
            gap = x1b - x2a
            if GAP_MIN_LEN_PX <= gap <= GAP_MAX_LEN_PX and abs(ya - yb) <= GAP_COLINEAR_TOL_PX:
                cx = midpoint(x2a, x1b)
                gaps.append([cx, ya])

    # --- Verticales: agrupar por x aproximado ---
    vert = [s for s in exterior_segments if is_vertical(s)]
    buckets = {}
    for s in vert:
        x = int(round((s["p1"][0] + s["p2"][0]) / 2))
        key = int(round(x / max(1, GAP_COLINEAR_TOL_PX)))
        buckets.setdefault(key, []).append(s)

    for key, segs in buckets.items():
        norm = []
        for s in segs:
            y1, y2 = sorted([s["p1"][1], s["p2"][1]])
            x = int(round((s["p1"][0] + s["p2"][0]) / 2))
            norm.append((y1, y2, x))
        norm.sort(key=lambda t: t[0])
        for (y1a, y2a, xa), (y1b, y2b, xb) in zip(norm, norm[1:]):
            gap = y1b - y2a
            if GAP_MIN_LEN_PX <= gap <= GAP_MAX_LEN_PX and abs(xa - xb) <= GAP_COLINEAR_TOL_PX:
                cy = midpoint(y2a, y1b)
                gaps.append([xa, cy])

    return dedup_points(gaps, radius=DEDUP_RADIUS_PX)


# =========================
# Export por imagen
# =========================
def export_scene(img_path: Path, px_per_m: float = 50.0):
    pil = Image.open(img_path).convert("RGB")
    img = np.array(pil)
    H, W, _ = img.shape

    # Máscaras por color con tolerancia (muros)
    mask_ext = mask_rgb_tol(img, RGB_EXT_WALL)
    mask_int = mask_rgb_tol(img, RGB_INT_WALL)

    # Segmentos de pared (Hough sobre el esqueleto)
    seg_ext = lines_from_mask(mask_ext)
    seg_int = lines_from_mask(mask_int)

    # Aperturas por COLOR (cian) + filtros geométricos
    wall_mask_total = (mask_ext | mask_int)
    openings = []

    mask_fd = mask_cyan_range(img)  # robusto para cian
    fd_pts = opening_centers(
        mask_fd, wall_mask_total,
        min_area=MIN_OPEN_AREA, max_area=MAX_OPEN_AREA,
        max_dist_to_wall=MAX_DIST_TO_WALL, min_aspect=MIN_OPEN_ASPECT
    )
    fd_pts = dedup_points(fd_pts, radius=DEDUP_RADIUS_PX)
    for c in fd_pts:
        openings.append({"type": "front_door", "center": c})

    # Fallback geométrico: si no hay nada por color, busca gaps en exterior
    if len(fd_pts) == 0:
        gap_pts = detect_gaps_on_exterior(seg_ext)
        for c in gap_pts:
            openings.append({"type": "front_door_gap", "center": c})

    scene = {
        "image": img_path.name,
        "size_px": [W, H],
        "px_per_m": px_per_m,
        "walls": {
            "exterior": seg_ext,
            "interior": seg_int
        },
        "openings": openings
    }
    return scene


# =========================
# CLI
# =========================
def main():
    if len(sys.argv) < 3:
        print("Usage: python export_scene_from_intermediate.py <input_folder> <output_folder> [px_per_m]")
        sys.exit(1)
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    px_per_m = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(list(in_dir.glob("*.PNG")) + list(in_dir.glob("*.png"))):
        scene = export_scene(img_path, px_per_m=px_per_m)
        (out_dir / f"{img_path.stem}.scene.json").write_text(
            json.dumps(scene, indent=2), encoding="utf-8"
        )
        print(f"[OK] {img_path.name} -> {img_path.stem}.scene.json")
        count += 1
    print(f"Done. {count} scenes written to {out_dir}")

if __name__ == "__main__":
    main()
