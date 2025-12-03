import numpy as np
from skimage.morphology import erosion, disk

FURNITURE_CATALOG = {
    "MasterRoom": [
        {   
            "type": "bed_master",
            "length_m": 2.0,
            "width_m": 1.6,
            "clearance_m": 0.4,
        },
    ],
}

import numpy as np
import cv2
from skimage.morphology import erosion, disk


def locate_furniture_in_room(gParam, room, furniture_item=FURNITURE_CATALOG["MasterRoom"][0]):
    """
    Coloca un mueble rectangular (por ejemplo la cama principal) dentro de una habitación arbitraria.
    """

    H, W = gParam["height"], gParam["width"]
    px_per_m = gParam["nPixelsPerMeter"]

    # 1) Dimensiones del mueble en píxeles
    length_px = int(round(furniture_item["length_m"] * px_per_m))  # lado largo
    width_px  = int(round(furniture_item["width_m"] * px_per_m))   # lado corto
    clearance_px = int(round(furniture_item.get("clearance_m", 0.0) * px_per_m))

    # Filtro rápido de área: si la habitación tiene muy pocos píxeles, ni lo intentamos
    approx_area_bed = length_px * width_px
    if len(room) < approx_area_bed * 0.7:  # factor optimista
        return None

    # 2) room_mask: forma real de la habitación
    room_mask = np.zeros((H, W), dtype=bool)
    for x, y in room:
        # Por seguridad, clamp a imagen
        if 0 <= x < H and 0 <= y < W:
            room_mask[x, y] = True

    if not room_mask.any():
        return None

    # 3) core_mask: zona de centros seguros usando erosión para respectar clearance
    if clearance_px > 0:
        selem = disk(clearance_px)
        core_mask = erosion(room_mask, selem)
    else:
        core_mask = room_mask.copy()

    if not core_mask.any():
        # No hay zona interior suficiente ni para el centro del mueble
        return None

    # 4) Imagen integral de room_mask para validar rectángulos rápido
    #    room_mask_bool -> 0/1 uint8
    room_mask_uint8 = room_mask.astype(np.uint8)
    # integral tiene tamaño (H+1, W+1)
    integral = cv2.integral(room_mask_uint8)

    def rect_sum(x1, y1, x2, y2):
        """
        Suma de room_mask en el rectángulo [x1:x2] x [y1:y2] (inclusive),
        usando la imagen integral de OpenCV (con offset +1).
        """
        # Ajuste por offset: en integral, (i,j) corresponde a suma hasta (i-1, j-1)
        X1, Y1 = x1,     y1
        X2, Y2 = x2 + 1, y2 + 1

        total = (
            integral[X2, Y2]
            - integral[X1, Y2]
            - integral[X2, Y1]
            + integral[X1, Y1]
        )
        return int(total)

    def rasterize_rectangle(cx, cy, height_px, width_px):
        """
        Rectángulo axis-aligned centrado en (cx, cy).
        height_px -> eje x (vertical)
        width_px  -> eje y (horizontal)
        Devuelve lista de píxeles [(x,y), ...] dentro de la imagen.
        """
        half_h = height_px // 2
        half_w = width_px // 2

        x1 = max(0, cx - half_h)
        x2 = min(H - 1, cx + half_h)
        y1 = max(0, cy - half_w)
        y2 = min(W - 1, cy + half_w)

        pixels = []
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                pixels.append((x, y))
        return pixels, x1, y1, x2, y2

    # 5) Candidatos de centro: píxeles de la habitación que están en core_mask
    #    (podemos muestrear para reducir coste, p.e. cada 2 o 3 píxeles)
    step = 2  # puedes ajustar este step
    candidate_centers = []
    for x, y in room:
        if core_mask[x, y] and (x % step == 0) and (y % step == 0):
            candidate_centers.append((x, y))

    if not candidate_centers:
        return None

    # 6) Probar cada centro con dos orientaciones: Vertical y Horizontal
    #    V: largo en eje x, ancho en eje y
    #    H: largo en eje y, ancho en eje x
    for cx, cy in candidate_centers:
        for orientation, h_px, w_px in [
            ("V", length_px, width_px),
            ("H", width_px, length_px),
        ]:
            # Comprobación básica de que el rectángulo cabe en la imagen
            half_h = h_px // 2
            half_w = w_px // 2
            x1 = cx - half_h
            x2 = cx + half_h
            y1 = cy - half_w
            y2 = cy + half_w

            if x1 < 0 or y1 < 0 or x2 >= H or y2 >= W:
                continue  # se sale de la imagen

            area_rect = (x2 - x1 + 1) * (y2 - y1 + 1)
            # Suma de room_mask dentro del rectángulo
            inside_count = rect_sum(x1, y1, x2, y2)

            if inside_count != area_rect:
                # Algún píxel del rectángulo no pertenece a la habitación
                continue

            # Si pasa el check con integral, generamos la lista de píxeles una vez
            furniture_pixels, _, _, _, _ = rasterize_rectangle(cx, cy, h_px, w_px)

            return {
                "type": furniture_item["type"],
                "cx": cx,
                "cy": cy,
                "height_px": h_px,
                "width_px": w_px,
                "orientation": orientation,
                "pixels": furniture_pixels,
            }

    # Si ningún candidato es válido: no cabe
    return None

