import numpy as np
import cv2
from skimage.morphology import disk, dilation, closing

# ==========================================
# 1. INTERPRETACIÓN DEL PLANO (DE VECTORES A IMAGEN)
# ==========================================

def parse_room_geometry(gParam, room_pixels, lPerimeter, pwalls):
    """ 
    Transforma la geometría vectorial (líneas) en mapas de bits (máscaras).
    
    MEJORA (v2): Clasifica estrictamente los obstáculos en 3 categorías:
    1. WALLS: Muros sólidos (Exterior 'EW' + Interior 'IW'). Aquí se pueden apoyar muebles.
    2. WINDOWS: Ventanas ('WN'). No se pueden tapar con armarios altos.
    3. DOORS: Puertas ('FD', 'ID'). Zona de paso obligatoria.
    """
    H, W = gParam["height"], gParam["width"]
    
    masks = {
        "room_floor": np.zeros((H, W), dtype=bool),
        "room_original": np.zeros((H, W), dtype=bool),  # NUEVO: Píxeles originales sin modificar
        "room_with_tolerance": np.zeros((H, W), dtype=bool),  # NUEVO: Con pequeño margen para muebles
        "walls": np.zeros((H, W), dtype=np.uint8),      # Muros sólidos (soportan muebles)
        "doors": np.zeros((H, W), dtype=np.uint8),      # Huecos de paso
        "windows": np.zeros((H, W), dtype=np.uint8),    # Huecos de luz
        "all_obstacles": np.zeros((H, W), dtype=np.uint8) # Todo lo anterior junto
    }

    # Función auxiliar para pintar líneas
    def draw_line(mask, x1, y1, x2, y2, val=1, thickness=1):
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.line(mask, pt1, pt2, val, thickness=thickness)

    # 1. PROCESAR PERÍMETRO EXTERIOR (lPerimeter)
    # Contiene: 'EW' (Exterior Wall), 'WN' (Window), 'FD' (Front Door)
    if lPerimeter:
        for x1, y1, x2, y2, tag in lPerimeter:
            draw_line(masks["all_obstacles"], x1, y1, x2, y2, thickness=1)
            
            if tag == 'WN':
                draw_line(masks["windows"], x1, y1, x2, y2, thickness=2)
            elif tag == 'FD':
                draw_line(masks["doors"], x1, y1, x2, y2, thickness=2)
            else: 
                # EW (Exterior Wall)
                draw_line(masks["walls"], x1, y1, x2, y2, thickness=2)

    # 2. PROCESAR MUROS INTERIORES (pwalls)
    # Contiene: 'IW' (Interior Wall), 'ID' (Interior Door), 'EW' (Structural Wall)
    if pwalls:
        for x1, y1, x2, y2, tag in pwalls:
            draw_line(masks["all_obstacles"], x1, y1, x2, y2, thickness=1)
            
            if tag == 'ID':
                draw_line(masks["doors"], x1, y1, x2, y2, thickness=2)
            elif tag == 'WN':
                draw_line(masks["windows"], x1, y1, x2, y2, thickness=2)
            else:
                # IW, EW (Interior Wall, Exterior Wall)
                draw_line(masks["walls"], x1, y1, x2, y2, thickness=2)

    # 3. PINTAR EL SUELO (Walkable Area)
    if room_pixels:
        rows, cols = zip(*room_pixels)
        r_arr, c_arr = np.array(rows), np.array(cols)
        valid = (r_arr >= 0) & (r_arr < H) & (c_arr >= 0) & (c_arr < W)
        
        # Guardar los píxeles ORIGINALES sin modificar
        masks["room_original"][r_arr[valid], c_arr[valid]] = True
        
        # room_floor se mantiene igual para compatibilidad
        masks["room_floor"][r_arr[valid], c_arr[valid]] = True

    # 4. REFINAR MÁSCARAS
    # Dilatamos un poco los obstáculos en la máscara general para seguridad
    masks["all_obstacles"] = dilation(masks["all_obstacles"], disk(1))
    
    # Aseguramos que el suelo toque las paredes (llenando gaps de rasterización)
    wall_footprint = (masks["walls"] | masks["windows"] | masks["doors"])
    wall_footprint_dilated = dilation(wall_footprint, disk(2))
    
    masks["room_floor"] = masks["room_floor"] | wall_footprint_dilated.astype(bool)
    
    # MEJORA: Usamos un closing más suave (disk(1) en vez de 2) para evitar fugar
    # el suelo a la habitación vecina si el muro es muy fino.
    masks["room_floor"] = closing(masks["room_floor"], disk(1))
    
    # 5. NUEVO: Calcular room_with_tolerance DESPUÉS de procesar paredes
    # Esto asegura que no se extienda más allá de las paredes del lPerimeter
    if room_pixels:
        # Dilatamos room_original para dar margen a muebles que tocan paredes
        room_expanded = dilation(masks["room_original"], disk(5))
        
        # CLAVE: Recortamos contra las paredes dilatadas
        # Esto evita que room_with_tolerance se extienda más allá del perímetro
        wall_barrier = dilation(wall_footprint, disk(3))
        masks["room_with_tolerance"] = room_expanded & ~wall_barrier.astype(bool)

    return masks

def get_forbidden_mask(gParam, masks, config, include_walls=True, exclude_window_clearance=False):
    """ 
    Crea un mapa de "Zonas Prohibidas" basado en reglas de comodidad.
    Por ejemplo: el arco de apertura de una puerta o el espacio frente a una ventana.
    """
    px_per_m = gParam["nPixelsPerMeter"]
    
    # Empezamos con un mapa limpio
    forbidden = np.zeros_like(masks["room_floor"], dtype=bool)
    
    if include_walls:
        # A veces queremos considerar las paredes como zona prohibida (ej. para armarios)
        forbidden |= masks["walls"].astype(bool)

    # 1. Zona de respeto de la Puerta (para que pueda abrirse)
    if np.any(masks["doors"]):
        r_px = int(config["door_radius"] * px_per_m)
        if r_px > 0:
            forbidden |= dilation(masks["doors"], disk(r_px)).astype(bool)

    # 2. Zona de respeto de la Ventana (para poder asomarse)
    # Solo si NO estamos excluyéndola (ej. para poner cama debajo)
    if not exclude_window_clearance and np.any(masks["windows"]):
        r_px = int(config["window_clearance"] * px_per_m)
        if r_px > 0:
            forbidden |= dilation(masks["windows"], disk(r_px)).astype(bool)
            
    return forbidden

# ==========================================
# 2. SISTEMA DE VALIDACIÓN (EL ÁRBITRO)
# ==========================================

def check_placement(masks, forbidden, x1, y1, x2, y2, allow_windows=False, return_reason=False):
    """ 
    Verifica si un rectángulo (mueble) cabe en una posición.
    
    Args:
        return_reason: Si True, devuelve (bool, str) con el motivo del fallo/éxito
    """
    H, W = masks["room_floor"].shape
    
    # 1. Chequeo básico: ¿Está dentro de la imagen?
    if x1 < 0 or y1 < 0 or x2 >= H or y2 >= W:
        return (False, "bounds") if return_reason else False
    if x1 >= x2 or y1 >= y2:
        return (False, "bounds") if return_reason else False
    
    sl = (slice(x1, x2), slice(y1, y2))

    # 2. Colisión Física: Paredes y Puertas son IMPENETRABLES
    if np.any(masks["walls"][sl]):
        return (False, "walls") if return_reason else False
    if np.any(masks["doors"][sl]):
        return (False, "doors") if return_reason else False
    
    # Ventanas: Prohibidas por defecto, permitidas para camas compactas
    if not allow_windows:
        # IMPORTANTE: Las ventanas suelen estar en la pared (fuera del suelo).
        # El mueble está en el suelo. Si solo miramos solapamiento directo, nunca chocarán.
        # Debemos verificar si el mueble está "pegado" a una ventana.
        
        # Opción A: Verificar dilatación de ventanas
        # Pero dilatar en tiempo real es costoso. 
        # Verificamos si hay alguna ventana en el borde dilatado del slice?
        
        # Mejor: Usamos una 'window_aura' precalculada si existe, o dilatamos el slice localmente.
        # Para ser eficientes, miramos si 'windows_aura' está en masks. Si no, dilatamos 'windows'.
        
        window_mask_to_check = masks.get("windows_aura", masks["windows"])
        
        # Si usamos windows normal (línea fina), necesitamos ampliar la búsqueda
        # Ampliamos el slice de búsqueda 5 píxeles para asegurar que no toque
        margin_w = 5
        x1_w, x2_w = max(0, x1-margin_w), min(H, x2+margin_w)
        y1_w, y2_w = max(0, y1-margin_w), min(W, y2+margin_w)
        
        if np.any(masks["windows"][x1_w:x2_w, y1_w:y2_w]):
             return (False, "windows") if return_reason else False

    # 3. Zonas de Respeto (radios de puertas, otros muebles)
    if np.any(forbidden[sl]):
        return (False, "forbidden") if return_reason else False
    
    # 4. Validación contra room_original (píxeles exactos de la imagen)
    if "room_original" in masks and np.any(masks["room_original"]):
        rect_original = masks["room_original"][sl]
        if rect_original.size > 0:
            coverage = np.sum(rect_original) / rect_original.size
            if coverage < 0.95:
                return (False, "room_original") if return_reason else False
        
    # 5. Validación de Suelo
    rect_floor = masks["room_floor"][sl]
    if rect_floor.size == 0:
        return (False, "floor") if return_reason else False
    if np.sum(rect_floor) < rect_floor.size:
        return (False, "floor") if return_reason else False
        
    return (True, "success") if return_reason else True

# ==========================================
# 3. BÚSQUEDA Y COLOCACIÓN DE MUEBLES (HELPERS)
# ==========================================

def try_place_simple(masks, forbidden, points, size_m, px_per_m, allow_windows=False, debug=False):
    """
    Prueba a colocar el mueble en las 4 direcciones alrededor de cada punto de pared.
    Fuerza bruta inteligente: Solo comprueba si cabe y toca la pared por detrás.
    """
    H, W = masks["room_floor"].shape
    largo_m, profundo_m = size_m
    largo_px = int(largo_m * px_per_m)
    profundo_px = int(profundo_m * px_per_m)
    
    candidates = []
    step = 2
    limit = 3000
    points_to_try = list(points)
    
    # Gaps a probar (para robustez)
    gaps = [0, 1]
    
    for i in range(0, min(len(points_to_try), limit), step):
        wx, wy = points_to_try[i]
        
        # Probar las 4 orientaciones posibles "pegado" a este punto
        proposals = []
        
        # UP -> Mueble se extiende en +X, centrado en Y
        for g in gaps:
             tx1 = wx + g
             tx2 = wx + g + profundo_px
             ty1 = wy - largo_px // 2
             ty2 = wy + largo_px // 2
             proposals.append((tx1, tx2, ty1, ty2, "BACK_UP"))
        
        # DOWN -> Mueble se extiende en -X
        for g in gaps:
             tx2 = wx - g + 1
             tx1 = wx - g + 1 - profundo_px
             ty1 = wy - largo_px // 2
             ty2 = wy + largo_px // 2
             proposals.append((tx1, tx2, ty1, ty2, "BACK_DOWN"))

        # LEFT -> Mueble se extiende en +Y
        for g in gaps:
             tx1 = wx - largo_px // 2
             tx2 = wx + largo_px // 2
             ty1 = wy + g
             ty2 = wy + g + profundo_px
             proposals.append((tx1, tx2, ty1, ty2, "BACK_LEFT"))
             
        # RIGHT -> Mueble se extiende en -Y
        for g in gaps:
             tx1 = wx - largo_px // 2
             tx2 = wx + largo_px // 2
             ty2 = wy - g + 1
             ty1 = wy - g + 1 - profundo_px
             proposals.append((tx1, tx2, ty1, ty2, "BACK_RIGHT"))
        
        for p in proposals:
            px1, px2, py1, py2, pori = p
            
            # Check básico de colisiones
            if check_placement(masks, forbidden, px1, py1, px2, py2, allow_windows):
                # Check adicional: ¿Realmente toca pared por el lado "BACK"?
                touches = False
                margin = 3
                if pori == "BACK_UP":
                    touches = np.any(masks["walls"][max(0, px1-margin):px1+1, py1:py2])
                elif pori == "BACK_DOWN":
                    touches = np.any(masks["walls"][px2:min(H, px2+margin), py1:py2])
                elif pori == "BACK_LEFT":
                    touches = np.any(masks["walls"][px1:px2, max(0, py1-margin):py1+1])
                elif pori == "BACK_RIGHT":
                    touches = np.any(masks["walls"][px1:px2, py2:min(W, py2+margin)])
                
                if touches:
                    candidates.append({
                        "x1": int(px1), "x2": int(px2), 
                        "y1": int(py1), "y2": int(py2), 
                        "orientation": pori
                    })

        if len(candidates) >= 100: return candidates

    return candidates

def try_place_strict(masks, forbidden, points, size_m, px_per_m, gaps=[0, 1], allow_windows=False, debug=False):
    """
    Coloca mueble asegurando que el lado LARGO esté SIEMPRE paralelo a la pared.
    """
    H, W = masks["room_floor"].shape
    largo_m, profundo_m = size_m
    largo_px = int(largo_m * px_per_m)    # Lado que va PARALELO a la pared
    profundo_px = int(profundo_m * px_per_m)  # Lado PERPENDICULAR (hacia el interior)
    
    points_to_try = list(points)
    step = 2 
    limit = 3000 
    
    candidates = []
    
    # Contadores de debug
    dbg_no_wall = 0
    dbg_no_pts = 0
    dbg_no_floor = 0
    dbg_check_fail = 0
    
    for i in range(0, min(len(points_to_try), limit), step):
        px_pt, py_pt = points_to_try[i]
        
        # 1. ENCONTRAR EL PÍXEL DE PARED MÁS CERCANO
        r_search = 8
        r_min, r_max = max(0, px_pt-r_search), min(H, px_pt+r_search+1)
        c_min, c_max = max(0, py_pt-r_search), min(W, py_pt+r_search+1)
        local_walls = masks["walls"][r_min:r_max, c_min:c_max]
        wall_pts_local = np.argwhere(local_walls)
        
        if len(wall_pts_local) == 0:
            dbg_no_wall += 1
            continue
        
        # Convertir a coordenadas globales
        wall_pts_global = wall_pts_local + [r_min, c_min]
        dists = np.abs(wall_pts_global[:, 0] - px_pt) + np.abs(wall_pts_global[:, 1] - py_pt)
        idx_min = np.argmin(dists)
        wx, wy = wall_pts_global[idx_min]  # Punto de pared más cercano
        
        # 2. DETERMINAR ORIENTACIÓN DE LA PARED (Horizontal vs Vertical)
        win = 8
        w_r_min, w_r_max = max(0, wx-win), min(H, wx+win+1)
        w_c_min, w_c_max = max(0, wy-win), min(W, wy+win+1)
        neighborhood = masks["walls"][w_r_min:w_r_max, w_c_min:w_c_max]
        pts_n = np.argwhere(neighborhood)
        
        if len(pts_n) < 2:
            dbg_no_pts += 1
            continue
        
        span_x = np.max(pts_n[:, 0]) - np.min(pts_n[:, 0])
        span_y = np.max(pts_n[:, 1]) - np.min(pts_n[:, 1])
        
        # 3. GENERAR PROYECCIÓN ÚNICA BASADA EN ORIENTACIÓN
        projections = []
        
        if span_y > span_x: # Pared HORIZONTAL
            is_floor_below = False
            for offset in range(1, 9):
                if (wx+offset < H) and masks["room_original"][wx+offset, wy]:
                    is_floor_below = True
                    break
            
            is_floor_above = False
            if not is_floor_below:
                for offset in range(1, 9):
                    if (wx-offset >= 0) and masks["room_original"][wx-offset, wy]:
                        is_floor_above = True
                        break
            
            if is_floor_below:
                # Pared ARRIBA -> Mueble ABAJO
                found_gap = False
                for gap in gaps:
                    tx1 = wx + gap
                    tx2 = wx + gap + profundo_px
                    ty1 = wy - largo_px // 2
                    ty2 = wy + largo_px // 2
                    if check_placement(masks, forbidden, tx1, ty1, tx2, ty2, allow_windows):
                         projections.append((tx1, tx2, ty1, ty2, "BACK_UP"))
                         found_gap = True
                         break
                if not found_gap: dbg_check_fail += 1

            elif is_floor_above:
                # Pared ABAJO -> Mueble ARRIBA
                found_gap = False
                for gap in gaps:
                    tx2 = wx - gap + 1
                    tx1 = wx - gap + 1 - profundo_px
                    ty1 = wy - largo_px // 2
                    ty2 = wy + largo_px // 2
                    if check_placement(masks, forbidden, tx1, ty1, tx2, ty2, allow_windows):
                         projections.append((tx1, tx2, ty1, ty2, "BACK_DOWN"))
                         found_gap = True
                         break
                if not found_gap: dbg_check_fail += 1
            else:
                dbg_no_floor += 1
                
        else: # Pared VERTICAL
            is_floor_right = False
            for offset in range(1, 9):
                if (wy+offset < W) and masks["room_original"][wx, wy+offset]:
                    is_floor_right = True
                    break
            
            is_floor_left = False
            if not is_floor_right:
                for offset in range(1, 9):
                    if (wy-offset >= 0) and masks["room_original"][wx, wy-offset]:
                        is_floor_left = True
                        break
            
            if is_floor_right:
                # Pared IZQUIERDA -> Mueble DERECHA
                found_gap = False
                for gap in gaps:
                    tx1 = wx - largo_px // 2
                    tx2 = wx + largo_px // 2
                    ty1 = wy + gap
                    ty2 = wy + gap + profundo_px
                    if check_placement(masks, forbidden, tx1, ty1, tx2, ty2, allow_windows):
                         projections.append((tx1, tx2, ty1, ty2, "BACK_LEFT"))
                         found_gap = True
                         break
                if not found_gap: dbg_check_fail += 1

            elif is_floor_left:
                # Pared DERECHA -> Mueble IZQUIERDA
                found_gap = False
                for gap in gaps:
                    tx1 = wx - largo_px // 2
                    tx2 = wx + largo_px // 2
                    ty2 = wy - gap + 1
                    ty1 = wy - gap + 1 - profundo_px
                    if check_placement(masks, forbidden, tx1, ty1, tx2, ty2, allow_windows):
                         projections.append((tx1, tx2, ty1, ty2, "BACK_RIGHT"))
                         found_gap = True
                         break
                if not found_gap: dbg_check_fail += 1
            else:
                dbg_no_floor += 1
        
        for px1, px2, py1, py2, pori in projections:
             candidates.append({
                    "x1": int(px1), "x2": int(px2), 
                    "y1": int(py1), "y2": int(py2), 
                    "orientation": pori
             })
             if len(candidates) >= 300: return candidates

    if debug:
        print(f"   [DEBUG] try_place_strict (Global): rejections: no_wall={dbg_no_wall}, no_pts={dbg_no_pts}, no_floor={dbg_no_floor}, check_fail={dbg_check_fail}")
    
    return candidates

# ==========================================
# 4. BÚSQUEDA Y COLOCACIÓN DE MUEBLES (ESPECÍFICOS)
# ==========================================

def find_wardrobe_for_bed(gParam, masks, forbidden, bed_data, config):
    """ 
    Intenta colocar un armario en la habitación, respetando la cama que ya se puso.
    Busca colocarlo siempre con la espalda pegada a la pared.
    """
    px_per_m = gParam["nPixelsPerMeter"]
    
    current_forbidden = forbidden.copy()
    use_margin = int(0.60 * px_per_m) 
    
    # "Quemamos" la cama en el mapa de prohibidos para que el armario no la pise
    bed_mask = np.zeros_like(masks["room_floor"], dtype=np.uint8)
    bx1, bx2 = max(0, bed_data["x1"]), min(masks["room_floor"].shape[0], bed_data["x2"])
    by1, by2 = max(0, bed_data["y1"]), min(masks["room_floor"].shape[1], bed_data["y2"])
    bed_mask[bx1:bx2 + 1, by1:by2 + 1] = 1 
    
    # Añadimos un pequeño margen de confort alrededor de la cama
    bed_aura = dilation(bed_mask, disk(use_margin))
    current_forbidden |= bed_aura.astype(bool)
    
    depth_px = int(0.60 * px_per_m) # Profundidad estándar de un armario (60cm)
    
    # Definimos anchos de armario según el nivel de lujo configurado
    if config["name"] == "Luxury": widths = [2.0, 1.8, 1.5]
    elif config["name"] == "Standard": widths = [1.5, 1.2, 1.0]
    elif config["name"] == "Compact": widths = [1.0, 0.8]
    else: widths = [0.8, 0.6] 
    
    # Buscamos puntos cercanos a las paredes (dentro de room_original)
    wall_aura = dilation(masks["walls"], disk(5)) & masks["room_original"]
    candidates = np.argwhere(wall_aura)
    if len(candidates) == 0: return None
    
    wall_margin = 3
    step = 4
    
    for width_m in widths:
        w_px = int(width_m * px_per_m) # Ancho del armario (Lado Largo)
        
        for i in range(0, len(candidates), step):
            cx, cy = candidates[i]
            
            # --- LÓGICA DE ORIENTACIÓN INTELIGENTE ---
            # No probamos al azar. Miramos dónde está la pared respecto al punto (cx, cy)
            # para orientar el armario correctamente (espalda contra pared).
            
            # Definimos las 4 orientaciones posibles
            possible_rects = [
                # 1. Pared Arriba -> Armario horizontal pegado arriba
                (cx + wall_margin, cx + wall_margin + depth_px, cy - w_px//2, cy + w_px//2, "BACK_UP"),
                
                # 2. Pared Abajo -> Armario horizontal pegado abajo
                (cx - depth_px - wall_margin, cx - wall_margin, cy - w_px//2, cy + w_px//2, "BACK_DOWN"),
                
                # 3. Pared Izquierda -> Armario vertical pegado a la izquierda
                (cx - w_px//2, cx + w_px//2, cy + wall_margin, cy + wall_margin + depth_px, "BACK_LEFT"),
                
                # 4. Pared Derecha -> Armario vertical pegado a la derecha
                (cx - w_px//2, cx + w_px//2, cy - depth_px - wall_margin, cy - wall_margin, "BACK_RIGHT")
            ]
            
            # FILTRADO: Miramos qué hay alrededor del punto para elegir la mejor opción primero
            priority_rects = []
            secondary_rects = []
            
            H, W = masks["walls"].shape
            # Detectamos muros adyacentes
            is_wall_up = (cx > 0) and masks["walls"][cx-1, cy]
            is_wall_down = (cx < H-1) and masks["walls"][cx+1, cy]
            is_wall_left = (cy > 0) and masks["walls"][cx, cy-1]
            is_wall_right = (cy < W-1) and masks["walls"][cx, cy+1]
            
            # Si hay pared arriba, priorizamos poner la espalda arriba, etc.
            if is_wall_up: priority_rects.append(possible_rects[0])
            elif is_wall_down: priority_rects.append(possible_rects[1])
            elif is_wall_left: priority_rects.append(possible_rects[2])
            elif is_wall_right: priority_rects.append(possible_rects[3])
            
            # Guardamos las otras opciones por si acaso (fallback)
            for r in possible_rects:
                if r not in priority_rects:
                    secondary_rects.append(r)
            
            # Probamos las opciones lógicas primero
            all_trials = priority_rects + secondary_rects
            
            for x1, x2, y1, y2, ori in all_trials:
                if check_placement(masks, current_forbidden, x1, y1, x2, y2):
                    return {
                        "type": "wardrobe",
                        "x1": int(x1), "x2": int(x2), "y1": int(y1), "y2": int(y2),
                        "width_m": width_m,
                        "orientation": ori 
                    }
    return None

def solve_master_bedroom(gParam, room_pixels, lPerimeter, pwalls, debug=False):
    """ 
    Función Principal: Orquesta todo el proceso. 
    
    REFACTORIZADO: Usa estrategia de capas (layered) como solve_single_bedroom.
    """
    masks = parse_room_geometry(gParam, room_pixels, lPerimeter, pwalls)
    px_per_m = gParam["nPixelsPerMeter"]
    
    if not room_pixels or len(room_pixels) == 0:
        return None
    
    # DEBUG: Info básica de la habitación
    if debug:
        room_area_px = np.sum(masks["room_original"])
        room_area_m2 = room_area_px / (px_per_m ** 2)
        print(f"   [DEBUG] MasterBedroom: Área={room_area_m2:.2f}m²")

    # Refinamiento máscaras
    masks = masks.copy()
    room_context = dilation(masks["room_original"], disk(10))
    masks["doors"] = masks["doors"] & room_context

    # Definimos niveles de calidad, de mejor a peor
    # Tamaños de cama: (Ancho, Largo)
    bed_configs = [
        {"name": "Luxury",   "bed_size": (1.60, 2.00)},
        {"name": "Standard", "bed_size": (1.50, 1.90)},
        {"name": "Compact",  "bed_size": (1.35, 1.90)},
        {"name": "Emergency","bed_size": (1.20, 1.80)},
    ]
    
    # Tamaños de armario: (Ancho, Profundidad)
    wardrobe_sizes = [(1.50, 0.60), (1.20, 0.60), (1.00, 0.55), (0.80, 0.50)]
    
    # Zona prohibida (Puertas)
    config = {"door_radius": 0.70, "window_clearance": 0.0}
    forbidden = get_forbidden_mask(gParam, masks, config, include_walls=True)
    current_forbidden = forbidden.copy()
    
    items = []
    
    # Generar puntos de pared por capas (1-7 px)
    wall_points_layered = []
    for dist in range(1, 8):
        current_layer = dilation(masks["walls"], disk(dist))
        previous_layer = dilation(masks["walls"], disk(dist-1))
        layer_mask = (current_layer ^ previous_layer) & masks["room_original"]
        points_in_layer = np.argwhere(layer_mask)
        if len(points_in_layer) > 0:
            wall_points_layered.extend(list(points_in_layer))
            
    wall_points = list(wall_points_layered)
    
    if not wall_points:
        if debug: print("   [DEBUG] MasterBedroom: No se encontraron puntos de pared.")
        return None

    # 1. COLOCAR CAMA DOBLE (Permite ventana, cualquier orientación)
    bed_placed = False
    config_used = "Unknown"
    
    for bed_cfg in bed_configs:
        bed_size = bed_cfg["bed_size"]
        
        # Usamos try_place_simple: solo chequea que toque pared y quepa
        cands = try_place_simple(masks, current_forbidden, wall_points, bed_size, px_per_m, 
                                  allow_windows=True, debug=False)
        if cands:
            best = cands[0]
            items.append({
                "type": "bed_double",
                "x1": best["x1"], "x2": best["x2"], "y1": best["y1"], "y2": best["y2"],
                "orientation": best["orientation"],
                "config": f"Cama {bed_size}"
            })
            
            # Actualizar forbidden con margen
            cmargin = int(0.40 * px_per_m)  # 40cm margen
            gx1, gx2 = max(0, best["x1"]-cmargin), min(masks["all_obstacles"].shape[0], best["x2"]+cmargin)
            gy1, gy2 = max(0, best["y1"]-cmargin), min(masks["all_obstacles"].shape[1], best["y2"]+cmargin)
            current_forbidden[gx1:gx2, gy1:gy2] = True
            
            if debug:
                print(f"   [DEBUG] MasterBedroom: Cama colocada ({bed_size}) en {best['orientation']}")
            bed_placed = True
            config_used = bed_cfg["name"]
            break
    
    if not bed_placed:
        if debug: print("   [DEBUG] MasterBedroom: No se pudo colocar la cama.")
        return None

    # 2. COLOCAR ARMARIO (NO ventana, lado LARGO paralelo a pared)
    wardrobe_placed = False
    
    # Primero intentamos try_place_strict
    for wardrobe_size in wardrobe_sizes:
        cands = try_place_strict(masks, current_forbidden, wall_points, wardrobe_size, px_per_m,
                                  gaps=[0, 1, 2, 3, 4, 5], allow_windows=False)
        if cands:
            best = cands[0]
            items.append({
                "type": "wardrobe",
                "x1": best["x1"], "x2": best["x2"], "y1": best["y1"], "y2": best["y2"],
                "orientation": best["orientation"],
                "config": f"Armario {wardrobe_size}"
            })
            
            if debug:
                print(f"   [DEBUG] MasterBedroom: Armario colocado ({wardrobe_size}) en {best['orientation']}")
            wardrobe_placed = True
            break
    
    # Fallback: try_place_simple si strict falla
    if not wardrobe_placed:
        if debug: print("   [DEBUG] MasterBedroom: Strict falló para armario, probando simple...")
        for wardrobe_size in wardrobe_sizes:
            cands = try_place_simple(masks, current_forbidden, wall_points, wardrobe_size, px_per_m,
                                      allow_windows=False, debug=False)
            if cands:
                best = cands[0]
                items.append({
                    "type": "wardrobe",
                    "x1": best["x1"], "x2": best["x2"], "y1": best["y1"], "y2": best["y2"],
                    "orientation": best["orientation"],
                    "config": f"Armario {wardrobe_size} (Simple)"
                })
                if debug:
                    print(f"   [DEBUG] MasterBedroom: Armario (Simple) colocado ({wardrobe_size})")
                wardrobe_placed = True
                break
    
    if not wardrobe_placed:
        if debug: print("   [DEBUG] MasterBedroom: No se pudo colocar armario, solo cama.")
    
    return {
        "success": wardrobe_placed,  # success=True solo si tenemos ambos
        "config_level": config_used + (" (Completo)" if wardrobe_placed else " (Solo Cama)"),
        "items": items
    }


# ==========================================
# 4. DINING ROOM - Mesa en el centro
# ==========================================

def solve_dining_room(gParam, room_pixels, lPerimeter, pwalls, debug=False):
    """
    Coloca una mesa en el comedor (Dining Room).
    
    Estrategia:
    1. Primero intenta el centro (centroide)
    2. Si no cabe, busca posiciones cerca de paredes
    """
    masks = parse_room_geometry(gParam, room_pixels, lPerimeter, pwalls)
    px_per_m = gParam["nPixelsPerMeter"]
    
    # Configuración de la mesa
    table_configs = [
        {"name": "ExtraLarge", "size": (2.0, 1.5)},
        {"name": "Large", "size": (1.5, 1.0)},
        {"name": "Medium", "size": (1.20, 0.80)},
        {"name": "Small", "size": (1.00, 0.60)},    
    ]
    
    if not room_pixels or len(room_pixels) == 0:
        return None
    
    rows, cols = zip(*room_pixels)
    centroid_x = int(np.mean(rows))
    centroid_y = int(np.mean(cols))
    
    # MEJORA: Buscar el punto más alejado de paredes (mejor centro real)
    # Usamos erosión para encontrar el "núcleo" de la habitación
    from scipy.ndimage import distance_transform_edt
    
    # Crear máscara de obstáculos (paredes + ventanas + puertas)
    obstacles = masks["walls"] | masks["windows"] | masks["doors"]
    
    # Calcular distancia de cada punto de room_original a los obstáculos
    safe_zone = masks["room_original"].astype(np.uint8)
    safe_zone[obstacles > 0] = 0  # Quitar obstáculos del suelo
    
    if np.any(safe_zone):
        # distance_transform_edt da la distancia al borde más cercano (0s)
        dist_map = distance_transform_edt(safe_zone)
        
        # El punto con máxima distancia es el "centro más seguro"
        max_dist_idx = np.unravel_index(np.argmax(dist_map), dist_map.shape)
        best_center_x, best_center_y = max_dist_idx
        max_dist_px = dist_map[best_center_x, best_center_y]
    else:
        best_center_x, best_center_y = centroid_x, centroid_y
        max_dist_px = 0
    
    if debug:
        room_area_m2 = len(room_pixels) / (px_per_m ** 2)
        print(f"   [DEBUG] Dining Room: Área={room_area_m2:.2f}m², Centroide=({centroid_x}, {centroid_y}), MejorCentro=({best_center_x}, {best_center_y}), DistMax={max_dist_px/px_per_m:.2f}m")
    
    # Zona prohibida mínima (solo puertas)
    config_minimal = {"door_radius": 0.50, "window_clearance": 0.0}
    forbidden = get_forbidden_mask(gParam, masks, config_minimal, include_walls=False)
    
    def try_place_table(cx, cy, cfg, debug_table=False):
        """Intenta colocar mesa centrada en (cx, cy)"""
        table_w_m, table_l_m = cfg["size"]
        w_px = int(table_w_m * px_per_m)
        l_px = int(table_l_m * px_per_m)
        
        for dim_x, dim_y, ori in [(l_px, w_px, "H"), (w_px, l_px, "V")]:
            x1 = cx - dim_x // 2
            x2 = x1 + dim_x  # Usar x1 + dim para mantener dimensión exacta
            y1 = cy - dim_y // 2
            y2 = y1 + dim_y  # Usar y1 + dim para mantener dimensión exacta
            
            valid, reason = check_placement(masks, forbidden, x1, y1, x2, y2, allow_windows=True, return_reason=True)
            if debug_table:
                print(f"   [DEBUG] Mesa {cfg['name']} ori={ori}: ({x1},{y1})-({x2},{y2}) -> {reason}")
            
            if valid:
                return {
                    "type": "dining_table",
                    "x1": int(x1), "x2": int(x2), 
                    "y1": int(y1), "y2": int(y2),
                    "orientation": ori,
                    "size_m": cfg["size"]
                }
        return None
    
    # Estrategia 1: Centro de la habitación (USAR CENTROIDE)
    for cfg in table_configs:
        result = try_place_table(centroid_x, centroid_y, cfg, debug_table=debug)
        if result:
            return {"success": True, "config_level": cfg["name"] + " (Centro)", "items": [result]}
    
    if debug:
        print("   [DEBUG] Mesa no cabe en centro, buscando en paredes...")
    
    # Estrategia 2: Cerca de paredes (dentro de room_original)
    wall_aura = dilation(masks["walls"], disk(8)) & masks["room_original"]
    candidates = np.argwhere(wall_aura)
    
    if len(candidates) > 0:
        step = 1  # Muestrear todos los puntos
        for cfg in table_configs:
            for i in range(0, len(candidates), step):
                cx, cy = candidates[i]
                result = try_place_table(cx, cy, cfg, debug_table=False)  # Sin debug para no llenar la consola
                if result:
                    return {"success": True, "config_level": cfg["name"] + " (Pared)", "items": [result]}
    
    if debug:
        print("   [DEBUG] No se encontró posición válida para la mesa")
    
    return None


# ==========================================
# 5. KITCHEN - Encimera y Nevera
# ==========================================

def solve_kitchen(gParam, room_pixels, lPerimeter, pwalls, debug=False):
    """
    Coloca muebles de cocina: encimera y nevera pegados a la pared.
    
    Estrategias de distribución:
    1. Busca la pared más larga para la encimera (más superficie de trabajo)
    2. Prioriza esquinas para la nevera (ocupa menos espacio útil)
    3. Intenta poner nevera en pared diferente a la encimera
    """
    masks = parse_room_geometry(gParam, room_pixels, lPerimeter, pwalls)
    px_per_m = gParam["nPixelsPerMeter"]
    
    if not room_pixels or len(room_pixels) == 0:
        if debug:
            print(f"   [DEBUG] Kitchen: No hay room_pixels")
        return None
    
    room_area_m2 = len(room_pixels) / (px_per_m ** 2)
    if debug:
        print(f"   [DEBUG] Kitchen: Área={room_area_m2:.2f}m²")

    # PRE-PROCESAMIENTO: Filtrar ventanas locales
    # Para evitar que ventanas de habitaciones vecinas afecten por proximidad,
    # limitamos la máscara 'windows' a la zona cercana a 'room_original'.
    # Usamos una copia local de masks para no afectar a otras llamadas.
    masks = masks.copy()
    
    # Dilatamos room_original 10px para capturar el perímetro y un margen de seguridad
    room_context = dilation(masks["room_original"], disk(10))
    # También filtramos PUERTAS para evitar radios de apertura fantasma de vecinos
    masks["doors"] = masks["doors"] & room_context
    
    # REFINAMIENTO DE VENTANAS (SOLO COCINA): Usar dilatación rectangular
    # El usuario pide que la ventana invalide como un rectángulo estricto.
    from skimage.morphology import square
    # Usamos un cuadrado de tamaño considerable para asegurar que el margen de 5px
    # (que usaremos en check_placement) esté visualmente representado en el aura si fuera necesario,
    # aunque check_placement usa la máscara 'windows' base + margen dinámico.
    # Pero si alguna lógica usa 'windows_aura', que sea cuadrada.
    masks["windows_aura"] = dilation(masks["windows"], square(5))
    # doors_aura se genera dentro de get_forbidden_mask, así que con filtrar masks["doors"] basta
    # SI get_forbidden_mask usa masks["doors"] directamente.
    
    # Configuración de muebles - Tamaño base más pequeño para cocinas pequeñas
    base_size = (1.0, 0.55) if room_area_m2 < 4 else (1.2, 0.60)
    
    # Zona prohibida: door_radius
    door_r = 0.40 if room_area_m2 < 4 else (0.50 if room_area_m2 < 5 else 0.70)
    
    config = {"door_radius": door_r, "window_clearance": 0.0}
    forbidden = get_forbidden_mask(gParam, masks, config, include_walls=True)
    
    items = []
    current_forbidden = forbidden.copy()
    wall_margin = 0 
    H, W = masks["room_floor"].shape
    
    # Clasificar puntos de pared por orientación y capas
    wall_points_layered = []
    for dist in range(1, 7):
        current_layer = dilation(masks["walls"], disk(dist))
        previous_layer = dilation(masks["walls"], disk(dist-1))
        layer_mask = (current_layer ^ previous_layer) & masks["room_original"]
        points_in_layer = np.argwhere(layer_mask)
        if len(points_in_layer) > 0:
            wall_points_layered.extend(list(points_in_layer))
            
    wall_points = np.array(wall_points_layered) if wall_points_layered else np.zeros((0, 2), dtype=int)
    
    if debug:
        print(f"   [DEBUG] Kitchen: wall_points encontrados={len(wall_points)}")
    
    if len(wall_points) == 0:
        if debug:
            print(f"   [DEBUG] Kitchen: FALLO - No hay wall_points")
        return None
    
    # ---------------------------------------------------------
    # ESTRATEGIA MODULAR: Bloque Único Expandido + Subdivisión
    # ---------------------------------------------------------
    
    # 1. Encontrar Posición Base (ignorando ventanas por petición usuario)
    base_item = None
    
    # Definimos try_place (reutilizamos la existente o asumimos que está disponible en scope)
    # Como está definida DENTRO de solve_kitchen en el código original, necesitamos redefinirla o confiar.
    # El replacement anterior reemplazo el cuerpo PERO mantenía try_place dentro?
    # NO, try_place es una inner function. Debo incluirla si reemplazo todo el cuerpo.
    # O esperar, el replace anterior fue parcial.
    # SI ESTOY REEMPLAZANDO DESDE LINEA 535, he perdido try_place si no la incluyo.
    # PERO try_place estaba definida mas abajo en el original? Sí, linea 595.
    # Así que DEBO incluir la definición de try_place y las utilidades.
    
    # MEJOR ESTRATEGIA PARA NO REPETIR CODIGO GIGANTE:
    # Usar las funciones auxiliares que ya están ahí si no las he borrado.
    # Pero el usuario pide un reemplazo total de la lógica.
    
    # Voy a asumir que try_place y sorted_walls están disponibles O las redefino brevemente.
    # Para seguridad, usaré la lógica "inline" simplificada o mantendré la estructura.
    
    # Requerimos wall_groups y sorted_walls para try_place
    rows, cols = zip(*room_pixels)
    cx, cy = int(np.mean(rows)), int(np.mean(cols))
    wall_groups = {"UP": [], "DOWN": [], "LEFT": [], "RIGHT": []}
    for px, py in wall_points:
        if px < cx - 5: wall_groups["UP"].append((px, py))
        elif px > cx + 5: wall_groups["DOWN"].append((px, py))
        elif py < cy - 5: wall_groups["LEFT"].append((px, py))
        elif py > cy + 5: wall_groups["RIGHT"].append((px, py))
    sorted_walls = sorted(wall_groups.items(), key=lambda x: len(x[1]), reverse=True)

    # Buscar bloque base - Evaluamos TODOS los candidatos
    all_wall_pts = list(wall_points)
    
    if debug:
        print(f"   [DEBUG] Kitchen: Buscando con tamaño base={base_size}")
    
    # Intento 1: Strict Gaps [0, 1]
    candidates = try_place_strict(masks, current_forbidden, all_wall_pts, base_size, px_per_m, gaps=[0, 1], allow_windows=True, debug=debug)
    
    # Fallback 1: Larger Gaps si falla estricto
    if not candidates:
        if debug: print("   [DEBUG] Falló gap estricto, probando gaps más grandes [0..4]...")
        candidates = try_place_strict(masks, current_forbidden, all_wall_pts, base_size, px_per_m, gaps=[0, 1, 2, 3, 4], allow_windows=True, debug=debug)
    
    # Fallback 2: Tamaño menor (Solo si falla lo anterior)
    if not candidates:
        # Solo probar compacto si base_size no era ya compacto
        if base_size != (1.0, 0.55):
             if debug: print("   [DEBUG] Falló base size, probando compacta (1.0x0.55)...")
             candidates = try_place_strict(masks, current_forbidden, all_wall_pts, (1.0, 0.55), px_per_m, gaps=[0, 1], allow_windows=True, debug=debug)
             if not candidates:
                 # Fallback 3: Tamaño menor + Gaps grandes
                 if debug: print("   [DEBUG] Falló compacta estricta, probando compacta gaps grandes...")
                 candidates = try_place_strict(masks, current_forbidden, all_wall_pts, (1.0, 0.55), px_per_m, gaps=[0, 1, 2, 3, 4], allow_windows=True, debug=debug)
    
    # Fallback 4: Tamaño mini
    if not candidates:
         if debug: print("   [DEBUG] Falló compacta, probando mini (0.8x0.55)...")
         candidates = try_place_strict(masks, current_forbidden, all_wall_pts, (0.8, 0.55), px_per_m, gaps=[0, 1, 2, 3, 4], allow_windows=True, debug=debug)
    
    # ULTIMO RECURSO: Fuerza bruta simple (try_place_simple)
    if not candidates:
        if debug: print("   [DEBUG] Strict falló. INTENTO DE ULTIMO RECURSO: Fuerza bruta simple...")
        candidates = try_place_simple(masks, current_forbidden, all_wall_pts, base_size, px_per_m, allow_windows=True, debug=debug)
        if not candidates and base_size != (1.0, 0.55):
            candidates = try_place_simple(masks, current_forbidden, all_wall_pts, (1.0, 0.55), px_per_m, allow_windows=True, debug=debug)
        if not candidates:
             candidates = try_place_simple(masks, current_forbidden, all_wall_pts, (0.8, 0.55), px_per_m, allow_windows=True, debug=debug)

    if not candidates:
        if debug:
            print(f"   [DEBUG] Kitchen: FALLO - No hay candidatos válidos ni con fuerza bruta")
        return None

        
    # Función auxiliar para verificar que el mueble sigue tocando pared durante expansión
    def still_touches_wall(x1, x2, y1, y2, ori):
        """Verifica que el borde trasero del mueble siga tocando la pared."""
        margin = 2  # Pequeño margen para detectar pared cercana
        if ori == "BACK_UP":
            back_edge = masks["walls"][max(0, x1-margin):x1+margin, y1:y2]
        elif ori == "BACK_DOWN":
            back_edge = masks["walls"][x2-margin:min(H, x2+margin), y1:y2]
        elif ori == "BACK_LEFT":

            back_edge = masks["walls"][x1:x2, max(0, y1-margin):y1+margin]
        elif ori == "BACK_RIGHT":
            back_edge = masks["walls"][x1:x2, y2-margin:min(W, y2+margin)]
        else:
            return False
        return back_edge.size > 0 and np.sum(back_edge) > 0
        
    # Evaluar expansión para cada candidato y elegir el mejor (Max Length)
    best_candidate = None
    max_len_found = -1
    
    for cand in candidates:
        # Simular expansión
        x1, x2, y1, y2 = cand["x1"], cand["x2"], cand["y1"], cand["y2"]
        ori = cand["orientation"]
        expand_step = int(0.1 * px_per_m)
        
        # Simulación temporal (sin modificar cand original hasta elegir)
        tx1, tx2, ty1, ty2 = x1, x2, y1, y2
        
        if ori in ["BACK_UP", "BACK_DOWN"]:  # Pared horizontal -> expandir en Y (izq/der)
            # Expandir hacia la izquierda (Y negativo)
            while ty1 - expand_step >= 0:
                new_y1 = ty1 - expand_step
                if check_placement(masks, current_forbidden, tx1, new_y1, tx2, ty2, allow_windows=True):
                    if still_touches_wall(tx1, tx2, new_y1, ty2, ori):
                        ty1 = new_y1
                    else:
                        break
                else:
                    break
            # Expandir hacia la derecha (Y positivo)
            while ty2 + expand_step < W:
                new_y2 = ty2 + expand_step
                if check_placement(masks, current_forbidden, tx1, ty1, tx2, new_y2, allow_windows=True):
                    if still_touches_wall(tx1, tx2, ty1, new_y2, ori):
                        ty2 = new_y2
                    else:
                        break
                else:
                    break
            curr_len = ty2 - ty1
        else:  # Pared vertical (BACK_LEFT, BACK_RIGHT) -> expandir en X (arriba/abajo)
            # Expandir hacia arriba (X negativo)
            while tx1 - expand_step >= 0:
                new_x1 = tx1 - expand_step
                if check_placement(masks, current_forbidden, new_x1, ty1, tx2, ty2, allow_windows=True):
                    if still_touches_wall(new_x1, tx2, ty1, ty2, ori):
                        tx1 = new_x1
                    else:
                        break
                else:
                    break
            # Expandir hacia abajo (X positivo)
            while tx2 + expand_step < H:
                new_x2 = tx2 + expand_step
                if check_placement(masks, current_forbidden, tx1, ty1, new_x2, ty2, allow_windows=True):
                    if still_touches_wall(tx1, new_x2, ty1, ty2, ori):
                        tx2 = new_x2
                    else:
                        break
                else:
                    break
            curr_len = tx2 - tx1
        
        if debug and curr_len > 0:
            print(f"   [DEBUG] Kitchen: Candidato ori={ori}, len_expandida={curr_len/px_per_m:.2f}m")
            
        if curr_len > max_len_found:
            max_len_found = curr_len
            best_candidate = cand.copy()
            # Guardamos también las coords expandidas para no recalcular
            best_candidate["ex_x1"], best_candidate["ex_x2"] = tx1, tx2
            best_candidate["ex_y1"], best_candidate["ex_y2"] = ty1, ty2

    if debug and best_candidate:
        print(f"   [DEBUG] Kitchen: MEJOR candidato ori={best_candidate['orientation']}, len={max_len_found/px_per_m:.2f}m")

    # Aplicar el mejor
    if not best_candidate:
        return None
        
    x1, x2, y1, y2 = best_candidate["ex_x1"], best_candidate["ex_x2"], best_candidate["ex_y1"], best_candidate["ex_y2"]
    ori = best_candidate["orientation"]
            
    # 3. Subdividir en 4 Módulos
    total_len_px = max(x2-x1, y2-y1)
    # Definir módulos: Nevera, Inducción, Pica, Armario
    # Si espacio es muy pequeño, reducir módulos.
    # Ancho mínimo por módulo: 60cm?
    min_mod_px = int(0.50 * px_per_m)
    num_modules = max(1, min(4, total_len_px // min_mod_px))
    
    module_types = ["Nevera", "Inducción", "Pica", "Almacenaje"][:num_modules]
    
    # División
    mod_len = total_len_px / num_modules
    
    curr_start = 0
    for i, m_type in enumerate(module_types):
        if "UP" in ori or "DOWN" in ori:
            # Horizontal: Dividimos Y
            # Nevera en extremo? Si i=0.
            my1 = int(y1 + i * mod_len)
            my2 = int(y1 + (i+1) * mod_len)
            mx1, mx2 = x1, x2
        else:
            # Vertical: Dividimos X
            mx1 = int(x1 + i * mod_len)
            mx2 = int(x1 + (i+1) * mod_len)
            my1, my2 = y1, y2
            
        final_type = "fridge" if m_type == "Nevera" else "countertop"
        config_name = m_type
        
        items.append({
            "type": final_type,
            "x1": mx1, "x2": mx2, "y1": my1, "y2": my2,
            "orientation": ori,
            "config": config_name
        })
        
        # Actualizar forbidden
        cmargin = int(0.2 * px_per_m)
        gx1, gx2 = max(0, mx1-cmargin), min(H, mx2+cmargin)
        gy1, gy2 = max(0, my1-cmargin), min(W, my2+cmargin)
        current_forbidden[gx1:gx2, gy1:gy2] = True

    return {
        "success": True,
        "config_level": f"Modular {num_modules} pcs (MaxLen: {total_len_px/px_per_m:.2f}m)",
        "items": items
    }


# ==========================================
# 6. BATHROOM - Baño Completo (Dutxa, Meadero, Pica)
# ==========================================

def solve_bathroom(gParam, room_pixels, lPerimeter, pwalls, debug=False):
    """
    Coloca elementos de baño: Ducha, Inodoro (Meadero) y Lavabo (Pica).
    
    Reglas:
    - Dutxa y Meadero pueden tocar ventana.
    - Pica preferiblemente no (espejo).
    - Todos estrictamente pegados a pared.
    - Estrategia: "De afuera hacia adentro por capas" (Layered).
    """
    masks = parse_room_geometry(gParam, room_pixels, lPerimeter, pwalls)
    px_per_m = gParam["nPixelsPerMeter"]
    
    if not room_pixels or len(room_pixels) == 0:
        return None
        
    if debug:
        print(f"   [DEBUG] Bathroom: Iniciando resolución...")

    # Refinamiento máscaras
    masks = masks.copy()
    room_context = dilation(masks["room_original"], disk(10))
    masks["doors"] = masks["doors"] & room_context
    
    # Configuración de tamaños (Ancho x Profundo respecto a la pared)
    # Ducha: Añadimos tamaño grande 100x100
    shower_sizes = [(1.0, 1.0), (0.90, 0.90), (0.80, 0.80)]
    
    # Inodoro: Aprox 40cm ancho x 65cm profundo
    toilet_size = (0.40, 0.65)
    
    # Pica: Añadimos tamaño grande 80x50
    sink_sizes = [(0.80, 0.50), (0.60, 0.45)]
    
    # Zona prohibida (Puertas)
    config = {"door_radius": 0.50, "window_clearance": 0.0}
    
    forbidden = get_forbidden_mask(gParam, masks, config, include_walls=True)
    current_forbidden = forbidden.copy()
    
    items = []
    
    # Generar puntos de pared por capas (de 1 a 7 píxeles de distancia)
    wall_points_layered = []
    for dist in range(1, 8):
        current_layer = dilation(masks["walls"], disk(dist))
        previous_layer = dilation(masks["walls"], disk(dist-1))
        layer_mask = (current_layer ^ previous_layer) & masks["room_original"]
        points_in_layer = np.argwhere(layer_mask)
        if len(points_in_layer) > 0:
            wall_points_layered.extend(list(points_in_layer))
            
    wall_points = list(wall_points_layered)
    
    if not wall_points:
        if debug: print("   [DEBUG] Bathroom: No se encontraron puntos de pared.")
        return None

    # Funcion auxiliar para colocar un item
    def place_item(name, type_str, sizes_list, allow_win, usage_depth_m=0.0):
        for size in sizes_list:
            # Usamos try_place_strict (Global)
            # gaps=[0, 1, 2] para tener cierta flexibilidad inicial
            cands = try_place_strict(masks, current_forbidden, wall_points, size, px_per_m, gaps=[0, 1, 2], allow_windows=allow_win)
            
            if not cands:
                # Intento relajado con más gaps
                cands = try_place_strict(masks, current_forbidden, wall_points, size, px_per_m, gaps=[0, 1, 2, 3, 4], allow_windows=allow_win)
                
            if cands:
                best = cands[0]
                
                # Guardar resultado
                items.append({
                    "type": type_str,
                    "x1": best["x1"], "x2": best["x2"], "y1": best["y1"], "y2": best["y2"],
                    "orientation": best["orientation"],
                    "config": name
                })
                
                # 1. Actualizar forbidden con el propio objeto + margen
                cmargin = int(0.15 * px_per_m) # 15cm margen entre sanitarios (Aumentado)
                gx1, gx2 = max(0, best["x1"]-cmargin), min(masks["all_obstacles"].shape[0], best["x2"]+cmargin)
                gy1, gy2 = max(0, best["y1"]-cmargin), min(masks["all_obstacles"].shape[1], best["y2"]+cmargin)
                current_forbidden[gx1:gx2, gy1:gy2] = True
                
                # 2. Bloquear zona de USO (Frontal) si se especifica
                # Esto evita poner la pica justo delante de la ducha ocupando su espacio de salida
                if usage_depth_m > 0:
                    u_px = int(usage_depth_m * px_per_m)
                    ux1, ux2, uy1, uy2 = best["x1"], best["x2"], best["y1"], best["y2"]
                    ori = best["orientation"]
                    
                    if ori == "BACK_UP": # Front is DOWN (+X)
                        ux1, ux2 = best["x2"], min(masks["all_obstacles"].shape[0], best["x2"] + u_px)
                    elif ori == "BACK_DOWN": # Front is UP (-X)
                        ux2, ux1 = best["x1"], max(0, best["x1"] - u_px)
                    elif ori == "BACK_LEFT": # Front is RIGHT (+Y)
                        uy1, uy2 = best["y2"], min(masks["all_obstacles"].shape[1], best["y2"] + u_px)
                    elif ori == "BACK_RIGHT": # Front is LEFT (-Y)
                        uy2, uy1 = best["y1"], max(0, best["y1"] - u_px)
                    
                    if ux2 > ux1 and uy2 > uy1:
                         current_forbidden[ux1:ux2, uy1:uy2] = True
                         if debug: print(f"   [DEBUG] Bloqueada zona de uso frontal de {name} ({usage_depth_m}m)")

                if debug: 
                    print(f"   [DEBUG] Bathroom: Colocado {name} ({size}) en {best['orientation']}")
                return True
        return False

    # 1. Colocar DUCHA 
    # Zona de uso amplia (80cm) para que no pongan nada delante
    if not place_item("Shower", "shower", shower_sizes, allow_win=True, usage_depth_m=0.80):
        if debug: print("   [DEBUG] Bathroom: Falló colocación Ducha")
        return None 
        
    # 2. Colocar INODORO 
    # Zona de uso estándar (60cm)
    if not place_item("Toilet", "toilet", [toilet_size], allow_win=True, usage_depth_m=0.60):
        if debug: print("   [DEBUG] Bathroom: Falló colocación WC")
        
    # 3. Colocar PICA
    # Pica no necesita bloquear zona frontal para el futuro (es el último)
    if not place_item("Sink", "sink", sink_sizes, allow_win=False):
        if debug: print("   [DEBUG] Bathroom: Falló Pica sin ventana, probando con ventana...")
        place_item("Sink", "sink", sink_sizes, allow_win=True)

    return {
        "success": True,
        "config_level": "Standard Bathroom",
        "items": items
    }


# ==========================================
# 7. SINGLE BEDROOM - Dormitorio Individual
# ==========================================

def solve_single_bedroom(gParam, room_pixels, lPerimeter, pwalls, debug=False):
    """
    Coloca muebles en dormitorio individual: Cama Individual + Armario.
    
    Estrategia:
    - Búsqueda por capas (layered) desde la pared.
    - Cama individual puede estar bajo ventana.
    - Armario NO puede estar bajo ventana (lado LARGO pegado a pared).
    """
    masks = parse_room_geometry(gParam, room_pixels, lPerimeter, pwalls)
    px_per_m = gParam["nPixelsPerMeter"]
    
    if not room_pixels or len(room_pixels) == 0:
        return None
        
    if debug:
        room_area_m2 = len(room_pixels) / (px_per_m ** 2)
        print(f"   [DEBUG] SingleBedroom: Área={room_area_m2:.2f}m²")

    # Refinamiento máscaras
    masks = masks.copy()
    room_context = dilation(masks["room_original"], disk(10))
    masks["doors"] = masks["doors"] & room_context
    
    # Configuración de Tamaños
    # Cama Individual: varios tamaños
    bed_sizes = [(0.90, 1.90), (0.80, 1.80), (0.75, 1.70)]
    
    # Armario: lado largo paralelo a pared (ancho x profundidad)
    wardrobe_sizes = [(1.20, 0.60), (1.00, 0.55), (0.80, 0.50)]
    
    # Zona prohibida (Puertas)
    config = {"door_radius": 0.60, "window_clearance": 0.0}
    forbidden = get_forbidden_mask(gParam, masks, config, include_walls=True)
    current_forbidden = forbidden.copy()
    
    items = []
    
    # Generar puntos de pared por capas (1-7 px)
    wall_points_layered = []
    for dist in range(1, 8):
        current_layer = dilation(masks["walls"], disk(dist))
        previous_layer = dilation(masks["walls"], disk(dist-1))
        layer_mask = (current_layer ^ previous_layer) & masks["room_original"]
        points_in_layer = np.argwhere(layer_mask)
        if len(points_in_layer) > 0:
            wall_points_layered.extend(list(points_in_layer))
            
    wall_points = list(wall_points_layered)
    
    if not wall_points:
        if debug: print("   [DEBUG] SingleBedroom: No se encontraron puntos de pared.")
        return None

    # 1. COLOCAR CAMA INDIVIDUAL (Permite ventana, NO importa orientación)
    # Usamos try_place_simple: solo chequea que toque pared y quepa
    bed_placed = False
    for bed_size in bed_sizes:
        cands = try_place_simple(masks, current_forbidden, wall_points, bed_size, px_per_m, 
                                  allow_windows=True, debug=False)
        if cands:
            best = cands[0]
            items.append({
                "type": "bed_single",
                "x1": best["x1"], "x2": best["x2"], "y1": best["y1"], "y2": best["y2"],
                "orientation": best["orientation"],
                "config": f"Cama {bed_size}"
            })
            
            # Actualizar forbidden con margen pequeño
            cmargin = int(0.30 * px_per_m)  # 30cm margen
            gx1, gx2 = max(0, best["x1"]-cmargin), min(masks["all_obstacles"].shape[0], best["x2"]+cmargin)
            gy1, gy2 = max(0, best["y1"]-cmargin), min(masks["all_obstacles"].shape[1], best["y2"]+cmargin)
            current_forbidden[gx1:gx2, gy1:gy2] = True
            
            if debug:
                print(f"   [DEBUG] SingleBedroom: Cama colocada ({bed_size}) en {best['orientation']}")
            bed_placed = True
            break
    
    if not bed_placed:
        if debug: print("   [DEBUG] SingleBedroom: No se pudo colocar la cama.")
        return None

    # 2. COLOCAR ARMARIO (NO ventana, lado LARGO paralelo a pared)
    # Usamos try_place_strict para orientación correcta, gaps más amplios
    wardrobe_placed = False
    for wardrobe_size in wardrobe_sizes:
        cands = try_place_strict(masks, current_forbidden, wall_points, wardrobe_size, px_per_m,
                                  gaps=[0, 1, 2, 3, 4, 5], allow_windows=False)
        if cands:
            best = cands[0]
            items.append({
                "type": "wardrobe",
                "x1": best["x1"], "x2": best["x2"], "y1": best["y1"], "y2": best["y2"],
                "orientation": best["orientation"],
                "config": f"Armario {wardrobe_size}"
            })
            
            if debug:
                print(f"   [DEBUG] SingleBedroom: Armario colocado ({wardrobe_size}) en {best['orientation']}")
            wardrobe_placed = True
            break
    
    if not wardrobe_placed:
        # Fallback: intentar con try_place_simple para el armario si strict falla
        if debug: print("   [DEBUG] SingleBedroom: Strict falló para armario, probando simple...")
        for wardrobe_size in wardrobe_sizes:
            cands = try_place_simple(masks, current_forbidden, wall_points, wardrobe_size, px_per_m,
                                      allow_windows=False, debug=False)
            if cands:
                best = cands[0]
                items.append({
                    "type": "wardrobe",
                    "x1": best["x1"], "x2": best["x2"], "y1": best["y1"], "y2": best["y2"],
                    "orientation": best["orientation"],
                    "config": f"Armario {wardrobe_size} (Simple)"
                })
                if debug:
                    print(f"   [DEBUG] SingleBedroom: Armario (Simple) colocado ({wardrobe_size})")
                wardrobe_placed = True
                break
    
    if not wardrobe_placed:
        if debug: print("   [DEBUG] SingleBedroom: No se pudo colocar armario, solo cama.")
    
    return {
        "success": True,
        "config_level": "Standard Single Bedroom" if wardrobe_placed else "Minimal Single Bedroom",
        "items": items
    }


# ==========================================
# 8. LIVING ROOM - Salón (Sofá + Mueble TV)
# ==========================================


# ==========================================
# 8. LIVING ROOM - Salón (Sofá + Mueble TV)
# ==========================================

def solve_living_room(gParam, room_pixels, lPerimeter, pwalls, debug=False):
    """
    Coloca Sofá y Mueble TV enfrentados.
    
    Estrategia "Anchor + Projection":
    1. Buscar candidatos estrictos en pared para TV y Sofá.
    2. Intentar parearlos (Strict-Strict).
    3. Si no hay pareo estricto, usar uno como "Ancla" (en pared) y proyectar el otro (flotante).
         - Ancla TV -> Proyectar Sofá (Prioridad)
         - Ancla Sofá -> Proyectar TV
    4. Maximizar score: (Ambos en pared) > (Distancia ideal ~2.5m).
    """
    masks = parse_room_geometry(gParam, room_pixels, lPerimeter, pwalls)
    px_per_m = gParam["nPixelsPerMeter"]
    
    if not room_pixels or len(room_pixels) == 0:
        return None
        
    if debug:
        print(f"   [DEBUG] LivingRoom: Iniciando resolución (Anchor Method)...")

    # Refinamiento máscaras
    masks = masks.copy()
    room_context = dilation(masks["room_original"], disk(10))
    masks["doors"] = masks["doors"] & room_context
    
    # Configuración de Tamaños
    # Sofá: Wall-attached sizes (puede ser grande)
    sofa_sizes = [(2.20, 0.90), (2.00, 0.90), (1.80, 0.85), (1.60, 0.80), (1.40, 0.75)]
    # Sofá: Floating sizes (más pequeño para dejar paso por los lados)
    sofa_sizes_floating = [(1.60, 0.80), (1.40, 0.75), (1.20, 0.70)]
    # TV:
    tv_sizes = [(1.80, 0.45), (1.50, 0.45), (1.20, 0.40)]
    
    # Zona prohibida (Puertas)
    config = {"door_radius": 0.50, "window_clearance": 0.0}
    forbidden = get_forbidden_mask(gParam, masks, config, include_walls=True)
    current_forbidden = forbidden.copy()
    
    # Generar puntos de pared por capas
    wall_points_layered = []
    for dist in range(1, 8):
        current_layer = dilation(masks["walls"], disk(dist))
        previous_layer = dilation(masks["walls"], disk(dist-1))
        layer_mask = (current_layer ^ previous_layer) & masks["room_original"]
        points_in_layer = np.argwhere(layer_mask)
        if len(points_in_layer) > 0:
            wall_points_layered.extend(list(points_in_layer))
            
    wall_points = list(wall_points_layered)
    
    if not wall_points: return None

    # 1. Obtener candidatos ESTRICTOS (Pared)
    candidates_sofa = []
    for size in sofa_sizes:
        cands = try_place_strict(masks, current_forbidden, wall_points, size, px_per_m, gaps=[0, 1, 2, 3], allow_windows=True)
        for c in cands:
            c["size_obj"] = size
            candidates_sofa.append(c)
        if len(candidates_sofa) > 50: break 

    candidates_tv = []
    for size in tv_sizes:
        cands = try_place_strict(masks, current_forbidden, wall_points, size, px_per_m, gaps=[0, 1, 2, 3], allow_windows=False)
        for c in cands:
            c["size_obj"] = size
            candidates_tv.append(c)
        if len(candidates_tv) > 50: break

    # Función auxiliar para comprobar "Check Placement" de un candidato generado ad-hoc
    def check_floating(x1, x2, y1, y2):
        if x1 < 0 or y1 < 0 or x2 >= masks["all_obstacles"].shape[0] or y2 >= masks["all_obstacles"].shape[1]:
            return False
        # Chequear colisión con paredes y puertas (forbidden)
        # check_placement usa: x1, y1, x2, y2, allow_windows
        return check_placement(masks, current_forbidden, x1, y1, x2, y2, True) # Allow windows for floating items usually OK

    best_pair = None
    best_score = -100.0
    
    # Rangos de distancia a probar (de cerca a lejos)
    # 1.5m a 4.0m
    dist_steps = np.arange(1.5, 4.1, 0.25)
    
    opposites = { "BACK_UP": "BACK_DOWN", "BACK_DOWN": "BACK_UP", "BACK_LEFT": "BACK_RIGHT", "BACK_RIGHT": "BACK_LEFT" }

    # === ESTRATEGIA 1: PAREO ESTRICTO (Wall to Wall) ===
    # Intentamos primero lo mejor (ambos en pared)
    for s in candidates_sofa:
        for t in candidates_tv:
            if opposites.get(s["orientation"]) != t["orientation"]: continue
            
            # Alineación y Distancia
            scx, scy = (s["x1"]+s["x2"])/2, (s["y1"]+s["y2"])/2
            tcx, tcy = (t["x1"]+t["x2"])/2, (t["y1"]+t["y2"])/2
            
            dist = 0
            aligned = False
            
            if s["orientation"] in ["BACK_UP", "BACK_DOWN"]:
                overlap_y = min(s["y2"], t["y2"]) - max(s["y1"], t["y1"])
                if overlap_y > min(s["y2"]-s["y1"], t["y2"]-t["y1"]) * 0.5: aligned = True
                dist = abs(scx - tcx)
            else:
                overlap_x = min(s["x2"], t["x2"]) - max(s["x1"], t["x1"])
                if overlap_x > min(s["x2"]-s["x1"], t["x2"]-t["x1"]) * 0.5: aligned = True
                dist = abs(scy - tcy)
            
            if not aligned: continue
            dist_m = dist / px_per_m
            
            if 1.4 <= dist_m <= 4.5:
                # Score: Base 100 (Strict) + Size - Penalty Dist
                base_val = 100
                len_val = (s["size_obj"][0] + t["size_obj"][0])
                dist_pen = abs(dist_m - 2.5) * 2.0 
                score = base_val + len_val - dist_pen
                
                if score > best_score:
                    best_score = score
                    best_pair = (s, t, "Strict-Strict")

    if debug: print(f"   [DEBUG] Best Strict Score: {best_score}")

    # === ESTRATEGIA 2: ANCLA TV -> PROYECTAR SOFÁ (Semi-Strict) ===
    # Si el score estricto no es muy alto, o no existe, probamos proyectar
    # Solo si no tenemos ya un par perfecto (score > 105, ej: strict + sizes grandes + buena dist)
    if best_score < 105:
        if debug: print("   [DEBUG] Probando Anchor TV -> Floating Sofa...")
        for t in candidates_tv:
            # Probar distancias
            for d_m in dist_steps:
                d_px = int(d_m * px_per_m)
                
                # Usar sofás PEQUEÑOS para dejar paso lateral
                for s_w_m, s_d_m in sofa_sizes_floating:
                    s_w_px = int(s_w_m * px_per_m)
                    s_d_px = int(s_d_m * px_per_m)
                    
                    # Calcular coords proyectadas
                    # TV orientation t["orientation"] -> Sofa must be opposite
                    # Si TV es BACK_UP (mira abajo), Sofa debe estar ABAJO y ser BACK_DOWN (mira arriba)
                   
                    valid_proj = False
                    sx1, sx2, sy1, sy2 = 0,0,0,0
                    sori = opposites.get(t["orientation"])
                    
                    tcx, tcy = (t["x1"]+t["x2"])/2, (t["y1"]+t["y2"])/2
                    
                    if t["orientation"] == "BACK_UP": # TV arriba, mira +X. Sofa debe estar en +X
                         # Sofa BACK_DOWN: top edge is "back". 
                         # Distancia d_px es entre frentes? O centros? Centros es mas facil.
                         # Center Sofa X = Center TV X + d_px ?? No, TV mira +X (Abajo) -> cx + d
                         scx = tcx + d_px
                         scy = tcy # Alineado perfecto
                         
                         sx1 = int(scx - s_d_px/2); sx2 = int(scx + s_d_px/2) # Ojo dimension prof
                         # BACK_DOWN: prof es X. largo es Y.
                         sx2 = int(scx + s_d_px/2) # Back edge (abajo) ??
                         # BACK_DOWN -> El mueble se extiende hacia ARRIBA (-X) desde su back.
                         # Su back está en X mayor.
                         # center X = back_x - prof/2. -> back_x = center + prof/2
                         
                         # Simplificación: Proyectar caja centrada en (tcx + d, tcy)
                         # Orientación BACK_DOWN: Dimensiones (prof, largo)
                         sx1 = int(scx - s_d_px/2); sx2 = int(scx + s_d_px/2)
                         sy1 = int(scy - s_w_px/2); sy2 = int(scy + s_w_px/2)
                         # Ajustar orientation logic
                         # Si es BACK_DOWN, el respaldo está en sx2.
                         
                    elif t["orientation"] == "BACK_DOWN": # TV abajo, mira -X. Sofa en -X
                         scx = tcx - d_px
                         scy = tcy
                         sx1 = int(scx - s_d_px/2); sx2 = int(scx + s_d_px/2)
                         sy1 = int(scy - s_w_px/2); sy2 = int(scy + s_w_px/2)
                         
                    elif t["orientation"] == "BACK_LEFT": # TV izq, mira +Y. Sofa en +Y
                         scx = tcx; scy = tcy + d_px
                         sx1 = int(scx - s_w_px/2); sx2 = int(scx + s_w_px/2)
                         sy1 = int(scy - s_d_px/2); sy2 = int(scy + s_d_px/2)
                         
                    elif t["orientation"] == "BACK_RIGHT": # TV der, mira -Y. Sofa en -Y
                         scx = tcx; scy = tcy - d_px
                         sx1 = int(scx - s_w_px/2); sx2 = int(scx + s_w_px/2)
                         sy1 = int(scy - s_d_px/2); sy2 = int(scy + s_d_px/2)
                    
                    if check_floating(sx1, sx2, sy1, sy2):
                        # Score: Base 50 (Semi) + Size - Penalty
                        base_val = 50
                        len_val = (t["size_obj"][0] + s_w_m)
                        dist_pen = abs(d_m - 2.5) * 2.0
                        score = base_val + len_val - dist_pen
                        
                        if score > best_score:
                            best_score = score
                            # Create fake candidate for sofa
                            fake_sofa = { "x1": sx1, "x2": sx2, "y1": sy1, "y2": sy2, "orientation": sori, "size_obj": (s_w_m, s_d_m) }
                            best_pair = (fake_sofa, t, "AnchorTV-FloatSofa")
                        
                        # Si encontramos uno valido, quizas no hace falta iterar mas tamaños de sofa para esta dist?
                        # O si, buscamos el mas grande.
    
    # === ESTRATEGIA 3: ANCLA SOFA -> PROYECTAR TV (Semi-Strict) ===
    # Menos prioritario (cables TV), pero valido si falla lo demas
    if best_score < 50: # Solo si no encontramos nada decente antes
        if debug: print("   [DEBUG] Probando Anchor Sofa -> Floating TV...")
        for s in candidates_sofa:
             for d_m in dist_steps:
                d_px = int(d_m * px_per_m)
                for t_w_m, t_d_m in tv_sizes:
                    t_w_px = int(t_w_m * px_per_m)
                    t_d_px = int(t_d_m * px_per_m)
                    
                    scx, scy = (s["x1"]+s["x2"])/2, (s["y1"]+s["y2"])/2
                    tori = opposites.get(s["orientation"])
                    
                    if s["orientation"] == "BACK_UP": # Sofa arriba, mira +X
                         tcx = scx + d_px; tcy = scy
                         tx1 = int(tcx - t_d_px/2); tx2 = int(tcx + t_d_px/2)
                         ty1 = int(tcy - t_w_px/2); ty2 = int(tcy + t_w_px/2)
                    elif s["orientation"] == "BACK_DOWN": # Sofa bajo, mira -X
                         tcx = scx - d_px; tcy = scy
                         tx1 = int(tcx - t_d_px/2); tx2 = int(tcx + t_d_px/2)
                         ty1 = int(tcy - t_w_px/2); ty2 = int(tcy + t_w_px/2)
                    elif s["orientation"] == "BACK_LEFT":
                         tcx = scx; tcy = scy + d_px
                         tx1 = int(tcx - t_w_px/2); tx2 = int(tcx + t_w_px/2)
                         ty1 = int(tcy - t_d_px/2); ty2 = int(tcy + t_d_px/2)
                    elif s["orientation"] == "BACK_RIGHT":
                         tcx = scx; tcy = scy - d_px
                         tx1 = int(tcx - t_w_px/2); tx2 = int(tcx + t_w_px/2)
                         ty1 = int(tcy - t_d_px/2); ty2 = int(tcy + t_d_px/2)
                         
                    if check_floating(tx1, tx2, ty1, ty2):
                        base_val = 40 # Menos que TV Anchor
                        len_val = (s["size_obj"][0] + t_w_m)
                        dist_pen = abs(d_m - 2.5) * 2.0
                        score = base_val + len_val - dist_pen
                        
                        if score > best_score:
                            best_score = score
                            fake_tv = { "x1": tx1, "x2": tx2, "y1": ty1, "y2": ty2, "orientation": tori, "size_obj": (t_w_m, t_d_m) }
                            best_pair = (s, fake_tv, "AnchorSofa-FloatTV")

    # === ESTRATEGIA 4: AMBOS FLOTANTES (Full Floating) ===
    # Si todo lo anterior falla, intentamos colocar ambos en el centro de la habitación
    if best_score < 30:
        if debug: print("   [DEBUG] Probando Full Floating (Centro de habitación)...")
        
        # Calcular centroide de la habitación
        room_pts = np.argwhere(masks["room_original"])
        if len(room_pts) > 0:
            centroid_x = int(np.mean(room_pts[:, 0]))
            centroid_y = int(np.mean(room_pts[:, 1]))
            
            # Probar orientaciones horizontales y verticales
            for ori_tv, ori_sofa in [("BACK_UP", "BACK_DOWN"), ("BACK_DOWN", "BACK_UP"), 
                                      ("BACK_LEFT", "BACK_RIGHT"), ("BACK_RIGHT", "BACK_LEFT")]:
                for d_m in [2.0, 2.5, 3.0, 1.8]:
                    d_px = int(d_m * px_per_m)
                    
                    for t_w_m, t_d_m in tv_sizes:
                        for s_w_m, s_d_m in sofa_sizes_floating:
                            t_w_px = int(t_w_m * px_per_m)
                            t_d_px = int(t_d_m * px_per_m)
                            s_w_px = int(s_w_m * px_per_m)
                            s_d_px = int(s_d_m * px_per_m)
                            
                            # Posicionar TV y Sofa centrados, separados por d_px/2 cada uno
                            if ori_tv in ["BACK_UP", "BACK_DOWN"]:
                                # Horizontal layout
                                tx1 = int(centroid_x - d_px/2 - t_d_px/2)
                                tx2 = int(centroid_x - d_px/2 + t_d_px/2)
                                ty1 = int(centroid_y - t_w_px/2)
                                ty2 = int(centroid_y + t_w_px/2)
                                
                                sx1 = int(centroid_x + d_px/2 - s_d_px/2)
                                sx2 = int(centroid_x + d_px/2 + s_d_px/2)
                                sy1 = int(centroid_y - s_w_px/2)
                                sy2 = int(centroid_y + s_w_px/2)
                            else:
                                # Vertical layout
                                tx1 = int(centroid_x - t_w_px/2)
                                tx2 = int(centroid_x + t_w_px/2)
                                ty1 = int(centroid_y - d_px/2 - t_d_px/2)
                                ty2 = int(centroid_y - d_px/2 + t_d_px/2)
                                
                                sx1 = int(centroid_x - s_w_px/2)
                                sx2 = int(centroid_x + s_w_px/2)
                                sy1 = int(centroid_y + d_px/2 - s_d_px/2)
                                sy2 = int(centroid_y + d_px/2 + s_d_px/2)
                            
                            if check_floating(tx1, tx2, ty1, ty2) and check_floating(sx1, sx2, sy1, sy2):
                                base_val = 80  # PRIORIDAD ALTA: Centro es preferido
                                len_val = (t_w_m + s_w_m)
                                dist_pen = abs(d_m - 2.5) * 1.5
                                score = base_val + len_val - dist_pen
                                
                                if score > best_score:
                                    best_score = score
                                    fake_tv = {"x1": tx1, "x2": tx2, "y1": ty1, "y2": ty2, "orientation": ori_tv, "size_obj": (t_w_m, t_d_m)}
                                    fake_sofa = {"x1": sx1, "x2": sx2, "y1": sy1, "y2": sy2, "orientation": ori_sofa, "size_obj": (s_w_m, s_d_m)}
                                    best_pair = (fake_sofa, fake_tv, "FullFloating")

    if best_pair:
        s, t, ptype = best_pair
        if debug:
            dist_m = np.sqrt((s['x1']-t['x1'])**2 + (s['y1']-t['y1'])**2) / px_per_m
            print(f"   [DEBUG] LivingRoom: WINNER -> {ptype} Score={best_score:.2f}, Dist={dist_m:.2f}m")

        items = []
        items.append({ "type": "sofa", "x1": s["x1"], "x2": s["x2"], "y1": s["y1"], "y2": s["y2"], "orientation": s["orientation"], "config": f"Sofa {s['size_obj']}" })
        items.append({ "type": "tv_unit", "x1": t["x1"], "x2": t["x2"], "y1": t["y1"], "y2": t["y2"], "orientation": t["orientation"], "config": f"TV {t['size_obj']}" })
        
        return { "success": True, "config_level": f"Living ({ptype})", "items": items }
    
    # === ESTRATEGIA 5: SOLO SOFÁ (Fallback) ===
    # Si no cabe ninguna pareja, intentar poner solo el sofá
    if debug: print("   [DEBUG] Probando Solo Sofa (Fallback)...")
    
    # Primero intentar en pared
    if candidates_sofa:
        best_sofa = candidates_sofa[0]  # El primero es el mejor (más pegado a pared)
        if debug: print(f"   [DEBUG] LivingRoom: Solo Sofa en pared")
        return {
            "success": True,
            "config_level": "Living (SoloSofa-Wall)",
            "items": [{ "type": "sofa", "x1": best_sofa["x1"], "x2": best_sofa["x2"], "y1": best_sofa["y1"], "y2": best_sofa["y2"], "orientation": best_sofa["orientation"], "config": f"Sofa {best_sofa['size_obj']}" }]
        }
    
    # Si no hay candidatos en pared, poner sofá en centro
    room_pts = np.argwhere(masks["room_original"])
    if len(room_pts) > 0:
        centroid_x = int(np.mean(room_pts[:, 0]))
        centroid_y = int(np.mean(room_pts[:, 1]))
        
        for s_w_m, s_d_m in sofa_sizes_floating:
            s_w_px = int(s_w_m * px_per_m)
            s_d_px = int(s_d_m * px_per_m)
            
            sx1 = int(centroid_x - s_d_px/2)
            sx2 = int(centroid_x + s_d_px/2)
            sy1 = int(centroid_y - s_w_px/2)
            sy2 = int(centroid_y + s_w_px/2)
            
            if check_floating(sx1, sx2, sy1, sy2):
                if debug: print(f"   [DEBUG] LivingRoom: Solo Sofa flotante en centro")
                return {
                    "success": True,
                    "config_level": "Living (SoloSofa-Center)",
                    "items": [{ "type": "sofa", "x1": sx1, "x2": sx2, "y1": sy1, "y2": sy2, "orientation": "BACK_UP", "config": f"Sofa ({s_w_m}, {s_d_m})" }]
                }
    
    return None

