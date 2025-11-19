# En window_placement_utils.py
from random import random
from Analysis_StyleGAN_functions_03_pindatafinal24_Relaxed import getPointsLineBetweenXY


def segmentate_perimeter_into_walls_by_room(gParam, lPerimeter, lConnexComponents):
    """
    Segmenta muros exteriores ('EW') en tramos contiguos usando las paredes interiores.
    Si una pared interior intersecta un muro exterior, este se parte en dos segmentos.
    Devuelve [(x1,y1,x2,y2,label)].
    """
    import numpy as np

    height = gParam['height']
    width = gParam['width']
    idxIW = gParam['idxNilColorInteriorWall']

    MIN_SEGMENT_LEN = 3
    # Distancia máxima para considerar que una pared interior "toca" el perímetro
    PROXIMITY_THRESHOLD = 3

    # --- 1) Obtener todos los píxeles de paredes interiores ---
    interior_wall_pixels = set()
    if len(lConnexComponents) > idxIW and isinstance(lConnexComponents[idxIW], tuple):
        cc_list = lConnexComponents[idxIW][2]
        for cc in cc_list:
            for (x, y) in cc:
                interior_wall_pixels.add((x, y))

    print(f"[SEGMENT] Total interior wall pixels: {len(interior_wall_pixels)}")

    def get_wall_intersection_points(x1, y1, x2, y2):
        """
        Detecta puntos donde una pared interior intersecta con el segmento del perímetro.
        Retorna una lista de coordenadas promedio donde hay intersección.
        """
        intersection_clusters = []

        if x1 == x2:  # Muro vertical
            x = int(x1)
            ya, yb = (int(y1), int(y2)) if y1 < y2 else (int(y2), int(y1))

            # Buscar píxeles de pared interior cercanos al muro
            cluster = []
            for y in range(ya, yb + 1):
                # Revisar píxeles adyacentes (izquierda y derecha)
                nearby_wall = False
                for dx in range(-PROXIMITY_THRESHOLD, PROXIMITY_THRESHOLD + 1):
                    for dy in range(-1, 2):  # Solo un pequeño rango vertical
                        check_x, check_y = x + dx, y + dy
                        if (check_x, check_y) in interior_wall_pixels:
                            nearby_wall = True
                            break
                    if nearby_wall:
                        break

                if nearby_wall:
                    cluster.append(y)
                elif cluster:
                    # Fin del cluster, calcular punto medio
                    if len(cluster) >= 2:
                        avg_y = int(np.mean(cluster))
                        intersection_clusters.append(avg_y)
                    cluster = []

            # Último cluster
            if cluster and len(cluster) >= 2:
                avg_y = int(np.mean(cluster))
                intersection_clusters.append(avg_y)

        else:  # Muro horizontal
            y = int(y1)
            xa, xb = (int(x1), int(x2)) if x1 < x2 else (int(x2), int(x1))

            cluster = []
            for x in range(xa, xb + 1):
                # Revisar píxeles adyacentes (arriba y abajo)
                nearby_wall = False
                for dy in range(-PROXIMITY_THRESHOLD, PROXIMITY_THRESHOLD + 1):
                    for dx in range(-1, 2):  # Solo un pequeño rango horizontal
                        check_x, check_y = x + dx, y + dy
                        if (check_x, check_y) in interior_wall_pixels:
                            nearby_wall = True
                            break
                    if nearby_wall:
                        break

                if nearby_wall:
                    cluster.append(x)
                elif cluster:
                    if len(cluster) >= 2:
                        avg_x = int(np.mean(cluster))
                        intersection_clusters.append(avg_x)
                    cluster = []

            if cluster and len(cluster) >= 2:
                avg_x = int(np.mean(cluster))
                intersection_clusters.append(avg_x)

        return intersection_clusters

    # --- 2) Procesar cada segmento del perímetro ---
    new_perimeter = []

    for (x1, y1, x2, y2, label) in lPerimeter:
        if label != 'EW':
            new_perimeter.append((int(x1), int(y1), int(x2), int(y2), label))
            continue

        # Detectar puntos de intersección con paredes interiores
        intersections = get_wall_intersection_points(x1, y1, x2, y2)

        if not intersections:
            # No hay intersecciones, mantener el segmento completo
            new_perimeter.append((int(x1), int(y1), int(x2), int(y2), label))
            continue

        # --- Muro vertical ---
        if x1 == x2:
            x = int(x1)
            ya, yb = (int(y1), int(y2)) if y1 < y2 else (int(y2), int(y1))

            # Ordenar intersecciones
            intersections.sort()

            # Crear segmentos entre intersecciones
            current_y = ya
            for cut_y in intersections:
                if cut_y - current_y >= MIN_SEGMENT_LEN:
                    new_perimeter.append((x, current_y, x, cut_y, label))
                current_y = cut_y + 1

            # Último segmento
            if yb - current_y >= MIN_SEGMENT_LEN:
                new_perimeter.append((x, current_y, x, yb, label))

        # --- Muro horizontal ---
        elif y1 == y2:
            y = int(y1)
            xa, xb = (int(x1), int(x2)) if x1 < x2 else (int(x2), int(x1))

            intersections.sort()

            current_x = xa
            for cut_x in intersections:
                if cut_x - current_x >= MIN_SEGMENT_LEN:
                    new_perimeter.append((current_x, y, cut_x, y, label))
                current_x = cut_x + 1

            if xb - current_x >= MIN_SEGMENT_LEN:
                new_perimeter.append((current_x, y, xb, y, label))

        else:
            # Segmento diagonal (no axial)
            new_perimeter.append((int(x1), int(y1), int(x2), int(y2), label))

    print(
        f"[SEGMENT] Original walls: {len(lPerimeter)} -> Segmented walls: {len(new_perimeter)}")
    return [(int(x1), int(y1), int(x2), int(y2), lbl) for (x1, y1, x2, y2, lbl) in new_perimeter]


def _is_wall_adjacent_to_room(wall_segment_pixels, room_pixels_set, max_dist=2):
    for p in wall_segment_pixels:
        wx, wy = p[:2]
        for dx in range(-max_dist, max_dist + 1):
            for dy in range(-max_dist, max_dist + 1):
                if (wx + dx, wy + dy) in room_pixels_set:
                    return True
    return False


def place_windows_heuristic(gParam, lPerimeter, lConnexComponentsCentroids,
                            lConnexComponents,
                            allow_multi_windows_per_wall=False,
                            min_len_factor=0.9,     # antes 1.5
                            max_dist_adj=3,         # antes 2
                            end_margin_m=0.15):     # margen a esquinas
    import numpy as np
    nppm = gParam['nPixelsPerMeter']
    WINDOW_SIZE_PX = int(1.2 * nppm)
    END_MARGIN_PX = int(end_margin_m * nppm)
    MIN_WALL_LEN_PX = max(int(min_len_factor * nppm),
                          WINDOW_SIZE_PX + 2*END_MARGIN_PX)

    new_perimeter = []
    processed_walls = set()
    total_placed = 0

    auxlPerimeter = segmentate_perimeter_into_walls_by_room(
        gParam, lPerimeter, lConnexComponents)

    print(
        f"[WINDOW] nppm={nppm} WINDOW={WINDOW_SIZE_PX}px MIN_WALL={MIN_WALL_LEN_PX}px MAX_DIST={max_dist_adj}")

    for idx_cc, (room_type, _, room_pixels) in enumerate(lConnexComponentsCentroids):
        room_pixels_set = set(room_pixels)
        best = None
        best_len = 0.0
        reasons = []

        for wall_idx, (x1, y1, x2, y2, label) in enumerate(auxlPerimeter):
            if label != 'EW':
                reasons.append((wall_idx, "not_EW"))
                continue
            if (not allow_multi_windows_per_wall) and (wall_idx in processed_walls):
                reasons.append((wall_idx, "already_used"))
                continue

            wall_len = float(np.hypot(x2-x1, y2-y1))
            if wall_len < MIN_WALL_LEN_PX:
                reasons.append(
                    (wall_idx, f"short({wall_len:.1f}<{MIN_WALL_LEN_PX})"))
                continue

            wall_pixels = getPointsLineBetweenXY(x1, y1, x2, y2)
            if not _is_wall_adjacent_to_room(wall_pixels, room_pixels_set, max_dist=max_dist_adj):
                reasons.append((wall_idx, "not_adjacent"))
                continue

            if wall_len > best_len:
                best = (wall_idx, (x1, y1, x2, y2, label), wall_len)
                best_len = wall_len

        if not best:
            # Muestra los 5 primeros motivos para esta habitación
            print(
                f"[ROOM {idx_cc}] type={room_type}: NO WINDOW. sample_reasons={reasons[:5]}")
            continue

        wall_idx, (x1, y1, x2, y2, _), wall_len = best
        v = np.array([x2-x1, y2-y1], dtype=float)
        v_unit = v / wall_len

        usable = wall_len - 2*END_MARGIN_PX
        if usable < WINDOW_SIZE_PX:
            print(
                f"[ROOM {idx_cc}] best_wall={wall_idx} TOO_SHORT_FOR_WINDOW usable={usable:.1f}px < {WINDOW_SIZE_PX}px")
            continue

        mid = np.array([(x1+x2)/2.0, (y1+y2)/2.0], dtype=float)
        half = WINDOW_SIZE_PX/2.0
        p1 = mid - v_unit*half
        p2 = mid + v_unit*half

        win_x1, win_y1 = int(round(p1[0])), int(round(p1[1]))
        win_x2, win_y2 = int(round(p2[0])), int(round(p2[1]))

        new_perimeter.append((x1, y1, win_x1, win_y1, 'EW'))
        new_perimeter.append((win_x1, win_y1, win_x2, win_y2, 'WN'))
        new_perimeter.append((win_x2, win_y2, x2, y2, 'EW'))

        processed_walls.add(wall_idx)
        total_placed += 1
        print(f"[ROOM {idx_cc}] type={room_type}: WINDOW on wall {wall_idx} len={wall_len:.1f}px "
              f"({win_x1},{win_y1})->({win_x2},{win_y2})")

    # Añadir lo no modificado
    for wall_idx, seg in enumerate(auxlPerimeter):
        if wall_idx not in processed_walls:
            new_perimeter.append(seg)

    print(f"[WINDOW] Placed {total_placed} windows.")

    eval_info = hill_climbing_window_optimization(gParam, auxlPerimeter, lConnexComponentsCentroids)
    new_perimeter = eval_info['mejor_perimeter']
    return new_perimeter

def evaluar(gParam, lPerimeter, lConnexComponentsCentroids, score_weights=None):
    """
    Evalúa cobertura de habitaciones con ventanas.
    """
    import numpy as np
    from window_placement_utils import _is_wall_adjacent_to_room

    # --- 1) Extraer ventanas ---
    ventanas = []
    for (x1, y1, x2, y2, label) in lPerimeter:
        if label == 'WN':
            pixels = getPointsLineBetweenXY(x1, y1, x2, y2)
            ventanas.append(set((p[0], p[1]) for p in pixels))

    num_ventanas = len(ventanas)

    # --- 2) Filtrar habitaciones elegibles ---
    indices_excluir = {
        gParam['idxNilColorBackground'],
        gParam['idxNilColorExteriorWall'],
        gParam['idxNilColorFrontDoor'],
        gParam['idxNilColorInteriorWall'],
        gParam.get('idxNilColorInsideDoor', -1)
    }

    habitaciones = []
    for idx_cc, (idxColor, rgbColor, room_pixels) in enumerate(lConnexComponentsCentroids):
        if idxColor not in indices_excluir:
            habitaciones.append({
                'idx': idx_cc,
                'tipo': idxColor,
                'pixels': set(room_pixels)
            })

    total_habitaciones = len(habitaciones)

    if total_habitaciones == 0:
        return {
            'score_total': 0.0,
            'componentes': {'cobertura': 0.0},
            'penalizaciones': {},
            'debug_info': {
                'num_ventanas': num_ventanas,
                'num_habitaciones_total': 0,
                'habitaciones_con_ventana': 0
            }
        }

    # --- 3) Verificar adyacencia ---
    habitaciones_con_ventana = 0
    detalle = []

    for hab in habitaciones:
        tiene_ventana = False

        for ventana_pixels in ventanas:
            # Usar función existente
            if _is_wall_adjacent_to_room(ventana_pixels, hab['pixels'], max_dist=3):
                tiene_ventana = True
                break

        if tiene_ventana:
            habitaciones_con_ventana += 1

        detalle.append((hab['idx'], hab['tipo'], tiene_ventana))

    # --- 4) Calcular score ---
    score_cobertura = habitaciones_con_ventana / total_habitaciones

    return {
        'score_total': score_cobertura,
        'componentes': {
            'cobertura': score_cobertura
        },
        'penalizaciones': {},
        'debug_info': {
            'num_ventanas': num_ventanas,
            'num_habitaciones_total': total_habitaciones,
            'habitaciones_con_ventana': habitaciones_con_ventana,
            'habitaciones_sin_ventana': total_habitaciones - habitaciones_con_ventana,
            'detalle_habitaciones': detalle
        }
    }

def generar_vecino_mover_ventana(gParam, lPerimeter, max_desplazamiento=None):
    """
    Mueve UNA ventana aleatoria dentro de su mismo muro.
    """
    import numpy as np
    import random
    
    # 1) Buscar ventanas
    ventanas_indices = [i for i, seg in enumerate(lPerimeter) if seg[4] == 'WN']
    if not ventanas_indices:
        return lPerimeter, {"moved": False, "reason": "no_windows"}
    
    idx_seg = random.choice(ventanas_indices)
    
    # 2) Verificar contexto
    if idx_seg == 0 or idx_seg >= len(lPerimeter) - 1:
        return lPerimeter, {"moved": False, "reason": "window_at_boundary"}
    
    wall_before = lPerimeter[idx_seg - 1]
    wall_after = lPerimeter[idx_seg + 1]
    window = lPerimeter[idx_seg]
    
    if wall_before[4] != 'EW' or wall_after[4] != 'EW':
        return lPerimeter, {"moved": False, "reason": "no_EW_adjacent"}
    
    # 3) Reconstruir muro completo
    muro_x1, muro_y1 = wall_before[0], wall_before[1]
    muro_x2, muro_y2 = wall_after[2], wall_after[3]
    
    # 4) Determinar orientación
    is_vertical = (muro_x1 == muro_x2)
    
    nppm = gParam['nPixelsPerMeter']
    END_MARGIN_PX = int(0.15 * nppm)
    
    if is_vertical:
        # Muro VERTICAL → ventana se mueve en Y
        window_height = abs(window[3] - window[1])
        muro_height = abs(muro_y2 - muro_y1)
        
        if max_desplazamiento is None:
            max_desplazamiento = window_height // 2
        
        # Limitar desplazamiento al espacio disponible
        espacio_util = muro_height - window_height - 2 * END_MARGIN_PX
        if espacio_util <= 0:
            return lPerimeter, {"moved": False, "reason": "no_space"}
        
        max_desp_real = min(max_desplazamiento, espacio_util // 2)
        desplazamiento = random.randint(-max_desp_real, max_desp_real)
        
        # Nueva ventana (mueve en Y, X constante)
        new_window = (
            window[0],  # X no cambia
            window[1] + desplazamiento,
            window[2],  # X no cambia
            window[3] + desplazamiento,
            'WN'
        )
        
        # Validar límites
        min_y = min(muro_y1, muro_y2) + END_MARGIN_PX
        max_y = max(muro_y1, muro_y2) - END_MARGIN_PX
        
        if new_window[1] < min_y or new_window[3] > max_y:
            return lPerimeter, {"moved": False, "reason": "out_of_bounds"}
    
    else:
        # Muro HORIZONTAL → ventana se mueve en X
        window_width = abs(window[2] - window[0])
        muro_width = abs(muro_x2 - muro_x1)
        
        if max_desplazamiento is None:
            max_desplazamiento = window_width // 2
        
        espacio_util = muro_width - window_width - 2 * END_MARGIN_PX
        if espacio_util <= 0:
            return lPerimeter, {"moved": False, "reason": "no_space"}
        
        max_desp_real = min(max_desplazamiento, espacio_util // 2)
        desplazamiento = random.randint(-max_desp_real, max_desp_real)
        
        # Nueva ventana (mueve en X, Y constante)
        new_window = (
            window[0] + desplazamiento,
            window[1],  # Y no cambia
            window[2] + desplazamiento,
            window[3],  # Y no cambia
            'WN'
        )
        
        # Validar límites
        min_x = min(muro_x1, muro_x2) + END_MARGIN_PX
        max_x = max(muro_x1, muro_x2) - END_MARGIN_PX
        
        if new_window[0] < min_x or new_window[2] > max_x:
            return lPerimeter, {"moved": False, "reason": "out_of_bounds"}
    
    # 5) Reconstruir segmentos
    new_wall_before = (muro_x1, muro_y1, new_window[0], new_window[1], 'EW')
    new_wall_after = (new_window[2], new_window[3], muro_x2, muro_y2, 'EW')
    
    # 6) Crear nuevo perímetro
    new_perimeter = lPerimeter.copy()
    new_perimeter[idx_seg - 1] = new_wall_before
    new_perimeter[idx_seg] = new_window
    new_perimeter[idx_seg + 1] = new_wall_after
    
    return new_perimeter, {
        "moved": True,
        "idx": idx_seg,
        "desplazamiento": desplazamiento,
        "from": window,
        "to": new_window
    }

def generar_vecino_intercambiar_ventana_de_pared(gParam, lPerimeter, lConnexComponentsCentroids):
    """
    Mueve UNA ventana aleatoria a OTRO muro de la MISMA habitación.
    
    Args:
        gParam: Parámetros globales
        lPerimeter: Lista de segmentos actuales
        lConnexComponentsCentroids: Lista de habitaciones con sus píxeles
    
    Returns:
        new_perimeter: Nueva configuración con ventana en otro muro
        info: Dict con detalles del intercambio
    """
    import numpy as np
    import random
    
    nppm = gParam['nPixelsPerMeter']
    WINDOW_SIZE_PX = int(1.2 * nppm)
    END_MARGIN_PX = int(0.15 * nppm)
    MIN_WALL_LEN_PX = WINDOW_SIZE_PX + 2 * END_MARGIN_PX
    
    # --- 1) Extraer todas las ventanas con su habitación asociada ---
    ventanas_info = []
    
    for idx_seg, (x1, y1, x2, y2, label) in enumerate(lPerimeter):
        if label != 'WN':
            continue
        
        # Obtener píxeles de la ventana
        ventana_pixels = set((p[0], p[1]) for p in getPointsLineBetweenXY(x1, y1, x2, y2))
        
        # Buscar a qué habitación pertenece esta ventana
        habitacion_asociada = None
        
        for idx_cc, (idxColor, rgbColor, room_pixels) in enumerate(lConnexComponentsCentroids):
            # Excluir elementos no-habitación
            if idxColor in {
                gParam['idxNilColorBackground'],
                gParam['idxNilColorExteriorWall'],
                gParam['idxNilColorFrontDoor'],
                gParam['idxNilColorInteriorWall'],
                gParam.get('idxNilColorInsideDoor', -1)
            }:
                continue
            
            room_pixels_set = set(room_pixels)
            
            # Verificar si ventana está adyacente a esta habitación
            if _is_wall_adjacent_to_room(ventana_pixels, room_pixels_set, max_dist=3):
                habitacion_asociada = (idx_cc, idxColor, room_pixels_set)
                break
        
        if habitacion_asociada:
            ventanas_info.append({
                'idx_seg': idx_seg,
                'coords': (x1, y1, x2, y2),
                'habitacion': habitacion_asociada
            })
    
    if not ventanas_info:
        return lPerimeter, {"moved": False, "reason": "no_windows"}
    
    # --- 2) Elegir una ventana aleatoria ---
    ventana_elegida = random.choice(ventanas_info)
    idx_ventana_original = ventana_elegida['idx_seg']
    habitacion_idx, habitacion_tipo, room_pixels_set = ventana_elegida['habitacion']
    
    # --- 3) Buscar otros muros EW adyacentes a la MISMA habitación ---
    muros_candidatos = []
    
    for idx_seg, (x1, y1, x2, y2, label) in enumerate(lPerimeter):
        # Excluir: no-EW, la ventana actual, y muros con ventana ya
        if label != 'EW':
            continue
        
        # No elegir el mismo segmento o muros que ya tienen ventana adyacente
        if idx_seg == idx_ventana_original - 1 or idx_seg == idx_ventana_original + 1:
            continue  # Es el muro actual de la ventana
        
        # Calcular longitud
        wall_len = float(np.hypot(x2 - x1, y2 - y1))
        if wall_len < MIN_WALL_LEN_PX:
            continue
        
        # Verificar si es adyacente a la misma habitación
        wall_pixels = getPointsLineBetweenXY(x1, y1, x2, y2)
        if _is_wall_adjacent_to_room(wall_pixels, room_pixels_set, max_dist=3):
            muros_candidatos.append({
                'idx_seg': idx_seg,
                'coords': (x1, y1, x2, y2),
                'length': wall_len
            })
    
    if not muros_candidatos:
        return lPerimeter, {
            "moved": False, 
            "reason": "no_alternative_walls",
            "habitacion": habitacion_idx
        }
    
    # --- 4) Elegir muro candidato (preferir más largo) ---
    muro_destino = max(muros_candidatos, key=lambda m: m['length'])
    
    # --- 5) Eliminar ventana del muro original ---
    new_perimeter = []
    
    for i, seg in enumerate(lPerimeter):
        if i == idx_ventana_original - 1:
            # Segmento EW antes de la ventana
            continue
        elif i == idx_ventana_original:
            # La ventana misma
            continue
        elif i == idx_ventana_original + 1:
            # Segmento EW después de la ventana
            # Fusionar EW antes + EW después
            ew_before = lPerimeter[idx_ventana_original - 1]
            ew_after = seg
            
            # Crear muro completo sin ventana
            muro_completo = (ew_before[0], ew_before[1], ew_after[2], ew_after[3], 'EW')
            new_perimeter.append(muro_completo)
        else:
            new_perimeter.append(seg)
    
    # --- 6) Colocar ventana en el nuevo muro ---
    x1, y1, x2, y2 = muro_destino['coords']
    wall_len = muro_destino['length']
    
    v = np.array([x2 - x1, y2 - y1], dtype=float)
    v_unit = v / wall_len
    
    # Calcular centro del muro
    mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
    half = WINDOW_SIZE_PX / 2.0
    p1 = mid - v_unit * half
    p2 = mid + v_unit * half
    
    win_x1, win_y1 = int(round(p1[0])), int(round(p1[1]))
    win_x2, win_y2 = int(round(p2[0])), int(round(p2[1]))
    
    # --- 7) Insertar ventana partiendo el muro destino ---
    final_perimeter = []
    idx_destino = muro_destino['idx_seg']
    
    # Ajustar índice si eliminamos segmentos antes
    if idx_destino > idx_ventana_original:
        idx_destino -= 2  # Eliminamos 2 segmentos (EW antes, WN)
    
    for i, seg in enumerate(new_perimeter):
        if i == idx_destino:
            # Partir este muro en: EW | WN | EW
            final_perimeter.append((x1, y1, win_x1, win_y1, 'EW'))
            final_perimeter.append((win_x1, win_y1, win_x2, win_y2, 'WN'))
            final_perimeter.append((win_x2, win_y2, x2, y2, 'EW'))
        else:
            final_perimeter.append(seg)
    
    return final_perimeter, {
        "moved": True,
        "operador": "INTERCAMBIAR_PARED",
        "habitacion": habitacion_idx,
        "ventana_original_idx": idx_ventana_original,
        "muro_destino_idx": idx_destino,
        "from": ventana_elegida['coords'],
        "to": (win_x1, win_y1, win_x2, win_y2),
        "muros_disponibles": len(muros_candidatos)
    }
    
def generar_vecino_anadir_ventana(gParam, lPerimeter, lConnexComponentsCentroids):
    """
    Añade UNA ventana en un muro aleatorio que NO tenga ventana y cumpla requisitos mínimos.
    
    Args:
        gParam: Parámetros globales
        lPerimeter: Lista de segmentos actuales
        lConnexComponentsCentroids: Lista de habitaciones con sus píxeles
    
    Returns:
        new_perimeter: Nueva configuración con ventana adicional
        info: Dict con detalles de la ventana añadida
    """
    import numpy as np
    import random
    
    nppm = gParam['nPixelsPerMeter']
    WINDOW_SIZE_PX = int(1.2 * nppm)
    END_MARGIN_PX = int(0.15 * nppm)
    MIN_WALL_LEN_PX = WINDOW_SIZE_PX + 2 * END_MARGIN_PX
    
    # --- 1) Identificar muros EW que NO tienen ventana adyacente ---
    muros_sin_ventana = []
    
    for idx_seg, (x1, y1, x2, y2, label) in enumerate(lPerimeter):
        if label != 'EW':
            continue
        
        # Verificar longitud mínima
        wall_len = float(np.hypot(x2 - x1, y2 - y1))
        if wall_len < MIN_WALL_LEN_PX:
            continue
        
        # Verificar que NO tenga ventana adyacente
        tiene_ventana_adyacente = False
        
        # Mirar segmento anterior
        if idx_seg > 0 and lPerimeter[idx_seg - 1][4] == 'WN':
            tiene_ventana_adyacente = True
        
        # Mirar segmento siguiente
        if idx_seg < len(lPerimeter) - 1 and lPerimeter[idx_seg + 1][4] == 'WN':
            tiene_ventana_adyacente = True
        
        if not tiene_ventana_adyacente:
            muros_sin_ventana.append({
                'idx_seg': idx_seg,
                'coords': (x1, y1, x2, y2),
                'length': wall_len
            })
    
    if not muros_sin_ventana:
        return lPerimeter, {
            "added": False,
            "reason": "no_available_walls"
        }
    
    # --- 2) Elegir muro aleatorio ---
    muro_elegido = random.choice(muros_sin_ventana)
    idx_muro = muro_elegido['idx_seg']
    x1, y1, x2, y2 = muro_elegido['coords']
    wall_len = muro_elegido['length']
    
    # --- 3) Calcular posición de la nueva ventana (centrada) ---
    v = np.array([x2 - x1, y2 - y1], dtype=float)
    v_unit = v / wall_len
    
    # Centro del muro
    mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
    half = WINDOW_SIZE_PX / 2.0
    p1 = mid - v_unit * half
    p2 = mid + v_unit * half
    
    win_x1, win_y1 = int(round(p1[0])), int(round(p1[1]))
    win_x2, win_y2 = int(round(p2[0])), int(round(p2[1]))
    
    # --- 4) Verificar que la ventana queda dentro de los márgenes ---
    # Distancia del centro de la ventana a los extremos del muro
    dist_inicio = np.linalg.norm(mid - np.array([x1, y1]))
    dist_fin = np.linalg.norm(mid - np.array([x2, y2]))
    
    if dist_inicio < (END_MARGIN_PX + half) or dist_fin < (END_MARGIN_PX + half):
        return lPerimeter, {
            "added": False,
            "reason": "insufficient_margin",
            "wall_idx": idx_muro
        }
    
    # --- 5) Identificar habitación asociada (opcional, para info) ---
    habitacion_asociada = None
    wall_pixels = getPointsLineBetweenXY(x1, y1, x2, y2)
    
    for idx_cc, (idxColor, rgbColor, room_pixels) in enumerate(lConnexComponentsCentroids):
        if idxColor in {
            gParam['idxNilColorBackground'],
            gParam['idxNilColorExteriorWall'],
            gParam['idxNilColorFrontDoor'],
            gParam['idxNilColorInteriorWall'],
            gParam.get('idxNilColorInsideDoor', -1)
        }:
            continue
        
        room_pixels_set = set(room_pixels)
        if _is_wall_adjacent_to_room(wall_pixels, room_pixels_set, max_dist=3):
            habitacion_asociada = (idx_cc, idxColor)
            break
    
    # --- 6) Partir el muro e insertar ventana ---
    new_perimeter = []
    
    for i, seg in enumerate(lPerimeter):
        if i == idx_muro:
            # Reemplazar este muro EW por: EW | WN | EW
            new_perimeter.append((x1, y1, win_x1, win_y1, 'EW'))
            new_perimeter.append((win_x1, win_y1, win_x2, win_y2, 'WN'))
            new_perimeter.append((win_x2, win_y2, x2, y2, 'EW'))
        else:
            new_perimeter.append(seg)
    
    return new_perimeter, {
        "added": True,
        "operador": "ANADIR_VENTANA",
        "wall_idx": idx_muro,
        "wall_length": wall_len,
        "window_coords": (win_x1, win_y1, win_x2, win_y2),
        "habitacion": habitacion_asociada[0] if habitacion_asociada else None,
        "muros_disponibles": len(muros_sin_ventana)
    }

def generar_vecino_eliminar_ventana(gParam, lPerimeter, lConnexComponentsCentroids):
    """
    Elimina UNA ventana aleatoria fusionando los muros EW adyacentes.
    Útil para simplificar soluciones o eliminar ventanas redundantes/subóptimas.
    
    Args:
        gParam: Parámetros globales
        lPerimeter: Lista de segmentos actuales
        lConnexComponentsCentroids: Lista de habitaciones (para info de debug)
    
    Returns:
        new_perimeter: Nueva configuración sin la ventana
        info: Dict con detalles de la ventana eliminada
    """
    import numpy as np
    import random
    
    # --- 1) Buscar todas las ventanas ---
    ventanas_indices = []
    for idx, (x1, y1, x2, y2, label) in enumerate(lPerimeter):
        if label == 'WN':
            ventanas_indices.append(idx)
    
    if not ventanas_indices:
        return lPerimeter, {
            "removed": False,
            "reason": "no_windows"
        }
    
    # --- 2) Elegir ventana aleatoria ---
    idx_ventana = random.choice(ventanas_indices)
    window = lPerimeter[idx_ventana]
    
    # --- 3) Verificar que tiene muros EW adyacentes ---
    if idx_ventana == 0 or idx_ventana >= len(lPerimeter) - 1:
        return lPerimeter, {
            "removed": False,
            "reason": "window_at_boundary"
        }
    
    wall_before = lPerimeter[idx_ventana - 1]
    wall_after = lPerimeter[idx_ventana + 1]
    
    if wall_before[4] != 'EW' or wall_after[4] != 'EW':
        return lPerimeter, {
            "removed": False,
            "reason": "no_EW_walls_adjacent"
        }
    
    # --- 4) Obtener info de la habitación asociada (para debug) ---
    ventana_pixels = set((p[0], p[1]) for p in getPointsLineBetweenXY(window[0], window[1], window[2], window[3]))
    habitacion_asociada = None
    
    for idx_cc, (idxColor, rgbColor, room_pixels) in enumerate(lConnexComponentsCentroids):
        if idxColor in {
            gParam['idxNilColorBackground'],
            gParam['idxNilColorExteriorWall'],
            gParam['idxNilColorFrontDoor'],
            gParam['idxNilColorInteriorWall'],
            gParam.get('idxNilColorInsideDoor', -1)
        }:
            continue
        
        room_pixels_set = set(room_pixels)
        if _is_wall_adjacent_to_room(ventana_pixels, room_pixels_set, max_dist=3):
            habitacion_asociada = (idx_cc, idxColor)
            break
    
    # --- 5) Fusionar muros EW ---
    # Crear muro completo: inicio del EW antes → fin del EW después
    muro_completo = (
        wall_before[0], wall_before[1],  # Inicio del muro antes
        wall_after[2], wall_after[3],     # Fin del muro después
        'EW'
    )
    
    muro_length = float(np.hypot(
        muro_completo[2] - muro_completo[0],
        muro_completo[3] - muro_completo[1]
    ))
    
    # --- 6) Reconstruir perímetro sin la ventana ---
    new_perimeter = []
    
    for i, seg in enumerate(lPerimeter):
        if i == idx_ventana - 1:
            # Segmento EW antes de la ventana → skip
            continue
        elif i == idx_ventana:
            # La ventana misma → skip
            continue
        elif i == idx_ventana + 1:
            # Segmento EW después de la ventana → insertar muro fusionado
            new_perimeter.append(muro_completo)
        else:
            # Resto de segmentos sin cambios
            new_perimeter.append(seg)
    
    return new_perimeter, {
        "removed": True,
        "operador": "ELIMINAR_VENTANA",
        "window_idx": idx_ventana,
        "window_coords": window,
        "wall_merged_length": muro_length,
        "habitacion": habitacion_asociada[0] if habitacion_asociada else None,
        "ventanas_restantes": len(ventanas_indices) - 1
    }

def hill_climbing_window_optimization(gParam, lPerimeter_inicial, lConnexComponentsCentroids,
                                     max_iteraciones=500,
                                     max_sin_mejora=50,
                                     probabilidades_operadores=None,
                                     score_weights=None,
                                     verbose=True):
    """
    Optimiza la colocación de ventanas usando Hill Climbing con múltiples operadores.
    
    Args:
        gParam: Diccionario con parámetros globales
        lPerimeter_inicial: Perímetro inicial (generado por heurística)
        lConnexComponentsCentroids: Lista de habitaciones con sus píxeles
        max_iteraciones: Número máximo de iteraciones
        max_sin_mejora: Parar si no mejora en N iteraciones consecutivas
        probabilidades_operadores: Dict con probabilidades de cada operador
            Ejemplo: {'Op1': 0.45, 'Op2': 0.30, 'Op3': 0.15, 'Op4': 0.10}
            Si None, usa probabilidades adaptativas
        score_weights: Dict con pesos de cada componente del score
        verbose: Si True, imprime progreso
    
    Returns:
        dict con:
            'mejor_perimeter': Mejor configuración encontrada
            'mejor_score': Score de la mejor configuración
            'score_inicial': Score de la configuración inicial
            'mejora_absoluta': Diferencia entre score final e inicial
            'mejora_relativa': Porcentaje de mejora
            'iteraciones_totales': Número de iteraciones ejecutadas
            'historial': Lista de scores por iteración
            'estadisticas_operadores': Contadores de uso y éxito por operador
    """
    import numpy as np
    import random
    
    # --- Inicialización ---
    mejor_perimeter = lPerimeter_inicial
    eval_inicial = evaluar(gParam, mejor_perimeter, lConnexComponentsCentroids, score_weights)
    mejor_score = eval_inicial['score_total']
    score_inicial = mejor_score
    
    iteraciones_sin_mejora = 0
    historial = []
    
    # Estadísticas de operadores
    stats_ops = {
        'Op1': {'intentos': 0, 'exitos': 0, 'mejoras': 0},
        'Op2': {'intentos': 0, 'exitos': 0, 'mejoras': 0},
        'Op3': {'intentos': 0, 'exitos': 0, 'mejoras': 0},
        'Op4': {'intentos': 0, 'exitos': 0, 'mejoras': 0}
    }
    
    if verbose:
        print("=" * 70)
        print("HILL CLIMBING - Optimización de Ventanas")
        print("=" * 70)
        print(f"Score inicial: {mejor_score:.4f}")
        print(f"  Cobertura: {eval_inicial['componentes']['cobertura']:.2%}")
        print(f"  Ventanas: {eval_inicial['debug_info']['num_ventanas']}")
        print(f"  Habitaciones: {eval_inicial['debug_info']['num_habitaciones_total']}")
        print(f"  Habitaciones con ventana: {eval_inicial['debug_info']['habitaciones_con_ventana']}")
        print("-" * 70)
    
    # --- Loop principal de Hill Climbing ---
    for iteracion in range(max_iteraciones):
        
        # 1) Determinar probabilidades adaptativas si no se especificaron
        if probabilidades_operadores is None:
            eval_actual = evaluar(gParam, mejor_perimeter, lConnexComponentsCentroids, score_weights)
            score_cobertura = eval_actual['componentes']['cobertura']
            num_ventanas = eval_actual['debug_info']['num_ventanas']
            num_habitaciones = eval_actual['debug_info']['num_habitaciones_total']
            
            aux = True
            # Estrategia adaptativa
            if  aux: #score_cobertura < 0.7:
                # Baja cobertura → priorizar AÑADIR
                probs = {'Op1': 0.15, 'Op2': 0.35, 'Op3': 0.45, 'Op4': 0.05}
            elif num_ventanas > num_habitaciones * 1.5:
                # Muchas ventanas → priorizar ELIMINAR y MOVER
                probs = {'Op1': 0.40, 'Op2': 0.25, 'Op3': 0.05, 'Op4': 0.30}
            elif score_cobertura >= 0.9:
                # Buena cobertura → optimizar posición (MOVER/INTERCAMBIAR)
                probs = {'Op1': 0.50, 'Op2': 0.35, 'Op3': 0.10, 'Op4': 0.05}
            else:
                # Balanceado
                probs = {'Op1': 0.45, 'Op2': 0.30, 'Op3': 0.15, 'Op4': 0.10}
        else:
            probs = probabilidades_operadores
        
        # 2) Elegir operador aleatorio
        ops = list(probs.keys())
        pesos = list(probs.values())
        operador = random.choices(ops, weights=pesos)[0]
        
        stats_ops[operador]['intentos'] += 1
        
        # 3) Generar vecino según operador
        if operador == 'Op1':
            vecino_perimeter, info = generar_vecino_mover_ventana(gParam, mejor_perimeter)
        elif operador == 'Op2':
            vecino_perimeter, info = generar_vecino_intercambiar_ventana_de_pared(
                gParam, mejor_perimeter, lConnexComponentsCentroids
            )
        elif operador == 'Op3':
            vecino_perimeter, info = generar_vecino_anadir_ventana(
                gParam, mejor_perimeter, lConnexComponentsCentroids
            )
        elif operador == 'Op4':
            vecino_perimeter, info = generar_vecino_eliminar_ventana(
                gParam, mejor_perimeter, lConnexComponentsCentroids
            )
        else:
            continue
        
        # 4) Verificar si el vecino es válido
        movimiento_exitoso = info.get('moved', info.get('added', info.get('removed', False)))
        
        if not movimiento_exitoso:
            # Operador no pudo generar vecino válido
            historial.append({
                'iteracion': iteracion,
                'score': mejor_score,
                'operador': operador,
                'exito_operador': False,
                'mejora': False
            })
            continue
        
        stats_ops[operador]['exitos'] += 1
        
        # 5) Evaluar vecino
        eval_vecino = evaluar(gParam, vecino_perimeter, lConnexComponentsCentroids, score_weights)
        score_vecino = eval_vecino['score_total']
        
        # 6) Criterio de aceptación: Hill Climbing estricto (solo mejoras)
        if score_vecino > mejor_score:
            # ACEPTAR: el vecino es mejor
            mejor_perimeter = vecino_perimeter
            mejora = score_vecino - mejor_score
            mejor_score = score_vecino
            iteraciones_sin_mejora = 0
            stats_ops[operador]['mejoras'] += 1
            
            if verbose and (iteracion % 10 == 0 or mejora > 0.01):
                print(f"[Iter {iteracion:4d}] ✓ MEJORA | Score: {mejor_score:.4f} (+{mejora:.4f}) | "
                      f"Op: {operador} | Cobertura: {eval_vecino['componentes']['cobertura']:.2%}")
            
            historial.append({
                'iteracion': iteracion,
                'score': mejor_score,
                'operador': operador,
                'exito_operador': True,
                'mejora': True,
                'delta_score': mejora
            })
        else:
            # RECHAZAR: el vecino no mejora
            iteraciones_sin_mejora += 1
            
            historial.append({
                'iteracion': iteracion,
                'score': mejor_score,
                'operador': operador,
                'exito_operador': True,
                'mejora': False,
                'delta_score': score_vecino - mejor_score
            })
        
        # 7) Criterio de parada: sin mejoras durante N iteraciones
        if iteraciones_sin_mejora >= max_sin_mejora:
            if verbose:
                print(f"\n{'='*70}")
                print(f"⚠️  Parada temprana: {max_sin_mejora} iteraciones sin mejora")
            break
    
    # --- Evaluación final ---
    eval_final = evaluar(gParam, mejor_perimeter, lConnexComponentsCentroids, score_weights)
    
    mejora_absoluta = mejor_score - score_inicial
    mejora_relativa = (mejora_absoluta / score_inicial * 100) if score_inicial > 0 else 0
    
    if verbose:
        print("=" * 70)
        print("RESULTADOS FINALES")
        print("=" * 70)
        print(f"Score inicial:  {score_inicial:.4f}")
        print(f"Score final:    {mejor_score:.4f}")
        print(f"Mejora:         {mejora_absoluta:+.4f} ({mejora_relativa:+.1f}%)")
        print(f"Iteraciones:    {iteracion + 1} / {max_iteraciones}")
        print()
        print("Configuración final:")
        print(f"  Cobertura:              {eval_final['componentes']['cobertura']:.2%}")
        print(f"  Ventanas totales:       {eval_final['debug_info']['num_ventanas']}")
        print(f"  Habitaciones totales:   {eval_final['debug_info']['num_habitaciones_total']}")
        print(f"  Habitaciones c/ventana: {eval_final['debug_info']['habitaciones_con_ventana']}")
        print(f"  Habitaciones s/ventana: {eval_final['debug_info']['habitaciones_sin_ventana']}")
        print()
        print("Estadísticas de operadores:")
        print("-" * 70)
        for op, stats in stats_ops.items():
            intentos = stats['intentos']
            exitos = stats['exitos']
            mejoras = stats['mejoras']
            tasa_exito = (exitos / intentos * 100) if intentos > 0 else 0
            tasa_mejora = (mejoras / exitos * 100) if exitos > 0 else 0
            
            nombre_op = {
                'Op1': 'MOVER',
                'Op2': 'INTERCAMBIAR',
                'Op3': 'AÑADIR',
                'Op4': 'ELIMINAR'
            }[op]
            
            print(f"  {nombre_op:15s}: {intentos:4d} intentos | "
                  f"{exitos:4d} éxitos ({tasa_exito:5.1f}%) | "
                  f"{mejoras:4d} mejoras ({tasa_mejora:5.1f}%)")
        print("=" * 70)
    
    return {
        'mejor_perimeter': mejor_perimeter,
        'mejor_score': mejor_score,
        'score_inicial': score_inicial,
        'eval_inicial': eval_inicial,
        'eval_final': eval_final,
        'mejora_absoluta': mejora_absoluta,
        'mejora_relativa': mejora_relativa,
        'iteraciones_totales': iteracion + 1,
        'iteraciones_sin_mejora': iteraciones_sin_mejora,
        'historial': historial,
        'estadisticas_operadores': stats_ops
    }