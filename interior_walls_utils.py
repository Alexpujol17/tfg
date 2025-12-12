import numpy as np
from skimage.segmentation import watershed, expand_labels
from skimage.morphology import skeletonize, binary_dilation, square
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt, binary_fill_holes
import cv2

# En window_placement_utils.py
from Analysis_StyleGAN_functions_03_pindatafinal24_Relaxed import getPointsLineBetweenXY
from interior_doors_placement_utils import build_adjacency_graph, place_interior_doors_mst

def build_interior_walls_from_borders(gParam, rgbImageNilColors, lConnexComponents,
                                      imageContourSmoothed=None,
                                      lPerimeter=None,
                                      min_segment_len=12, thickness_px=2,
                                      config=None):
    """
    Función Principal: Detecta dónde terminan las habitaciones y empiezan otras para
    crear los muros interiores. Luego los limpia, los convierte en líneas y 
    finalmente decide dónde van las puertas.
    """
    height = gParam['height']; width = gParam['width']
    listNilColors = gParam['listNilColors']
    idxBG  = gParam['idxNilColorBackground']
    idxEW  = gParam['idxNilColorExteriorWall']
    idxFD  = gParam['idxNilColorFrontDoor']
    idxIW  = gParam['idxNilColorInteriorWall']
    idxID  = gParam['idxNilColorInsideDoor']

    # ==========================================
    # 1. TRADUCCIÓN DE COLORES A IDENTIFICADORES
    # ==========================================
    # Convertimos la lista de píxeles RGB en una matriz 2D (mapa) donde cada
    # píxel tiene un número ID. Esto facilita saber quién es vecino de quién.
    idx_map = np.full((height, width), idxBG, dtype=np.int32)
    lc2idx = {tuple(listNilColors[i]): i for i in range(len(listNilColors))}
    
    for x in range(height):      # Barrido vertical
        base = x * width
        for y in range(width):   # Barrido horizontal
            idx_map[x, y] = lc2idx[rgbImageNilColors[base + y]]

    # ==========================================
    # 1.5 (ELIMINADO - Ahora usamos extensión geométrica de paredes)
    # ==========================================

    # ==========================================
    # 2. DETECCIÓN DE FRONTERAS (BORDERS)
    # ==========================================
    # Un muro existe donde un píxel es "Habitación" y su vecino es "Otra Habitación".
    # Ignoramos fondo, muros exteriores y puertas de entrada.
    mask_room = (idx_map != idxBG) & (idx_map != idxEW) & (idx_map != idxFD)
    
    # Creamos copias del mapa desplazadas para comparar vecinos rápidamente
    down  = np.zeros_like(idx_map); down[:-1,:]= idx_map[1:,:]
    right = np.zeros_like(idx_map); right[:,:-1]= idx_map[:,1:]

    # Lógica: Soy habitación Y mi vecino de abajo es diferente Y mi vecino no es fondo/exterior.
    b_down  = mask_room & (idx_map != down)  & (down  != idxBG) & (down!= idxEW) & (down!= idxFD) & (down != idxIW) & (down != idxID)
    b_right = mask_room & (idx_map != right) & (right != idxBG) & (right!= idxEW) & (right!= idxFD) & (right != idxIW) & (right != idxID)

    # Combinamos bordes horizontales y verticales
    border_mask = b_down | b_right 

    # 3. LIMPIEZA DE BORDES
    # Si nos pasan el contorno exterior limpio, lo usamos para no pintar muros encima del exterior.
    if imageContourSmoothed is not None:
        border_mask = border_mask & (imageContourSmoothed.astype(bool)) 

    # 'Closing' cierra pequeños agujeritos en la línea detectada para que sea continua.
    from skimage.morphology import closing, square
    border_mask = closing(border_mask, square(3))
    
    # ==========================================
    # 4. ADELGAZAMIENTO (SKELETONIZATION)
    # ==========================================
    # La frontera detectada puede ser gruesa. La convertimos en una línea fina
    # de 1 píxel de ancho (el "esqueleto") para poder vectorizarla bien.
    if np.any(border_mask):
        skel = skeletonize(border_mask.astype(bool))
    else:
        skel = border_mask

    # ==========================================
    # 5. FILTRADO DE RUIDO
    # ==========================================
    # Eliminamos líneas minúsculas (manchas de polvo) que sean menores a 'min_segment_len'.
    lbl = label(skel, connectivity=2)
    keep = np.zeros_like(skel, dtype=bool)
    for r in regionprops(lbl):
        if r.area >= min_segment_len:
            keep[lbl == r.label] = True

    wall_mask = keep
    # Si queremos visualizarlos más gordos en la máscara final:
    if thickness_px and thickness_px > 1:
        wall_mask = binary_dilation(keep, square(thickness_px))

    # ==========================================
    # 6. GUARDAR DATOS (PIXEL PIXEL)
    # ==========================================
    # Guardamos estas paredes en la lista global de componentes de la casa.
    lbl_w = label(wall_mask, connectivity=2)
    cc_list = []
    for r in regionprops(lbl_w):
        coords = r.coords  
        cc_pixels = [(int(x), int(y)) for (x, y) in coords]
        cc_list.append(cc_pixels)

    # Insertamos o actualizamos la entrada de 'Interior Walls' en la lista principal
    if len(lConnexComponents) > idxIW and isinstance(lConnexComponents[idxIW], tuple):
        lConnexComponents[idxIW][2].extend(cc_list)
    else:
        rgbIW = listNilColors[idxIW]
        lConnexComponents.insert(idxIW, (idxIW, rgbIW, cc_list))

    # ==========================================
    # 7. VECTORIZACIÓN Y CORTE (GEOMETRÍA)
    # ==========================================
    # Convertimos los píxeles en líneas matemáticas (x1, y1, x2, y2)
    pwalls = []
    for i, cc in enumerate(cc_list):
        pwalls.extend(pixels_to_walls(cc, padding=5))

    if lPerimeter is not None:
        print("lPerimeter:")
        for perim_wall in lPerimeter:
            print("\n", perim_wall)
        print("Interior walls before fragmentation:")
        for wall in pwalls:
            print("\n", wall)
    
    # ==========================================
    # 7.5 EXTENSIÓN DE PAREDES HASTA EL PERÍMETRO
    # ==========================================
    # Solo extendemos paredes cuyos extremos estaban "cerca" del perímetro
    # pero ahora no lo tocan (debido al suavizado)
    if lPerimeter is not None and pwalls:
        pwalls = extend_walls_to_perimeter(pwalls, lPerimeter, threshold=7)
    
    # IMPORTANTE: Cortamos las paredes donde se cruzan (Intersecciones T o X).
    # Esto es vital para que el grafo entienda las conexiones correctamente.
    pwalls_final = fragment_walls_at_intersections(pwalls)

    for wall in pwalls_final:
        print(" Fragmented: ", wall)

    # ==========================================
    # 8. COLOCACIÓN DE PUERTAS (LÓGICA DE GRAFOS)
    # ==========================================
    # Aquí llamamos al código anterior: Creamos el grafo y calculamos el MST.
    G, start_node = build_adjacency_graph(gParam, lConnexComponents, pwalls_final, lPerimeter)

    pwalls_final = place_interior_doors_mst(gParam, pwalls_final, G, start_node, config=config)

    print(f"Habitación de Entrada (Start Node): {start_node}")
    print("Conexiones detectadas:")
    for u, v, data in G.edges(data=True):
        wall_idx = data['wall_idx']
        length = data['length']
        print(f"   - Room {u} <--> Room {v} (via Wall {wall_idx}, len={length:.1f}px)")

    for wall in pwalls_final:
        print(" Walls: ", wall)

    return lConnexComponents, pwalls_final

import numpy as np
import cv2
from skimage.morphology import skeletonize

def pixels_to_walls(pixels, padding=5):
    """
    Convierte un montón de píxeles desordenados en una línea recta limpia (vector).
    Usa la Transformada de Hough.
    """
    if not pixels:
        return []

    # A. Preparamos un lienzo negro temporal
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    height = max_r - min_r + (padding * 2)
    width = max_c - min_c + (padding * 2)
    img = np.zeros((height, width), dtype=np.uint8)

    # B. Pintamos los píxeles
    for r, c in pixels:
        img[r - min_r + padding, c - min_c + padding] = 255

    # C. Volvemos a adelgazar (Skeletonize) por seguridad
    skeleton = skeletonize(img // 255).astype(np.uint8) * 255
    
    # D. DETECCIÓN DE LÍNEAS (HoughLinesP)
    # Este algoritmo busca patrones lineales en la imagen.
    lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180, threshold=10, minLineLength=0, maxLineGap=10)
    
    wall_segments = []
    
    if lines is not None:
        for line in lines:
            x1_local, y1_local, x2_local, y2_local = line[0]
            
            # E. Ajustamos coordenadas locales a globales de la casa
            real_x1 = x1_local - padding + min_c
            real_y1 = y1_local - padding + min_r
            real_x2 = x2_local - padding + min_c
            real_y2 = y2_local - padding + min_r
            
            # 'IW' = Interior Wall
            wall_segments.append((real_x1, real_y1, real_x2, real_y2, 'IW'))
            
    return wall_segments

import numpy as np
import cv2
from skimage.morphology import skeletonize

def fragment_walls_at_intersections(pwalls, tolerance=5):
    """
    Corta las paredes largas cuando son atravesadas por otra pared.
    Si tienes una 'T', convierte la pared superior de la T en dos segmentos (izq y der).
    Esto es necesario para saber exactamente qué tramo de pared conecta qué habitaciones.
    """
    if not pwalls:
        return []

    # 0. Limpieza inicial de datos
    walls_clean = []
    for w in pwalls:
        walls_clean.append((int(w[0]), int(w[1]), int(w[2]), int(w[3]), w[4]))

    # Preparamos un almacén de puntos de corte para cada pared
    cuts = {i: set() for i in range(len(walls_clean))}

    # 1. BÚSQUEDA DE INTERSECCIONES
    # Comparamos cada pared con todas las demás
    for i in range(len(walls_clean)):
        for j in range(i + 1, len(walls_clean)):
            wall_A = walls_clean[i]
            wall_B = walls_clean[j]
            
            xA1, yA1, xA2, yA2, _ = wall_A
            xB1, yB1, xB2, yB2, _ = wall_B
            
            # Determinamos si son Horizontales (H) o Verticales (V)
            type_A = 'H' if abs(yA1 - yA2) < abs(xA1 - xA2) else 'V'
            type_B = 'H' if abs(yB1 - yB2) < abs(xB1 - xB2) else 'V'

            # Solo nos interesan los cruces perpendiculares (H con V)
            if type_A == type_B:
                continue

            # Identificamos quién es quién
            if type_A == 'H':
                H_wall, H_idx = wall_A, i
                V_wall, V_idx = wall_B, j
            else:
                H_wall, H_idx = wall_B, j
                V_wall, V_idx = wall_A, i

            xH1, yH1, xH2, yH2, _ = H_wall
            xV1, yV1, xV2, yV2, _ = V_wall

            # Calculamos el punto donde deberían cruzarse
            hy = int(round((yH1 + yH2) / 2.0))
            vx = int(round((xV1 + xV2) / 2.0))

            hx_min, hx_max = min(xH1, xH2), max(xH1, xH2)
            vy_min, vy_max = min(yV1, yV2), max(yV1, yV2)

            # Verificamos si realmente se cruzan (con un margen de tolerancia)
            intersects_x = (hx_min - tolerance) <= vx <= (hx_max + tolerance)
            intersects_y = (vy_min - tolerance) <= hy <= (vy_max + tolerance)

            if not (intersects_x and intersects_y):
                continue

            # Aseguramos que el corte esté dentro de los límites de la pared
            cut_x = max(hx_min, min(vx, hx_max))
            cut_y = max(vy_min, min(hy, vy_max))

            # Añadimos el punto de corte a la lista de "tareas pendientes" de cada pared
            cuts[H_idx].add(cut_x) # Pared Horizontal se corta en X
            cuts[V_idx].add(cut_y) # Pared Vertical se corta en Y

    # 2. RECONSTRUCCIÓN (CORTAR Y PEGAR)
    new_segments = []
    
    for i, wall in enumerate(walls_clean):
        x1, y1, x2, y2, tag = wall
        is_horiz = abs(y1 - y2) < abs(x1 - x2)
        
        points_to_cut = sorted(list(cuts[i]))
        
        # Si no hay cortes, la pared pasa tal cual
        if not points_to_cut:
            new_segments.append(wall)
            continue
            
        if is_horiz:
            # Cortar pared Horizontal en trocitos
            curr_x = min(x1, x2)
            end_x = max(x1, x2)
            fixed_y = int(round((y1 + y2) / 2.0))
            
            segment_points = [curr_x] + points_to_cut + [end_x]
            
            for k in range(len(segment_points) - 1):
                p_start = segment_points[k]
                p_end   = segment_points[k + 1]
                if abs(p_start - p_end) > 1:
                    new_segments.append((p_start, fixed_y, p_end, fixed_y, tag))
                
        else:
            # Cortar pared Vertical en trocitos
            curr_y = min(y1, y2)
            end_y  = max(y1, y2)
            fixed_x = int(round((x1 + x2) / 2.0))
            
            segment_points = [curr_y] + points_to_cut + [end_y]
            
            for k in range(len(segment_points) - 1):
                p_start = segment_points[k]
                p_end   = segment_points[k + 1]
                if abs(p_start - p_end) > 1:
                    new_segments.append((fixed_x, p_start, fixed_x, p_end, tag))

    return new_segments


def extend_walls_to_perimeter(pwalls, lPerimeter, threshold=5):
    """
    Extiende las paredes interiores hasta el perímetro, pero SOLO si:
    - El extremo de la pared está "cerca" del perímetro (< threshold px)
    - Pero no lo toca exactamente (distancia > 0)
    
    NOTA: lPerimeter usa formato NumPy (x=Fila, y=Columna)
          pwalls usa formato OpenCV (x=Columna, y=Fila)
          Convertimos lPerimeter a formato OpenCV para comparar.
    """
    import math
    
    # Convertir lPerimeter a formato OpenCV (x=Col, y=Row)
    lPerimeter_cv = []
    for seg in lPerimeter:
        row1, col1, row2, col2, tag = seg
        # Intercambiamos: (row, col) → (col, row) = (x, y) en OpenCV
        lPerimeter_cv.append((col1, row1, col2, row2, tag))
    
    def point_to_segment_distance(px, py, x1, y1, x2, y2):
        """Distancia de un punto (px,py) a un segmento (x1,y1)-(x2,y2)"""
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def min_distance_to_perimeter(px, py):
        """Distancia mínima de un punto al perímetro"""
        min_dist = float('inf')
        for seg in lPerimeter_cv:
            x1, y1, x2, y2, _ = seg
            dist = point_to_segment_distance(px, py, x1, y1, x2, y2)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    def find_intersection_with_perimeter(px, py, direction):
        """
        Encuentra la intersección de una línea desde (px,py) en 'direction'
        con el perímetro. direction = 'left', 'right', 'up', 'down'
        
        NOTA: El perímetro tiene grosor 3px (centro en el medio).
        Añadimos offset de 2px para parar en el borde interno.
        """
        PERIMETER_HALF_THICKNESS = 2  # Grosor/2 + 1px de margen
        
        best_point = None
        best_dist = float('inf')
        
        for seg in lPerimeter_cv:
            sx1, sy1, sx2, sy2, _ = seg
            
            # Determinar si el segmento del perímetro es H o V
            seg_is_horiz = abs(sy1 - sy2) < abs(sx1 - sx2)
            
            if direction in ['left', 'right']:
                # Buscamos segmentos verticales del perímetro
                if seg_is_horiz:
                    continue
                seg_x = (sx1 + sx2) / 2
                seg_y_min = min(sy1, sy2)
                seg_y_max = max(sy1, sy2)
                
                # El punto debe estar en el rango Y del segmento
                if not (seg_y_min - 5 <= py <= seg_y_max + 5):
                    continue
                
                if direction == 'left' and seg_x < px:
                    dist = px - seg_x
                    if dist < best_dist:
                        best_dist = dist
                        # Offset: parar 2px DESPUÉS del centro (hacia el interior)
                        best_point = (seg_x + PERIMETER_HALF_THICKNESS, py)
                elif direction == 'right' and seg_x > px:
                    dist = seg_x - px
                    if dist < best_dist:
                        best_dist = dist
                        # Offset: parar 2px ANTES del centro (hacia el interior)
                        best_point = (seg_x - PERIMETER_HALF_THICKNESS, py)
            
            else:  # 'up' or 'down'
                # Buscamos segmentos horizontales del perímetro
                if not seg_is_horiz:
                    continue
                seg_y = (sy1 + sy2) / 2
                seg_x_min = min(sx1, sx2)
                seg_x_max = max(sx1, sx2)
                
                # El punto debe estar en el rango X del segmento
                if not (seg_x_min - 5 <= px <= seg_x_max + 5):
                    continue
                
                if direction == 'up' and seg_y < py:
                    dist = py - seg_y
                    if dist < best_dist:
                        best_dist = dist
                        # Offset: parar 2px DESPUÉS del centro (hacia el interior)
                        best_point = (px, seg_y + PERIMETER_HALF_THICKNESS)
                elif direction == 'down' and seg_y > py:
                    dist = seg_y - py
                    if dist < best_dist:
                        best_dist = dist
                        # Offset: parar 2px ANTES del centro (hacia el interior)
                        best_point = (px, seg_y - PERIMETER_HALF_THICKNESS)
        
        return best_point
    
    extended_walls = []
    
    for wall in pwalls:
        x1, y1, x2, y2, tag = wall
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
        
        # Determinar si la pared es H o V
        is_horiz = abs(y1 - y2) < abs(x1 - x2)
        
        # Calcular distancia de cada extremo al perímetro
        dist1 = min_distance_to_perimeter(x1, y1)
        dist2 = min_distance_to_perimeter(x2, y2)
        
        if is_horiz:  # Pared horizontal
            # Extremo izquierdo (x más pequeño)
            left_x = min(x1, x2)
            right_x = max(x1, x2)
            
            if 0 < dist1 < threshold and x1 == left_x:
                new_point = find_intersection_with_perimeter(x1, y1, 'left')
                if new_point:
                    new_x1 = new_point[0]
            elif 0 < dist1 < threshold and x1 == right_x:
                new_point = find_intersection_with_perimeter(x1, y1, 'right')
                if new_point:
                    new_x1 = new_point[0]
            
            if 0 < dist2 < threshold and x2 == left_x:
                new_point = find_intersection_with_perimeter(x2, y2, 'left')
                if new_point:
                    new_x2 = new_point[0]
            elif 0 < dist2 < threshold and x2 == right_x:
                new_point = find_intersection_with_perimeter(x2, y2, 'right')
                if new_point:
                    new_x2 = new_point[0]
        
        else:  # Pared vertical
            # Extremo superior (y más pequeño)
            top_y = min(y1, y2)
            bottom_y = max(y1, y2)
            
            if 0 < dist1 < threshold and y1 == top_y:
                new_point = find_intersection_with_perimeter(x1, y1, 'up')
                if new_point:
                    new_y1 = new_point[1]
            elif 0 < dist1 < threshold and y1 == bottom_y:
                new_point = find_intersection_with_perimeter(x1, y1, 'down')
                if new_point:
                    new_y1 = new_point[1]
            
            if 0 < dist2 < threshold and y2 == top_y:
                new_point = find_intersection_with_perimeter(x2, y2, 'up')
                if new_point:
                    new_y2 = new_point[1]
            elif 0 < dist2 < threshold and y2 == bottom_y:
                new_point = find_intersection_with_perimeter(x2, y2, 'down')
                if new_point:
                    new_y2 = new_point[1]
        
        extended_walls.append((new_x1, new_y1, new_x2, new_y2, tag))
    
    return extended_walls