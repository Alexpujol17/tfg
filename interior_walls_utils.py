
import numpy as np
from skimage.morphology import skeletonize, binary_dilation, square
from skimage.measure import label, regionprops

# En window_placement_utils.py
from Analysis_StyleGAN_functions_03_pindatafinal24_Relaxed import getPointsLineBetweenXY
from interior_doors_placement_utils import build_adjacency_graph, place_interior_doors_mst

def build_interior_walls_from_borders(gParam, rgbImageNilColors, lConnexComponents,
                                      imageContourSmoothed=None,
                                      lPerimeter=None,
                                      min_segment_len=12, thickness_px=2):
    """
    Genera muros interiores como fronteras entre salas (no fondo, no exterior, no puerta),
    los convierte en CC [(x,y), ...] y los añade a lConnexComponents[idxIntWall][2].
    """
    height = gParam['height']; width = gParam['width']
    listNilColors = gParam['listNilColors']
    idxBG  = gParam['idxNilColorBackground']
    idxEW  = gParam['idxNilColorExteriorWall']
    idxFD  = gParam['idxNilColorFrontDoor']
    idxIW  = gParam['idxNilColorInteriorWall']
    idxID  = gParam['idxNilColorInsideDoor']

    # 1) Mapa de índices por píxel (width x height) con el convenio (x,y) del repo
    idx_map = np.full((height, width), idxBG, dtype=np.int32)
    lc2idx = {tuple(listNilColors[i]): i for i in range(len(listNilColors))}
    # rgbImageNilColors es LIST de length width*height con tuplas RGB. Convertimos:
    for x in range(height):      # lee de arriba a abajo
        base = x * width
        for y in range(width):   # de izq a dcha
            idx_map[x, y] = lc2idx[rgbImageNilColors[base + y]]

    # 2) Candidatos a muro: frontera entre colores (4-conexión) excluyendo BG/EW/FD
    mask_room = (idx_map != idxBG) & (idx_map != idxEW) & (idx_map != idxFD)
    # Vecinos con padding “seguro”
    #up    = np.zeros_like(idx_map); up[1:,:]   = idx_map[:-1,:]
    down  = np.zeros_like(idx_map); down[:-1,:]= idx_map[1:,:]
    #left  = np.zeros_like(idx_map); left[:,1:] = idx_map[:,:-1]
    right = np.zeros_like(idx_map); right[:,:-1]= idx_map[:,1:]

    # Frontera: difiere con un vecino y ambos son “room”
    #b_up    = mask_room & (idx_map != up)    & (up    != idxBG) & (up  != idxEW) & (up  != idxFD)
    b_down  = mask_room & (idx_map != down)  & (down  != idxBG) & (down!= idxEW) & (down!= idxFD) & (down != idxIW) & (down != idxID)
    #b_left  = mask_room & (idx_map != left)  & (left  != idxBG) & (left!= idxEW) & (left!= idxFD)
    b_right = mask_room & (idx_map != right) & (right != idxBG) & (right!= idxEW) & (right!= idxFD) & (right != idxIW) & (right != idxID)

    border_mask = b_down | b_right 

    # 3) Quita contorno exterior si te lo pasan (0=contorno, 1=resto en imageContourSmoothed)
    if imageContourSmoothed is not None:
        border_mask = border_mask & (imageContourSmoothed.astype(bool))  # evita pisar el exterior

    from skimage.morphology import closing, square
    border_mask = closing(border_mask, square(3))
    # 4) Esqueleto fino (1 px)
    if np.any(border_mask):
        skel = skeletonize(border_mask.astype(bool))
    else:
        skel = border_mask

    # 5) Filtra segmentos muy pequeños y opcionalmente engrosa
    #    Etiqueta CC sobre el esqueleto y queda con los suficientemente largos
    lbl = label(skel, connectivity=2)
    keep = np.zeros_like(skel, dtype=bool)
    for r in regionprops(lbl):
        if r.area >= min_segment_len:
            keep[lbl == r.label] = True

    wall_mask = keep
    if thickness_px and thickness_px > 1:
        wall_mask = binary_dilation(keep, square(thickness_px))

    # 6) Convierte wall_mask a lista de CC en formato [(x,y), ...] y añade a lConnexComponents[IntWall][2]
    lbl_w = label(wall_mask, connectivity=2)
    cc_list = []
    for r in regionprops(lbl_w):
        coords = r.coords  # array Nx2 con [x,y] ya en el mismo convenio
        # Convertimos a lista de tuplas (x,y)
        cc_pixels = [(int(x), int(y)) for (x, y) in coords]
        cc_list.append(cc_pixels)

    

    # Asegura que lConnexComponents tiene entrada IntWall con la tupla (idxIW, rgb, list_cc)
    # Si ya existe, extendemos; si no, la construimos.
    if len(lConnexComponents) > idxIW and isinstance(lConnexComponents[idxIW], tuple):
        # Tupla (idx_color, rgb_color, lista_de_componentes)
        lConnexComponents[idxIW][2].extend(cc_list)
    else:
        rgbIW = listNilColors[idxIW]
        lConnexComponents.insert(idxIW, (idxIW, rgbIW, cc_list))

    # 7) Convierte cada CC de píxeles a segmentos de muro (x1,y1,x2,y2,'IW')
    pwalls = []
    for i, cc in enumerate(cc_list):
        pwalls.extend(pixels_to_walls(cc, padding=5))
    #extender paredes necesarias

    if lPerimeter is not None:
        print("lPerimeter:")
        for perim_wall in lPerimeter:
            print("\n", perim_wall)
        print("Interior walls before fragmentation:")
        for wall in pwalls:
            print("\n", wall)
    
    pwalls_final = fragment_walls_at_intersections(pwalls)

    for wall in pwalls_final:
        print(" Fragmented: ", wall)

    # En tu función principal:
    G, start_node = build_adjacency_graph(gParam, lConnexComponents, pwalls_final, lPerimeter)

    pwalls_final= place_interior_doors_mst(gParam, pwalls_final, G, start_node)

    print(f"🏠 Habitación de Entrada (Start Node): {start_node}")
    print("🔗 Conexiones detectadas:")
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
    if not pixels:
        return []

    # A. Determinar el tamaño del lienzo
    # Asumimos formato (Y, X) o (Row, Col) basado en tu data
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    # Creamos un lienzo negro lo suficientemente grande
    height = max_r - min_r + (padding * 2)
    width = max_c - min_c + (padding * 2)
    img = np.zeros((height, width), dtype=np.uint8)

    # B. Pintar los pixeles en el lienzo (offset para que quepan)
    for r, c in pixels:
        img[r - min_r + padding, c - min_c + padding] = 255

    # C. Adelgazamiento (Skeletonization)
    # Convertimos a formato booleano para scikit-image, luego vuelta a uint8
    skeleton = skeletonize(img // 255).astype(np.uint8) * 255
    
    # D. Detección de líneas con HoughLinesP
    # rho=1, theta=np.pi/180 (1 grado), threshold=baja tolerancia, minLineLength=longitud minima
    lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180, threshold=10, minLineLength=0, maxLineGap=10)
    
    wall_segments = []
    
    if lines is not None:
        for line in lines:
            x1_local, y1_local, x2_local, y2_local = line[0]
            
            # E. Convertir coordenadas locales de vuelta a las globales
            # Nota: Hough devuelve (x, y), pero nosotros ajustamos (col, row)
            # x es columna, y es fila en imagen
            
            real_x1 = x1_local - padding + min_c
            real_y1 = y1_local - padding + min_r
            real_x2 = x2_local - padding + min_c
            real_y2 = y2_local - padding + min_r
            
            # Formato solicitado
            wall_segments.append((real_x1, real_y1, real_x2, real_y2, 'IW'))
            
    return wall_segments

import numpy as np
import cv2
from skimage.morphology import skeletonize

def fragment_walls_at_intersections(pwalls, tolerance=5):
    """
    Divide las paredes largas en segmentos más pequeños en cada intersección
    entre paredes aproximadamente horizontales y verticales.
    Usa 'tolerance' (en píxeles) para permitir que se corten aunque haya
    pequeños huecos o desalineaciones.
    """
    if not pwalls:
        return []

    # 0. Normalizar tipos
    walls_clean = []
    for w in pwalls:
        walls_clean.append((int(w[0]), int(w[1]), int(w[2]), int(w[3]), w[4]))

    # Puntos de corte por pared (índice -> set de coords)
    cuts = {i: set() for i in range(len(walls_clean))}

    # 1. Detectar intersecciones H–V con tolerancia
    for i in range(len(walls_clean)):
        for j in range(i + 1, len(walls_clean)):
            wall_A = walls_clean[i]
            wall_B = walls_clean[j]
            
            xA1, yA1, xA2, yA2, _ = wall_A
            xB1, yB1, xB2, yB2, _ = wall_B
            
            # Clasificación aproximada
            type_A = 'H' if abs(yA1 - yA2) < abs(xA1 - xA2) else 'V'
            type_B = 'H' if abs(yB1 - yB2) < abs(xB1 - xB2) else 'V'

            # Sólo cruces perpendiculares
            if type_A == type_B:
                continue

            # Identificar cuál es H y cuál V
            if type_A == 'H':
                H_wall, H_idx = wall_A, i
                V_wall, V_idx = wall_B, j
            else:
                H_wall, H_idx = wall_B, j
                V_wall, V_idx = wall_A, i

            xH1, yH1, xH2, yH2, _ = H_wall
            xV1, yV1, xV2, yV2, _ = V_wall

            # Punto "ideal" de cruce: (vx, hy) usando el punto medio
            hy = int(round((yH1 + yH2) / 2.0))
            vx = int(round((xV1 + xV2) / 2.0))

            hx_min, hx_max = min(xH1, xH2), max(xH1, xH2)
            vy_min, vy_max = min(yV1, yV2), max(yV1, yV2)

            # ¿Este punto (vx, hy) cae cerca de ambos segmentos?
            intersects_x = (hx_min - tolerance) <= vx <= (hx_max + tolerance)
            intersects_y = (vy_min - tolerance) <= hy <= (vy_max + tolerance)

            if not (intersects_x and intersects_y):
                continue

            # Clamp para asegurar que el corte cae DENTRO del segmento real
            cut_x = max(hx_min, min(vx, hx_max))
            cut_y = max(vy_min, min(hy, vy_max))

            # Añadir corte a la horizontal (por X)
            cuts[H_idx].add(cut_x)

            # Añadir corte a la vertical (por Y)
            cuts[V_idx].add(cut_y)

    # 2. Reconstruir paredes fragmentadas
    new_segments = []
    
    for i, wall in enumerate(walls_clean):
        x1, y1, x2, y2, tag = wall
        is_horiz = abs(y1 - y2) < abs(x1 - x2)
        
        points_to_cut = sorted(list(cuts[i]))
        
        # Si no hay cortes, dejamos la pared tal cual
        if not points_to_cut:
            new_segments.append(wall)
            continue
            
        if is_horiz:
            # Segmentos horizontales
            curr_x = min(x1, x2)
            end_x = max(x1, x2)
            # Usamos la y media para evitar pequeños desajustes
            fixed_y = int(round((y1 + y2) / 2.0))
            
            segment_points = [curr_x] + points_to_cut + [end_x]
            
            for k in range(len(segment_points) - 1):
                p_start = segment_points[k]
                p_end   = segment_points[k + 1]
                if abs(p_start - p_end) > 1:
                    new_segments.append((p_start, fixed_y, p_end, fixed_y, tag))
                
        else:
            # Segmentos verticales
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
