import numpy as np
import networkx as nx
import math
from collections import Counter

def build_adjacency_graph(gParam, lConnexComponents, pwalls, lPerimeter):
    height = gParam['height']
    width = gParam['width']
    idxBG = gParam['idxNilColorBackground']
    idxEW = gParam['idxNilColorExteriorWall']
    
    # Mapa rasterizado de IDs únicos (UID)
    room_map = np.full((height, width), -1, dtype=np.int32)
    
    # Diccionario: UID -> TIPO DE HABITACIÓN (COLOR ID)
    # Necesario para saber si el UID 5 es un baño o una cocina.
    uid_to_room_type = {} 
    
    current_uid = 0
    
    # 1. GENERAR MAPA Y METADATA
    for type_index_in_list, component in enumerate(lConnexComponents):
        real_room_type_id = component[0] # Este es el ID del color (0=Living, etc)
        
        if real_room_type_id in [idxBG, idxEW]: 
            continue
            
        if len(component) >= 3:
            list_of_ccs = component[2]
            
            for cc in list_of_ccs:
                this_room_uid = current_uid
                
                # Guardamos el tipo real
                uid_to_room_type[this_room_uid] = real_room_type_id 
                current_uid += 1
                
                for pixel in cc:
                    if len(pixel) >= 2:
                        r, c = pixel[0], pixel[1]
                        if 0 <= r < height and 0 <= c < width:
                            room_map[r, c] = this_room_uid

    # 2. INICIALIZAR GRAFO
    G = nx.Graph()
    valid_rooms = np.unique(room_map)
    valid_rooms = valid_rooms[valid_rooms != -1]
    G.add_nodes_from(valid_rooms)

    # 3. ANALIZAR PAREDES
    for w_idx, wall in enumerate(pwalls):
        x1, y1, x2, y2, tag = wall
        wall_len = math.hypot(x2 - x1, y2 - y1)
        
        if wall_len < 5: continue

        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        is_horiz = abs(x1 - x2) > abs(y1 - y2)
        
        # Radar
        neighbors_side_A = []
        neighbors_side_B = []
        scan_range = range(2, 8)
        
        for offset in scan_range:
            if is_horiz:
                pt_A = (mx, my - offset); pt_B = (mx, my + offset)
            else:
                pt_A = (mx - offset, my); pt_B = (mx + offset, my)
            
            if 0 <= pt_A[0] < width and 0 <= pt_A[1] < height:
                rid = room_map[int(pt_A[1]), int(pt_A[0])]
                if rid != -1: neighbors_side_A.append(rid)
            if 0 <= pt_B[0] < width and 0 <= pt_B[1] < height:
                rid = room_map[int(pt_B[1]), int(pt_B[0])]
                if rid != -1: neighbors_side_B.append(rid)

        room_A = -1; room_B = -1
        if neighbors_side_A: room_A = Counter(neighbors_side_A).most_common(1)[0][0]
        if neighbors_side_B: room_B = Counter(neighbors_side_B).most_common(1)[0][0]
            
        if room_A != -1 and room_B != -1 and room_A != room_B:
            # Recuperar TIPOS para calcular prioridad
            type_A = uid_to_room_type.get(room_A, -1)
            type_B = uid_to_room_type.get(room_B, -1)
            
            # CALCULAR PRIORIDAD ARQUITECTÓNICA
            priority_score = get_architectural_priority(type_A, type_B, wall_len)
            
            # Añadir al grafo (Maximizando el score)
            if G.has_edge(room_A, room_B):
                prev_score = G[room_A][room_B]['weight']
                if priority_score > prev_score:
                    G.add_edge(room_A, room_B, wall_idx=w_idx, weight=priority_score, length=wall_len)
            else:
                G.add_edge(room_A, room_B, wall_idx=w_idx, weight=priority_score, length=wall_len)

    # 4. START NODE (Detección de Entrada)
    start_room_uid = -1
    
    if lPerimeter:
        fd_walls = [w for w in lPerimeter if w[4] == 'FD']
        if fd_walls:
            fd = fd_walls[0]
            fx, fy = (fd[0] + fd[2]) // 2, (fd[1] + fd[3]) // 2
            center_x, center_y = width // 2, height // 2
            dir_x = 1 if center_x > fx else -1; dir_y = 1 if center_y > fy else -1
            
            for k in range(2, 35): 
                move_x = dir_x * k if abs(fx - center_x) > abs(fy - center_y) else 0
                move_y = dir_y * k if abs(fy - center_y) >= abs(fx - center_x) else 0
                chk_x, chk_y = int(fx + move_x), int(fy + move_y)
                
                if 0 <= chk_x < width and 0 <= chk_y < height:
                    rid = room_map[chk_y, chk_x]
                    if rid != -1: start_room_uid = rid; break

    # Fallback Start Node (Por prioridad lógica)
    if start_room_uid == -1 and uid_to_room_type:
        # 1. Entrance (9)
        for uid, rtype in uid_to_room_type.items():
            if rtype == 9: start_room_uid = uid; break
        # 2. Living (0)
        if start_room_uid == -1:
             for uid, rtype in uid_to_room_type.items():
                if rtype == 0: start_room_uid = uid; break
    
    # 3. Cualquiera válido
    if start_room_uid == -1 and len(valid_rooms) > 0:
        start_room_uid = valid_rooms[0]

    return G, start_room_uid

import networkx as nx
import math

def place_interior_doors_mst(gParam, pwalls, G, start_room_id):
    """
    Coloca el MÍNIMO número de puertas posible (N-1) para conectar todas las habitaciones.
    Usa un Maximum Spanning Tree ponderado por la longitud de la pared para
    priorizar poner puertas en las paredes más grandes y cómodas.
    """
    
    # --- CONFIGURACIÓN FÍSICA ---
    # Ajusta esto según tu escala (25px = 1m)
    DOOR_SIZE_PX = 9       # ~80cm
    CORNER_MARGIN_PX = 2    # ~15cm (marco + seguridad)
    MIN_WALL_LEN = DOOR_SIZE_PX + (CORNER_MARGIN_PX * 2)

    # 1. PRE-FILTRADO DEL GRAFO (Física)
    # Eliminamos del grafo las paredes que son físicamente demasiado cortas para una puerta.
    # Así el MST no intentará elegir una pared de 10px solo porque es la única conexión.
    for u, v, data in list(G.edges(data=True)):
        wall_idx = data['wall_idx']
        x1, y1, x2, y2, tag = pwalls[wall_idx]
        wall_len = math.hypot(x2 - x1, y2 - y1)
        
        if wall_len < MIN_WALL_LEN:
            G.remove_edge(u, v)
    # Verificación de seguridad: ¿Sigue conectado el grafo?
    # Si no, significa que hay habitaciones a las que es imposible entrar (paredes muy pequeñas)
    if not nx.is_connected(G):
        # print("⚠️ Aviso: Algunas habitaciones son inaccesibles físicamente (paredes muy cortas).")
        # Para evitar crash, calculamos el MST sobre las componentes conectadas más grandes
        pass

    # 2. CÁLCULO DEL MAXIMUM SPANNING TREE (Topología)
    # Buscamos el subgrafo que conecta TODOS los nodos con el MAYOR peso total (longitud de paredes)
    # Esto evita pasillos estrechos y prioriza paredes anchas.
    mst = nx.maximum_spanning_tree(G, weight='weight')

    # Convertimos las aristas del MST en un set de índices de pared para búsqueda rápida
    walls_with_door_indices = set()
    for u, v, data in mst.edges(data=True):
        walls_with_door_indices.add(data['wall_idx'])

    # 3. INSERTAR PUERTAS FÍSICAMENTE (Geometría)
    final_walls_list = []

    for i, wall in enumerate(pwalls):
        # Si esta pared no fue elegida por el MST, se queda como muro ciego ('IW')
        if i not in walls_with_door_indices:
            final_walls_list.append(wall)
            continue
            
        # --- CORTAR LA PARED E INSERTAR 'ID' ---
        x1, y1, x2, y2, tag = wall
        
        # Calcular centro geométrico
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_door = DOOR_SIZE_PX / 2
        
        # Determinar orientación
        if abs(x1 - x2) > abs(y1 - y2): 
            # --- HORIZONTAL (varía X) ---
            orientation = 'H'
            wx_min, wx_max = min(x1, x2), max(x1, x2)
            fixed_y = y1
            
            # Posición ideal: Centro
            d_start = cx - half_door
            d_end   = cx + half_door
            
            # Corrección de Márgenes (Offset inteligente)
            # Si choca con la izquierda...
            if d_start < (wx_min + CORNER_MARGIN_PX):
                d_start = wx_min + CORNER_MARGIN_PX
                d_end = d_start + DOOR_SIZE_PX
            
            # Si choca con la derecha...
            if d_end > (wx_max - CORNER_MARGIN_PX):
                d_end = wx_max - CORNER_MARGIN_PX
                d_start = d_end - DOOR_SIZE_PX

            # Clamping final (por si la pared es justa justa)
            d_start = max(wx_min, d_start)
            d_end = min(wx_max, d_end)

            # Generar los 3 tramos
            # 1. Muro Izquierdo
            if (d_start - wx_min) > 1:
                final_walls_list.append((int(wx_min), int(fixed_y), int(d_start), int(fixed_y), 'IW'))
            
            # 2. La Puerta
            final_walls_list.append((int(d_start), int(fixed_y), int(d_end), int(fixed_y), 'ID'))
            
            # 3. Muro Derecho
            if (wx_max - d_end) > 1:
                final_walls_list.append((int(d_end), int(fixed_y), int(wx_max), int(fixed_y), 'IW'))

        else:
            # --- VERTICAL (varía Y) ---
            orientation = 'V'
            wy_min, wy_max = min(y1, y2), max(y1, y2)
            fixed_x = x1
            
            d_start = cy - half_door
            d_end   = cy + half_door
            
            # Corrección de Márgenes
            if d_start < (wy_min + CORNER_MARGIN_PX):
                d_start = wy_min + CORNER_MARGIN_PX
                d_end = d_start + DOOR_SIZE_PX
            
            if d_end > (wy_max - CORNER_MARGIN_PX):
                d_end = wy_max - CORNER_MARGIN_PX
                d_start = d_end - DOOR_SIZE_PX
                
            d_start = max(wy_min, d_start)
            d_end = min(wy_max, d_end)

            # Generar los 3 tramos
            # 1. Muro Arriba
            if (d_start - wy_min) > 1:
                final_walls_list.append((int(fixed_x), int(wy_min), int(fixed_x), int(d_start), 'IW'))
                
            # 2. La Puerta
            final_walls_list.append((int(fixed_x), int(d_start), int(fixed_x), int(d_end), 'ID'))
            
            # 3. Muro Abajo
            if (wy_max - d_end) > 1:
                final_walls_list.append((int(fixed_x), int(d_end), int(fixed_x), int(wy_max), 'IW'))

    return final_walls_list


def get_architectural_priority(type_A, type_B, wall_length):
    """
    Calcula el PESO de una conexión. El MST buscará MAXIMIZAR este valor.
    
    Fórmula: Longitud_Pared * Multiplicador_Semántico
    
    Esto asegura que una pared pequeña que conecta el Salón con la Cocina (x15)
    tenga más peso que un ventanal enorme que conecta el Dormitorio con el Balcón (x0.001).
    """
    
    # Índices basados en tu lista (Canal 2)
    # 0:Living, 1:Master, 2:Kitchen, 3:Bath, 4:Dining, 5:Child, 
    # 6:Second, 7:Guest, 8:Balcony, 9:Entrance, 10:Storage, 11:BackGrd, 12:ExtWall
    
    multiplier = 1.0
    
    # Grupos lógicos
    bedrooms = {1, 5, 6, 7}
    service = {2, 3, 10}
    public = {0, 4, 9}
    
    # --- REGLA 0: FILTRO DE SEGURIDAD ---
    # Si alguno es fondo o pared exterior (por error), prioridad nula.
    if 11 in (type_A, type_B) or 12 in (type_A, type_B):
        return 0.0

    # --- REGLA 1: EL BALCÓN ES UN "CUL-DE-SAC" ---
    # Solo queremos conectar con el balcón si NO HAY OTRA SALIDA.
    # Prioridad infinitesimal.
    if 8 in (type_A, type_B):
        return 0.0001 * wall_length 

    # --- REGLA 2: LA ENTRADA (Entrance - 9) ---
    # La entrada debe conectar fuertemente con las zonas públicas.
    if 9 in (type_A, type_B):
        if 0 in (type_A, type_B): return wall_length * 100.0 # Entrance -> Living (Prioridad Absoluta)
        if 4 in (type_A, type_B): return wall_length * 80.0  # Entrance -> Dining
        return wall_length * 50.0 # Entrance -> Pasillo/Otros

    # --- REGLA 3: EL SALÓN (Living - 0) ES EL HUB ---
    # El salón distribuye a casi todo.
    elif 0 in (type_A, type_B):
        if type_A in bedrooms or type_B in bedrooms: multiplier = 20.0
        elif 2 in (type_A, type_B): multiplier = 15.0 # Cocina
        elif 4 in (type_A, type_B): multiplier = 15.0 # Comedor
        else: multiplier = 10.0

    # --- REGLA 4: SUITE (Master - 1 <-> Bath - 3) ---
    # Conexión deseable.
    elif (type_A == 1 and type_B == 3) or (type_A == 3 and type_B == 1):
        multiplier = 15.0 

    # --- REGLA 5: ZONA DE DÍA (Kitchen - 2 <-> Dining - 4) ---
    elif (type_A == 2 and type_B == 4) or (type_A == 4 and type_B == 2):
        multiplier = 12.0
        
    # --- REGLA 6: PENALIZACIONES (OLORES/RUIDO) ---
    # Evitar conectar Cocina directamente a Dormitorio si es posible ir por otro lado.
    elif 2 in (type_A, type_B) and (type_A in bedrooms or type_B in bedrooms):
        multiplier = 0.5 
    
    # Evitar conectar Baño a Cocina
    elif 2 in (type_A, type_B) and 3 in (type_A, type_B):
        multiplier = 0.2

    return wall_length * multiplier

import networkx as nx
import math
