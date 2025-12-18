import numpy as np
import networkx as nx
import math
from collections import Counter

def solicitar_config_puertas_usuario():
    """Pregunta al usuario por la configuración de puertas."""
    print("\n=== CONFIGURACIÓN DE PUERTAS ===")
    print("¿Dónde prefieres alinear las puertas en la pared?")
    print("   1: Izquierda (Inicio de pared)")
    print("   2: Centro")
    print("   3: Derecha (Final de pared)")
    
    try:
        opcion = input("   Opción [1]: ") or "1"
    except KeyboardInterrupt:
        opcion = "1"
        
    mapping = {'1': 'LEFT', '2': 'CENTER', '3': 'RIGHT'}
    posicion = mapping.get(opcion, 'LEFT')
    
    print(f"   ✅ Alinear puertas: {posicion}")
    return {'posicion_puertas': posicion}

def build_adjacency_graph(gParam, lConnexComponents, pwalls, lPerimeter):
    """
    Construye el 'Mapa de Relaciones' (Grafo) de la casa.
    Analiza qué habitaciones son vecinas y comparten pared.
    """
    height = gParam['height']
    width = gParam['width']
    idxBG = gParam['idxNilColorBackground']
    idxEW = gParam['idxNilColorExteriorWall']
    
    # Creamos un mapa vacío (una cuadrícula de píxeles) para pintar las habitaciones
    room_map = np.full((height, width), -1, dtype=np.int32)
    
    # Diccionario para recordar qué ID numérico corresponde a qué tipo de habitación
    # Ej: El ID 5 es una Cocina.
    uid_to_room_type = {} 
    
    current_uid = 0
    
    # ==========================================
    # 1. PINTAR EL MAPA (RASTERIZACIÓN)
    # ==========================================
    # Convertimos la lista de formas geométricas en una imagen de píxeles.
    # Así sabremos que el píxel (10, 20) pertenece al Salón.
    for type_index_in_list, component in enumerate(lConnexComponents):
        real_room_type_id = component[0] # Tipo real (0=Salón, 2=Cocina, etc.)
        
        # Ignoramos el fondo negro y los muros exteriores
        if real_room_type_id in [idxBG, idxEW]: 
            continue
            
        if len(component) >= 3:
            list_of_ccs = component[2]
            
            for cc in list_of_ccs:
                this_room_uid = current_uid
                
                # Guardamos qué es esta habitación
                uid_to_room_type[this_room_uid] = real_room_type_id 
                current_uid += 1
                
                # Rellenamos los píxeles en el mapa
                for pixel in cc:
                    if len(pixel) >= 2:
                        r, c = pixel[0], pixel[1]
                        if 0 <= r < height and 0 <= c < width:
                            room_map[r, c] = this_room_uid

    # ==========================================
    # 2. CREAR LOS PUNTOS DEL GRAFO (NODOS)
    # ==========================================
    G = nx.Graph()  # Mantenemos grafo no dirigido, pero guardamos direccion en edges
    valid_rooms = np.unique(room_map)
    # Quitamos el -1 (que es espacio vacío)
    valid_rooms = valid_rooms[valid_rooms != -1]
    G.add_nodes_from(valid_rooms)

    # Calcular área de cada habitación (en píxeles)
    room_areas = {}
    for uid in valid_rooms:
        room_areas[uid] = np.sum(room_map == uid)
    
    for uid in valid_rooms:
        r_type = uid_to_room_type.get(uid, -1)
        G.nodes[uid]['room_type'] = r_type
        G.nodes[uid]['room_area'] = room_areas.get(uid, 0)

    # ==========================================
    # 3. ANALIZAR LAS PAREDES (ENLACES)
    # ==========================================
    for w_idx, wall in enumerate(pwalls):
        x1, y1, x2, y2, tag = wall
        wall_len = math.hypot(x2 - x1, y2 - y1)
        
        # Si la pared mide menos de 5 píxeles, es ruido, la ignoramos.
        if wall_len < 5: continue

        # Calculamos el punto medio de la pared
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        is_horiz = abs(x1 - x2) > abs(y1 - y2)
        
        # --- EL RADAR DE VECINOS ---
        # Desde el centro de la pared, miramos a izquierda y derecha (o arriba y abajo)
        # para ver qué habitaciones están tocando esta pared.
        neighbors_side_A = []
        neighbors_side_B = []
        scan_range = range(2, 8) # Miramos entre 2 y 8 píxeles de distancia
        
        for offset in scan_range:
            if is_horiz:
                # Si pared horizontal, miramos arriba (A) y abajo (B)
                pt_A = (mx, my - offset); pt_B = (mx, my + offset)
            else:
                # Si pared vertical, miramos izquierda (A) y derecha (B)
                pt_A = (mx - offset, my); pt_B = (mx + offset, my)
            
            # Verificamos qué habitación hay en esos puntos
            if 0 <= pt_A[0] < width and 0 <= pt_A[1] < height:
                rid = room_map[int(pt_A[1]), int(pt_A[0])]
                if rid != -1: neighbors_side_A.append(rid)
            if 0 <= pt_B[0] < width and 0 <= pt_B[1] < height:
                rid = room_map[int(pt_B[1]), int(pt_B[0])]
                if rid != -1: neighbors_side_B.append(rid)

        # Decidimos cuál es la habitación predominante a cada lado
        room_A = -1; room_B = -1
        if neighbors_side_A: room_A = Counter(neighbors_side_A).most_common(1)[0][0]
        if neighbors_side_B: room_B = Counter(neighbors_side_B).most_common(1)[0][0]
            
        # Si tenemos dos habitaciones distintas compartiendo pared...
        if room_A != -1 and room_B != -1 and room_A != room_B:
            # Recuperamos qué TIPO son (ej: Cocina y Salón)
            type_A = uid_to_room_type.get(room_A, -1)
            type_B = uid_to_room_type.get(room_B, -1)
            
            # --- DETERMINAR DIRECCIÓN DE APERTURA ---
            # La puerta abre HACIA (swing_into) la habitación destino
            # Reglas de prioridad:
            #   1. Habitaciones privadas (dormitorios, baños) - abre hacia ellas
            #   2. Si ninguna es privada o ambas lo son - abre hacia la más grande
            private_rooms = {1, 3, 5, 6, 7}  # 1:Principal, 3:Baño, 5:Niño, 6:Segunda, 7:Invitados
            
            A_is_private = type_A in private_rooms
            B_is_private = type_B in private_rooms
            
            area_A = room_areas.get(room_A, 0)
            area_B = room_areas.get(room_B, 0)
            
            if A_is_private and not B_is_private:
                swing_into = room_A  # Abre hacia habitación privada A
            elif B_is_private and not A_is_private:
                swing_into = room_B  # Abre hacia habitación privada B
            else:
                # Ambas privadas o ninguna: hacia la más grande
                swing_into = room_A if area_A >= area_B else room_B
            
            # --- PUNTUACIÓN DE LA CONEXIÓN ---
            # Calculamos cuán buena idea es poner una puerta aquí.
            priority_score = get_architectural_priority(type_A, type_B, wall_len)
            
            # Añadimos la conexión al Grafo con info de dirección.
            # Si ya existía conexión (otra pared entre las mismas habitaciones),
            # nos quedamos solo con la que tenga mejor puntuación.
            edge_data = {
                'wall_idx': w_idx, 
                'weight': priority_score, 
                'length': wall_len,
                'swing_into': swing_into,  # NUEVO: hacia qué habitación abre
                'room_A': room_A,  # NUEVO: referencia para saber lados
                'room_B': room_B
            }
            
            if G.has_edge(room_A, room_B):
                prev_score = G[room_A][room_B]['weight']
                if priority_score > prev_score:
                    G.add_edge(room_A, room_B, **edge_data)
            else:
                G.add_edge(room_A, room_B, **edge_data)

    # ==========================================
    # 4. BUSCAR LA ENTRADA PRINCIPAL (Start Node)
    # ==========================================
    start_room_uid = -1
    
    # Intentamos encontrar la puerta de entrada ('FD') en el perímetro
    if lPerimeter:
        fd_walls = [w for w in lPerimeter if w[4] == 'FD']
        if fd_walls:
            fd = fd_walls[0]
            # Lanzamos un rayo desde la puerta hacia el centro de la casa
            # para ver qué habitación es la primera que toca (el Recibidor).
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

    # Si falla el rayo, usamos lógica por tipo de habitación
    if start_room_uid == -1 and uid_to_room_type:
        # 1. Buscamos si hay algo etiquetado como "Entrance" (9)
        for uid, rtype in uid_to_room_type.items():
            if rtype == 9: start_room_uid = uid; break
        # 2. Si no, empezamos por el Salón (0)
        if start_room_uid == -1:
             for uid, rtype in uid_to_room_type.items():
                if rtype == 0: start_room_uid = uid; break
    
    # 3. Si todo falla, cogemos la primera habitación que haya
    if start_room_uid == -1 and len(valid_rooms) > 0:
        start_room_uid = valid_rooms[0]

    return G, start_room_uid

def place_interior_doors_mst(gParam, pwalls, G, start_room_id, config=None):
    """
    ALGORITMO MAESTRO DE PUERTAS INTERIORES:
    Usa el Grafo de Adyacencias para decidir dónde poner puertas interiores.
    - Balcones: Puerta grande y CENTRADA.
    - Resto: Puerta estándar y posición según CONFIGURACIÓN (Left/Center/Right).
    """
    
    # --- CONFIGURACIÓN ---
    STANDARD_DOOR_SIZE = 9
    BALCONY_DOOR_SIZE = 25
    CORNER_MARGIN_PX = 2    
    MIN_WALL_LEN = STANDARD_DOOR_SIZE + (CORNER_MARGIN_PX * 2)

    # Preferencia de posición del usuario (Default: LEFT)
    user_pos = config.get('posicion_puertas', 'LEFT') if config else 'LEFT'

    # 1. LIMPIEZA DEL GRAFO 
    for u, v, data in list(G.edges(data=True)):
        wall_idx = data['wall_idx']
        x1, y1, x2, y2, tag = pwalls[wall_idx]
        wall_len = math.hypot(x2 - x1, y2 - y1)
        if wall_len < MIN_WALL_LEN:
            G.remove_edge(u, v)

    if not nx.is_connected(G):
        pass # Manejo de grafo desconectado (opcional)

    # 2. MST (Esqueleto)
    mst = nx.maximum_spanning_tree(G, weight='weight')

    # Diccionario para mapear pared -> (habitaciones, info de apertura)
    walls_to_rooms_map = {} 
    for u, v, data in mst.edges(data=True):
        w_idx = data['wall_idx']
        swing_into = data.get('swing_into', u)  # Default: abre hacia u
        room_A = data.get('room_A', u)
        room_B = data.get('room_B', v)
        walls_to_rooms_map[w_idx] = {
            'room_u': u, 
            'room_v': v,
            'swing_into': swing_into,
            'room_A': room_A,
            'room_B': room_B
        } 

    # 3. CONSTRUCCIÓN GEOMÉTRICA
    final_walls_list = []

    for i, wall in enumerate(pwalls):
        # Si la pared no es elegida, es muro ciego
        if i not in walls_to_rooms_map:
            final_walls_list.append(wall)
            continue
            
        # --- PREPARACIÓN DE LA PUERTA ---
        edge_info = walls_to_rooms_map[i]
        room_u = edge_info['room_u']
        room_v = edge_info['room_v']
        swing_into = edge_info['swing_into']
        room_A = edge_info['room_A']
        room_B = edge_info['room_B']
        
        type_u = G.nodes[room_u].get('room_type', -1)
        type_v = G.nodes[room_v].get('room_type', -1)
        
        # Detectamos si es una conexión de balcón (puerta corredera, sin arco)
        is_balcony_connection = (type_u == 8 or type_v == 8)
        
        x1, y1, x2, y2, tag = wall
        w_len_pixel = math.hypot(x2-x1, y2-y1)
        is_horiz = abs(x1 - x2) > abs(y1 - y2)
        
        # --- DETERMINAR TAG DE PUERTA ---
        # Para paredes horizontales: lado A está ARRIBA (y menor), lado B está ABAJO (y mayor)
        # Para paredes verticales: lado A está IZQUIERDA (x menor), lado B está DERECHA (x mayor)
        # El tag indica hacia qué LADO abre la puerta (donde está el arco de 90°)
        if is_balcony_connection:
            door_tag = 'ID'  # Balcón: sin dirección (corredera)
        else:
            # Determinamos si swing_into corresponde a room_A o room_B
            # room_A está en side A (arriba/izquierda), room_B en side B (abajo/derecha)
            if swing_into == room_A:
                door_tag = 'ID_A'  # Abre hacia lado A (arriba/izquierda)
            else:
                door_tag = 'ID_B'  # Abre hacia lado B (abajo/derecha)

        # A) DEFINIR TAMAÑO Y ESTRATEGIA
        if is_balcony_connection:
            # Lógica BALCÓN: Grande y Centrada (Siempre)
            target_width = BALCONY_DOOR_SIZE
            placement_mode = 'CENTER'
        else:
            # Lógica NORMAL: Estándar y POSICIÓN CONFIGURABLE
            target_width = STANDARD_DOOR_SIZE
            placement_mode = user_pos # 'LEFT', 'CENTER', o 'RIGHT'

        # Seguridad: Si la pared es muy pequeña, forzamos tamaño estándar
        # para evitar errores geométricos, aunque sea un balcón.
        if w_len_pixel < (target_width + 4):
            target_width = STANDARD_DOOR_SIZE
            # Si no cabe la grande, quizás convenga centrar la pequeña
            # pero mantendremos la lógica pedida o fallback a centro si es muy chica.

        # B) CÁLCULO DE COORDENADAS
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if abs(x1 - x2) > abs(y1 - y2): 
            # --- HORIZONTAL ---
            wx_min, wx_max = min(x1, x2), max(x1, x2)
            fixed_y = y1
            
            # --- AQUÍ APLICAMOS LA LÓGICA DE POSICIÓN ---
            if placement_mode == 'CENTER':
                d_start = cx - (target_width / 2)
            elif placement_mode == 'RIGHT':
                d_start = wx_max - target_width - CORNER_MARGIN_PX
            else: # 'LEFT'
                d_start = wx_min + CORNER_MARGIN_PX
            
            d_end = d_start + target_width
            
            # --- CORRECCIONES DE LÍMITES (CLAMPING) ---
            # Si al esquinar nos salimos (o si centramos mal), ajustamos:
            
            # 1. No salirse por la izquierda
            if d_start < (wx_min + CORNER_MARGIN_PX):
                d_start = wx_min + CORNER_MARGIN_PX
                d_end = d_start + target_width
            
            # 2. No salirse por la derecha
            if d_end > (wx_max - CORNER_MARGIN_PX):
                d_end = wx_max - CORNER_MARGIN_PX
                d_start = d_end - target_width
                # Si tras empujar hacia atrás nos salimos por la izq, la pared es demasiado corta.
                # Aseguramos el mínimo:
                d_start = max(wx_min, d_start)

            # --- GENERAR SEGMENTOS ---
            # Trozo Izquierdo
            if (d_start - wx_min) > 1:
                final_walls_list.append((int(wx_min), int(fixed_y), int(d_start), int(fixed_y), 'IW'))
            
            # La Puerta (con dirección de apertura)
            final_walls_list.append((int(d_start), int(fixed_y), int(d_end), int(fixed_y), door_tag))
            
            # Trozo Derecho
            if (wx_max - d_end) > 1:
                final_walls_list.append((int(d_end), int(fixed_y), int(wx_max), int(fixed_y), 'IW'))

        else:
            # --- VERTICAL ---
            wy_min, wy_max = min(y1, y2), max(y1, y2)
            fixed_x = x1
            
            # --- AQUÍ APLICAMOS LA LÓGICA DE POSICIÓN ---
            if placement_mode == 'CENTER':
                d_start = cy - (target_width / 2)
            elif placement_mode == 'RIGHT':
                d_start = wy_max - target_width - CORNER_MARGIN_PX
            else: # 'LEFT' ('TOP' en vertical)
                d_start = wy_min + CORNER_MARGIN_PX
                
            d_end = d_start + target_width
            
            # --- CORRECCIONES DE LÍMITES ---
            if d_start < (wy_min + CORNER_MARGIN_PX):
                d_start = wy_min + CORNER_MARGIN_PX
                d_end = d_start + target_width
            
            if d_end > (wy_max - CORNER_MARGIN_PX):
                d_end = wy_max - CORNER_MARGIN_PX
                d_start = d_end - target_width
                d_start = max(wy_min, d_start)

            # --- GENERAR SEGMENTOS ---
            if (d_start - wy_min) > 1:
                final_walls_list.append((int(fixed_x), int(wy_min), int(fixed_x), int(d_start), 'IW'))
                
            final_walls_list.append((int(fixed_x), int(d_start), int(fixed_x), int(d_end), door_tag))
            
            if (wy_max - d_end) > 1:
                final_walls_list.append((int(fixed_x), int(d_end), int(fixed_x), int(wy_max), 'IW'))

    return final_walls_list

def get_architectural_priority(type_A, type_B, wall_length):
    """
    EL CEREBRO DE DISEÑO:
    Calcula una puntuación de "Deseabilidad" para una conexión.
    Cuanto más alto el número, más ganas tiene el algoritmo de poner una puerta ahí.
    
    Fórmula: Longitud de la pared * Factor de conveniencia.
    """
    
    # Mapa de IDs para referencia:
    # 0:Salón, 1:Principal, 2:Cocina, 3:Baño, 4:Comedor, 5:Niño, 
    # 6:Segunda, 7:Invitados, 8:Balcón, 9:Entrada, 10:Trastero...
    
    multiplier = 1.0
    
    # Agrupamos tipos de habitación por lógica
    bedrooms = {1, 5, 6, 7}
    service = {2, 3, 10}
    public = {0, 4, 9}
    
    # --- REGLA 0: SEGURIDAD ---
    # Nunca poner puertas hacia el vacío o paredes exteriores incorrectas.
    if 11 in (type_A, type_B) or 12 in (type_A, type_B):
        return 0.0

    # --- REGLA 1: EL BALCÓN (Callejón sin salida) ---
    # Solo conectamos con el balcón si no hay otra opción. Prioridad muy baja.
    if 8 in (type_A, type_B):
        return 0.0001 * wall_length 

    # --- REGLA 2: LA ENTRADA (9) ---
    # La entrada DEBE conectar con el Salón o zonas comunes. Prioridad máxima.
    if 9 in (type_A, type_B):
        if 0 in (type_A, type_B): return wall_length * 100.0 # Entrada -> Salón (¡Obligatorio si se puede!)
        if 4 in (type_A, type_B): return wall_length * 80.0  # Entrada -> Comedor
        return wall_length * 50.0 # Entrada -> Pasillo u otros

    # --- REGLA 3: EL SALÓN (0) ES EL REY ---
    # El salón es el distribuidor central de la casa.
    elif 0 in (type_A, type_B):
        if type_A in bedrooms or type_B in bedrooms: multiplier = 20.0 # Salón -> Habitaciones
        elif 2 in (type_A, type_B): multiplier = 15.0 # Salón -> Cocina
        elif 4 in (type_A, type_B): multiplier = 15.0 # Salón -> Comedor
        else: multiplier = 10.0

    # --- REGLA 4: SUITE PRIVADA (Habitación Principal -> Baño) ---
    # Es muy deseable tener el baño conectado a la habitación principal.
    elif (type_A == 1 and type_B == 3) or (type_A == 3 and type_B == 1):
        multiplier = 15.0 

    # --- REGLA 5: ZONA DE DÍA (Cocina -> Comedor) ---
    # Para llevar la comida fácilmente.
    elif (type_A == 2 and type_B == 4) or (type_A == 4 and type_B == 2):
        multiplier = 12.0
        
    # --- REGLA 6: PENALIZACIONES (Olores y Privacidad) ---
    # ¿Cocina directa a Dormitorio? Mala idea por olores y ruidos. Bajamos prioridad.
    elif 2 in (type_A, type_B) and (type_A in bedrooms or type_B in bedrooms):
        multiplier = 0.5 
    
    # ¿Baño directo a Cocina? Antihigiénico. Evitar si es posible.
    elif 2 in (type_A, type_B) and 3 in (type_A, type_B):
        multiplier = 0.2

    return wall_length * multiplier

