# En window_placement_utils.py
from random import random
from Analysis_StyleGAN_functions_03_pindatafinal24_Relaxed import getPointsLineBetweenXY

# ============================================================
# CONFIGURACIÓN POR DEFECTO - El usuario puede personalizar
# ============================================================
DEFAULT_CONFIG = {
    # --- Orientación Solar ---
    'orientacion_norte': 'TOP',  # TOP, BOTTOM, LEFT, RIGHT
    'pesos_orientacion': {
        'SUR': 1.0,    # Luz directa todo el día
        'ESTE': 0.7,   # Sol de mañana
        'OESTE': 0.6,  # Sol de tarde
        'NORTE': 0.3   # Luz indirecta
    },
    
    # --- Pesos de Evaluación ---
    'score_weights': {
        'cobertura': 0.3,           # % habitaciones con ventana
        'orientacion_solar': 0.3,   # Bonus por orientación óptima
        'distribucion': 0.2,        # Equilibrio entre fachadas
        'simetria': 0.1,            # Simetría visual
        'tamaño_proporcional': 0.1  # Ventana proporcional al área
    },
    
    # --- Probabilidades de Operadores ---
    # Op1=Mover, Op2=Intercambiar, Op3=Añadir, Op4=Eliminar
    'probabilidades_operadores': {
        'Op1': 0.25,  # Mover
        'Op2': 0.25,  # Intercambiar
        'Op3': 0.35,  # Añadir
        'Op4': 0.15   # Eliminar
    },
    
    # --- Tamaño de Ventana ---
    'tamaño_ventana_m': 1.2,      # Metros
    'margen_esquina_m': 0.15,     # Distancia mínima a esquinas
    'tamaño_variable': False,     # Si True, usa tamaños por tipo
    'tamaños_por_tipo': {
        'BEDROOM': 1.2,
        'LIVING': 1.5,
        'KITCHEN': 1.0,
        'BATHROOM': 0.6
    },
    
    # --- Penalizaciones ---
    'penalizacion_balcon': 0.15,
    
    # --- Restricciones ---
    'max_ventanas_por_hab': 2,
    'habitaciones_obligatorias': []  # Lista vacía = ninguna obligatoria
}

def get_config(user_config=None):
    """Combina config del usuario con defaults."""
    if user_config is None:
        return DEFAULT_CONFIG.copy()
    
    config = DEFAULT_CONFIG.copy()
    for key, value in user_config.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def solicitar_config_usuario():
    """
    Función interactiva para que el usuario configure los parámetros
    de evaluación de ventanas desde la terminal.
    """
    print("\n" + "="*60)
    print("   CONFIGURACIÓN DE EVALUACIÓN DE VENTANAS")
    print("="*60)
    
    config = {}
    
    # 1. Orientación Norte
    print("\n📍 ORIENTACIÓN DEL PLANO")
    print("   ¿En qué lado del plano está el NORTE?")
    print("   Opciones: TOP (arriba), BOTTOM (abajo), LEFT (izq), RIGHT (der)")
    norte = input("   Norte [TOP]: ").strip().upper() or "TOP"
    if norte not in ['TOP', 'BOTTOM', 'LEFT', 'RIGHT']:
        print(f"   ⚠️ Valor inválido, usando TOP")
        norte = 'TOP'
    config['orientacion_norte'] = norte
    
    # 2. Pesos de evaluación
    print("\n⚖️  PESOS DE EVALUACIÓN (0-10, se normalizarán)")
    try:
        cobertura = int(input("   Cobertura (% habitaciones con ventana) [3]: ") or 3)
        solar = int(input("   Orientación Solar (bonus sol sur) [3]: ") or 3)
        distribucion = int(input("   Distribución (ventanas en varias fachadas) [2]: ") or 2)
        simetria = int(input("   Simetría (visual) [1]: ") or 1)
    except ValueError:
        print("   ⚠️ Valores inválidos, usando defaults")
        cobertura, solar, distribucion, simetria = 3, 3, 2, 1
    
    # Normalizar para que sumen 1.0
    total = cobertura + solar + distribucion + simetria
    if total > 0:
        config['score_weights'] = {
            'cobertura': cobertura / total,
            'orientacion_solar': solar / total,
            'distribucion': distribucion / total,
            'simetria': simetria / total
        }
    
    # 3. Pesos de orientación solar
    print("\n☀️  PESOS POR ORIENTACIÓN SOLAR (0-10)")
    try:
        peso_sur = int(input("   Sur (máximo sol) [10]: ") or 10) / 10
        peso_este = int(input("   Este (sol mañana) [7]: ") or 7) / 10
        peso_oeste = int(input("   Oeste (sol tarde) [6]: ") or 6) / 10
        peso_norte = int(input("   Norte (sin sol) [3]: ") or 3) / 10
    except ValueError:
        peso_sur, peso_este, peso_oeste, peso_norte = 1.0, 0.7, 0.6, 0.3
    
    config['pesos_orientacion'] = {
        'SUR': peso_sur,
        'ESTE': peso_este,
        'OESTE': peso_oeste,
        'NORTE': peso_norte
    }
    
    # 4. Tamaño de ventana
    print("\n📐 TAMAÑO DE VENTANA")
    try:
        tam = float(input("   Ancho de ventana en metros [1.2]: ") or 1.2)
    except ValueError:
        tam = 1.2
    config['tamaño_ventana_m'] = tam
    
    # 5. Probabilidades de Operadores
    print("\n🎲 PROBABILIDADES DE OPERADORES (0-100, se normalizarán)")
    print("   Definen con qué frecuencia se intenta cada acción.")
    try:
        p_mover = int(input("   Mover (ajuste fino) [25]: ") or 25)
        p_intercambiar = int(input("   Intercambiar (pared a pared) [25]: ") or 25)
        p_anadir = int(input("   Añadir (nueva ventana) [35]: ") or 35)
        p_eliminar = int(input("   Eliminar (quitar sobrantes) [15]: ") or 15)
    except ValueError:
        p_mover, p_intercambiar, p_anadir, p_eliminar = 25, 25, 35, 15
        
    total_probs = p_mover + p_intercambiar + p_anadir + p_eliminar
    if total_probs > 0:
        config['probabilidades_operadores'] = {
            'Op1': p_mover / total_probs,       # Mover
            'Op2': p_intercambiar / total_probs, # Intercambiar
            'Op3': p_anadir / total_probs,       # Añadir
            'Op4': p_eliminar / total_probs      # Eliminar
        }
    
    config['usar_probabilidades_usuario'] = True  # Flag para forzar uso
    
    # Resumen
    print("\n" + "-"*60)
    print("   ✅ CONFIGURACIÓN GUARDADA:")
    print(f"      Norte: {config['orientacion_norte']}")
    print(f"      Pesos: cobertura={config['score_weights']['cobertura']:.0%}, "
          f"solar={config['score_weights']['orientacion_solar']:.0%}")
    print(f"      Tamaño ventana: {config['tamaño_ventana_m']}m")
    print("-"*60 + "\n")
    
    return config
def segmentate_perimeter_into_walls_by_room(gParam, lPerimeter, lConnexComponents):
    """
    DIVISOR DE FACHADA:
    Toma los muros exteriores largos y los corta en segmentos pequeños
    justo donde las paredes interiores los tocan.
    
    Objetivo: Que cada tramo de pared exterior pertenezca a UNA sola habitación.
    """
    import numpy as np

    height = gParam['height']
    width = gParam['width']
    idxIW = gParam['idxNilColorInteriorWall']

    MIN_SEGMENT_LEN = 3
    # Distancia en píxeles para considerar que un tabique toca la fachada
    PROXIMITY_THRESHOLD = 7

    # --- 1) Mapear paredes interiores ---
    # Recopilamos todos los píxeles (x,y) que son muros interiores
    # para poder buscar colisiones rápidamente.
    interior_wall_pixels = set()
    if len(lConnexComponents) > idxIW and isinstance(lConnexComponents[idxIW], tuple):
        cc_list = lConnexComponents[idxIW][2]
        for cc in cc_list:
            for (x, y) in cc:
                interior_wall_pixels.add((x, y))

    print(f"[SEGMENT] Total interior wall pixels: {len(interior_wall_pixels)}")

    def get_wall_intersection_points(x1, y1, x2, y2):
        """
        Detecta dónde una pared interior golpea este segmento de fachada.
        Devuelve una lista de coordenadas (puntos de corte).
        """
        intersection_clusters = []

        if x1 == x2:  # --- Caso: Muro Exterior VERTICAL ---
            x = int(x1)
            ya, yb = (int(y1), int(y2)) if y1 < y2 else (int(y2), int(y1))

            # Escaneamos a lo largo del muro vertical
            cluster = []
            for y in range(ya, yb + 1):
                # Miramos a izquierda y derecha buscando un muro interior cercano
                nearby_wall = False
                for dx in range(-PROXIMITY_THRESHOLD, PROXIMITY_THRESHOLD + 1):
                    for dy in range(-1, 2):  # Pequeño margen vertical
                        check_x, check_y = x + dx, y + dy
                        if (check_x, check_y) in interior_wall_pixels:
                            nearby_wall = True
                            break
                    if nearby_wall:
                        break

                # Agrupamos píxeles contiguos de intersección (clusters)
                # porque una pared tiene grosor, no es un solo punto.
                if nearby_wall:
                    cluster.append(y)
                elif cluster:
                    # Fin del contacto, calculamos el centro del cruce
                    if len(cluster) >= 2:
                        avg_y = int(np.mean(cluster))
                        intersection_clusters.append(avg_y)
                    cluster = []

            # Procesar último cluster si existe
            if cluster and len(cluster) >= 2:
                avg_y = int(np.mean(cluster))
                intersection_clusters.append(avg_y)

        else:  # --- Caso: Muro Exterior HORIZONTAL ---
            y = int(y1)
            xa, xb = (int(x1), int(x2)) if x1 < x2 else (int(x2), int(x1))

            cluster = []
            for x in range(xa, xb + 1):
                # Miramos arriba y abajo buscando muro interior
                nearby_wall = False
                for dy in range(-PROXIMITY_THRESHOLD, PROXIMITY_THRESHOLD + 1):
                    for dx in range(-1, 2):  
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

    # --- 2) Cortar los muros ---
    new_perimeter = []

    for (x1, y1, x2, y2, label) in lPerimeter:
        # Solo procesamos muros exteriores ('EW')
        if label != 'EW':
            new_perimeter.append((int(x1), int(y1), int(x2), int(y2), label))
            continue

        # Buscamos puntos de corte
        intersections = get_wall_intersection_points(x1, y1, x2, y2)

        if not intersections:
            new_perimeter.append((int(x1), int(y1), int(x2), int(y2), label))
            continue

        # Si hay cortes, dividimos el segmento original
        if x1 == x2: # Vertical
            x = int(x1)
            ya, yb = (int(y1), int(y2)) if y1 < y2 else (int(y2), int(y1))
            intersections.sort()

            current_y = ya
            for cut_y in intersections:
                if cut_y - current_y >= MIN_SEGMENT_LEN:
                    new_perimeter.append((x, current_y, x, cut_y, label))
                current_y = cut_y + 1

            # Añadir el trozo final
            if yb - current_y >= MIN_SEGMENT_LEN:
                new_perimeter.append((x, current_y, x, yb, label))
            else:
                new_perimeter.append((x, current_y, x, yb, label))

        elif y1 == y2: # Horizontal
            y = int(y1)
            xa, xb = (int(x1), int(x2)) if x1 < x2 else (int(x2), int(x1))
            intersections.sort()

            current_x = xa
            for cut_x in intersections:
                if cut_x - current_x >= MIN_SEGMENT_LEN:
                    new_perimeter.append((current_x, y, cut_x, y, label))
                current_x = cut_x + 1

            # Añadir el trozo final
            if xb - current_x >= MIN_SEGMENT_LEN:
                new_perimeter.append((current_x, y, xb, y, label))
            else:
                new_perimeter.append((current_x, y, xb, y, label))

        else:
            # Si es diagonal, lo dejamos tal cual
            new_perimeter.append((int(x1), int(y1), int(x2), int(y2), label))

    print(
        f"[SEGMENT] Original walls: {len(lPerimeter)} -> Segmented walls: {len(new_perimeter)}")
    return [(int(x1), int(y1), int(x2), int(y2), lbl) for (x1, y1, x2, y2, lbl) in new_perimeter]

def _is_wall_adjacent_to_room(wall_segment_pixels, room_pixels_set, max_dist=2):
    """ Helper: Verifica si un trozo de pared toca los píxeles de una habitación. """
    for p in wall_segment_pixels:
        wx, wy = p[:2]
        for dx in range(-max_dist, max_dist + 1):
            for dy in range(-max_dist, max_dist + 1):
                if (wx + dx, wy + dy) in room_pixels_set:
                    return True
    return False

def place_windows_heuristic(gParam, lPerimeter, lConnexComponentsCentroids,
                            lConnexComponents,
                            config=None,
                            interactive=True,
                            allow_multi_windows_per_wall=False,
                            min_len_factor=0.9,     
                            max_dist_adj=3,        
                            end_margin_m=0.15):     
    """
    Función Maestra:
    1. Opcionalmente solicita configuración al usuario (si interactive=True)
    2. Segmenta la fachada por habitaciones.
    3. Ejecuta un algoritmo de optimización (Hill Climbing) para poner ventanas.
    4. Devuelve el perímetro final con ventanas ('WN').
    
    Parámetros:
    - config: Dict con configuración (si None y interactive=True, pregunta al usuario)
    - interactive: Si True, pregunta al usuario por la configuración
    """
    import numpy as np
    
    # Obtener configuración
    if config is None and interactive:
        config = solicitar_config_usuario()
    
    nppm = gParam['nPixelsPerMeter']
    WINDOW_SIZE_PX = int(1.2 * nppm)
    END_MARGIN_PX = int(end_margin_m * nppm)
    MIN_WALL_LEN_PX = max(int(min_len_factor * nppm),
                          WINDOW_SIZE_PX + 2*END_MARGIN_PX)

    new_perimeter = []

    auxlPerimeter = segmentate_perimeter_into_walls_by_room(
        gParam, lPerimeter, lConnexComponents)
    
    # Ejecutar optimización con la config
    eval_info = hill_climbing_window_optimization(gParam, auxlPerimeter, lConnexComponentsCentroids, config=config)
    
    new_perimeter = eval_info['mejor_perimeter']
    return new_perimeter

def evaluar(gParam, lPerimeter, lConnexComponentsCentroids, config=None):
    """
    EL JUEZ DE LUZ (Versión 5.0 - Completamente Configurable):
    - Usa config para personalizar pesos, orientación solar, etc.
    - Cada ventana se asigna SOLO a la habitación más cercana.
    - Calcula múltiples métricas combinadas según pesos.
    """
    import numpy as np
    from window_placement_utils import getPointsLineBetweenXY, get_config
    
    # Obtener configuración (defaults + usuario)
    cfg = get_config(config)
    weights = cfg['score_weights']
    
    idx_balcon = gParam.get('idxNilColorBalcony', -999)
    height = gParam.get('height', 256)
    width = gParam.get('width', 256)

    # --- 1) Encontrar ventanas y calcular orientación ---
    ventanas = []
    for (x1, y1, x2, y2, label) in lPerimeter:
        if label == 'WN':
            pixels = getPointsLineBetweenXY(x1, y1, x2, y2)
            centro_x = (x1 + x2) / 2
            centro_y = (y1 + y2) / 2
            orientacion = _calcular_orientacion(x1, y1, x2, y2, height, width, cfg['orientacion_norte'])
            ventanas.append({
                'pixels': set((p[0], p[1]) for p in pixels),
                'centro': (centro_x, centro_y),
                'coords': (x1, y1, x2, y2),
                'orientacion': orientacion
            })

    num_ventanas = len(ventanas)

    # --- 2) Identificar habitaciones ---
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
            if room_pixels:
                avg_x = np.mean([p[0] for p in room_pixels])
                avg_y = np.mean([p[1] for p in room_pixels])
            else:
                avg_x, avg_y = 0, 0
            
            habitaciones.append({
                'idx': idx_cc,
                'tipo': idxColor,
                'pixels': set(room_pixels),
                'centro': (avg_x, avg_y),
                'es_balcon': (idxColor == idx_balcon),
                'tiene_ventana': False
            })

    total_habitables = sum(1 for h in habitaciones if not h['es_balcon'])

    # --- CASO VACÍO ---
    if total_habitables == 0:
        return {
            'score_total': 0.0,
            'componentes': {'cobertura': 0.0, 'orientacion_solar': 0.0, 'distribucion': 0.0},
            'penalizaciones': {},
            'debug_info': {'num_ventanas': num_ventanas, 'num_habitaciones_total': 0, 
                          'habitaciones_con_ventana': 0, 'habitaciones_sin_ventana': 0}
        }

    # --- 3) ASIGNACIÓN: Cada ventana → habitación más cercana ---
    MAX_DIST_ADYACENCIA = 10
    
    for ventana in ventanas:
        vx, vy = ventana['centro']
        mejor_dist = float('inf')
        mejor_hab = None
        
        for hab in habitaciones:
            hx, hy = hab['centro']
            dist = np.sqrt((vx - hx)**2 + (vy - hy)**2)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor_hab = hab
        
        if mejor_hab is not None and mejor_dist < MAX_DIST_ADYACENCIA * gParam.get('nPixelsPerMeter', 10):
            mejor_hab['tiene_ventana'] = True

    # --- 4) CALCULAR MÉTRICAS ---
    
    # 4a) Cobertura
    habitaciones_con_ventana = sum(1 for h in habitaciones if h['tiene_ventana'] and not h['es_balcon'])
    balcones_con_ventana = sum(1 for h in habitaciones if h['tiene_ventana'] and h['es_balcon'])
    score_cobertura = habitaciones_con_ventana / total_habitables if total_habitables > 0 else 0
    
    # 4b) Orientación Solar (Calidad + Cantidad)
    if num_ventanas > 0:
        pesos_orient = cfg['pesos_orientacion']
        
        # 1. Calidad Promedio (0-1)
        # ¿Cómo de buenas son mis ventanas en promedio?
        calidad_promedio = sum(pesos_orient.get(v['orientacion'], 0.5) for v in ventanas) / num_ventanas
        
        # 2. Factor Cantidad (Bonus 0-1)
        # Incentiva poner más ventanas si son buenas.
        # Definimos "ventana buena" como aquella con peso > 0.5 (Sur/Este/Oeste)
        # Tope de bonus: 1.0 (se alcanza con 10 ventanas buenas)
        num_ventanas_buenas = sum(1 for v in ventanas if pesos_orient.get(v['orientacion'], 0) > 0.5)
        factor_cantidad = min(1.0, num_ventanas_buenas * 0.1)
        
        # Score final combinado (70% Calidad, 30% Cantidad)
        score_solar = (calidad_promedio * 0.7) + (factor_cantidad * 0.3)
    else:
        score_solar = 0.0
    
    # 4c) Distribución por Fachada
    score_distribucion = _calcular_distribucion_fachada(ventanas)
    
    # 4d) Simetría (simplificado)
    score_simetria = _calcular_simetria(ventanas, width, height)
    
    # --- 5) SCORE FINAL PONDERADO ---
    score_final = (
        weights.get('cobertura', 0.3) * score_cobertura +
        weights.get('orientacion_solar', 0.3) * score_solar +
        weights.get('distribucion', 0.2) * score_distribucion +
        weights.get('simetria', 0.1) * score_simetria
    )
    
    # Penalización balcones
    score_final -= balcones_con_ventana * cfg.get('penalizacion_balcon', 0.15)
    score_final = max(0.0, min(1.0, score_final))

    detalle = [(h['idx'], h['tipo'], 'BALCON' if h['es_balcon'] else 'ROOM', h['tiene_ventana']) for h in habitaciones]

    return {
        'score_total': score_final,
        'componentes': {
            'cobertura': score_cobertura,
            'orientacion_solar': score_solar,
            'distribucion': score_distribucion,
            'simetria': score_simetria
        },
        'penalizaciones': {'balcones_erroneos': balcones_con_ventana},
        'debug_info': {
            'num_ventanas': num_ventanas,
            'num_habitaciones_total': total_habitables,
            'habitaciones_con_ventana': habitaciones_con_ventana,
            'habitaciones_sin_ventana': total_habitables - habitaciones_con_ventana,
            'orientaciones': [v['orientacion'] for v in ventanas],
            'detalle_habitaciones': detalle
        }
    }


def _calcular_orientacion(x1, y1, x2, y2, height, width, norte):
    """Determina la orientación de una ventana (N/S/E/O) basándose en su posición."""
    # Centro de la ventana
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Determinar en qué fachada está (borde más cercano)
    dist_top = cy
    dist_bottom = height - cy
    dist_left = cx
    dist_right = width - cx
    
    min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
    
    if min_dist == dist_top:
        fachada = 'TOP'
    elif min_dist == dist_bottom:
        fachada = 'BOTTOM'
    elif min_dist == dist_left:
        fachada = 'LEFT'
    else:
        fachada = 'RIGHT'
    
    # Mapear fachada a orientación según dónde está el norte
    orientacion_map = {
        'TOP': {'TOP': 'NORTE', 'BOTTOM': 'SUR', 'LEFT': 'OESTE', 'RIGHT': 'ESTE'},
        'BOTTOM': {'TOP': 'SUR', 'BOTTOM': 'NORTE', 'LEFT': 'ESTE', 'RIGHT': 'OESTE'},
        'LEFT': {'TOP': 'ESTE', 'BOTTOM': 'OESTE', 'LEFT': 'NORTE', 'RIGHT': 'SUR'},
        'RIGHT': {'TOP': 'OESTE', 'BOTTOM': 'ESTE', 'LEFT': 'SUR', 'RIGHT': 'NORTE'}
    }
    
    return orientacion_map.get(norte, orientacion_map['TOP']).get(fachada, 'SUR')


def _calcular_distribucion_fachada(ventanas):
    """Calcula score de distribución: penaliza si todas las ventanas están en una fachada."""
    if len(ventanas) == 0:
        return 0.0
    
    orientaciones = [v['orientacion'] for v in ventanas]
    unique = set(orientaciones)
    
    # Bonus por tener ventanas en múltiples fachadas
    if len(unique) >= 4:
        return 1.0
    elif len(unique) == 3:
        return 0.8
    elif len(unique) == 2:
        return 0.6
    else:
        return 0.3  # Todas en una sola fachada


def _calcular_simetria(ventanas, width, height):
    """Calcula score de simetría respecto al eje central."""
    if len(ventanas) == 0:
        return 1.0
    
    centro_x = width / 2
    desviaciones = [abs(v['centro'][0] - centro_x) for v in ventanas]
    avg_desviacion = sum(desviaciones) / len(desviaciones)
    
    # Normalizar: 0 desviación = 1.0, max desviación = 0.0
    max_desviacion = width / 2
    return 1.0 - (avg_desviacion / max_desviacion)

def generar_vecino_mover_ventana(gParam, lPerimeter, max_desplazamiento=None):
    """
    MOVIMIENTO 1: DESLIZAR (Slide)
    Elige una ventana al azar y la mueve lateralmente dentro de su misma pared.
    Útil para ajustar márgenes o evitar obstaculos menores.
    """
    import numpy as np
    import random
    
    # 1) Buscar ventanas existentes
    ventanas_indices = [i for i, seg in enumerate(lPerimeter) if seg[4] == 'WN']
    if not ventanas_indices:
        return lPerimeter, {"moved": False, "reason": "no_windows"}
    
    idx_seg = random.choice(ventanas_indices)
    
    # 2) Validar que tiene muro antes y después (que no está en una esquina rota)
    if idx_seg == 0 or idx_seg >= len(lPerimeter) - 1:
        return lPerimeter, {"moved": False, "reason": "window_at_boundary"}
    
    wall_before = lPerimeter[idx_seg - 1]
    wall_after = lPerimeter[idx_seg + 1]
    window = lPerimeter[idx_seg]
    
    if wall_before[4] != 'EW' or wall_after[4] != 'EW':
        return lPerimeter, {"moved": False, "reason": "no_EW_adjacent"}
    
    # 3) Obtener coordenadas del "Hueco" total (Muro + Ventana + Muro)
    muro_x1, muro_y1 = wall_before[0], wall_before[1]
    muro_x2, muro_y2 = wall_after[2], wall_after[3]
    
    # 4) Calcular desplazamiento
    is_vertical = (muro_x1 == muro_x2)
    
    nppm = gParam['nPixelsPerMeter']
    END_MARGIN_PX = int(0.15 * nppm)
    
    if is_vertical:
        # --- MOVIMIENTO VERTICAL (Eje Y) ---
        window_height = abs(window[3] - window[1])
        muro_height = abs(muro_y2 - muro_y1)
        
        if max_desplazamiento is None:
            max_desplazamiento = window_height // 2
        
        # Calcular espacio libre real
        espacio_util = muro_height - window_height - 2 * END_MARGIN_PX
        if espacio_util <= 0:
            return lPerimeter, {"moved": False, "reason": "no_space"}
        
        max_desp_real = min(max_desplazamiento, espacio_util // 2)
        desplazamiento = random.randint(-max_desp_real, max_desp_real)
        
        # Calcular nueva posición
        new_window = (
            window[0],  
            window[1] + desplazamiento,
            window[2],  
            window[3] + desplazamiento,
            'WN'
        )
        
        # Comprobar que no nos salimos del muro
        min_y = min(muro_y1, muro_y2) + END_MARGIN_PX
        max_y = max(muro_y1, muro_y2) - END_MARGIN_PX
        
        if new_window[1] < min_y or new_window[3] > max_y:
            return lPerimeter, {"moved": False, "reason": "out_of_bounds"}
    
    else:
        # --- MOVIMIENTO HORIZONTAL (Eje X) ---
        window_width = abs(window[2] - window[0])
        muro_width = abs(muro_x2 - muro_x1)
        
        if max_desplazamiento is None:
            max_desplazamiento = window_width // 2
        
        espacio_util = muro_width - window_width - 2 * END_MARGIN_PX
        if espacio_util <= 0:
            return lPerimeter, {"moved": False, "reason": "no_space"}
        
        max_desp_real = min(max_desplazamiento, espacio_util // 2)
        desplazamiento = random.randint(-max_desp_real, max_desp_real)
        
        new_window = (
            window[0] + desplazamiento,
            window[1], 
            window[2] + desplazamiento,
            window[3], 
            'WN'
        )
        
        min_x = min(muro_x1, muro_x2) + END_MARGIN_PX
        max_x = max(muro_x1, muro_x2) - END_MARGIN_PX
        
        if new_window[0] < min_x or new_window[2] > max_x:
            return lPerimeter, {"moved": False, "reason": "out_of_bounds"}
    
    # 5) Reconstruir la lista de paredes
    # Ajustamos los segmentos de muro a los lados de la nueva ventana
    new_wall_before = (muro_x1, muro_y1, new_window[0], new_window[1], 'EW')
    new_wall_after = (new_window[2], new_window[3], muro_x2, muro_y2, 'EW')
    
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

def generar_vecino_intercambiar_ventana_de_pared(gParam, lPerimeter, lConnexComponentsCentroids, window_size_m=1.2):
    """
    MOVIMIENTO 2: TELETRANSPORTAR (Swap)
    Quita una ventana de su pared actual y la pone en OTRA pared exterior
    que pertenezca a la MISMA habitación.
    
    Ejemplo: Mover ventana de la pared Norte a la pared Este de un dormitorio.
    """
    import numpy as np
    import random
    
    nppm = gParam['nPixelsPerMeter']
    WINDOW_SIZE_PX = int(window_size_m * nppm)
    END_MARGIN_PX = int(0.15 * nppm)
    MIN_WALL_LEN_PX = WINDOW_SIZE_PX + 2 * END_MARGIN_PX
    
    # --- 1) Inventario de ventanas ---
    ventanas_info = []
    
    for idx_seg, (x1, y1, x2, y2, label) in enumerate(lPerimeter):
        if label != 'WN': continue
        
        # Localizamos a qué habitación pertenece esta ventana
        ventana_pixels = set((p[0], p[1]) for p in getPointsLineBetweenXY(x1, y1, x2, y2))
        habitacion_asociada = None
        
        for idx_cc, (idxColor, rgbColor, room_pixels) in enumerate(lConnexComponentsCentroids):
            # Filtramos tipos no validos
            if idxColor in {
                gParam['idxNilColorBackground'], gParam['idxNilColorExteriorWall'],
                gParam['idxNilColorFrontDoor'], gParam['idxNilColorInteriorWall'],
                gParam.get('idxNilColorInsideDoor', -1)
            }: continue
            
            room_pixels_set = set(room_pixels)
            # Check de adyacencia
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
    
    # --- 2) Elegir víctima (ventana a mover) ---
    ventana_elegida = random.choice(ventanas_info)
    idx_ventana_original = ventana_elegida['idx_seg']
    habitacion_idx, habitacion_tipo, room_pixels_set = ventana_elegida['habitacion']
    
    # --- 3) Buscar destino (otros muros de la MISMA habitación) ---
    muros_candidatos = []
    
    for idx_seg, (x1, y1, x2, y2, label) in enumerate(lPerimeter):
        if label != 'EW': continue
        
        # No podemos moverla a la misma pared donde ya está
        if idx_seg == idx_ventana_original - 1 or idx_seg == idx_ventana_original + 1:
            continue 
        
        # La pared nueva debe ser suficientemente larga
        wall_len = float(np.hypot(x2 - x1, y2 - y1))
        if wall_len < MIN_WALL_LEN_PX:
            continue
        
        # Verificar que la pared destino toca la misma habitación
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
    
    # --- 4) Selección del destino (preferimos paredes grandes) ---
    muro_destino = max(muros_candidatos, key=lambda m: m['length'])
    
    # --- 5) RECONSTRUCCIÓN: Cerrar el hueco viejo ---
    # Convertimos (Muro-Ventana-Muro) en (Muro Largo Sólido)
    new_perimeter = []
    
    for i, seg in enumerate(lPerimeter):
        if i == idx_ventana_original - 1:
            continue # Saltamos tramo 1
        elif i == idx_ventana_original:
            continue # Saltamos la ventana vieja
        elif i == idx_ventana_original + 1:
            # En el tramo 2, fusionamos todo
            ew_before = lPerimeter[idx_ventana_original - 1]
            ew_after = seg
            muro_completo = (ew_before[0], ew_before[1], ew_after[2], ew_after[3], 'EW')
            new_perimeter.append(muro_completo)
        else:
            new_perimeter.append(seg)
    
    # --- 6) ABRIR HUECO NUEVO ---
    # Calculamos las coordenadas para centrar la ventana en el muro destino
    x1, y1, x2, y2 = muro_destino['coords']
    wall_len = muro_destino['length']
    
    v = np.array([x2 - x1, y2 - y1], dtype=float)
    v_unit = v / wall_len
    
    mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
    half = WINDOW_SIZE_PX / 2.0
    p1 = mid - v_unit * half
    p2 = mid + v_unit * half
    
    win_x1, win_y1 = int(round(p1[0])), int(round(p1[1]))
    win_x2, win_y2 = int(round(p2[0])), int(round(p2[1]))
    
    # --- 7) Insertar ventana en la lista ---
    final_perimeter = []
    idx_destino = muro_destino['idx_seg']
    
    # Ajustar índice porque la lista se encogió al borrar la ventana vieja
    if idx_destino > idx_ventana_original:
        idx_destino -= 2 
    
    for i, seg in enumerate(new_perimeter):
        if i == idx_destino:
            # Rompemos el muro sólido en: EW | WN | EW
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

def generar_vecino_anadir_ventana(gParam, lPerimeter, lConnexComponentsCentroids, window_size_m=1.2):
    """
    MOVIMIENTO 3: AÑADIR (Add)
    Busca una pared exterior "ciega" (sin ventanas) y le abre un hueco nuevo en el centro.
    Fundamental para habitaciones oscuras que necesitan ganar puntuación.
    """
    import numpy as np
    import random
    
    nppm = gParam['nPixelsPerMeter']
    WINDOW_SIZE_PX = int(window_size_m * nppm)
    END_MARGIN_PX = int(0.15 * nppm)
    MIN_WALL_LEN_PX = WINDOW_SIZE_PX + 2 * END_MARGIN_PX
    
    # --- 1) BUSCAR CANDIDATOS ---
    # Identificamos todos los muros exteriores (EW) que sean "vírgenes" (sin ventanas pegadas).
    muros_sin_ventana = []
    
    for idx_seg, (x1, y1, x2, y2, label) in enumerate(lPerimeter):
        if label != 'EW':
            continue
        
        # Filtro 1: ¿Cabe una ventana físicamente?
        wall_len = float(np.hypot(x2 - x1, y2 - y1))
        if wall_len < MIN_WALL_LEN_PX:
            continue
        
        # Filtro 2: ¿Ya tiene ventana al lado?
        # Miramos el segmento anterior y el siguiente en la lista del perímetro.
        tiene_ventana_adyacente = False
        
        if idx_seg > 0 and lPerimeter[idx_seg - 1][4] == 'WN':
            tiene_ventana_adyacente = True
        
        if idx_seg < len(lPerimeter) - 1 and lPerimeter[idx_seg + 1][4] == 'WN':
            tiene_ventana_adyacente = True
        
        if not tiene_ventana_adyacente:
            muros_sin_ventana.append({
                'idx_seg': idx_seg,
                'coords': (x1, y1, x2, y2),
                'length': wall_len
            })
    
    # Si no hay dónde poner ventanas, abortamos
    if not muros_sin_ventana:
        return lPerimeter, {
            "added": False,
            "reason": "no_available_walls"
        }
    
    # --- 2) SELECCIÓN ALEATORIA ---
    muro_elegido = random.choice(muros_sin_ventana)
    idx_muro = muro_elegido['idx_seg']
    x1, y1, x2, y2 = muro_elegido['coords']
    wall_len = muro_elegido['length']
    
    # --- 3) GEOMETRÍA: CALCULAR EL CENTRO ---
    # Convertimos la pared en un vector para encontrar su punto medio exacto.
    v = np.array([x2 - x1, y2 - y1], dtype=float)
    v_unit = v / wall_len
    
    mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
    half = WINDOW_SIZE_PX / 2.0
    
    # Definimos los puntos de inicio y fin de la nueva ventana
    p1 = mid - v_unit * half
    p2 = mid + v_unit * half
    
    win_x1, win_y1 = int(round(p1[0])), int(round(p1[1]))
    win_x2, win_y2 = int(round(p2[0])), int(round(p2[1]))
    
    # --- 4) VERIFICACIÓN DE MÁRGENES ---
    # Asegurarnos de no poner la ventana pegada a la esquina de la casa.
    dist_inicio = np.linalg.norm(mid - np.array([x1, y1]))
    dist_fin = np.linalg.norm(mid - np.array([x2, y2]))
    
    if dist_inicio < (END_MARGIN_PX + half) or dist_fin < (END_MARGIN_PX + half):
        return lPerimeter, {
            "added": False,
            "reason": "insufficient_margin",
            "wall_idx": idx_muro
        }
    
    # --- 5) INFO EXTRA (Opcional) ---
    # Buscamos a qué habitación le acabamos de regalar luz (para estadísticas).
    habitacion_asociada = None
    wall_pixels = getPointsLineBetweenXY(x1, y1, x2, y2)
    
    for idx_cc, (idxColor, rgbColor, room_pixels) in enumerate(lConnexComponentsCentroids):
        # Ignorar elementos estructurales
        if idxColor in {
            gParam['idxNilColorBackground'], gParam['idxNilColorExteriorWall'],
            gParam['idxNilColorFrontDoor'], gParam['idxNilColorInteriorWall'],
            gParam.get('idxNilColorInsideDoor', -1)
        }:
            continue
        
        room_pixels_set = set(room_pixels)
        if _is_wall_adjacent_to_room(wall_pixels, room_pixels_set, max_dist=3):
            habitacion_asociada = (idx_cc, idxColor)
            break
    
    # --- 6) CIRUGÍA: INSERTAR VENTANA ---
    new_perimeter = []
    
    for i, seg in enumerate(lPerimeter):
        if i == idx_muro:
            # Sustituimos el Muro Sólido por: [Muro Corto] - [Ventana] - [Muro Corto]
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
    MOVIMIENTO 4: ELIMINAR (Remove)
    Elige una ventana y la "tapia". Fusiona los dos trozos de muro adyacentes
    para crear una pared sólida continua.
    Útil si hay exceso de ventanas o están mal puestas.
    """
    import numpy as np
    import random
    
    # --- 1) ENCONTRAR VENTANAS ---
    ventanas_indices = []
    for idx, (x1, y1, x2, y2, label) in enumerate(lPerimeter):
        if label == 'WN':
            ventanas_indices.append(idx)
    
    if not ventanas_indices:
        return lPerimeter, {
            "removed": False,
            "reason": "no_windows"
        }
    
    # --- 2) ELEGIR VÍCTIMA ---
    idx_ventana = random.choice(ventanas_indices)
    window = lPerimeter[idx_ventana]
    
    # --- 3) VERIFICAR ENTORNO ---
    # Necesitamos que tenga muro a izq y derecha para poder fusionarlos.
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
    
    # --- 4) INFO EXTRA (Debug) ---
    ventana_pixels = set((p[0], p[1]) for p in getPointsLineBetweenXY(window[0], window[1], window[2], window[3]))
    habitacion_asociada = None
    
    for idx_cc, (idxColor, rgbColor, room_pixels) in enumerate(lConnexComponentsCentroids):
        # Filtro de seguridad de tipos
        if idxColor in {
            gParam['idxNilColorBackground'], gParam['idxNilColorExteriorWall'],
            gParam['idxNilColorFrontDoor'], gParam['idxNilColorInteriorWall'],
            gParam.get('idxNilColorInsideDoor', -1)
        }:
            continue
        
        room_pixels_set = set(room_pixels)
        if _is_wall_adjacent_to_room(ventana_pixels, room_pixels_set, max_dist=3):
            habitacion_asociada = (idx_cc, idxColor)
            break
    
    # --- 5) ALBAÑILERÍA: FUSIONAR MUROS ---
    # Creamos un muro nuevo que va desde el inicio del anterior hasta el final del posterior.
    muro_completo = (
        wall_before[0], wall_before[1],  # Inicio
        wall_after[2], wall_after[3],    # Fin
        'EW'
    )
    
    muro_length = float(np.hypot(
        muro_completo[2] - muro_completo[0],
        muro_completo[3] - muro_completo[1]
    ))
    
    # --- 6) RECONSTRUIR PERÍMETRO ---
    new_perimeter = []
    
    for i, seg in enumerate(lPerimeter):
        if i == idx_ventana - 1:
            continue # Ignoramos el trozo de muro izquierdo
        elif i == idx_ventana:
            continue # Ignoramos la ventana
        elif i == idx_ventana + 1:
            # Aquí insertamos el muro nuevo y grande
            new_perimeter.append(muro_completo)
        else:
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
                                      max_sin_mejora=100,
                                      config=None,
                                      verbose=True):
    """
    CEREBRO DE OPTIMIZACIÓN (HILL CLIMBING):
    Este algoritmo intenta mejorar la distribución de ventanas iterativamente.
    Prueba cambios al azar y solo acepta los que mejoran la puntuación global.
    
    Parámetros:
    - config: Diccionario con configuración (usar DEFAULT_CONFIG si None)
    """
    import numpy as np
    import random
    from window_placement_utils import get_config
    
    # Obtener configuración
    cfg = get_config(config)
    probabilidades_operadores = cfg.get('probabilidades_operadores', None)
    window_size_m = cfg.get('tamaño_ventana_m', 1.2)
    
    # --- Inicialización ---
    mejor_perimeter = lPerimeter_inicial
    # Calculamos la nota inicial de la casa
    eval_inicial = evaluar(gParam, mejor_perimeter, lConnexComponentsCentroids, cfg)
    mejor_score = eval_inicial['score_total']
    score_inicial = mejor_score
    
    iteraciones_sin_mejora = 0
    historial = []
    
    # Contadores para saber qué movimientos funcionan mejor
    stats_ops = {
        'Op1': {'intentos': 0, 'exitos': 0, 'mejoras': 0}, # Mover
        'Op2': {'intentos': 0, 'exitos': 0, 'mejoras': 0}, # Intercambiar
        'Op3': {'intentos': 0, 'exitos': 0, 'mejoras': 0}, # Añadir
        'Op4': {'intentos': 0, 'exitos': 0, 'mejoras': 0}  # Eliminar
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
    
    # --- BUCLE DE OPTIMIZACIÓN ---
    for iteracion in range(max_iteraciones):
        
        # 1. ESTRATEGIA DE PROBABILIDADES
        # Si el usuario definió probabilidades explícitas, las usamos.
        # Si no, usamos lógica adaptativa.
        
        if cfg.get('usar_probabilidades_usuario', False) and probabilidades_operadores:
            probs = probabilidades_operadores
        elif probabilidades_operadores is not None and not cfg.get('usar_probabilidades_usuario', False):
             # Caso legacy: probabilidades pasadas como argumento pero sin flag
             probs = probabilidades_operadores
        else:
            # --- ESTRATEGIA ADAPTATIVA ---
            eval_actual = evaluar(gParam, mejor_perimeter, lConnexComponentsCentroids, cfg)
            score_cobertura = eval_actual['componentes']['cobertura']
            num_ventanas = eval_actual['debug_info']['num_ventanas']
            num_habitaciones = eval_actual['debug_info']['num_habitaciones_total']
            
            if num_ventanas > num_habitaciones * 1.5:
                # Demasiadas ventanas -> Fomentar Eliminar
                probs = {'Op1': 0.40, 'Op2': 0.25, 'Op3': 0.05, 'Op4': 0.30}
            elif score_cobertura >= 0.9:
                # Casi perfecto -> Solo ajustes finos
                probs = {'Op1': 0.50, 'Op2': 0.35, 'Op3': 0.10, 'Op4': 0.05}
            else:
                # Balanceado (ligeramente agresivo en añadir)
                probs = {'Op1': 0.15, 'Op2': 0.35, 'Op3': 0.45, 'Op4': 0.05}
        
        # 2. ELEGIR UN MOVIMIENTO
        ops = list(probs.keys())
        pesos = list(probs.values())
        operador = random.choices(ops, weights=pesos)[0]
        
        stats_ops[operador]['intentos'] += 1
        
        # 3. EJECUTAR EL MOVIMIENTO (Generar Vecino)
        if operador == 'Op1':
            vecino_perimeter, info = generar_vecino_mover_ventana(gParam, mejor_perimeter)
        elif operador == 'Op2':
            vecino_perimeter, info = generar_vecino_intercambiar_ventana_de_pared(
                gParam, mejor_perimeter, lConnexComponentsCentroids, window_size_m
            )
        elif operador == 'Op3':
            vecino_perimeter, info = generar_vecino_anadir_ventana(
                gParam, mejor_perimeter, lConnexComponentsCentroids, window_size_m
            )
        elif operador == 'Op4':
            vecino_perimeter, info = generar_vecino_eliminar_ventana(
                gParam, mejor_perimeter, lConnexComponentsCentroids
            )
        else:
            continue
        
        # 4. VALIDAR SI EL MOVIMIENTO FUE FÍSICAMENTE POSIBLE
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
        
        # 5. EVALUAR EL NUEVO DISEÑO (VECINO)
        eval_vecino = evaluar(gParam, vecino_perimeter, lConnexComponentsCentroids, cfg)
        score_vecino = eval_vecino['score_total']
        
        # 6. DECISIÓN (HILL CLIMBING ESTRICTO)
        # Solo aceptamos el cambio si la puntuación MEJORA.
        if score_vecino > mejor_score:
            # ¡Mejora encontrada! Nos quedamos con este perímetro.
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
            # El cambio fue a peor (o igual), lo descartamos y probamos otro.
            iteraciones_sin_mejora += 1
            
            historial.append({
                'iteracion': iteracion,
                'score': mejor_score,
                'operador': operador,
                'exito_operador': True,
                'mejora': False,
                'delta_score': score_vecino - mejor_score
            })
        
        # 7. PARADA TEMPRANA
        # Si llevamos N intentos sin conseguir mejorar nada, asumimos que hemos llegado al máximo.
        if iteraciones_sin_mejora >= max_sin_mejora:
            if verbose:
                print(f"\n{'='*70}")
                print(f"⚠️  Parada temprana: {max_sin_mejora} iteraciones sin mejora")
            break
    
    # --- RESULTADOS FINALES ---
    eval_final = evaluar(gParam, mejor_perimeter, lConnexComponentsCentroids, cfg)
    
    mejora_absoluta = mejor_score - score_inicial
    mejora_relativa = (mejora_absoluta / score_inicial * 100) if score_inicial > 0 else 0
    
    if verbose:
        print("=" * 70)
        print("RESULTADOS FINALES")
        print("=" * 70)
        print(f"Score inicial:    {score_inicial:.4f}")
        print(f"Score final:      {mejor_score:.4f}")
        print(f"Mejora:           {mejora_absoluta:+.4f} ({mejora_relativa:+.1f}%)")
        print(f"Iteraciones:      {iteracion + 1} / {max_iteraciones}")
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