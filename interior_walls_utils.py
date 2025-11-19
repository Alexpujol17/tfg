
import numpy as np
from skimage.morphology import skeletonize, binary_dilation, square
from skimage.measure import label, regionprops

# En window_placement_utils.py
from Analysis_StyleGAN_functions_03_pindatafinal24_Relaxed import getPointsLineBetweenXY

def build_interior_walls_from_borders(gParam, rgbImageNilColors, lConnexComponents,
                                      imageContourSmoothed=None,
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
    idx_map = np.full((width, height), idxBG, dtype=np.int32)
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

    print ('  - Interior walls added - ')

    return lConnexComponents, wall_mask
