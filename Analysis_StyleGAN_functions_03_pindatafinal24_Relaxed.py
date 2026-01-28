
import sys, time
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

#from skimage.measure import label, regionprops
from skimage import measure #, data, filters, morphology
from skimage import segmentation
from skimage.feature import corner_harris, corner_peaks #, corner_fast, corner_subpix

from queue import PriorityQueue

####################################################################################################################
### Constructs the resulting image
####################################################################################################################

def getPointsLineBetweenXY (x1, y1, x2, y2):

    ### Points in the line between two points
    x  = x1;        y = y1;
    dx = x2 - x1;  dy = y2 - y1;
    #
    lPointsLineP1P2 = []
    if dx == 0 and dy == 0:
        lPointsLineP1P2.append( (int(x),int(y),'-') )
    elif np.abs(dx) > np.abs(dy):
        ### Line between both points following x
        lPointsLineP1P2.append( (int(x),int(y),'X') )
        m  = dy/dx
        while x != x2:
            x += np.sign(dx);  y += m*np.sign(dx);
            lPointsLineP1P2.append( (int(x),int(y),'X') )
    else:
        ### Line between both points following y
        lPointsLineP1P2.append( (int(x),int(y),'Y') )
        m  = dx/dy
        while y != y2:
            x += m*np.sign(dy);  y += np.sign(dy);
            lPointsLineP1P2.append( (int(x),int(y),'Y') )

    return lPointsLineP1P2


def drawLine (arrImageNilColorsRes, xC1, yC1, xC2, yC2, rgbColor, nPx):

    ### Draws a line of nPx pixels between two points

    lPointsLineP1P2 = getPointsLineBetweenXY (xC1, yC1, xC2, yC2)
    #print(lPointsLineXY)
    for (x,y,dr) in lPointsLineP1P2:
        for k in range(3):
            for p in range(nPx):
                if dr == 'X':
                    arrImageNilColorsRes[x, y + p - nPx//2, k] = rgbColor[k]
                else:
                    arrImageNilColorsRes[x + p - nPx//2, y, k] = rgbColor[k]

    return arrImageNilColorsRes    


###
from furniture_placement_utils import solve_master_bedroom, solve_dining_room, solve_kitchen, solve_bathroom, solve_living_room, solve_single_bedroom

def constructResultingImage (gParam, lConnexComponents, lCentroids=None, lAdjacencies=None,
                             lCorners=None, lPerimeter=None, plotPoint = False, pwalls=None):

    height                  = gParam['height']
    width                   = gParam['width']
    listNilColors           = gParam['listNilColors']
    idxNilColorBackground   = gParam['idxNilColorBackground']
    idxNilColorExteriorWall = gParam['idxNilColorExteriorWall']
    idxNilColorFrontDoor    = gParam['idxNilColorFrontDoor']
    idxNilColorInteriorWall = gParam['idxNilColorInteriorWall']
    idxNilColorInsideDoor   = gParam['idxNilColorInsideDoor']
    idxNilColorAux1         = gParam['idxNilColorAux1']
    idxNilColorAux2         = gParam['idxNilColorAux2']
   
    
    grayAdjPlot             = gParam['grayAdjPlot']
    dxyCentroid             = gParam['dxyCentroid']
    dxyCorner               = gParam['dxyCorner']
    dxyPoint                = gParam['dxyPoint']
    dxyFrontDoor            = gParam['dxyFrontDoor']

    ### Matrix of RGB colors (in the channels)
    arrImageNilColorsRes = np.zeros([width,height,3],int)

    # Add background to the whole image
    for x in range(height):     ### Reads from top to bottom, from left to right
        for y in range(width):
            for k in range(3):
                arrImageNilColorsRes[x,y,k] = listNilColors[idxNilColorBackground][k]

    ## Add interior walls from lConnexComponents
    lConnexComponentsInteriorWall = lConnexComponents[idxNilColorInteriorWall][2]
    for m in range(len(lConnexComponentsInteriorWall)):
        pixels = lConnexComponentsInteriorWall[m]
        for (x,y) in pixels:
            for k in range(3):
                arrImageNilColorsRes[x,y,k] = listNilColors[idxNilColorInteriorWall][k]

    # Add the adjacencies (if required) from lAdjacencies
    if lAdjacencies is not None:
        for (idxCentroid1, idxCentroid2) in lAdjacencies:
            centroid1 = lCentroids[idxCentroid1]
            centroid2 = lCentroids[idxCentroid2]
            (xC1,yC1) = centroid1[2]
            (xC2,yC2) = centroid2[2]
            ### Draws a line of 1 pixel between the two centroids
            arrImageNilColorsRes = drawLine(arrImageNilColorsRes,xC1,yC1,xC2,yC2,grayAdjPlot,2)

    # Add the centroids (if required) from lCentroids
    if lCentroids is not None:
        for tCentroid in lCentroids:
            idxColor = tCentroid[0]
            rgbColor = tCentroid[1]
            (xC,yC)  = tCentroid[2]
            for dx in np.arange(-dxyCentroid,dxyCentroid):
                for dy in np.arange(-dxyCentroid,dxyCentroid):
                    for k in range(3):
                        arrImageNilColorsRes[xC+dx,yC+dy,k] = listNilColors[idxColor][k]

    # Add the front door (if required) from lFrontDoors
    if lPerimeter is not None:
        try:
            idx_window = gParam['listNilColorsNames'].index("Window")
            color_window = listNilColors[idx_window]
        except (ValueError, IndexError):
            color_window = (0, 180, 255) # Color por defecto si no se encuentra
        #print(lPerimeter)
        for (xC1,yC1,xC2,yC2,idColor) in lPerimeter:
            ### Draws a line of 1 pixel between the two points
            if   idColor == 'EW':
                colorPer = listNilColors[idxNilColorExteriorWall]
            elif idColor == 'FD':
                colorPer = listNilColors[idxNilColorFrontDoor]
            elif idColor == 'WN':
                colorPer = color_window 
            else:
                print('ERROR: constructResultingImage: idColor not implemented'); sys.exit()
            arrImageNilColorsRes = drawLine(arrImageNilColorsRes,xC1,yC1,xC2,yC2,colorPer,3)
    else:
        # Add exteriors walls from lConnexComponents
        lConnexComponentsExteriorWall = lConnexComponents[idxNilColorExteriorWall][2]
        for m in range(len(lConnexComponentsExteriorWall)):
            pixels = lConnexComponentsExteriorWall[m]
            for (x,y) in pixels:
                for k in range(3):
                    arrImageNilColorsRes[x,y,k] = listNilColors[idxNilColorExteriorWall][k]

        # Add Connex components front door from lConnexComponents
        lConnexComponentsFrontDoor = lConnexComponents[idxNilColorFrontDoor][2]
        for m in range(len(lConnexComponentsFrontDoor)):
            pixels = lConnexComponentsFrontDoor[m]
            for (x,y) in pixels:
                for dx in np.arange(-dxyFrontDoor,dxyFrontDoor+1):
                    for dy in np.arange(-dxyFrontDoor,dxyFrontDoor+1):
                        for k in range(3):
                            arrImageNilColorsRes[x+dx,y+dy,k] = listNilColors[idxNilColorFrontDoor][k]
    

    if pwalls is not None:
        import math
        
        # Iteramos con índice para poder mirar anterior/siguiente (vecinos)
        for i in range(len(pwalls)):
            (y1, x1, y2, x2, id) = pwalls[i]
            
            # --- DIBUJO DE LÍNEAS DE PARED/PUERTA ---
            if id == 'IW':
                arrImageNilColorsRes = drawLine(arrImageNilColorsRes, x1, y1, x2, y2,
                                        listNilColors[idxNilColorInteriorWall], 3)
            if id == 'ID':
                arrImageNilColorsRes = drawLine(arrImageNilColorsRes, x1, y1, x2, y2,
                                        listNilColors[idxNilColorInsideDoor], 3)
            if id == 'ID_A':
                arrImageNilColorsRes = drawLine(arrImageNilColorsRes, x1, y1, x2, y2,
                                        listNilColors[idxNilColorAux1], 3)
                
                # --- VISUALIZACIÓN APERTURA (ID_A) ---
                # 1. Determinar Posición de la Bisagra analizando vecinos
                # El segmento actual es (y1, x1) -> (y2, x2)  [Tupla es (col, row)]
                
                len_before = 0
                len_after = 0
                
                # Chequear Vecino Anterior (i-1)
                # Debe terminar donde empieza este: Prev(y2, x2) == Curr(y1, x1)
                if i > 0:
                    py1, px1, py2, px2, pid = pwalls[i-1]
                    # Check conectividad (tolerancia 1px)
                    if abs(py2 - y1) <= 1 and abs(px2 - x1) <= 1 and 'IW' in pid:
                        len_before = math.hypot(py2-py1, px2-px1)
                
                # Chequear Vecino Siguiente (i+1)
                # Debe empezar donde termina este: Next(y1, x1) == Curr(y2, x2)
                if i < len(pwalls) - 1:
                    ny1, nx1, ny2, nx2, nid = pwalls[i+1]
                    if abs(ny1 - y2) <= 1 and abs(nx1 - x2) <= 1 and 'IW' in nid:
                        len_after = math.hypot(ny2-ny1, nx2-nx1)
                
                # LÓGICA DE BISAGRA:
                # - Si len_before < len_after: Puerta a la Izquierda -> Bisagra en Start (y1, x1)
                # - Si len_before > len_after: Puerta a la Derecha -> Bisagra en End (y2, x2)
                # - Empate: Default Start
                
                if len_before > len_after:
                     # Right Aligned -> Hinge at End
                     h_x, h_y = x2, y2
                else:
                     # Left Aligned or Center -> Hinge at Start (Default)
                     # NOTA: (x1, y1) suele ser Top/Left en la generación.
                     h_x, h_y = x1, y1
                     
                # Calcular Punto Indicador (ID_A = Arriba/Izquierda)
                # USUARIO: "El punto tiene que estar en la puerta, no pongas ofset ni nada"
                # Y AHORA: "se tiene que pintar tan solo un pixel en la puerta, nada mas"
                
                color_point = [0, 0, 0]
                p_x, p_y = h_x, h_y

                # Pintar 1 solo pixel
                px, py = int(p_x), int(p_y)
                if 0 <= px < height and 0 <= py < width:
                     arrImageNilColorsRes[px, py] = color_point


            if id == 'ID_B':
                arrImageNilColorsRes = drawLine(arrImageNilColorsRes, x1, y1, x2, y2,
                                        listNilColors[idxNilColorInsideDoor], 3)
                
                # --- VISUALIZACIÓN APERTURA (ID_B) ---
                len_before = 0
                len_after = 0
                
                if i > 0:
                    py1, px1, py2, px2, pid = pwalls[i-1]
                    if abs(py2 - y1) <= 1 and abs(px2 - x1) <= 1 and 'IW' in pid:
                        len_before = math.hypot(py2-py1, px2-px1)
                
                if i < len(pwalls) - 1:
                    ny1, nx1, ny2, nx2, nid = pwalls[i+1]
                    if abs(ny1 - y2) <= 1 and abs(nx1 - x2) <= 1 and 'IW' in nid:
                        len_after = math.hypot(ny2-ny1, nx2-nx1)
                
                if len_before > len_after:
                     h_x, h_y = x2, y2
                else:
                     h_x, h_y = x1, y1
                
                color_point = [0, 0, 0]
                # PINTAR EN LA PUERTA (SIN OFFSET)
                p_x, p_y = h_x, h_y

                # Pintar 1 solo pixel
                px, py = int(p_x), int(p_y)
                if 0 <= px < height and 0 <= py < width:
                     arrImageNilColorsRes[px, py] = color_point

                # ---------------------------------------------------------
        # BLOQUE DE EJECUCIÓN Y PINTADO (Run Script)
        # ---------------------------------------------------------
        
        result = solve_master_bedroom(gParam, lConnexComponents[1][2][0], lPerimeter, pwalls, debug=True)

        if result: 
            # Entramos aquí si hay ÉXITO TOTAL o ÉXITO PARCIAL (Solo cama)
                
            if result["success"]:
                print(f"   [OK] Solución Completa: {result['config_level']}")
            else:
                print(f"   [WARN] Solución Parcial: {result['config_level']} (No cupo el armario)")

            # Pintamos TODOS los items que vengan en la lista, sean 1 o 2
            for item in result["items"]:
                # Extraemos coordenadas
                x1, x2 = item["x1"], item["x2"]
                y1, y2 = item["y1"], item["y2"]
                    
                # Color: Rojo para cama, Verde para armario
                if "bed" in item["type"]:
                    color = [255, 100, 100] 
                else:
                    color = [100, 255, 100] 
                        
                # CLAMP: Aseguramos que no se salga de la imagen (Evita errores de índice)
                H_img, W_img, _ = arrImageNilColorsRes.shape
                x1_safe, x2_safe = max(0, x1), min(H_img, x2)
                y1_safe, y2_safe = max(0, y1), min(W_img, y2)
                    
                # Pintamos
                arrImageNilColorsRes[x1_safe:x2_safe, y1_safe:y2_safe] = color

        else:
            print("   [ERROR] Habitación inamueblable (Ni siquiera cabe una cama pequeña).")

        # ---------------------------------------------------------
        # DINING ROOM - Mesa en el centro
        # ---------------------------------------------------------
        # Índice 4 = DiningR (color ocre/magenta según dataset)
        if len(lConnexComponents) > 4 and len(lConnexComponents[4][2]) > 0:
            dining_pixels = lConnexComponents[4][2][0]  # Primera componente conexa del DiningR
                
            result_dining = solve_dining_room(gParam, dining_pixels, lPerimeter, pwalls, debug=True)
                
            if result_dining and result_dining["success"]:
                print(f"   [OK] Dining Room: Mesa {result_dining['config_level']} colocada")
                    
                for item in result_dining["items"]:
                    x1, x2 = item["x1"], item["x2"]
                    y1, y2 = item["y1"], item["y2"]
                        
                    # Color azul para mesa de comedor
                    color = [100, 100, 255]
                        
                    H_img, W_img, _ = arrImageNilColorsRes.shape
                    x1_safe, x2_safe = max(0, x1), min(H_img, x2)
                    y1_safe, y2_safe = max(0, y1), min(W_img, y2)
                        
                    arrImageNilColorsRes[x1_safe:x2_safe, y1_safe:y2_safe] = color
            else:
                print("   [WARN] Dining Room: No se pudo colocar mesa")

            # ---------------------------------------------------------
            # KITCHEN - Encimera y Nevera
            # ---------------------------------------------------------
            # Índice 2 = Kitchen (color verde)
        if len(lConnexComponents) > 2 and len(lConnexComponents[2][2]) > 0:
            kitchen_pixels = lConnexComponents[2][2][0]
                
            result_kitchen = solve_kitchen(gParam, kitchen_pixels, lPerimeter, pwalls, debug=True)
                
            if result_kitchen:
                if result_kitchen["success"]:
                    print(f"   [OK] Kitchen: {result_kitchen['config_level']}")
                else:
                    print(f"   [WARN] Kitchen: {result_kitchen['config_level']} (incompleto)")
                    
                for item in result_kitchen["items"]:
                    x1, x2 = item["x1"], item["x2"]
                    y1, y2 = item["y1"], item["y2"]
                        
                    # Colores: Marrón para encimera, Blanco para nevera
                    if item["type"] == "countertop":
                        color = [139, 90, 43]  # Marrón
                    else:  # fridge
                        color = [220, 220, 220]  # Gris claro (nevera)
                        
                    H_img, W_img, _ = arrImageNilColorsRes.shape
                    x1_safe, x2_safe = max(0, x1), min(H_img, x2)
                    y1_safe, y2_safe = max(0, y1), min(W_img, y2)
                        
                    arrImageNilColorsRes[x1_safe:x2_safe, y1_safe:y2_safe] = color
            else:
                print("   [WARN] Kitchen: No se pudo colocar muebles")

            # ---------------------------------------------------------
            # BATHROOM - Baño (Ducha, WC, Pica)
            # ---------------------------------------------------------
            # Índice 3 = Bathroom (color cyan/azul-verdoso)
        if len(lConnexComponents) > 3 and len(lConnexComponents[3][2]) > 0:
            bathroom_pixels = lConnexComponents[3][2][0]
                
            result_bathroom = solve_bathroom(gParam, bathroom_pixels, lPerimeter, pwalls, debug=True)
                
            if result_bathroom:
                if result_bathroom["success"]:
                    print(f"   [OK] Bathroom: {result_bathroom['config_level']}")
                else:
                    print(f"   [WARN] Bathroom: {result_bathroom['config_level']} (incompleto)")
                    
                for item in result_bathroom["items"]:
                    x1, x2 = item["x1"], item["x2"]
                    y1, y2 = item["y1"], item["y2"]
                        
                    # Colores
                    if item["type"] == "shower":
                        color = [0, 255, 255]    # Cyan (Ducha)
                    elif item["type"] == "toilet":
                        color = [0, 0, 255]      # Azul (WC)
                    elif item["type"] == "sink":
                        color = [139, 90, 43] # Marron (Pica)
                    else:
                        color = [128, 128, 128]  # Gris (Fallback)
                        
                    H_img, W_img, _ = arrImageNilColorsRes.shape
                    x1_safe, x2_safe = max(0, x1), min(H_img, x2)
                    y1_safe, y2_safe = max(0, y1), min(W_img, y2)
                        
                    arrImageNilColorsRes[x1_safe:x2_safe, y1_safe:y2_safe] = color
            else:
                print("   [WARN] Bathroom: No se pudo resolver la colocación")

        # ---------------------------------------------------------
        # LIVING ROOM - Salón (Sofá + TV)
        # ---------------------------------------------------------
        # Índice 5 = LivingRoom (color rojo/naranja según dataset, en listaNilColors puede variar)
        if len(lConnexComponents) > 5 and len(lConnexComponents[0][2]) > 0:
            living_pixels = lConnexComponents[0][2][0]
                
            result_living = solve_living_room(gParam, living_pixels, lPerimeter, pwalls, debug=True)
                
            if result_living and result_living["success"]:
                print(f"   [OK] LivingRoom: {result_living['config_level']}")
                    
                for item in result_living["items"]:
                    x1, x2 = item["x1"], item["x2"]
                    y1, y2 = item["y1"], item["y2"]
                        
                    if item["type"] == "sofa":
                        color = [0, 0, 128] # Azul Oscuro
                    elif item["type"] == "tv_unit":
                        color = [50, 50, 50] # Gris Oscuro
                    else:
                        color = [100, 100, 100]
                            
                    H_img, W_img, _ = arrImageNilColorsRes.shape
                    x1_safe, x2_safe = max(0, x1), min(H_img, x2)
                    y1_safe, y2_safe = max(0, y1), min(W_img, y2)
                        
                    arrImageNilColorsRes[x1_safe:x2_safe, y1_safe:y2_safe] = color
            else:
                print("   [WARN] LivingRoom: No se pudo colocar Sofá/TV")

        # ---------------------------------------------------------
        # SINGLE BEDROOM - Dormitorio Individual
        # ---------------------------------------------------------
        # Índice 6 = Single Bedroom (o segundo dormitorio)
        if len(lConnexComponents) > 6 and len(lConnexComponents[6][2]) > 0:
            single_pixels = lConnexComponents[6][2][0]
                
            result_single = solve_single_bedroom(gParam, single_pixels, lPerimeter, pwalls, debug=True)
                
            if result_single and result_single["success"]:
                print(f"   [OK] SingleBedroom: {result_single['config_level']}")
                    
                for item in result_single["items"]:
                    x1, x2 = item["x1"], item["x2"]
                    y1, y2 = item["y1"], item["y2"]
                        
                    if item["type"] == "bed_single":
                        color = [100, 149, 237]  # Cornflower Blue (Cama)
                    elif item["type"] == "wardrobe":
                        color = [139, 69, 19]    # Saddle Brown (Armario)
                    else:
                        color = [150, 150, 150]
                            
                    H_img, W_img, _ = arrImageNilColorsRes.shape
                    x1_safe, x2_safe = max(0, x1), min(H_img, x2)
                    y1_safe, y2_safe = max(0, y1), min(W_img, y2)
                        
                    arrImageNilColorsRes[x1_safe:x2_safe, y1_safe:y2_safe] = color
            else:
                print("   [WARN] SingleBedroom: No se pudo colocar muebles")

    # Add the corners (if required) from lCorners
    if lCorners is not None:
        for i in range(len(lCorners)):
            dataCorner = lCorners[i]
            typeCorner = dataCorner[0]
            sizeCorner = dataCorner[1]
            xC         = dataCorner[2][0]
            yC         = dataCorner[2][1]
            #pctEqual   = dataCorner[3]
            #ldataCorIm = dataCorner[4]
            for dx in np.arange(-dxyCorner,dxyCorner):
                for dy in np.arange(-dxyCorner,dxyCorner):
                    for k in range(3):
                        arrImageNilColorsRes[xC+dx,yC+dy,k] = listNilColors[idxNilColorAux1][k]

    ### Add a particular point (for testing)
    if plotPoint:
        #(xC,yC)  = (gParam['MinFrameX'],10)  ### First component: vertical / Second component: horizontal
        (xC,yC)  = (30,30)
        for dx in np.arange(-dxyPoint,dxyPoint+1):
            for dy in np.arange(-dxyPoint,dxyPoint+1):
                for k in range(3):
                    arrImageNilColorsRes[xC+dx,yC+dy,k] = listNilColors[idxNilColorAux2][k]

                        
    ### List of integers
    intNilColorsResImage = getIntImageFromArrImage(arrImageNilColorsRes)
                
    return intNilColorsResImage


####################################################################################################################
### RESULTING IMAGES (with centroids + adjacencies + front door + corners)
####################################################################################################################

def plotSequenceResults (gParam, oriPilImage, intImageNilColors, lConnexComponents,
                         lCentroids=None, lAdjacencies=None, lCorners=None, lPerimeter=None):


    height = gParam['height']
    width  = gParam['width']

    nPlots = 2
    if lCentroids   is not None: nPlots += 1
    if lAdjacencies is not None: nPlots += 1
    if lPerimeter   is not None: nPlots += 1
    if lCorners     is not None: nPlots += 1

    fig, axs = plt.subplots(nrows=1, ncols=nPlots, figsize=(20, 30)) #,layout="constrained")
    nPlot = -1

    nPlot += 1
    axs[nPlot].imshow(oriPilImage); axs[nPlot].title.set_text("Original Image")
    #
    nPlot += 1
    newPilImage = Image.frombytes('RGB', (width, height), bytes(intImageNilColors))
    axs[nPlot].imshow(newPilImage); axs[nPlot].title.set_text("NilColors Image")
    #
    if lCentroids is not None:
        nPlot += 1
        intResImage = constructResultingImage (gParam, lConnexComponents, \
          lCentroids=lCentroids, lAdjacencies=None, lCorners=None, lPerimeter=None)
        pilResImage = Image.frombytes('RGB', (width, height), bytes(intResImage))
        axs[nPlot].imshow(pilResImage); axs[nPlot].title.set_text("Centroids")
    #
    if lAdjacencies is not None:
        nPlot += 1
        intResImage = constructResultingImage (gParam, lConnexComponents, \
          lCentroids=lCentroids, lAdjacencies=lAdjacencies, lCorners=None, lPerimeter=None)
        pilResImage = Image.frombytes('RGB', (width, height), bytes(intResImage))
        axs[nPlot].imshow(pilResImage); axs[nPlot].title.set_text("+ Adjacencies")
    #
    if lCorners is not None:
        nPlot += 1
        intResImage = constructResultingImage (gParam, lConnexComponents, \
          lCentroids=lCentroids, lAdjacencies=lAdjacencies, lCorners=lCorners, lPerimeter=None)
          #lCentroids=lCentroids, lAdjacencies=lAdjacencies, lCorners=lCorners, lPerimeter=None)
        pilResImage = Image.frombytes('RGB', (width, height), bytes(intResImage))
        axs[nPlot].imshow(pilResImage); axs[nPlot].title.set_text("+ Corners")
    #
    if lPerimeter is not None:
        nPlot += 1
        intResImage = constructResultingImage (gParam, lConnexComponents, \
          lCentroids=lCentroids, lAdjacencies=lAdjacencies, lCorners=None, lPerimeter=lPerimeter)
        pilResImage = Image.frombytes('RGB', (width, height), bytes(intResImage))
        axs[nPlot].imshow(pilResImage); axs[nPlot].title.set_text("+ Perimeter")
    #
    return


####################################################################################################################
### Transformation functions
####################################################################################################################

def getArrImageFromPilImage (pilImage):
    ### Converts a PIL image to numpy ARRAY of size [width,height,nchannels]
    ###   In each position [i][j][:] we have the [r,g,b] values
    arrImage = np.asarray(pilImage, dtype=np.int16)  
    #arrImage = np.asarray(pilImage)
    return arrImage

###

def getRgbImageFromArrImage (arrImage):
    ### Obtains the LIST of TUPLES with rgb colors of the numpy array image
    [width,height,nchannels] = arrImage.shape
    #print(width,height,nchannels)
    rgbImage = []
    for x in range(height):     ### Reads from top to bottom, from left to right
        for y in range(width):
            rgbImage.append( (arrImage[x][y][0], arrImage[x][y][1], arrImage[x][y][2]) )
    return rgbImage

###

def getArrImageFromRgbImage (gParam, rgbImage):
    ### Obtains the ARRAY of size [width,height,nchannels] from the list of TUPLES with rgb colors of the image
    height    = gParam['height']
    width     = gParam['width']
    nchannels = 3
    arrImage = np.zeros([width,height,nchannels],int)
    for x in range(height):     ### Reads from top to bottom, from left to right
        for y in range(width):
            rgbColor = rgbImage[x*width + y]
            for k in range(nchannels):
                arrImage[x,y,k] = rgbColor[k]
    return arrImage

###

def getIntImageFromRgbImage (rgbImage):
    ### Obtains the LIST of integers from the list of TUPLES with rgb colors of the image
    intImage = []
    for c in rgbImage:
        intImage.extend(c)
    return intImage

###

def getRgbImageFromIntImage (intImage):
    ### Obtains the LIST of TUPLES with rgb colors from the LIST of INTEGERS of the image
    rgbImage = []
    for n in range(len(intImage)//3):
        rgbImage.append( (intImage[3*n], intImage[3*n+1], intImage[3*n+2]) )
    return rgbImage

###

def getIntImageFromArrImage (arrImage):
    ### Obtains the LIST of integers from the numpy ARRAY of size [width,height,nchannels]
    [width,height,nchannels] = arrImage.shape
    intImage = []
    for x in range(height):     ### Reads from top to bottom, from left to right
        for y in range(width):
            for k in range(nchannels):
                intImage.append(arrImage[x,y,k])
    return intImage


####################################################################################################################
### Standard functions
####################################################################################################################

def norm1_3D (x, y):
    s = abs(x[0]-y[0]) + abs(x[1]-y[1]) + abs(x[2]-y[2])
    return s / 3.0

###

def norm2_3D (x, y):
    s = (x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2
    return np.sqrt(s) / 3.0

###

def boundingBox (lPositions):
    XMin = 9999999; YMin = XMin; XMax = -9999999; YMax = XMax;
    for (x,y) in lPositions:
        if x < XMin: XMin = x
        if y < YMin: YMin = y
        if x > XMax: XMax = x
        if y > YMax: YMax = y

    return XMin, YMin, XMax, YMax

###

def saveStringList (filename,lString):
    txtfile = open(filename, "w")
    for n in range(len(lString)):
        s = lString[n]        
        if n > 0:
            txtfile.write('\n')
        txtfile.write(s)
    #txtfile.write('\n')    ### Descomentar si se quiere un carry return al final (Graph2Plan no lo quiere)
    txtfile.close()

###

def saveDataToMatlab(pfileName,data):
    import scipy.io as scipyIO
    #print(data)
    fileName = pfileName + '.mat'
    #
    scipyIO.savemat(fileName,{'data':data})
    #sio.savemat('./data/data_train_converted.mat',{'data':data_converted,'nameList':names_train,'trainTF':trainTF})

###

def saveDataToPickle(pfileName,data):
    import pickle
    #print(data)
    fileName = pfileName + '.pkl'
    #
    with open(fileName, 'wb') as f:
        pickle.dump(data, f)

    ### with open('fileName.pkl','rb') as f:
    ###    x = pickle.load(f)
        

####################################################################################################################
### Assignation of the nearest color of the predefined list used in the original transformation for every pixel
####################################################################################################################

def transformToNearestListColors (gParam, rgbImage, nListColors, listColors, idxcolorBackground):

    strNorm = gParam['normDistColors']
    if   strNorm == 'norm1_3D':
        norm = norm1_3D
    elif strNorm == 'norm2_3D':
        norm = norm2_3D
    else:
        norm = norm2_NotImplemented
    
    ### For each rgb color in the image, obtains the nearest predefined color in listColors
    rgbImageListColors = []
    for c1 in rgbImage:
        minsim  = 9999999
        iminsim = -1
        for n in range(nListColors):
            c2 = listColors[n]
            sim = norm(c1,c2)
            if sim < minsim:
                minsim  = sim
                iminsim = n
        rgbImageListColors.append(listColors[iminsim]) ### List of tuples

    return rgbImageListColors

###

def obtainImagesNilColors (gParam, oriPilImage):
    
    height                = gParam['height']
    width                 = gParam['width']
    listNilColors         = gParam['listNilColors']
    nListNilColors        = gParam['nListNilColors']
    idxNilColorBackground = gParam['idxNilColorBackground']
    #
    extraFrameX           = gParam['extraFrameX']
    extraFrameY           = gParam['extraFrameY'] 

    listColorsOriImage = oriPilImage.getcolors(maxcolors=2**16)
    print("   Number of different colors in the original image:",len(listColorsOriImage))

    ### Obtains the rgbOriImage from the oriPilImage
    rgbOriImage       = getRgbImageFromArrImage(getArrImageFromPilImage(oriPilImage))
    ### Transforms the rbgOriImage according to the nearest colors of listNilColors
    rgbImageNilColors = transformToNearestListColors (gParam, rgbOriImage, nListNilColors, listNilColors, idxNilColorBackground)

    ### Selects the largest connex components different from background
    lConnexComponentsNoBackgroundAll = getConnexComponentsNoBackground (gParam, rgbImageNilColors)
    lConnexComponentsNoBackground    = getLargestConnexComponents (lConnexComponentsNoBackgroundAll, N=1)
    
    ### Changes the color to background outside the selected connex components
    print('   Changing to background everything outside the largest connex components...')
    ## First constructs a boolean matrix with True in the pixels that are background
    mPixelsBackground = np.ndarray([width,height],bool)
    mPixelsBackground[:,:] = True
    for connexComponentsNoBackground in lConnexComponentsNoBackground:
        for (x,y) in connexComponentsNoBackground:
            mPixelsBackground[x,y] = False
    ## Now puts the background color in True positions
    for x in range(height):        ### Reads from top to bottom, from left to right
        for y in range(width):
            if mPixelsBackground[x,y]:
                rgbImageNilColors[x*width + y] = listNilColors[idxNilColorBackground]

    ### Computes the frame of the picture
    print('   Computing the frame of the picture...')
    XMin = 9999999; XMax = -9999999; YMin = XMin; YMax = XMax;
    for x in range(height):        ### Reads from top to bottom, from left to right
        for y in range(width):
            if not mPixelsBackground[x,y]:
                if x < XMin: XMin = x
                if x > XMax: XMax = x
                if y < YMin: YMin = y
                if y > YMax: YMax = y
    #print(XMin, XMax, YMin, YMax)
    ## Now compute the final values of the frame
    gParam['MinFrameX'] = max(XMin - extraFrameX, 0)
    gParam['MaxFrameX'] = min(XMax + extraFrameX, height)
    gParam['MinFrameY'] = max(YMin - extraFrameY, 0)
    gParam['MaxFrameY'] = min(YMax + extraFrameY, width)

    ### Converts the list of tuples into the list of integers
    intImageNilColors = getIntImageFromRgbImage(rgbImageNilColors)

    ### Converts the list of tuples into the array of size [width,height,nchannels]
    arrImageNilColors = getArrImageFromRgbImage (gParam, rgbImageNilColors)

    #print(len(intImageNilColors))    ### intImageNilColors is a LIST of width * height * nchannels  integers
    #print(len(rgbImageNilColors))    ### rgbImageNilColors is a LIST of width * height              tuples (RGB)
    #print(arrImageNilColors.shape)   ### arrImageNilColors is an ARRAY of [width,height,3] integers

    return gParam, intImageNilColors, rgbImageNilColors, arrImageNilColors


####################################################################################################################
### Search of the connex componentes with different criteria:
###   - Same Color (connex components of the same color)
###   - No Background
###   - ...
####################################################################################################################

def nonVisitedPosition (gParam, visitedPositions, nIters=1):
    if  nIters is None or nIters <= 0:
        nIters = 1

    height = gParam['height']
    width  = gParam['width']

    n = 0
    for x in range(height):
        for y in range(width):
            if not visitedPositions[x,y]:
                n += 1
                if n == nIters:
                    return (x,y)
    return (None,None)

###

def getLargestConnexComponents (lConnexComponents, N):

    #print(lConnexComponents)
    pq = PriorityQueue()
    for c in range(len(lConnexComponents)):
        s = len(lConnexComponents[c])
        pq.put((-s,c))  ### The minus is for reverse the queue

    lConnexComponentsSelected = []
    i = 0
    while i < N and not pq.empty():
        idxCC = pq.get()[1]
        lConnexComponentsSelected.append(lConnexComponents[idxCC])
        i += 1

    ### Returns the N largest connex components
    return lConnexComponentsSelected

###

def getConnexComponentFrom (gParam, x, y, visitedPositions):

    height = gParam['height']
    width  = gParam['width']

    lConnexComponent = []  ### list of pixels
    qPos = []              ### queue for the search of the pixels in the connex component
    qPos.append((x,y))
    while len(qPos) > 0:
        (x,y) = qPos.pop(0)
        if not visitedPositions[x,y]:
            visitedPositions[x,y] = True
            lConnexComponent.append((x,y))
            xx = x-1;  yy = y;
            if xx >= 0 and xx < width and yy >= 0 and yy < height and not visitedPositions[xx,yy]:
                qPos.append((xx,yy))
            xx = x+1;  yy = y;
            if xx >= 0 and xx < width and yy >= 0 and yy < height and not visitedPositions[xx,yy]:
                qPos.append((xx,yy))
            xx = x;    yy = y-1;
            if xx >= 0 and xx < width and yy >= 0 and yy < height and not visitedPositions[xx,yy]:
                qPos.append((xx,yy))
            xx = x;    yy = y+1;
            if xx >= 0 and xx < width and yy >= 0 and yy < height and not visitedPositions[xx,yy]:
                qPos.append((xx,yy))
        #print("--- ",len(qPos),len(lConnexComponent))
    #
    return lConnexComponent, visitedPositions

###

def getConnexComponentsAllColors (gParam, rgbImage):

    print('   Computing connex components by color...')
    print('   ',end="")

    height         = gParam['height']
    width          = gParam['width']
    listNilColors  = gParam['listNilColors']
    nListNilColors = gParam['nListNilColors']

    lConnexComponents = []             ### list (one element for every index/color/list) of tuples
    for n in range(nListNilColors):
        c1 = listNilColors[n]
        #
        lConnexComponentsColor = []    ### list (one element for every connex component) of lists of pixels
        #
        visitedPositions = np.ndarray([width,height],bool)  ### boolean matrix to control the search
        for x in range(height):        ### Reads from top to bottom, from left to right
            for y in range(width):
                c2 = rgbImage[x*width + y]
                if c1 == c2:
                    visitedPositions[x,y] = False
                else:
                    visitedPositions[x,y] = True
        #
        (x,y) = nonVisitedPosition (gParam,visitedPositions)
        while x is not None:
            #print(x,y)
            lConnexComponentColorPixels, visitedPositions = getConnexComponentFrom (gParam,x,y,visitedPositions)
            #print(lConnexComponentColorPixels)
            lConnexComponentsColor.append(lConnexComponentColorPixels)
            (x,y) = nonVisitedPosition (gParam,visitedPositions)
        #
        lConnexComponents.append((n,c1,lConnexComponentsColor))  ### Tuple (index, color, connex components list)

        print(" .",len(lConnexComponents),end="")
    print()

    return lConnexComponents

###

def getConnexComponentsNoBackground (gParam, rgbImage):

    print('   Computing connex components different from background...')
    print('   ',end="")

    height                = gParam['height']
    width                 = gParam['width']
    listNilColors         = gParam['listNilColors']
    idxNilColorBackground = gParam['idxNilColorBackground']

    lConnexComponents = []  ### list (one element for every connex component) of lists
    visitedPositions = np.ndarray([width,height],bool)  ### boolean matrix to control the search
    for x in range(height):        ### Reads from top to bottom, from left to right
        for y in range(width):
            c2 = rgbImage[x*width + y]
            if c2 == listNilColors[idxNilColorBackground]:  ### We only want connex components "with a color different from background"
                visitedPositions[x,y] = True
            else:
                visitedPositions[x,y] = False
    #
    (x,y) = nonVisitedPosition (gParam,visitedPositions)
    while x is not None:
        lConnexComponentPixels, visitedPositions = getConnexComponentFrom (gParam,x,y,visitedPositions)
        #print(lConnexComponentColorPixels)
        lConnexComponents.append(lConnexComponentPixels)
        (x,y) = nonVisitedPosition (gParam,visitedPositions)
        print(" .",len(lConnexComponents),end="")
    #
    print()

    return lConnexComponents


###

def getConnexComponentsFrontDoor (gParam, rgbImage):

    print('   Computing connex components for the Front Door...')
    print('   ',end="")

    height               = gParam['height']
    width                = gParam['width']
    listNilColors        = gParam['listNilColors']
    idxNilColorFrontDoor = gParam['idxNilColorFrontDoor']

    lConnexComponents = []  ### list (one element for every connex component) of lists
    visitedPositions = np.ndarray([width,height],bool)  ### boolean matrix to control the search
    for x in range(height):        ### Reads from top to bottom, from left to right
        for y in range(width):
            c2 = rgbImage[x*width + y]
            if c2 == listNilColors[idxNilColorFrontDoor]:  ### We only want connex components with this color
                visitedPositions[x,y] = False
            else:
                visitedPositions[x,y] = True
    #
    (x,y) = nonVisitedPosition (gParam,visitedPositions)
    while x is not None:
        lConnexComponentPixels, visitedPositions = getConnexComponentFrom (gParam,x,y,visitedPositions)
        #print(lConnexComponentColorPixels)
        lConnexComponents.append(lConnexComponentPixels)
        (x,y) = nonVisitedPosition (gParam,visitedPositions)
        print(" .",len(lConnexComponents),end="")
    #
    print()

    return lConnexComponents


####################################################################################################################
### Obtains the centroids of the "rooms with centroid", by selecting the connex components of colors
### in "listNilColorsRoomsCentroid" with the minimum number of pixels required "minPixelsRoomCentroid"
####################################################################################################################

def sanityCheckCentroids (gParam, rgbImage, lCentroids, lConnexComponentsCentroids):

    height              = gParam['height']
    width               = gParam['width']
    idxNilColorEntrance = gParam['idxNilColorEntrance']
    idxNilColorBalcony  = gParam['idxNilColorBalcony']
    nPixelsPerMeter     = gParam['nPixelsPerMeter']
    nPixelsPerMeter2    = gParam['nPixelsPerMeter2']

    sanityCheckOKCentroids = True

    ### RESTRICCION RELAJADA: se añade la excepción de la terraza
    ### Every centroid has the same color as its connex component
    if sanityCheckOKCentroids:
        for idxC in range(len(lCentroids)):
            centroid           = lCentroids[idxC]
            idxColorCentroid   = centroid[0]
            if idxColorCentroid != idxNilColorEntrance:     ### El pasillo puede tener formas muy diferentes de un cuadrado
                if idxColorCentroid != idxNilColorBalcony:  ### RESTRICCION RELAJADA: La terraza también...
                    colorCentroidCC    = centroid[1]
                    (xC,yC)            = centroid[2]
                    colorCentroidImage = rgbImage[xC*width + yC]
                    if colorCentroidCC != colorCentroidImage:
                        sanityCheckOKCentroids = False
                        print(xC,yC,colorCentroidCC,colorCentroidImage)
                        print('   ERROR: A centroid has not the same color than its connex component'); #sys.exit()

    ### Condiciones de habitabilidad
    ##
    nPixelsLivingKitchenDining = 0
    pixelsLivingDining = []
    pixelsKitchen      = []
    pixelsBath         = []
    pixelsEntrance     = []
    lOnlyRooms         = []
    if sanityCheckOKCentroids:
        for idxC in range(len(lCentroids)):
            centroid                 = lCentroids[idxC]
            idxColorCentroid         = centroid[0]
            connexComponentsCentroid = lConnexComponentsCentroids[idxC][2]
            if idxColorCentroid == 0:   ### LivingR
                nPixelsLivingKitchenDining += len(connexComponentsCentroid)
                pixelsLivingDining.extend(connexComponentsCentroid)
            if idxColorCentroid == 2:   ### Kitchen
                nPixelsLivingKitchenDining += len(connexComponentsCentroid)
                pixelsKitchen.extend(connexComponentsCentroid)
            if idxColorCentroid == 3:   ### BathR
                pixelsBath.extend(connexComponentsCentroid)
            if idxColorCentroid == 4:   ### DiningR
                nPixelsLivingKitchenDining += len(connexComponentsCentroid)
                pixelsLivingDining.extend(connexComponentsCentroid)
            if idxColorCentroid == 9:   ### Entrance
                pixelsEntrance.extend(connexComponentsCentroid)
            if idxColorCentroid in [1, 5, 6, 7]:   ### MasterR, ChildR, SecondR, GuestR
                lOnlyRooms.append(idxC)

    ### RESTRICCION RELAJADA: de 20 a 15
    ## Living+Kitchen+Dining
    if sanityCheckOKCentroids:
        # Superficie mínina Living+Kitchen+Dining = 20m^2
        if nPixelsLivingKitchenDining < 15*nPixelsPerMeter2:
            sanityCheckOKCentroids = False
            print('   ERROR: Superficie mínima Living+Kitchen+Dining < 15m^2',nPixelsLivingKitchenDining/nPixelsPerMeter2); #sys.exit()

    ## Living+Dining
    if sanityCheckOKCentroids:
        # Anchura mínina Living+Dining = 1.60m en las dos direcciones
        x0bbLD, y0bbLD, x1bbLD, y1bbLD = boundingBox(pixelsLivingDining)
        if not (np.abs(x0bbLD - x1bbLD) >= 1.60 * nPixelsPerMeter  and  np.abs(y0bbLD - y1bbLD) >= 1.60 * nPixelsPerMeter):
            sanityCheckOKCentroids = False
            print('ERROR: Anchura mínima Living+Dining < 1.60m',np.abs(x0bbLD - x1bbLD)/nPixelsPerMeter,np.abs(y0bbLD - y1bbLD)/nPixelsPerMeter); #sys.exit()
        ### RESTRICCION RELAJADA: de 2.80 a 2.20
        # Diámetro circulo mínino Living+Dining = 2.80m
        if not (np.abs(x0bbLD - x1bbLD) >= 2.20 * nPixelsPerMeter  and  np.abs(y0bbLD - y1bbLD) >= 2.20 * nPixelsPerMeter):
            sanityCheckOKCentroids = False
            print('   ERROR: Diámetro círculo mínimo Living+Dining < 2.20m',np.abs(x0bbLD - x1bbLD)/nPixelsPerMeter,np.abs(y0bbLD - y1bbLD)/nPixelsPerMeter); #sys.exit()

    ### RESTRICCION ELIMINADA
    ## Kitchen
    #if sanityCheckOKCentroids:
    #    # Anchura mínina Kitchen = 1.60m en las dos direcciones
    #    x0bbK, y0bbK, x1bbK, y1bbK = boundingBox(pixelsKitchen)
    #    if not (np.abs(x0bbK - x1bbK) >= 1.60 * nPixelsPerMeter  and  np.abs(y0bbK - y1bbK) >= 1.60 * nPixelsPerMeter):
    #        sanityCheckOKCentroids = False
    #        print('   ERROR: Anchura mínima Kitchen < 1.60m',np.abs(x0bbK - x1bbK)/nPixelsPerMeter,np.abs(y0bbK - y1bbK)/nPixelsPerMeter); #sys.exit()

    ### RESTRICCION RELAJADA: de 1.10 a 0.80
    ## Bathroom
    if sanityCheckOKCentroids:
        # Diámetro circulo mínino Bathroom = 1.20m en las dos direcciones (Lluís hace trampas y pone 1.10 en lugar de 1.20)
        x0bbB, y0bbB, x1bbB, y1bbB = boundingBox(pixelsBath)
        if not (np.abs(x0bbB - x1bbB) >= 0.80 * nPixelsPerMeter  and  np.abs(y0bbB - y1bbB) >= 0.80 * nPixelsPerMeter):
            sanityCheckOKCentroids = False
            print('   ERROR: Diámetro círculo mínimo BathRoom < 0.80m',np.abs(x0bbB - x1bbB)/nPixelsPerMeter,np.abs(y0bbB - y1bbB)/nPixelsPerMeter); #sys.exit()

    ### RESTRICCION RELAJADA: de 1.00 a 0.60
    ## Entrance
    if sanityCheckOKCentroids:
        # Anchura mínina Entrance = 1.00m en las dos direcciones
        x0bbE, y0bbE, x1bbE, y1bbE = boundingBox(pixelsEntrance)
        if not (np.abs(x0bbE - x1bbE) >= 0.60 * nPixelsPerMeter  and  np.abs(y0bbE - y1bbE) >= 0.60 * nPixelsPerMeter):
            sanityCheckOKCentroids = False
            print('   ERROR: Anchura mínima Entrance < 0.60m',np.abs(x0bbE - x1bbE)/nPixelsPerMeter,np.abs(y0bbE - y1bbE)/nPixelsPerMeter); #sys.exit()

    ### RESTRICCION RELAJADA: de 2.60 a 2.20 y de 2.00 a 1.70
    ## Rooms
    if sanityCheckOKCentroids:
        # Solo una - anchura minima = 2.60m / Más de una - anchura minima = 2.00m cada una (en las dos direcciones)
        if   len(lOnlyRooms) == 1:
            minWideR = 2.20
        elif len(lOnlyRooms) > 1:
            minWideR = 1.70
        for idxC in lOnlyRooms:
            centroid             = lCentroids[idxC]
            idxColorCentroid     = centroid[0]
            connexComponentsRoom = lConnexComponentsCentroids[idxC][2]
            x0bbR, y0bbR, x1bbR, y1bbR = boundingBox(connexComponentsRoom)
            if not (np.abs(x0bbR - x1bbR) >= minWideR * nPixelsPerMeter  and  np.abs(y0bbR - y1bbR) >= minWideR * nPixelsPerMeter):
                sanityCheckOKCentroids = False
                print('   ERROR: Anchura mínina Room',idxColorCentroid,'insuficiente',minWideR,centroid,np.abs(x0bbR - x1bbR)/nPixelsPerMeter,np.abs(y0bbR - y1bbR)/nPixelsPerMeter); #sys.exit()

    return sanityCheckOKCentroids


def getConnexComponentsCentroids (gParam, rgbImage):

    print('   Computing centroids...')

    height                                  = gParam['height']
    width                                   = gParam['width']
    listColorsRoomsWithCentroid             = gParam['listNilColorsRoomsWithCentroid']
    listNilColorsRoomsWithCentroidSmall     = gParam['listNilColorsRoomsWithCentroidSmall']
    minPixelsConnexComponentCentroidSmall   = gParam['minPixelsConnexComponentCentroidSmall']
    listNilColorsRoomsWithCentroidLarge     = gParam['listNilColorsRoomsWithCentroidLarge']
    minPixelsConnexComponentCentroidLarge   = gParam['minPixelsConnexComponentCentroidLarge']
    listNilColorsRoomsWithCentroidAnySize   = gParam['listNilColorsRoomsWithCentroidAnySize']
    minPixelsConnexComponentCentroidAnySize = gParam['minPixelsConnexComponentCentroidAnySize']
    
    lConnexComponents = getConnexComponentsAllColors (gParam, rgbImage)

    lCentroids = []
    lConnexComponentsCentroids = []
    for n in listColorsRoomsWithCentroid:
        idxColor = lConnexComponents[n][0]   ### idxColor is the type of the room
        rgbColor = lConnexComponents[n][1]
        lConnexComponentsColor = lConnexComponents[n][2]
        for m in range(len(lConnexComponentsColor)):
            nPixelsConnexComponentsColor = len(lConnexComponentsColor[m])
            #print(idxColor,nPixelsConnexComponentsColor,minPixelsConnexComponentCentroidSmall,minPixelsConnexComponentCentroidLarge)
            ### Only considers connex components larger than the minimum allowed (that depends on the type of room)
            if (idxColor in listNilColorsRoomsWithCentroidSmall    and  nPixelsConnexComponentsColor > minPixelsConnexComponentCentroidSmall) or \
               (idxColor in listNilColorsRoomsWithCentroidLarge    and  nPixelsConnexComponentsColor > minPixelsConnexComponentCentroidLarge) or \
               (idxColor in listNilColorsRoomsWithCentroidAnySize  and  nPixelsConnexComponentsColor > minPixelsConnexComponentCentroidAnySize):
                xC = 0
                yC = 0
                for (x,y) in lConnexComponentsColor[m]:
                    xC += x
                    yC += y
                xC //= nPixelsConnexComponentsColor
                yC //= nPixelsConnexComponentsColor
                lConnexComponentsCentroids.append( (idxColor,rgbColor, lConnexComponentsColor[m]) )
                lCentroids.append( (idxColor, rgbColor, (xC,yC)) )

    return lCentroids, lConnexComponentsCentroids, lConnexComponents


####################################################################################################################
### Search of the adjacencies among the connex components of the centroids
####################################################################################################################

def adjacencyCentroidFound (gParam, x, y, dx, dy, visitedPositions, mapPointsIdxCC, interiorPointsCC):
    
    height             = gParam['height']
    width              = gParam['width']
    maxNegMapPointsAdj = gParam['maxNegMapPointsAdj']    
    
    idxCentroid1 = mapPointsIdxCC[x,y]

    adjacencyFound = False
    idxCentroid2   = None
    #
    xx = x; yy = y;
    endSearch = False
    nNegMapPoints = 0    ### Number of consecutive negative values in mapPointsIdxCC
    while not endSearch:
        xx += dx; yy+= dy
        if xx < 0 or yy < 0 or xx >= height or yy >= width:
            endSearch = True        ### Out of bounds
        elif interiorPointsCC[xx,yy]:
            endSearch = True        ### No adjacency in this way
        elif mapPointsIdxCC[xx,yy] >= 0 and idxCentroid1 == mapPointsIdxCC[xx,yy]:
            endSearch = True        ### No adjacency in this way
        elif mapPointsIdxCC[xx,yy] >= 0 and idxCentroid1 != mapPointsIdxCC[xx,yy]:
            endSearch = True
            adjacencyFound = True   ### Adjacency found!
            idxCentroid2   = mapPointsIdxCC[xx,yy]
        elif mapPointsIdxCC[xx,yy] < 0:
            nNegMapPoints += 1
            if nNegMapPoints >= maxNegMapPointsAdj:
                endSearch = True    ### No adjacency in this way (OR A VERY WIDE WALL)
        else:
            print('ERROR! - adjacencyFound'); sys.exit()
    
    return (adjacencyFound, idxCentroid2)

###

def getAdjacenciesCentroids (gParam, lConnexComponents, lConnexComponentsCentroids, lCentroids):
    
    print('   Computing adjacencies among centroids...')

    height = gParam['height']
    width  = gParam['width']

    visitedPositions = np.ndarray([width,height],bool)  ### boolean matrix to control the search
    mapPointsIdxCC   = np.ndarray([width,height],int)   ### integer matrix to store the indexes of the CC
    interiorPointsCC = np.ndarray([width,height],bool)  ### boolean matrix to store the interior points of the CC

    ### First informs mapPointsIdxCC
    for x in range(height):     ### Reads from top to bottom, from left to right
        for y in range(width):
            mapPointsIdxCC[x,y] = -1
    for idxC in range(len(lCentroids)):
        centroid         = lCentroids[idxC]
        #idxColorCentroid = centroid[0]  ### = lConnexComponentsCentroids[c][0]
        #colorCentroid    = centroid[1]  ### = lConnexComponentsCentroids[c][1]
        #posXYCentroid    = centroid[2]
        lposXYConnexC    = lConnexComponentsCentroids[idxC][2]
        #
        for (x,y) in lposXYConnexC:
            mapPointsIdxCC[x,y] = idxC
     
    ### Now informs interiorPointsCC and visitedPositions
    for x in range(height):     ### Reads from top to bottom, from left to right
        for y in range(width):
            idxCentroid = mapPointsIdxCC[x,y]
            if idxCentroid < 0:
                interiorPointsCC[x,y] = False
                visitedPositions[x,y] = True   ### We don't want to visit points outside the connex components
            elif x == 0 or y == 0 or x == height-1 or y == width-1:
                interiorPointsCC[x,y] = False
                visitedPositions[x,y] = True   ### We don't want to visit the borders of the image
            elif mapPointsIdxCC[x-1,y]   == idxCentroid  and  mapPointsIdxCC[x,y-1]   == idxCentroid  and \
                 mapPointsIdxCC[x-1,y-1] == idxCentroid  and  mapPointsIdxCC[x+1,y]   == idxCentroid  and \
                 mapPointsIdxCC[x,y+1]   == idxCentroid  and  mapPointsIdxCC[x+1,y+1] == idxCentroid:
                interiorPointsCC[x,y] = True
                visitedPositions[x,y] = True   ### We don't want to visit interior points
            else:
                interiorPointsCC[x,y] = False
                visitedPositions[x,y] = False  ### These are the points we want to visit: border points in a CC
                
    ### Finally, searches for the adjacencies
    sAdjacenciesCC = set()
    (x,y) = nonVisitedPosition (gParam,visitedPositions)
    while x is not None:
        idxCentroid1 = mapPointsIdxCC[x,y]
        dx = +1; dy = 0;
        (found, idxCentroid2) = adjacencyCentroidFound (gParam,x,y,dx,dy,visitedPositions,mapPointsIdxCC,interiorPointsCC)
        if found:
            sAdjacenciesCC.add((idxCentroid1,idxCentroid2))
        dx = -1; dy = 0;
        (found, idxCentroid2) = adjacencyCentroidFound (gParam,x,y,dx,dy,visitedPositions,mapPointsIdxCC,interiorPointsCC)
        if found:
            sAdjacenciesCC.add((idxCentroid1,idxCentroid2))
        dx =  0; dy = +1;
        (found, idxCentroid2) = adjacencyCentroidFound (gParam,x,y,dx,dy,visitedPositions,mapPointsIdxCC,interiorPointsCC)
        if found:
            sAdjacenciesCC.add((idxCentroid1,idxCentroid2))
        dx =  0; dy = -1;
        (found, idxCentroid2) = adjacencyCentroidFound (gParam,x,y,dx,dy,visitedPositions,mapPointsIdxCC,interiorPointsCC)
        if found:
            sAdjacenciesCC.add((idxCentroid1,idxCentroid2))
        #
        visitedPositions[x,y] = True
        (x,y) = nonVisitedPosition (gParam,visitedPositions)
    #
    ### And returns a sorted list
    lAdjacenciesCC = list(sAdjacenciesCC)
    lAdjacenciesCC.sort()

    return lAdjacenciesCC


####################################################################################################################
### Kernels: Kernels for cleaning and smoothing / Ideal kernels in the exterior wall
####################################################################################################################

def rotate90Clock (m):
    ### Rotate 90 degrees a matrix
    Nr,Nc = m.shape
    #if Nr != Nc:
    #    print('ERROR in rotate90Clock: matrix not squared'); sys.exit()
    m90 = np.zeros([Nc,Nr])
    # Convers row i in column Nr-i
    for i in range(Nr):
        for j in range(Nc):
            m90[j,Nr-i-1] = m[i,j]
    #
    return m90

###

def horizontalMirror (m):
    ### Mirrors matrix wrt the horizontal axis
    Nr,Nc = m.shape
    hm = np.zeros([Nr,Nc])
    for i in range(Nr):
        for j in range(Nc):
            hm[i,j] = m[Nr-i-1,j]
    #
    return hm

###

def verticalMirror (m):
    ### Mirrors matrix wrt the vertical axis
    Nr,Nc = m.shape
    vm = np.zeros([Nr,Nc])
    for i in range(Nr):
        for j in range(Nc):
            vm[i,j] = m[i,Nc-j-1]
    #
    return vm

###

def defineKernelsForCleanAndSmoothContour():

    ###
    ### Kernels for cleaning and smoothing the contour
    ###

    def rotationsKernels(lKernels):
        
        lKernelsOut = []
        for kernel0 in lKernels:
            kernel1 = rotate90Clock(kernel0)
            kernel2 = rotate90Clock(kernel1)
            kernel3 = rotate90Clock(kernel2)
            lKernelsOut.append(kernel0)
            lKernelsOut.append(kernel1)
            lKernelsOut.append(kernel2)
            lKernelsOut.append(kernel3)

        return lKernelsOut

    ###
    
    def mirrorsKernels(lKernels):
        
        lKernelsOut = []
        for kernel0 in lKernels:
            kernel1 = horizontalMirror(kernel0)
            kernel2 = verticalMirror(kernel0)
            lKernelsOut.append(kernel0)
            lKernelsOut.append(kernel1)
            lKernelsOut.append(kernel2)

        return lKernelsOut
            
    ############################################################

    print('   Computing kernels for cleaning and smoothing...',end="")
    
    lKernelsSearch = []  ### kernels to search and clean/smooth
    lKernelsSmooth = []  ### cleaned/smoothed kernel

    #  .....     .....
    #  ..1..     .....
    #  11111  -> 11111
    krn = np.zeros([3,5]); krn[2,:] = 1; krn[1,2]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,5]); krn[2,:] = 1;             lKernelsSmooth.append(krn)

    #  ....     ....
    #  ..1.     ....
    #  1111  -> 1111
    krn = np.zeros([3,4]); krn[2,:] = 1; krn[1,2]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:] = 1;             lKernelsSmooth.append(krn)

    #  ....     ....
    #  .1..     ....
    #  1111  -> 1111
    krn = np.zeros([3,4]); krn[2,:] = 1; krn[1,1]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:] = 1;             lKernelsSmooth.append(krn)

    #  .....     .....
    #  ..1..     .....
    #  11.11  -> 11111
    krn = np.zeros([3,5]); krn[2,[0,1,3,4]] = 1; krn[1,2]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,5]); krn[2,:]         = 1;             lKernelsSmooth.append(krn)

    #  ....     ....
    #  ..1.     ....
    #  11.1  -> 1111
    krn = np.zeros([3,4]); krn[2,[0,1,3]] = 1; krn[1,2]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:]       = 1;             lKernelsSmooth.append(krn)

    #  .....     .....
    #  .111.     .....
    #  11.11  -> 11111
    krn = np.zeros([3,5]); krn[2,[0,1,3,4]] = 1; krn[1,[1,2,3]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,5]); krn[2,:]         = 1;                   lKernelsSmooth.append(krn)

    #  ....     ....
    #  .111     ....
    #  11.1  -> 1111
    krn = np.zeros([3,4]); krn[2,[0,1,3]] = 1; krn[1,[1,2,3]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:]       = 1;                   lKernelsSmooth.append(krn)

    #  .....     .....
    #  ..11.     .....
    #  11.11  -> 11111
    krn = np.zeros([3,5]); krn[2,[0,1,3,4]] = 1; krn[1,[2,3]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,5]); krn[2,:]         = 1;                 lKernelsSmooth.append(krn)

    #  ....     ....
    #  ..11     ....
    #  11.1  -> 1111
    krn = np.zeros([3,4]); krn[2,[0,1,3]] = 1; krn[1,[2,3]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:]       = 1;                 lKernelsSmooth.append(krn)

    #  ....     ....
    #  .11.     ....
    #  1.11  -> 1111
    krn = np.zeros([3,4]); krn[2,[0,2,3]] = 1; krn[1,[1,2]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:]       = 1;                 lKernelsSmooth.append(krn)

    #  .....     .....
    #  ..11.     .....
    #  111.1  -> 11111
    krn = np.zeros([3,5]); krn[2,[0,1,2,4]] = 1; krn[1,[2,3]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,5]); krn[2,:]         = 1;                 lKernelsSmooth.append(krn)

    #  ....     ....
    #  .11.     ....
    #  11.1  -> 1111
    krn = np.zeros([3,4]); krn[2,[0,1,3]] = 1; krn[1,[1,2]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:]       = 1;                 lKernelsSmooth.append(krn)

    #  .....     .....
    #  .111.     .....
    #  11..1  -> 11111
    krn = np.zeros([3,5]); krn[2,[0,1,4]] = 1; krn[1,[1,2,3]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,5]); krn[2,:]       = 1;                   lKernelsSmooth.append(krn)

    #  ....     .....
    #  111.     .....
    #  1..1  -> 11111
    krn = np.zeros([3,4]); krn[2,[0,3]] = 1; krn[1,[0,1,2]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:]     = 1;                   lKernelsSmooth.append(krn)

    #  .....     .....
    #  .11..     .....
    #  11.11  -> 11111
    krn = np.zeros([3,5]); krn[2,[0,1,3,4]] = 1; krn[1,[1,2]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,5]); krn[2,:]         = 1;                 lKernelsSmooth.append(krn)

    #  ....     ....
    #  11..     ....
    #  1.11  -> 1111
    krn = np.zeros([3,4]); krn[2,[0,2,3]] = 1; krn[1,[0,1]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:]       = 1;                 lKernelsSmooth.append(krn)

    #  ....     ....
    #  .11.     ....
    #  11.1  -> 1111
    krn = np.zeros([3,4]); krn[2,[0,1,3]] = 1; krn[1,[1,2]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:]       = 1;                 lKernelsSmooth.append(krn)

    #  .....     .....
    #  .111.     .....
    #  11.11  -> 11111
    krn = np.zeros([3,5]); krn[2,[0,1,3,4]] = 1; krn[1,[1,2,3]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,5]); krn[2,:]         = 1;                   lKernelsSmooth.append(krn)

    #  ....     ....
    #  .111     ....
    #  11.1  -> 1111
    krn = np.zeros([3,4]); krn[2,[0,1,3]] = 1; krn[1,[1,2,3]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:]       = 1;                   lKernelsSmooth.append(krn)

    #  ....     ....
    #  .1..     ....
    #  .11.  -> .11.
    krn = np.zeros([3,4]); krn[2,[1,2]] = 1; krn[1,1]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,[1,2]] = 1;             lKernelsSmooth.append(krn)

    #  ....     ....
    #  .1..     ....
    #  111.  -> 111.
    krn = np.zeros([3,4]); krn[2,[0,1,2]] = 1; krn[1,1]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,[0,1,2]] = 1;             lKernelsSmooth.append(krn)

    #  ....     ....
    #  .1..     ....
    #  1111  -> 1111
    krn = np.zeros([3,4]); krn[2,:] = 1; krn[1,1]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:] = 1;             lKernelsSmooth.append(krn)

    #  ....     ....
    #  .11.     ....
    #  .11.  -> .11.
    krn = np.zeros([3,4]); krn[2,[1,2]] = 1; krn[1,[1,2]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,[1,2]] = 1;                 lKernelsSmooth.append(krn)

    #  ....     ....
    #  .11.     ....
    #  111.  -> 111.
    krn = np.zeros([3,4]); krn[2,[0,1,2]] = 1; krn[1,[1,2]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,[0,1,2]] = 1;                 lKernelsSmooth.append(krn)

    #  ....     ....
    #  .11.     ....
    #  1111  -> 1111
    krn = np.zeros([3,4]); krn[2,:] = 1; krn[1,[1,2]]=1; lKernelsSearch.append(krn)
    krn = np.zeros([3,4]); krn[2,:] = 1;                 lKernelsSmooth.append(krn)

    ###
    
    #  ...11     .1111
    #  ..1..  -> .1...
    #  .1...     .1...
    #  .1...     .1...
    krn = np.zeros([4,5]); krn[[2,3],1] = 1; krn[1,2]=1; krn[0,[3,4]]     = 1; lKernelsSearch.append(krn)
    krn = np.zeros([4,5]); krn[:,1]     = 1;             krn[0,[1,2,3,4]] = 1; lKernelsSmooth.append(krn)

    #  ..111     .1111
    #  .1...  -> .1...
    #  .1...     .1...
    #  .1...     .1...
    krn = np.zeros([4,5]); krn[[1,2,3],1] = 1; krn[0,[2,3,4]]   = 1; lKernelsSearch.append(krn)
    krn = np.zeros([4,5]); krn[:,1]       = 1; krn[0,[1,2,3,4]] = 1; lKernelsSmooth.append(krn)

    #  ..111     .1111
    #  .11..  -> .1...
    #  .1...     .1...
    #  .1...     .1...
    krn = np.zeros([4,5]); krn[[1,2,3],1] = 1; krn[1,2] = 1; krn[0,[2,3,4]]   = 1; lKernelsSearch.append(krn)
    krn = np.zeros([4,5]); krn[:,1]       = 1;               krn[0,[1,2,3,4]] = 1; lKernelsSmooth.append(krn)

    #  11..     ....
    #  1111     1111
    #  1...  -> 1...
    #  1...     1...
    krn = np.zeros([4,4]); krn[:,0] = 1;       krn[1,:] = 1; krn[0,1] = 1; lKernelsSearch.append(krn)
    krn = np.zeros([4,4]); krn[[1,2,3],0] = 1; krn[1,:] = 1;               lKernelsSmooth.append(krn)

    #  1111     .111
    #  11..  -> .1..
    #  .1..     .1..
    #  .1..     .1..
    krn = np.zeros([4,4]); krn[:,1] = 1; krn[0,[1,2,3]] = 1; krn[[0,1],0] = 1; lKernelsSearch.append(krn)
    krn = np.zeros([4,4]); krn[:,1] = 1; krn[0,[1,2,3]] = 1;                   lKernelsSmooth.append(krn)

    #  ...     ...
    #  ..1     ...
    #  .11  -> ...
    #  1.1     111
    krn = np.zeros([4,3]); krn[3,0] = 1; krn[2,1] = 1; krn[[1,2,3],2] = 1; lKernelsSearch.append(krn)
    krn = np.zeros([4,3]); krn[3,:] = 1;                                   lKernelsSmooth.append(krn)
    
    ### Rotations
    lKernelsSearch = rotationsKernels(lKernelsSearch)
    lKernelsSmooth = rotationsKernels(lKernelsSmooth)

    ### Mirroring
    lKernelsSearch = mirrorsKernels(lKernelsSearch)
    lKernelsSmooth = mirrorsKernels(lKernelsSmooth)

    ### Removes duplicate kernels
    #print(len(lKernelsSearch))
    lKernelsSearchUnique = []
    lKernelsSmoothUnique = []
    for n in range(len(lKernelsSearch)):
        krn1 = lKernelsSearch[n]
        found = False
        for krn2 in lKernelsSearchUnique:
            if krn1.shape == krn2.shape:
                if (krn1[:,:] == krn2[:,:]).all():
                    #print('---',krn1,krn2,krn1-krn2)
                    #print(n," ",end="")
                    found = True
                    break
        if not found:
            lKernelsSearchUnique.append(lKernelsSearch[n])
            lKernelsSmoothUnique.append(lKernelsSmooth[n])
    #print(len(lKernelsSearchUnique))

    for n in range(len(lKernelsSearchUnique)):
        krn1 = lKernelsSearchUnique[n]
        krn2 = lKernelsSmoothUnique[n]
        if (krn1[:,:] == krn2[:,:]).all():
            print(krn1,krn2)
            print('ERROR defineKernelsForCleanAndSmoothContour: krn1[:,:] == krn2[:,:])',n); sys.exit()

    print(' ',len(lKernelsSearchUnique),'kernels constructed')

    return lKernelsSearchUnique, lKernelsSmoothUnique

###

def defineCornersIdeal_EWS1_NoIR (sizeCorner):
    ### 0: Background
    ### 1: Exterior wall (EWS1)
    ### 0: Interior region (NoIR)
    
    if sizeCorner % 2 == 0:
        print('ERROR: defineCornersIdeal_EWS1_NoIR - sizeCorner % 2 == 0'); sys.exit()

    lCornersIdeal = np.zeros([4,sizeCorner,sizeCorner],int)

    ###
    ### Indexes of the directions of every corner
    ###
    ###   Left Top     = 0,4
    ###   Right Top    = 1,5
    ###   Right Bottom = 2,6
    ###   Left Bottom  = 3,7
    ###
    ### *** DO NOT CHANGE!!!
    ###

    ### Base Corner
    lCornersIdeal[0, 0:sizeCorner//2+1, 0:sizeCorner//2+1] = 1;  ### Exterior Wall
    lCornersIdeal[0, 0:sizeCorner//2,   0:sizeCorner//2]   = 0;  ### *NO Interior region*
    # Rotations of Pi/2, Pi and -Pi/2
    lCornersIdeal[1,:,:] = rotate90Clock(lCornersIdeal[0,:,:])
    lCornersIdeal[2,:,:] = rotate90Clock(lCornersIdeal[1,:,:])
    lCornersIdeal[3,:,:] = rotate90Clock(lCornersIdeal[2,:,:])

    return lCornersIdeal

###

def defineCornersIdeal_AllSizes (gParam):

    lCornersIdeal = []
    for sizeCorner in gParam['sizeIdealCorner']:
        lCornersIdealSizeCorner = defineCornersIdeal_EWS1_NoIR (sizeCorner)
        nCornersIdeal = lCornersIdealSizeCorner.shape[0]
        for typeC in range(nCornersIdeal):
            lCornersIdeal.append( (typeC, lCornersIdealSizeCorner[typeC,:,:]) )

    return lCornersIdeal


####################################################################################################################
### Search of Valid elements in the image (Contour, Corners, Perimeter, FrontDoor,...)
####################################################################################################################

################
### Contour
################

def getContour_from_SkimageContour (gParam, sParamTry, rgbImageNilColors):

    ###
    ### Get contour (from SkimageContour) and equivalent structures...
    ###

    print('   Computing contour from SkimageContour...')

    listNilColors           = gParam['listNilColors']
    idxNilColorBackground   = gParam['idxNilColorBackground']
    idxNilColorExteriorWall = gParam['idxNilColorExteriorWall']
    #idxNilColorFrontDoor    = gParam['idxNilColorFrontDoor']
    colorBackground         = listNilColors[idxNilColorBackground]
    colorExteriorWall       = listNilColors[idxNilColorExteriorWall]
    #colorFrontDoor          = listNilColors[idxNilColorFrontDoor]

    levelFindContour               = sParamTry['levelFindContour']
    fullyConnectedFindContour      = sParamTry['fullyConnectedFindContour']
    positiveOrientationFindContour = sParamTry['positiveOrientationFindContour']
    
    width  = gParam['width']
    height = gParam['height']

    ### Image 2D with 0 in the background and 1 in the rest
    backgroundImage = np.zeros([width,height],int)
    for x in range(gParam['height']):     ### Reads from top to bottom, from left to right
        for y in range(gParam['width']):
            c = rgbImageNilColors[x*gParam['width'] + y]
            #if c == colorExteriorWall or c == colorFrontDoor:
            if c != colorBackground:
                backgroundImage[x,y] = 1

    labelBackgroundImage = measure.label(backgroundImage)   ### Conectivity
    contourBackgroundImage = measure.find_contours\
               (labelBackgroundImage == 1, level=levelFindContour, fully_connected=fullyConnectedFindContour, positive_orientation=positiveOrientationFindContour)[0]
             #  (labelBackgroundImage == 1, level=0.5, fully_connected='high', positive_orientation='low')[0]
    
    ### List of points with the contour
    intContourBackgroundImage = []
    for xyP in contourBackgroundImage:
        xP = int(xyP[0])
        yP = int(xyP[1])
        intContourBackgroundImage.append([xP,yP])
    
    ### intImage (LIST of integers with rgb colors) of the contour
    intRegionContour = []
    for x in range(height):     ### Reads from top to bottom, from left to right
        for y in range(width):
            if [x,y] in intContourBackgroundImage:
                intRegionContour.extend(colorExteriorWall)
            else:
                intRegionContour.extend(colorBackground)

    ### Obtain equivalent structures
    rgbRegionContour = getRgbImageFromIntImage (intRegionContour)
    imageContour = np.ones([gParam['width'],gParam['height']],int)
    for x in range(imageContour.shape[1]):     ### Reads from top to bottom, from left to right
        for y in range(imageContour.shape[0]):
            if [x,y] in intContourBackgroundImage:
                imageContour[x,y] = 0
                
    return intContourBackgroundImage, imageContour, intRegionContour, rgbRegionContour

###

def cleanAndSmoothContour (imageContour, lKernelsSearch, lKernelsSmooth):

    ###
    ### Clean and smooth the contour
    ###

    print('   Cleaning and smoothing the contour...')

    height = imageContour.shape[0]
    width  = imageContour.shape[1]

    ### Transforms the image to the same codes than the kernels
    imageContourSmoothed = 1 - imageContour

    ### Obtains the list of positions that are "around the contour"
    ## First gets the maximum size of the kernels
    maxSizeKernels = 0
    for krn in lKernelsSearch:
        sizeKernel1,sizeKernel2 = krn.shape
        maxSizeKernels = max(maxSizeKernels,max(sizeKernel1,sizeKernel2))
    ## Now makes the perimeter wider
    imageContourSmoothedWidePerimeter = imageContourSmoothed.copy()
    imageContourSmoothedWidePerimeter[:,:] = 0
    dxy = maxSizeKernels + 3   ### 3 pixels of margin...
    for x in range(height):        ### Reads from top to bottom, from left to right
        for y in range(width):
            if imageContourSmoothed[x,y] == 1:
                imageContourSmoothedWidePerimeter[x-dxy:x+dxy+1,y-dxy:y+dxy+1] = 1
    ## And finally constructs the list
    lPositionsCloseContour = []
    for x in range(height):        ### Reads from top to bottom, from left to right
        for y in range(width):
            if imageContourSmoothedWidePerimeter[x,y] == 1:
                lPositionsCloseContour.append( ( x,y ) )
    
    ###
    ### Search and replace
    ###
    nReplaces = np.inf
    while nReplaces > 0:    ### If 0, it will do the maximum mumber of loops / If 999, it will only do one loop (NO ES NECESARIO SI YA VA RAPIDO)
        print('      .')
        nReplaces = 0
        for nk in range(len(lKernelsSearch)):
            ###
            ### Gets the kernel and takes its properties
            ###
            kernelSearch            = lKernelsSearch[nk]
            kernelSmooth            = lKernelsSmooth[nk]
            sizeKernel1,sizeKernel2 = kernelSearch.shape
            sizeKernelX,sizeKernelY = kernelSmooth.shape
            if sizeKernel1 != sizeKernelX:
                print(kernelSearch,kernelSmooth)
                print('ERROR: sizeKernel1 != sizeKernelX'); sys.exit()
            if sizeKernel2 != sizeKernelY:
                print(kernelSearch,kernelSmooth)
                print('ERROR: sizeKernel2 != sizeKernelY'); sys.exit()

            ###
            ### Search every kernel in the valid positions of the image
            ###

            ## Fast
            for x,y in lPositionsCloseContour:
                if x >= 0  and  y >= 0  and  x+sizeKernel1 <= height  and  y+sizeKernel2 <= width:  ### We still have to check this is a valid region (a[x:x+N] does not give any error for arrays with size smaller than N)
                    regionContour = imageContourSmoothed[x:x+sizeKernel1,y:y+sizeKernel2]
                    #print(x,y,x+sizeKernel1,y+sizeKernel2,sizeKernel1,sizeKernel2,width,height)
                    #print(regionContour[:,:])
                    #print(kernelSearch[:,:])
                    #print(regionContour[:,:] == kernelSearch[:,:])
                    if (regionContour[:,:] == kernelSearch[:,:]).all():
                        ### replace
                        nReplaces += 1
                        print('      cleaning and smoothing at...',x,y,nk)
                        #print(kernelSearch,kernelSmooth)
                        imageContourSmoothed[x:x+sizeKernel1,y:y+sizeKernel2] = kernelSmooth[:,:]

                    ## Slow
                    # for x in range(height-sizeKernel1):      ### Reads from top to bottom, from left to right
                    #     for y in range(width-sizeKernel2):
                    #         if x >= 0  and  y >= 0  and  x+sizeKernel1 <= height  and  y+sizeKernel2 <= width:  ### We still have to check this is a valid region (a[x:x+N] does not give any error for arrays with size smaller than N)
                    #             regionContour = imageContourSmoothed[x:x+sizeKernel1,y:y+sizeKernel2]
                    #             #print(regionContour[:,:])
                    #             #print(kernelSearch[:,:])
                    #             #print(regionContour[:,:] == kernelSearch[:,:])
                    #             if (regionContour[:,:] == kernelSearch[:,:]).all():
                    #                 ### replace
                    #                 nReplaces += 1
                    #                 print('      cleaning and smoothing at...',x,y,nk)
                    #                 #print(kernelSearch,kernelSmooth)
                    #                 imageContourSmoothed[x:x+sizeKernel1,y:y+sizeKernel2] = kernelSmooth[:,:]
                    
    ### Reverses the transformation
    imageContourSmoothed = 1 - imageContourSmoothed

    return imageContourSmoothed

###

def getConnexComponentsInteriorContour (gParam, imageContourSmoothed):

    print('   Computing main connex component of the interior region of the contour...')

    height = gParam['height']
    width  = gParam['width']

    ### First makes the perimeter thicker
    imageContourSmoothedWidePerimeter = imageContourSmoothed.copy()
    imageContourSmoothedWidePerimeter[:,:] = 1
    dxy = 2
    for x in range(height):        ### Reads from top to bottom, from left to right
        for y in range(width):
            if imageContourSmoothed[x,y] == 0:
                imageContourSmoothedWidePerimeter[x-dxy:x+dxy+1,y-dxy:y+dxy+1] = 0

    ### Now computes the interior region
    ## flood returns an array of booleans, and "1 - arrBool" is an array of integers
    interiorRegionContour = 1 - segmentation.flood(imageContourSmoothedWidePerimeter,seed_point=(0,0))

    return interiorRegionContour

###

def sanityCheckContourSmoothed (gParam, imageContourSmoothed, lCentroids):

    interiorRegionContour = getConnexComponentsInteriorContour (gParam, imageContourSmoothed)
    
    ### The contour must contain all the centroids
    allCentroidsInsideTheContour = True
    for idxC in range(len(lCentroids)):
        centroid           = lCentroids[idxC]
        idxColorCentroid   = centroid[0]
        colorCentroidCC    = centroid[1]
        (xC,yC)            = centroid[2]
        #
        if not interiorRegionContour[xC,yC]:
            print('   sanityCheckContourSmoothed: Centroid',xC,yC,'is not in the interior region of the contour')
        #
        allCentroidsInsideTheContour = allCentroidsInsideTheContour and interiorRegionContour[xC,yC]
    #
    if not allCentroidsInsideTheContour:
        print('   UNSUCCESSFUL sanityCheckContourSmoothed: there are centroids outside the contour')

    sanityCheckOK = allCentroidsInsideTheContour

    return sanityCheckOK

###
            
def walkContourInOrder (gParam, imageContourSmoothed):

    ###
    ### Recorre el contorno en orden y lo guarda en una lista
    ###

    print('   Walking the contour in order...')
    
    def getDxDyFromDirection (dr):

        if   dr ==  'R':  dx =  0;  dy = +1;
        elif dr ==  'L':  dx =  0;  dy = -1;
        elif dr ==  'B':  dx = +1;  dy =  0;
        elif dr ==  'T':  dx = -1;  dy =  0; 
        elif dr == 'BR':  dx = +1;  dy = +1;
        elif dr == 'BL':  dx = +1;  dy = -1;
        elif dr == 'TR':  dx = -1;  dy = +1;
        elif dr == 'TL':  dx = -1;  dy = -1;
        else:
            print(dr); print('ERROR: direction not defined'); sys.exit()
        return dx, dy

    
    def validDirectionsFromXY (x,y,visitedPosition):
        validDirections = []
        #
        dr = 'R';  dx, dy = getDxDyFromDirection (dr);  xx = x + dx;  yy = y + dy;
        if not visitedPosition[xx,yy]: validDirections.append(dr)
        dr = 'L';  dx, dy = getDxDyFromDirection (dr);  xx = x + dx;  yy = y + dy;
        if not visitedPosition[xx,yy]: validDirections.append(dr)
        dr = 'B';  dx, dy = getDxDyFromDirection (dr);  xx = x + dx;  yy = y + dy;
        if not visitedPosition[xx,yy]: validDirections.append(dr)
        dr = 'T';  dx, dy = getDxDyFromDirection (dr);  xx = x + dx;  yy = y + dy;
        if not visitedPosition[xx,yy]: validDirections.append(dr)
        dr = 'BR'; dx, dy = getDxDyFromDirection (dr);  xx = x + dx;  yy = y + dy;
        if not visitedPosition[xx,yy]: validDirections.append(dr)
        dr = 'BL'; dx, dy = getDxDyFromDirection (dr);  xx = x + dx;  yy = y + dy;
        if not visitedPosition[xx,yy]: validDirections.append(dr)
        dr = 'TR'; dx, dy = getDxDyFromDirection (dr);  xx = x + dx;  yy = y + dy;
        if not visitedPosition[xx,yy]: validDirections.append(dr)
        dr = 'TL'; dx, dy = getDxDyFromDirection (dr);  xx = x + dx;  yy = y + dy;
        if not visitedPosition[xx,yy]: validDirections.append(dr)
        #
        return validDirections

    ###
    height = gParam['height']
    width  = gParam['width']

    ### Boolean matrix to control the search
    visitedPositions = np.ndarray([width,height],bool)
    for x in range(height):        ### Reads from top to bottom, from left to right
        for y in range(width):
            if imageContourSmoothed[x,y] == 0:   ### The contour has 0 values, and the background has 1s
                visitedPositions[x,y] = False
            else:
                visitedPositions[x,y] = True
    #
    ### Statistics of the number of valid directions at every point
    nValidDireccions = np.zeros(5)
    for x in range(height):        ### Reads from top to bottom, from left to right
        for y in range(width):
            if imageContourSmoothed[x,y] == 0:   ### The contour has 0 values, and the background has 1s
                validDirections = validDirectionsFromXY (x,y,visitedPositions)
                nValidDireccions[min(len(nValidDireccions)-1,len(validDirections))] += 1
    for i in range(len(nValidDireccions)):
        print('      Number of points with',i,'valid directions:',nValidDireccions[i])
    #
    ### Search
    ## Start
    # We only can start if we have 2 or less valid directions
    n = 1
    (x,y) = nonVisitedPosition (gParam,visitedPositions,n)
    validDirections = validDirectionsFromXY (x,y,visitedPositions)
    while len(validDirections) > 2:
        n += 1
        (x,y) = nonVisitedPosition (gParam,visitedPositions,n)
        validDirections = validDirectionsFromXY (x,y,visitedPositions)
    ## Loop
    drAnt = 'Start'
    xIni = x
    yIni = y
    n = 1
    orderContour = []
    loopCompleted = False
    while not loopCompleted:
        orderContour.append((x,y))
        validDirections = validDirectionsFromXY (x,y,visitedPositions)
        #
        if   len(validDirections) == 1:
            dr = validDirections[0]
        elif len(validDirections) == 2:
            dr1, dr2 = validDirections[0], validDirections[1]
            ### Se prioriza ir recto y no en diagonal
            if   len(dr1) == 1 and len(dr2) == 2:
                dr = dr1
            elif len(dr2) == 1 and len(dr1) == 2:
                dr = dr2
            elif len(dr1) == 1 and len(dr2) == 1:
                if   dr1 == drAnt or drAnt == 'Start':
                    dr = dr1
                elif dr2 == drAnt or drAnt == 'Start':
                    dr = dr2
                else:
                    print('   ---',x,y,validDirections)
                    print('   UNSUCCESSFUL walkContourInOrder: ambiguous valid directions (1)'); return False, orderContour; # sys.exit()
            else:
                print('   ---',x,y,validDirections)
                print('   UNSUCCESSFUL walkContourInOrder: ambiguous valid directions (2)'); return False, orderContour; # sys.exit()
        elif len(validDirections) == 3:
            dr1, dr2, dr3 = validDirections[0], validDirections[1], validDirections[2]
            ### Se prioriza ir recto y no en diagonal
            if   len(dr1) == 1 and len(dr2) == 2 and len(dr3) == 2:
                dr = dr1
            elif len(dr2) == 1 and len(dr1) == 2 and len(dr3) == 2:
                dr = dr2
            elif len(dr3) == 1 and len(dr1) == 2 and len(dr2) == 2:
                dr = dr3
            else:
                print('   ---',x,y,validDirections)
                print('   UNSUCCESSFUL walkContourInOrder: ambiguous valid directions (3)'); return False, orderContour; # sys.exit()
        else:
            print('   ---',x,y,validDirections)
            print('   UNSUCCESSFUL walkContourInOrder: ambiguous valid directions (4)'); return False, orderContour; # sys.exit()

        # Update, Move and control the end of the loop
        visitedPositions[x,y] = True
        drAnt = dr
        #
        dx, dy = getDxDyFromDirection (dr)
        x = x + dx;  y = y + dy;
        #
        n += 1
        if abs(x - xIni) <= 1 and abs(y - yIni) <= 1 and n > 20:  ### At least 20 steps
            orderContour.append((x,y))
            #print('   SUCCESSFUL walkContourInOrder')
            loopCompleted = True
        #
        if n > height*width:
            print('   UNSUCCESSFUL walkContourInOrder: too many iterations'); return False, orderContour; # sys.exit()
            
    return loopCompleted, orderContour

################
### Corners
################

def getCorners_from_SkimageHarris (sParamTry, imageContourSmoothed):

    kvalueHarris   = sParamTry['kvalueHarris']
    sigmaHarris    = sParamTry['sigmaHarris']
    mindistPeaks   = sParamTry['mindistPeaks']
    thresholdPeaks = sParamTry['thresholdPeaks']
    
    #responseImage = corner_peaks(corner_fast(image, n=8, threshold=0.05), min_distance=5, threshold_rel=0.01)
    responseImage = corner_harris(imageContourSmoothed, method='k', k=kvalueHarris, sigma=sigmaHarris)

    lCornersHarris = corner_peaks(responseImage, min_distance=mindistPeaks, threshold_rel=thresholdPeaks)

    return lCornersHarris

###

def checkCornersFound_wrt_CornersIdealInArray01 (gParam, imageContourSmoothed, lCornersFound, lCornersIdeal):
    ###
    ### Checks that the corners found elsewhere correspond to any idel corners
    ###
    
    height = gParam['height']
    width  = gParam['width']

    ### Transforms the image to the same codes than the kernels
    imageCheckCorners = 1 - imageContourSmoothed
    
    δFrame = 0  ### Por si se necesita...
    minFrameX = max(gParam['MinFrameX'] - δFrame, 0)
    maxFrameX = min(gParam['MaxFrameX'] + δFrame, height)
    minFrameY = max(gParam['MinFrameY'] - δFrame, 0)
    maxFrameY = min(gParam['MaxFrameY'] + δFrame, width)

    lCornersImageCentered = []
    nCornersFound = lCornersFound.shape[0]
    checkedCorners = []  ### Control variable so as not to include duplicates
    for ncf in range(nCornersFound):
        xCF, yCF = lCornersFound[ncf,:]
        if xCF < minFrameX or xCF > maxFrameX or yCF < minFrameY or yCF > maxFrameY:
            continue
        #print(xCF,yCF)
        pctEqualMax = -1
        for nci in range(len(lCornersIdeal)):
            typeC       = lCornersIdeal[nci][0]
            cornerIdeal = lCornersIdeal[nci][1]
            #print(typeC)
            #
            sizeCorner,sizeCorner2 = cornerIdeal.shape
            if sizeCorner != sizeCorner2:
                print('ERROR: Ideal corners are not squared'); sys.exit()

            ### We have to compare with some tolerance...
            dX = 3
            dY = 3
            for dx in range(-dX,dX+1):
                for dy in range(-dY,dY+1):
                    x1 = dx + xCF - sizeCorner//2
                    x2 = dx + xCF + sizeCorner//2
                    y1 = dy + yCF - sizeCorner//2
                    y2 = dy + yCF + sizeCorner//2
                    #
                    if x1 >= 0  and  y1 >= 0  and  x2+1 <= height  and y2+1 <= width:  ### We still have to check this is a valid region (a[x:x+N] does not give any error for arrays with size smaller than N)
                        regionToCompare = imageCheckCorners[x1:x2+1,y1:y2+1]
                        #print(xCF,yCF)
                        #print(regionToCompare)
                        #print(cornerIdeal)
                        ### Compara los 1s iguales
                        comparison = (regionToCompare[:,:] * cornerIdeal[:,:])
                        pctEqual = sum(sum(comparison))/sum(sum(cornerIdeal))
                        #print(pctEqual)
                        if pctEqual > pctEqualMax:
                            pctEqualMax   = pctEqual
                            typeC_OK      = typeC
                            sizeCorner_OK = sizeCorner
                            xCorner_OK    = x1 + (x2 - x1) // 2
                            yCorner_OK    = y1 + (y2 - y1) // 2

        ### We only append if it is not a duplicated point
        if (xCorner_OK, yCorner_OK) not in checkedCorners:
            checkedCorners.append( (xCorner_OK, yCorner_OK) )
            lCornersImageCentered.append( (typeC_OK, sizeCorner_OK, (xCorner_OK, yCorner_OK), pctEqualMax) )

    return lCornersImageCentered

###

def obtainCorrespondenceCornersOrderedContour (lCornersImage,orderContour):
    ###
    ### For each corner in lCornersImage, finds the closest element in orderContour
    ###   Returns the corners in order
    ###
    
    pq = PriorityQueue()  ### We can store the corners sorted by the *index* in orderContour
    for c in lCornersImage:       
        typeC            = c[0]
        sizeCorner       = c[1]
        xCorner, yCorner = c[2]
        pctEqualMax      = c[3]
        #
        minDist = 99999
        idxMin  = -1
        for n in range(len(orderContour)):
            x,y = orderContour[n]
            d = (x - xCorner)**2 + (y - yCorner)**2
            if d < minDist:
                minDist = d
                idxMin  = n
        pq.put((idxMin,c))

    ### Now obtains the corners in order
    orderNearestCorners = []
    while not pq.empty():
        c = pq.get()[1]
        orderNearestCorners.append(c)

    return orderNearestCorners

################
### Perimeter
################

def getPositionsFrontDoor (gParam, rgbImage):
    ###
    ### Obtains the positions of the FrontDoor
    ###

    height = gParam['height']
    width  = gParam['width']

    listNilColors        = gParam['listNilColors']
    idxNilColorFrontDoor = gParam['idxNilColorFrontDoor']
    colorFrontDoor       = listNilColors[idxNilColorFrontDoor]

    lPositionsFrontDoor = []
    for x in range(gParam['height']):     ### Reads from top to bottom, from left to right
        for y in range(gParam['width']):
            c = rgbImage[x*gParam['width'] + y]
            if c == colorFrontDoor:
                lPositionsFrontDoor.append( (x,y) )

    return lPositionsFrontDoor

###

def constructPerimeterWithoutFrontDoor (orderNearestCorners):
    ###
    ### Constructs the perimeter without the FrontDoor
    ###

    lPerimeterEW = []
    ### Exterior walls from orderNearestCorners
    xC1 = None
    yC1 = None
    for n in range(len(orderNearestCorners)):
        xC2, yC2 = orderNearestCorners[n][2]
        if xC1 is not None:
            lPerimeterEW.append ( (xC1,yC1,xC2,yC2,'EW') )   ### in orderNearestCorners we only have ExteriorWall
        else:
            xCi = xC2
            yCi = yC2
        xC1 = xC2
        yC1 = yC2
    #
    lPerimeterEW.append ( (xC1,yC1,xCi,yCi,'EW') )   ### in orderNearestCorners we only have ExteriorWall
    #print(lPerimeterEW)

    successfulPerimeter = True
    
    return successfulPerimeter, lPerimeterEW


def constructPerimeterWithFrontDoor (sParamTry, orderNearestCorners, lPositionsFrontDoor):
    ###
    ### Constructs the perimeter with the FrontDoor (if possible)
    ###

    sizeFrontDoor = sParamTry['sizeFrontDoor']

    ### Obtains the perimeter (without the FrontDoor, only exterior walls)
    successfulPerimeter, lPerimeterEW = constructPerimeterWithoutFrontDoor (orderNearestCorners)

    ### Now obtains the center and the wall of the FrontDoor
    idxMinDistPerimeter      = None
    idxMinDistInLineEW       = None
    xClosestPer, yClosestPer = None, None
    minSumDistPerimeter      = None
    for n in range(len(lPerimeterEW)):
        ### Finds the index of the ExteriorWall with the closest point to the positions with FrontDoor colors
        (xC1,yC1,xC2,yC2,t) = lPerimeterEW[n]
        lPointsLineC1C2 = getPointsLineBetweenXY (xC1, yC1, xC2, yC2)
        #
        idxMinDistLineEW       = None
        xClosestEW, yClosestEW = None, None
        minSumDistToFD         = None
        if len(lPositionsFrontDoor) != 0:
            for m in range(len(lPointsLineC1C2)):
                xEW,yEW,dr = lPointsLineC1C2[m]
                sumDistToFD = 0   ### sum of the distances from lPositionsFrontDoor to (xEW,yEW)
                for (xFD,yFD) in lPositionsFrontDoor:
                    sumDistToFD += np.sqrt( (xFD-xEW)**2 + (yFD-yEW)**2 )
                if minSumDistToFD is None or sumDistToFD < minSumDistToFD:
                    idxMinDistLineEW       = m
                    minSumDistToFD         = sumDistToFD
                    xClosestEW, yClosestEW = xEW, yEW
        if minSumDistToFD is not None and (minSumDistPerimeter is None or minSumDistToFD < minSumDistPerimeter):
            idxMinDistPerimeter      = n
            minSumDistPerimeter      = minSumDistToFD
            xClosestPer, yClosestPer = xClosestEW, yClosestEW
            idxMinDistInLineEW       = idxMinDistLineEW

    ### Finally, obtains the initial and final point of the door in the selected wall
    #print(idxMinDistPerimeter,xClosestPer,yClosestPer)
    if idxMinDistPerimeter is not None:
        (xC1,yC1,xC2,yC2,t) = lPerimeterEW[idxMinDistPerimeter]
        lPointsLineC1C2 = getPointsLineBetweenXY (xC1, yC1, xC2, yC2)
        #
        xEW1 = lPointsLineC1C2[idxMinDistInLineEW][0];  yEW1 = lPointsLineC1C2[idxMinDistInLineEW][1]
        xEW2 = lPointsLineC1C2[idxMinDistInLineEW][0];  yEW2 = lPointsLineC1C2[idxMinDistInLineEW][1]
        nPointsFrontDoor = 1
        nPointsFrontDoor_ant = 0
        im = 1;  ip = 1
        while nPointsFrontDoor < sizeFrontDoor and nPointsFrontDoor_ant != nPointsFrontDoor:
            nPointsFrontDoor_ant = nPointsFrontDoor
            if idxMinDistInLineEW - im >= 0:
                xEW1 = lPointsLineC1C2[idxMinDistInLineEW - im][0];  yEW1 = lPointsLineC1C2[idxMinDistInLineEW - im][1]
                nPointsFrontDoor += 1
                im += 1
            if idxMinDistInLineEW + ip < len(lPointsLineC1C2):
                xEW2 = lPointsLineC1C2[idxMinDistInLineEW + ip][0];  yEW2 = lPointsLineC1C2[idxMinDistInLineEW + ip][1]
                nPointsFrontDoor += 1
                ip += 1
        if nPointsFrontDoor < sizeFrontDoor:
            successfulFrontDoor = False
        else:
            successfulFrontDoor = True
            extWall1  = (xC1,yC1,xEW1,yEW1,'EW')
            frontDoor = (xEW1,yEW1,xEW2,yEW2,'FD')
            extWall2  = (xEW2,yEW2,xC2,yC2,'EW')
            #print(extWall1,frontDoor,extWall2)
    else:
        successfulFrontDoor = False

    ### Now constructs the final perimeter (maybe without the FrontDoor, but always with the walls)
    lPerimeterX = []
    for n in range(len(lPerimeterEW)):
        if n != idxMinDistPerimeter:
            lPerimeterX.append(lPerimeterEW[n])
        else:
            if successfulFrontDoor:
                lPerimeterX.append(extWall1)
                lPerimeterX.append(frontDoor)
                lPerimeterX.append(extWall2)
            else:
                lPerimeterX.append(lPerimeterEW[n])

    lPerimeter = lPerimeterX.copy()
    firstIteration = True
    for i in range(len(lPerimeter)+2):
        n = i % len(lPerimeter)
        (xP1,yP1,xP2,yP2,t) = lPerimeter[n]
        ### Adjusts xP1, yP1 with the previous value
        if not firstIteration:
            xP1,yP1 = xP2_ant,yP2_ant
        ### Compute the direction between P1 and P2 and aligns
        adx = np.abs(xP2 - xP1);  sdx = xP2 - xP1   ### absolute and real differences
        ady = np.abs(yP2 - yP1);  sdy = yP2 - yP1
        #
        if adx > ady:    ### up/down
            yP2 = yP1
        else:            ### right/left
            xP2 = xP1
        #
        lPerimeter[n]   = (xP1,yP1,xP2,yP2,t)
        xP2_ant,yP2_ant = xP2,yP2
        firstIteration  = False

    return successfulPerimeter, successfulFrontDoor, lPerimeter

################
### Interior Walls
################
from interior_walls_utils import build_interior_walls_from_borders


################
### Windows
################

from window_placement_utils import place_windows_heuristic




####################################################################################################################
### Matlab stuff
####################################################################################################################

def obtainDataMatlab (fileImage, lPerimeter, lCentroids, lConnexComponentsCentroids, lAdjacenciesCC):

    ###
    ### Transformation to the matlab format for retrieval
    ###
    ### Example of the structure needed
    ###   README.md in
    ###     ~/localhd/PPAL/Projects-Research/2021-ViviendaPublica-LluisOrtega/Software-Papers/Software/Graph2Plan/Graph2plan-master/DataPreparation/
    ###
    ### Software to obtain it (https://github.com/zzilch/RPLAN-Toolbox)
    ###   cd ~/localhd/PPAL/Projects-Research/2021-ViviendaPublica-LluisOrtega/Software-Papers/Software/Graph2Plan/RPLAN-Toolbox-master
    ###   cat readme.md
    ###   conda activate new_g2p
    ###   python
    ###     from rplan.floorplan import Floorplan
    ###     fp = Floorplan('./data/0.png')
    ###     data = fp.to_dict()
    ###     print(data.keys())
    ###     ... see readme.md
    ###
    ### Retrieve:
    ###   Ver scripts en ./Retrieve-Test
    ###     (modificados de ~/Arquitectura/Graph2Plan/Graph2plan-master/Test)
    ###     Son las mismas transformaciones (al menos de nombre) que en 'DataPreparation' + el test
    ###
    ### WE ONLY NEED:
    ###   - name
    ###   - boundary
    ###   - order (CREO QUE NO SE USA, PERO HAY QUE INFORMARLO)
    ###   - rType
    ###   - rEdge
    ### WE DON'T NEED (PERO HAY QUE INFORMARLO):
    ###   - rBoundary
    ###   - gtBox
    ###   - gtBoxNew
    ###

    ###

    def computeDataMatlab_boundary(lPerimeter):
        ###
        ### In RPLAN-Toolbox-master/rplan/floorplan.py
        ###   def get_exterior_boundary(self):
        ###      directions:
        ###        0 (right)
        ###        1 (down)
        ###        2 (left)
        ###        3 (up)
        ###
        ### Format (Graph2plan-master/DataPreparation/README.md):
        ###   (x,y,dir,isNew)
        ###     x,y are the coordinates of thge point
        ###     first two point indicate the front door
        ###     dir: 0(right)/1(down)/2(left)/3(up) for `dir`.
        ###     `isNew` means the point is not a corner point (usually a point of door)
        ###

        ### Gets the index of the front door
        idx_FrontDoor = None
        for idx in range(len(lPerimeter)):
            label = lPerimeter[idx][4]
            if label == 'FD':
                idx_FrontDoor = idx
        #
        if idx_FrontDoor is None:
            print('ERROR: computeDataMatlab_boundary: There is no front door'); sys.exit()

        ### Visit the perimeter starting at the front door
        idx = idx_FrontDoor
        boundaryMatlab = []
        for i in range(len(lPerimeter)):
            ### Each element in lPerimeter contains: starting point / ending point / label
            elemP    = lPerimeter[idx]
            xP1, yP1 = elemP[0], elemP[1]
            xP2, yP2 = elemP[2], elemP[3]
            ### Compute the direction between P1 and P2
            adx = np.abs(xP2 - xP1);  sdx = xP2 - xP1   ### absolute and real differences
            ady = np.abs(yP2 - yP1);  sdy = yP2 - yP1
            ### SOLO 4 DIRECCIONES: AQUI ESTA SUPONIENDO PERIMETRO PARALELO A LOS EJES, NO HAY DIAGONALES
            #
            if adx > ady:    ### up/down
                if sdx > 0:   ## down
                    direc = 1
                else:         ## up
                    direc = 3
            else:            ### right/left
                if sdy > 0:   ## right
                    direc = 0
                else:         ## left
                    direc = 2
            ###
            if idx == idx_FrontDoor or (idx == idx_FrontDoor+1 or (idx == 0 and idx_FrontDoor == len(lPerimeter)-1)):
                isNew = 1
            else:
                isNew = 0
            ###
            boundaryMatlab.append ( [yP1, xP1, direc, isNew] )  ### !!! (y,x) en vez de (x,y) - *Graph2Plan quiere que la primera componente sea horizontal*
            idx += 1
            if idx == len(lPerimeter): idx = 0

        return boundaryMatlab

    ###

    def computeDataMatlab_rType_rEdge(lCentroids, lConnexComponentsCentroids, lAdjacenciesCC):
        ###
        ### In RPLAN-Toolbox-master/rplan/utils.py
        ###   def get_edges(boxes,th=9):
        ###     relation = 5 #'surrounding'
        ###     relation = 4 #'inside'
        ###   def point_box_relation(u,vbox):
        ###     relation = 0 # 'left-above'
        ###     relation = 3 # 'above'
        ###     relation = 8 # 'right-above'
        ###     relation = 7 # 'right-of'
        ###     relation = 9 # 'right-below'
        ###     relation = 6 # 'below'
        ###     relation = 1 # 'left-below'
        ###     relation = 2 # 'left-of'
        ###     relation = 4 # 'inside'
        ###
        ### Format (Graph2plan-master/DataPreparation/README.md):
        ###   rType: room categories
        ###   rEdge: (u,v,r), room indices and relative position(u relative to v)
        ###
        
        
        def point_box_relation(u,vbox):
            ### COPIED FROM RPLAN-Toolbox-master/rplan/utils.py
            ###   u is the centroid of the ubox
            ###   vbox is the minimum container box that contains the room for v
            #
            #uy, ux = u      ### !!! (y,x) en vez de (x,y)  *NO* YA VIENE INVERTIDO
            #vy0, vx0, vy1, vx1 = vbox
            ux, uy = u
            vx0, vy0, vx1, vy1 = vbox
            #
            if (ux<vx0 and uy<=vy0) or (ux==vx0 and uy==vy0):
                relation = 0 # 'left-above'
            elif (vx0<=ux<vx1 and uy<=vy0):
                relation = 3 # 'above'
            elif (vx1<=ux and uy<vy0) or (ux==vx1 and uy==vy0):
                relation = 8 # 'right-above'
            elif (vx1<=ux and vy0<=uy<vy1):
                relation = 7 # 'right-of'
            elif (vx1<ux and vy1<=uy) or (ux==vx1 and uy==vy1):
                relation = 9 # 'right-below'
            elif (vx0<ux<=vx1 and vy1<=uy):
                relation = 6 # 'below'
            elif (ux<=vx0 and vy1<uy) or (ux==vx0 and uy==vy1):
                relation = 1 # 'left-below'
            elif(ux<=vx0 and vy0<uy<=vy1):
                relation = 2 # 'left-of'
            elif(vx0<ux<vx1 and vy0<uy<vy1):
                relation = 4 # 'inside'

            return relation
        
        rType = []
        for centroid in lCentroids:
            roomType = centroid[0]
            rType.append(roomType)
            
        rEdges = []
        for (idxCentroid1, idxCentroid2) in lAdjacenciesCC:
            if idxCentroid1 < idxCentroid2:   ### Elimina repetidos (i,j) / (j,i))
                #print(idxCentroid1,idxCentroid2)
                centroid1 = lCentroids[idxCentroid1]
                centroid2 = lCentroids[idxCentroid2]
                (xC1,yC1) = centroid1[2]   ### Position of centroid1
                (xC2,yC2) = centroid2[2]   ### Position of centroid2
                ### Computes the relation between centroid1 and centroid2
                ## Computes the bounding boxes
                #XMin, YMin, XMax, YMax = boundingBox (...)
                lPointsConnexComponent1 = lConnexComponentsCentroids[idxCentroid1][2]
                ux0, uy0, ux1, uy1  = boundingBox (lPointsConnexComponent1)
                lPointsConnexComponent2 = lConnexComponentsCentroids[idxCentroid2][2]
                vx0, vy0, vx1, vy1  = boundingBox (lPointsConnexComponent2)
                #
                #ubox = uy0, ux0, uy1, ux1     ### !!! (y,x) en vez de (x,y) - *Graph2Plan quiere que la primera componente sea horizontal*
                vbox = vy0, vx0, vy1, vx1     ### !!! (y,x) en vez de (x,y) - *Graph2Plan quiere que la primera componente sea horizontal*
                #
                #uc = (uy0+uy1)/2,(ux0+ux1)/2   ### ERM - Graph2Plan usa el centroide de la caja... 
                uc = yC1, xC1                  ### !!! (y,x) en vez de (x,y) - *Graph2Plan quiere que la primera componente sea horizontal*
                ##vc = yC2, xC2                 ### !!! (y,x) en vez de (x,y) - *Graph2Plan quiere que la primera componente sea horizontal*
                #
                ## Translated from def get_edges(boxes,th=9):
                if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                    relation = 5 #'surrounding'
                elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                    relation = 4 #'inside'
                else:
                    relation = point_box_relation(uc,vbox)
                rEdges.append( [idxCentroid1, idxCentroid2, relation] )
            
        return rType, rEdges

    
    def computeDataMatlab_gtBox_gtBoxNew(lCentroids, lConnexComponentsCentroids):

        gtBox    = []
        gtBoxNew = []
        for idxCentroid in range(len(lCentroids)):
            centroid = lCentroids[idxCentroid]
            ## Computes the bounding box
            #XMin, YMin, XMax, YMax = boundingBox (...)
            lPointsConnexComponent = lConnexComponentsCentroids[idxCentroid][2]
            x0, y0, x1, y1         = boundingBox (lPointsConnexComponent)
            vbox = [ y0, x0, y1, x1 ]    ### !!! (y,x) en vez de (x,y) - *Graph2Plan quiere que la primera componente sea horizontal*
            gtBox.append(vbox)
            gtBoxNew.append(vbox)

        return gtBox, gtBoxNew
        

    #############################
    dataMatlab = {}
    ### NO SE PUEDE GUARDAR 'None' EN NINGUNO DE LOS CAMPOS
    dataMatlab['name']      = fileImage.split('.')[0]
    dataMatlab['order']     = []  ### ERM - Se podría obtener de los ficheros matlab
    dataMatlab['boundary']  = computeDataMatlab_boundary(lPerimeter)
    rType, rEdges           = computeDataMatlab_rType_rEdge(lCentroids, lConnexComponentsCentroids, lAdjacenciesCC)
    dataMatlab['rType']     = rType
    dataMatlab['rEdge']     = rEdges
    dataMatlab['rBoundary'] = []
    gtBox, gtBoxNew         = computeDataMatlab_gtBox_gtBoxNew(lCentroids, lConnexComponentsCentroids)
    dataMatlab['gtBox']     = gtBox     ### ERM - Seguramente no son muy precisos (calculados a partir de los puntos de la componente conexa)
    dataMatlab['gtBoxNew']  = gtBoxNew  ### ERM - Seguramente no son muy precisos (calculados a partir de los puntos de la componente conexa)
    ### EXTRA DATA (for Venecia)
    dataMatlab['lPerimeter']     = lPerimeter
    dataMatlab['lCentroids']     = lCentroids
    dataMatlab['lAdjacenciesCC'] = lAdjacenciesCC

    return dataMatlab


####################################################################################################################
### Search of the ideal kernels in the corners already found
####################################################################################################################

###

def findValidElementsInImage (gParam, sParam, oriPilImage):

    sanityCheckOKCentroids = None
    sanityCheckOKContour = None
    successfulWalkContour = None
    successfulPerimeter = None
    successfulFrontDoor = None
    lCentroids = None
    lConnexComponents = None
    lConnexComponentsCentroids = None
    lAdjacenciesCC = None
    lCornersIdealImage = None
    lPerimeter = None
    pwalls = None
    
    ####################################################################################################
    #
    gParam, intImageNilColors, rgbImageNilColors, arrImageNilColors = obtainImagesNilColors (gParam,oriPilImage)
    #
    lCentroids, lConnexComponentsCentroids, lConnexComponents = getConnexComponentsCentroids (gParam, rgbImageNilColors)
    #
    sanityCheckOKCentroids = sanityCheckCentroids (gParam, rgbImageNilColors, lCentroids, lConnexComponentsCentroids)
    #
    lAdjacenciesCC = getAdjacenciesCentroids (gParam, lConnexComponents, lConnexComponentsCentroids, lCentroids)
    #
    ####################################################################################################

    if not sanityCheckOKCentroids:
        return sanityCheckOKCentroids, sanityCheckOKContour, successfulWalkContour, successfulPerimeter, successfulFrontDoor, \
               lCentroids, lConnexComponents, lConnexComponentsCentroids, lAdjacenciesCC, lCornersIdealImage, lPerimeter, pwalls
    
    print('   Finding (and computing) valid elements in the image...')

    np.random.seed(1)   ### No debería ser relevante, en el caso peor se prueban todas

    ###
    ### Generates all combinations of parameters in two priority queues with random weights
    ###
    pq1 = PriorityQueue()
    #
    for levelFindContour in sParam['levelFindContour']:
        for fullyConnectedFindContour in sParam['fullyConnectedFindContour']:
            for positiveOrientationFindContour in sParam['positiveOrientationFindContour']:
                sParamTry = {}
                sParamTry['levelFindContour']               = levelFindContour
                sParamTry['fullyConnectedFindContour']      = fullyConnectedFindContour
                sParamTry['positiveOrientationFindContour'] = positiveOrientationFindContour
                r = np.random.rand()   ### We want to store the parameters combinations in a random way
                pq1.put((r,sParamTry))
    #
    pq2 = PriorityQueue()
    for kvalueHarris in sParam['kvalueHarris']:
        for sigmaHarris in sParam['sigmaHarris']:
            for mindistPeaks in sParam['mindistPeaks']:
                for thresholdPeaks in sParam['thresholdPeaks']:
                    for sizeFrontDoor in sParam['sizeFrontDoor']:
                        #
                        sParamTry = {}
                        #
                        sParamTry['kvalueHarris']    = kvalueHarris
                        sParamTry['sigmaHarris']     = sigmaHarris
                        sParamTry['mindistPeaks']    = mindistPeaks
                        sParamTry['thresholdPeaks']  = thresholdPeaks
                        sParamTry['sizeFrontDoor']   = sizeFrontDoor
                        r = np.random.rand()   ### We want to store the parameters combinations in a random way
                        pq2.put((r,sParamTry))

    ##########
    ###
    ### Searches a combination of parameters that gives a good contour (1)
    ###
    startTime = time.time()
    print('   Search for contour params (1)')
    successfulSearch1 = False
    while not pq1.empty() and not successfulSearch1:
        print('      ...')
        gpq = pq1.get()
        if gParam['Debug']: print(gpq)
        sParamTry  = gpq[1]
        #
        ### Get contour (from SkimageContour)
        intContourBackgroundImage, imageContour, intRegionContour, rgbRegionContour = \
          getContour_from_SkimageContour (gParam, sParamTry, rgbImageNilColors)
        #
        ### Clean and smooth the contour
        lKernelsContourSearch, lKernelsContourSmooth = defineKernelsForCleanAndSmoothContour()
        imageContourSmoothed = cleanAndSmoothContour (imageContour, lKernelsContourSearch, lKernelsContourSmooth)
        #

        ##############################################3

        ### Sanity check of the smoothed contour
        sanityCheckOKContour = sanityCheckContourSmoothed (gParam, imageContourSmoothed, lCentroids)
        #
        ### Walk the smoothed contour in order (if possible)
        successfulWalkContour, orderContour = walkContourInOrder (gParam, imageContourSmoothed)
        #
        ### Success control
        successfulSearch1 = sanityCheckOKContour and successfulWalkContour
    #
    endTime = time.time()
    print('   Elapsed time (1):',endTime-startTime)

    ###
    if not successfulSearch1:

        return sanityCheckOKCentroids, sanityCheckOKContour, successfulWalkContour, successfulPerimeter, successfulFrontDoor, \
               lCentroids, lConnexComponents, lConnexComponentsCentroids, lAdjacenciesCC, lCornersIdealImage, lPerimeter

    ##########
    ###
    ### Searches a combination of parameters that gives a good contour (2)
    ###

    ### Define ideal kernels for the corners
    lCornersIdeal = defineCornersIdeal_AllSizes (gParam)

    startTime = time.time()
    print('   Search for contour params (2)')
    successfulSearch2 = False
    while not pq2.empty() and not successfulSearch2:
        print('      ...')
        gpq = pq2.get()
        if gParam['Debug']: print(gpq)
        sParamTry  = gpq[1]
        #
        ### Obtain the corners (form corners_harris)
        lCornersHarris = getCorners_from_SkimageHarris (sParamTry, imageContourSmoothed)
        #
        ### Checks that the corners found elsewhere correspond to any idel corners
        lCornersIdealImage = checkCornersFound_wrt_CornersIdealInArray01 \
                             (gParam, imageContourSmoothed, lCornersHarris, lCornersIdeal)
        #
        ### Sorts the corners in lCornersIdealImage according to orderContour
        orderNearestCorners = obtainCorrespondenceCornersOrderedContour (lCornersIdealImage, orderContour)
        #
        ###  Obtains the positions of the FrontDoor
        lPositionsFrontDoor = getPositionsFrontDoor (gParam, rgbImageNilColors)
        #
        ### Constructs the perimeter with the FrontDoor (if possible)
        successfulPerimeter, successfulFrontDoor, lPerimeter = \
            constructPerimeterWithFrontDoor (sParamTry, orderNearestCorners, lPositionsFrontDoor)
        
        # 0. CONFIGURACIÓN DE PUERTAS (Si es interactivo)
        from interior_doors_placement_utils import solicitar_config_puertas_usuario
        door_config = solicitar_config_puertas_usuario()

        lConnexComponents, pwalls = build_interior_walls_from_borders(gParam, rgbImageNilColors, lConnexComponents,
                                                                                imageContourSmoothed=imageContourSmoothed,
                                                                                lPerimeter=lPerimeter,
                                                                                min_segment_len=12,
                                                                                thickness_px=2,
                                                                                config=door_config
                                                                               )
        
        if successfulPerimeter:
            lPerimeter = place_windows_heuristic(gParam, lPerimeter, lConnexComponentsCentroids, lConnexComponents)
        ### Success control
        successfulSearch2 = successfulPerimeter and successfulFrontDoor
    #
    endTime = time.time()
    print('   Elapsed time (2):',endTime-startTime)

    ###
    return sanityCheckOKCentroids, sanityCheckOKContour, successfulWalkContour, successfulPerimeter, successfulFrontDoor, \
           lCentroids, lConnexComponents, lConnexComponentsCentroids, lAdjacenciesCC, lCornersIdealImage, lPerimeter, pwalls
        

####################################################################################################################
### GLOBAL PARAMETERS
####################################################################################################################

def defineGlobalAndSearchParameters (heightImage, widthImage):
    
    ###
    ### List of colors
    ###

    listColors = 2

    listNilColors = [];                   listNilColorsNames = []
    if   listColors == 1:  ### Predefined list of colors used in the original (Nil) transformation

        listNilColors.append((248,216,89));   listNilColorsNames.append("LivingR")  ###  0: Amarillo
        listNilColors.append((253,33,33));    listNilColorsNames.append("MasterR")  ###  1: Rojo1
        listNilColors.append((163,192,48));   listNilColorsNames.append("Kitchen")  ###  2: Verde1
        listNilColors.append((85,89,218));    listNilColorsNames.append("BathR")    ###  3: Azul
        listNilColors.append((188,161,48));   listNilColorsNames.append("DiningR")  ###  4: Ocre
        listNilColors.append((252,146,146));  listNilColorsNames.append("ChildR")   ###  5: Rosa
        listNilColors.append((120,36,36));    listNilColorsNames.append("SecondR")  ###  6: Marron1
        listNilColors.append((255,165,0));    listNilColorsNames.append("GuestR")   ###  7: Verde2
        listNilColors.append((76,119,77));    listNilColorsNames.append("Balcony")  ###  8: Verde3
        listNilColors.append((182,187,60));   listNilColorsNames.append("Entrance") ###  9: Verde4
        listNilColors.append((89,18,47));     listNilColorsNames.append("Storage")  ### 10: Marron2
        listNilColors.append((255,255,255));  listNilColorsNames.append("BackGrd")  ### 11: Blanco
        listNilColors.append((0,0,0));        listNilColorsNames.append("ExtWall")  ### 12: Negro
        listNilColors.append((255,0,0));      listNilColorsNames.append("FrontD")   ### 13: Rojo2
        listNilColors.append((215,165,159));  listNilColorsNames.append("IntWall")  ### 14: Topo
        listNilColors.append((180,1,1));      listNilColorsNames.append("IntDoor")  ### 15: Granate
        listNilColors.append((0, 180, 255));  listNilColorsNames.append("Window")    ### 16: Azul claro
        #listNilColors.append((999,999,999));  listNilColorsNames.append("StudyR")   ### # No usado
        #listNilColors.append((999,999,999));  listNilColorsNames.append("WallIn")   ### # No usado

    elif listColors == 2:  ### Predefined list of colors with "large distances" among the colors

        listNilColors.append((255,0,0));      listNilColorsNames.append("LivingR")  ###  0:  0 en canal 2
        listNilColors.append((0,255,0));      listNilColorsNames.append("MasterR")  ###  1:  1 en canal 2
        listNilColors.append((0,0,255));      listNilColorsNames.append("Kitchen")  ###  2:  2 en canal 2
        listNilColors.append((127,0,255));    listNilColorsNames.append("BathR")    ###  3:  3 en canal 2
        listNilColors.append((255,0,255));    listNilColorsNames.append("DiningR")  ###  4:  4 en canal 2
        listNilColors.append((139,69,69));    listNilColorsNames.append("ChildR")   ###  5:  5 en canal 2
        #listNilColors.append((128,0,0));      listNilColorsNames.append("StudyR")   ### # No usado -  6 en canal 2 
        listNilColors.append((0,0,127));      listNilColorsNames.append("SecondR")  ###  6:  7 en canal 2
        listNilColors.append((255,127,0));    listNilColorsNames.append("GuestR")   ###  7:  8 en canal 2
        listNilColors.append((0,127,0));      listNilColorsNames.append("Balcony")  ###  8:  9 en canal 2
        listNilColors.append((0,127,255));    listNilColorsNames.append("Entrance") ###  9: 10 en canal 2
        listNilColors.append((127,0,0));      listNilColorsNames.append("Storage")  ### 10: 11 en canal 2
        #listNilColors.append((255,68,0));     listNilColorsNames.append("WallIn")   ### # No usado - 12 en canal 2
        listNilColors.append((255,255,255));  listNilColorsNames.append("BackGrd")  ### 11: 13 en canal 2
        listNilColors.append((0,0,0));        listNilColorsNames.append("ExtWall")  ### 12: 14 en canal 2
        listNilColors.append((0,255,255));    listNilColorsNames.append("FrontD")   ### 13: 15 en canal 2
        listNilColors.append((192,192,68));   listNilColorsNames.append("IntWall")  ### 14: 16 en canal 2
        listNilColors.append((255,255,0));    listNilColorsNames.append("IntDoor")  ### 15: 17 en canal 2
        listNilColors.append((0, 180, 255));  listNilColorsNames.append("Window")    ### 16: 18 en canal 2
        
    nListNilColors = len(listNilColors)
    #
    idxNilColorBackground   = 11
    idxNilColorExteriorWall = 12
    idxNilColorFrontDoor    = 13
    idxNilColorInteriorWall = 14
    idxNilColorInsideDoor   = 15
    idxNilColorEntrance     =  9
    idxNilColorBalcony      =  8   
    idxNilColorAux1         =  7   ### A very little used color
    idxNilColorAux2         =  6   ### A very little used color
    idxNilColorWindow       = 16   ### Window color
    #

    ###
    ### GLOBAL DICTIONARIES
    ###
    gParam = {}   ### Global parameters
    sParam = {}   ### Search parameters (for finding valid elements inthe image)

    ### Distances in the image
    ## Number of pixels of one meter
    gParam['nPixelsPerMeter']  = 11
    ## Number of pixels of one squared meter
    gParam['nPixelsPerMeter2'] = gParam['nPixelsPerMeter'] * gParam['nPixelsPerMeter']  ### 121
    
    ### Colors
    gParam['listNilColors']                     = listNilColors
    gParam['listNilColorsNames']                = listNilColorsNames
    gParam['nListNilColors']                    = nListNilColors
    gParam['idxNilColorBackground']             = idxNilColorBackground
    gParam['idxNilColorExteriorWall']           = idxNilColorExteriorWall
    gParam['idxNilColorFrontDoor']              = idxNilColorFrontDoor
    gParam['idxNilColorInteriorWall']           = idxNilColorInteriorWall
    gParam['idxNilColorInsideDoor']             = idxNilColorInsideDoor
    gParam['idxNilColorEntrance']               = idxNilColorEntrance
    gParam['idxNilColorBalcony']                = idxNilColorBalcony
    gParam['idxNilColorAux1']                   = idxNilColorAux1
    gParam['idxNilColorAux2']                   = idxNilColorAux2
    gParam['idxNilColorWindow']                 = idxNilColorWindow
    ## Norma para calcular distancias entre colores
    gParam['normDistColors'] = 'norm2_3D'

    ### Parameters for plotting
    gParam['grayAdjPlot']  = (128,128,128)
    gParam['dxyCentroid']  = 3
    gParam['dxyCorner']    = 2
    gParam['dxyPoint']     = 3
    gParam['dxyFrontDoor'] = 1

    ### Height and width of the image
    gParam['height']      = heightImage
    gParam['width']       = widthImage

    ### Frames
    ## Extra pixels around the computed frame (see obtainImagesNilColors)
    gParam['extraFrameX'] = 5
    gParam['extraFrameY'] = 5 

    ### Centroids
    ## List of indexes of colors that represent "rooms with/without centroid"
    listNilColorsRoomsWithCentroid    = [0,1,2,3,4,5,6,7,8,9,10]
    listNilColorsRoomsWithoutCentroid = list(set(np.arange(len(listNilColors))) - set(listNilColorsRoomsWithCentroid))
    gParam['listNilColorsRoomsWithCentroid']    = listNilColorsRoomsWithCentroid
    gParam['listNilColorsRoomsWithoutCentroid'] = listNilColorsRoomsWithoutCentroid
    ## Minimum number of pixels in a connex component to be considered a "centroid of a room"
    ## DEPENDS ON THE TYPE OF ROOM
    gParam['listNilColorsRoomsWithCentroidSmall']      = [2,3,  8,  0,4,  1,5,6,7]
    gParam['minPixelsConnexComponentCentroidSmall']    = 2 * gParam['nPixelsPerMeter2']  ### Smallest ones
    gParam['listNilColorsRoomsWithCentroidLarge']      = []   ### Mejor que salga un error a que desaparezca de antemano
    gParam['minPixelsConnexComponentCentroidLarge']    = 6 * gParam['nPixelsPerMeter2']  ### Largest ones
    gParam['listNilColorsRoomsWithCentroidAnySize']    = [9,10]
    gParam['minPixelsConnexComponentCentroidAnySize']  = gParam['nPixelsPerMeter2']      ### Any Size (= minimum size)

    ### Adjacencies
    ## Maximo de pixels consecutivos vacíos para deducir NO adyacencia
    gParam['maxNegMapPointsAdj'] = 10

    ### Kernels
    # Tamaños de los kernels ideales en la búsqueda en el contorno
    gParam['sizeIdealCorner'] = [ 5, 7, 9, 11, 13, 15, 17 ]       ### No es un solo valor, hay que mirarlos todos!!!

    ### Parámetros de búsqueda para el cálculo del contorno
    sParam['levelFindContour']                = [0.5]
    sParam['fullyConnectedFindContour']       = ['high', 'low']   ### Probably 'high' is enough
    sParam['positiveOrientationFindContour']  = ['low', 'high']   ### Probably 'low' is enough

    ### Parámetros de búsqueda para el cálculo de esquinas y el perímetro
    # Parámetros para 'corner_harris' y 'corner_peaks'
    sParam['kvalueHarris']    = [ 0.05 ]
    sParam['sigmaHarris']     = [ 1.0 ]
    sParam['mindistPeaks']    = [ 1 ]
    sParam['thresholdPeaks']  = [ 0.20 ]
    # Tamaño de la puerta de entrada (en el perímetro)
    sParam['sizeFrontDoor'] = [ gParam['nPixelsPerMeter'] ]   ### [Lluis: 1 metro (11 pixels/metro)]

    del listNilColors, nListNilColors
    del idxNilColorBackground, idxNilColorExteriorWall, idxNilColorFrontDoor, idxNilColorInteriorWall
    del idxNilColorAux1, idxNilColorAux2
    del listNilColorsRoomsWithCentroid, listNilColorsRoomsWithoutCentroid

    gParam['MasiveRun'] = 1  ### 1: Search for good parameters / 2: One shot
    gParam['Debug'] = False

    return gParam, sParam


####################################################################################################################
### MASIVE PROCEDURE
####################################################################################################################
    
def masiveRun(pathImages,pathSaveImages,matlabFileName):

    from os import listdir, makedirs
    from os.path import isfile, join, exists
    
    ### Si no existe el directorio, lo crea
    if not exists(pathSaveImages):
        makedirs(pathSaveImages)

    filesImages = [f for f in listdir(pathImages) if isfile(join(pathImages, f))]
    filesImages.sort()

    lDataMatlab   = []
    lIdDataMatlab = []
    n = 0
    for fileImage in filesImages:
        n += 1
        #if n > 2: continue    ### ??? COMENTAR: Solo para pruebas
        #
        #fileImage = 'seed0001.png'
        #
        print()
        print('---',fileImage)
        print()
        #
        oriPilImage = Image.open(pathImages + '/' + str(fileImage), 'r')  ### read
        widthImage, heightImage = oriPilImage.size
        #
        gParam, sParam = defineGlobalAndSearchParameters(heightImage, widthImage)
        del heightImage, widthImage

        ####################################################################################################

        if gParam['MasiveRun'] == 1:

            sanityCheckOKCentroids, sanityCheckOKContour, successfulWalkContour, successfulPerimeter, successfulFrontDoor, \
            lCentroids, lConnexComponents, lConnexComponentsCentroids, lAdjacenciesCC, lCornersIdealImage, lPerimeter, pwalls = \
                findValidElementsInImage (gParam, sParam, oriPilImage)

        else:

            ####################################################################################################
            #
            gParam, intImageNilColors, rgbImageNilColors, arrImageNilColors = obtainImagesNilColors (gParam,oriPilImage)
            #
            lCentroids, lConnexComponentsCentroids, lConnexComponents = getConnexComponentsCentroids (gParam, rgbImageNilColors)
            #
            sanityCheckOKCentroids = sanityCheckCentroids (gParam, rgbImageNilColors, lCentroids, lConnexComponentsCentroids)
            #
            lAdjacenciesCC = getAdjacenciesCentroids (gParam, lConnexComponents, lConnexComponentsCentroids, lCentroids)
            #
            ####################################################################################################
            #
            sParamTry1 = {}
            #
            sParamTry1['levelFindContour']                = 0.5
            sParamTry1['fullyConnectedFindContour']       = 'high'
            sParamTry1['positiveOrientationFindContour']  = 'low'
            #
            intContourBackgroundImage, imageContour, intRegionContour, rgbRegionContour = \
              getContour_from_SkimageContour (gParam, sParamTry1, rgbImageNilColors)
            #
            lKernelsContourSearch, lKernelsContourSmooth = defineKernelsForCleanAndSmoothContour()
            imageContourSmoothed = cleanAndSmoothContour (imageContour, lKernelsContourSearch, lKernelsContourSmooth)
            #
            # lConnexComponents, wall_mask = build_interior_walls_from_borders(...)  <-- MOVED DOWN
            sanityCheckOKContour = sanityCheckContourSmoothed (gParam, imageContourSmoothed, lCentroids)
            #
            successfulWalkContour, orderContour = walkContourInOrder (gParam, imageContourSmoothed)
            #
            if not (sanityCheckOKContour and successfulWalkContour):

                successfulPerimeter = False
                successfulFrontDoor = False
                #
                lCornersIdealImage = None
                lPerimeter = None

            else:

                sParamTry2 = {}
                #
                sParamTry2['kvalueHarris']    = 0.05                       ### Solo un valor
                sParamTry2['sigmaHarris']     = 1.0                        ### Solo un valor
                sParamTry2['mindistPeaks']    = 1                          ### Solo un valor
                sParamTry2['thresholdPeaks']  = 0.20                       ### Solo un valor
                #
                sParamTry2['sizeIdealCorner'] = [5, 7, 9, 11, 13, 15, 17]  ### *** Lista de valores
                #
                sParamTry2['sizeFrontDoor']   = 15                         ### Solo un valor
                #
                lCornersIdeal = defineCornersIdeal_AllSizes (gParam)
                #
                lCornersHarris = getCorners_from_SkimageHarris (sParamTry2, imageContourSmoothed)
                #
                lCornersIdealImage = \
                  checkCornersFound_wrt_CornersIdealInArray01 (gParam, imageContourSmoothed, lCornersHarris, lCornersIdeal)
                #
                orderNearestCorners = obtainCorrespondenceCornersOrderedContour (lCornersIdealImage, orderContour)
                #
                lPositionsFrontDoor = getPositionsFrontDoor (gParam, rgbImageNilColors)
                #
                successfulPerimeter, successfulFrontDoor, lPerimeter = \
                  constructPerimeterWithFrontDoor (sParamTry2, orderNearestCorners, lPositionsFrontDoor)

            # Construimos muros interiores AHORA que tenemos el perímetro final
            # Esto permite rellenar huecos entre las habitaciones y el nuevo perímetro
            lConnexComponents,  pwalls = build_interior_walls_from_borders(gParam, rgbImageNilColors, lConnexComponents,
                                                                                imageContourSmoothed=imageContourSmoothed,
                                                                                lPerimeter=lPerimeter, 
                                                                                min_segment_len=12,
                                                                                thickness_px=1
                                                                            )

        ################################################
        #
        ### FINAL RESULT
        #
        failProcess = not (sanityCheckOKCentroids and \
                           sanityCheckOKContour and successfulWalkContour and \
                           successfulPerimeter and successfulFrontDoor)
        if failProcess:
            ### Cuando no ha habido exito, pinta un punto indicándolo
            print('   FAIL!!!!!',sanityCheckOKCentroids,sanityCheckOKContour,successfulWalkContour,successfulPerimeter,successfulFrontDoor)
        else:
            dataMatlab = obtainDataMatlab \
                           (fileImage, lPerimeter, lCentroids, lConnexComponentsCentroids, lAdjacenciesCC)
            lDataMatlab.append(dataMatlab)
            lIdDataMatlab.append(dataMatlab['name'])
            print('   SUCCESS!!!!!')

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(30, 20)) #,layout="constrained")
        #
        axs[0].title.set_text("Original Image"); axs[0].imshow(oriPilImage);
        #
        intResImage = constructResultingImage \
                        (gParam, lConnexComponents, lCentroids=None, lAdjacencies=None,
                         lCorners=None, lPerimeter=None, plotPoint=False, pwalls=None)
        pilResImage = Image.frombytes('RGB', (gParam['width'], gParam['height']), bytes(intResImage))
        axs[1].title.set_text("Intermediate Image"); axs[1].imshow(pilResImage);
        #
        intResImage = constructResultingImage \
                        (gParam, lConnexComponents, lCentroids=lCentroids, lAdjacencies=lAdjacenciesCC,
                        lCorners=lCornersIdealImage, lPerimeter=lPerimeter, plotPoint=failProcess, pwalls=None)
        pilResImage = Image.frombytes('RGB', (gParam['width'], gParam['height']), bytes(intResImage))
        axs[2].title.set_text("Centroids Image"); axs[2].imshow(pilResImage);
        #
        initResImage =  constructResultingImage \
                        (gParam, lConnexComponents, lCentroids=None, lAdjacencies=None,
                         lCorners=None, lPerimeter=lPerimeter, plotPoint=False, pwalls=pwalls)
        pilInitResImage = Image.frombytes('RGB', (gParam['width'], gParam['height']), bytes(initResImage))
        axs[3].title.set_text("Final Image"); axs[3].imshow(pilInitResImage);
        #
        fig.savefig(pathSaveImages + '/' + fileImage)
        #
        plt.close()

        ### At the end of the loop...
        sys.stdout.flush()
        if n % 20 == 0:
            ### GUARDA LOS DATOS OBTENIDOS EN FORMATO MATLAB
            saveDataToMatlab(pathSaveImages + '/' + matlabFileName,lDataMatlab)
            #saveStringList(pathSaveImages + '/' + 'train.txt', lIdDataMatlab)
            #saveStringList(pathSaveImages + '/' + 'test.txt',  lIdDataMatlab)
            #
            ### GUARDA LOS DATOS OBTENIDOS EN FORMATO PICKLE
            saveDataToPickle(pathSaveImages + '/' + matlabFileName,lDataMatlab)

    #
    ### GUARDA LOS DATOS OBTENIDOS EN FORMATO MATLAB
    saveDataToMatlab(pathSaveImages + '/' + matlabFileName,lDataMatlab)
    #saveStringList(pathSaveImages + '/' + 'train.txt', lIdDataMatlab)
    #saveStringList(pathSaveImages + '/' + 'test.txt',  lIdDataMatlab)
    #
    ### GUARDA LOS DATOS OBTENIDOS EN FORMATO PICKLE
    saveDataToPickle(pathSaveImages + '/' + matlabFileName,lDataMatlab)


####################################################################################################################

####################################################################################################################


if __name__ == '__main__':

    dataName = 'publicplanrgb08_24'

    pathImages = './pintadafinal_24/' + dataName
    pathSaveImages = 'Graph-' + dataName + '-Relaxed'
    matlabFileName = dataName + '-data4Retrieve'

    masiveRun(pathImages,pathSaveImages,matlabFileName)
    #
