# TFG — Pipeline para enriquecer planos generados por IA (vivienda social)

**Calificación obtenida:** **9,7/10** ✅

Este repositorio contiene el código y los recursos de mi Trabajo de Fin de Grado (TFG): un **pipeline automatizado** que transforma **planos generados por IA** (imágenes raster 256×256 con codificación por colores) en planos **enriquecidos** con estructura arquitectónica (perímetro, muros, aperturas) y **elementos interiores** (puertas, ventanas, mobiliario), aplicando heurísticas geométricas y verificación de colisiones mediante máscaras.


## Motivación
El diseño y enriquecimiento manual de planos (especialmente en contextos de **vivienda social**) consume tiempo y recursos. Con la llegada de modelos generativos capaces de proponer distribuciones, surge el reto de **convertir layouts “crudos”** en planos utilizables: consistentes, con aperturas realistas, mobiliario plausible y métricas verificables.

---

## Qué hace el proyecto
A partir de una imagen segmentada por clases (por ejemplo, suelo/estancias/obstáculos codificados por color), el sistema:

- ✅ **Reconstruye geometría** de perímetros y habitaciones.
- ✅ Genera **muros interiores** a partir de bordes/segmentación.
- ✅ Inserta **puertas y ventanas** con reglas geométricas y de conectividad.
- ✅ Añade **mobiliario** respetando espacio libre y colisiones (máscaras raster).
- ✅ Exporta **outputs** listos para análisis/visualización y evaluación cuantitativa.

---

## Resultados
Incluye:
- Ejemplos **before/after** (inputs generados vs. outputs enriquecidos).

---

## Arquitectura del pipeline
1. **Parseo del layout** (raster → componentes/estancias)
2. **Extracción y suavizado de perímetro**
3. **Generación de muros interiores** y segmentación por estancia
4. **Colocación de aperturas** (puertas/ventanas) con validación
5. **Creación de máscaras raster internas** (suelo, paredes, puertas, ventanas, obstáculos)
6. **Colocación de mobiliario** con verificación de colisiones
7. **Evaluación** (cobertura, conectividad, fragmentación, ortogonalidad, etc.)

---



