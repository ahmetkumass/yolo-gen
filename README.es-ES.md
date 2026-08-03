

<p align="center">
  <img src="assets/yolo-gen-banner.png" alt="YoloGen">
</p>

# YoloGen

**Entrena YOLO + VLM con un solo comando. Sin etiquetado adicional.**

```
Imagen + etiquetas YOLO → Auto-generación de datos de entrenamiento VLM → Modelo afinado
```

Entrena la detección de objetos y una "segunda opinión" de VLM a partir de un conjunto de datos estándar de YOLO. Los datos de entrenamiento del VLM se generan automáticamente a partir de las etiquetas de YOLO, incluyendo **negativos difíciles** extraídos directamente de tus imágenes existentes, sin necesidad de ejecutar el detector.

## Casos de Uso

YOLO localiza objetos (bbox) → VLM analiza la región enmarcada en rojo y ya sea **describe** o **verifica**:

| Escenario | Modo Descriptivo | Modo de Verificación |
|----------|------------------|-------------------|
| **Detección de defectos** | `{"defect": true, "type": "scratch", "size": "2mm"}` | `Yes` / `No` |
| **Detección de armas** | `{"weapon": true, "type": "rifle"}` | `Yes` / `No` |
| **Daños en vehículos** | `{"damaged": true, "part": "front bumper"}` | `Yes` / `No` |
| **Imagenología médica** | `{"finding": true, "type": "nodule", "size": "6mm"}` | `Yes` / `No` |

## ¿Por qué YOLO + VLM?

- **Solo YOLO**: Rápido, pero insuficiente para una precisión de nivel producción
- **Solo VLM**: Inteligente, pero demasiado lento para producción
- **YOLO + VLM**: Detección rápida + el VLM añade descripciones detalladas, clasificación, **y filtrado de falsos positivos**

## Dos modos de entrenamiento del VLM

YoloGen admite dos modos de entrenamiento para la etapa del VLM. Elige según lo que consuma tu sistema aguas abajo.

### 1. Modo descriptivo (predeterminado)

Generación de descripciones basada en plantillas. Dada una imagen con caja roja, el VLM produce una descripción legible por humanos o JSON estructurado. Ideal cuando necesitas metadatos ricos por cada detección.

```yaml
vlm_dataset:
  qa_format: descriptive      # default
  prompts:
    - question: "What is in the red marked area?"
      answer:   "The red marked area contains a {class}. {detail}"
```

### 2. Modo de verificación (binario Sí/No)

Supervisión Sí/No por caja delimitadora (bbox). Para cada bbox de verdad territorial (GT) y cada clase en tu conjunto de datos, YoloGen genera una muestra: clase coincidente → `"Yes"`, otras clases → `"No"`. Los negativos difíciles interclase se generan automáticamente.

```yaml
vlm_dataset:
  qa_format: binary_multiclass
  class_prompts:
    handgun: |
      Decide if the red bounding box contains a handgun.
      Answer Yes or No only.
    rifle: |
      Decide if the red bounding box contains a long gun (rifle or shotgun).
      Answer Yes or No only.
```

El modo de verificación está diseñado para combinarse con un detector existente: usa YOLO para proponer regiones y el VLM afinado para **rechazar falsos positivos** en tiempo de inferencia.

## Minería de negativos difíciles — Sin detector

El modo de verificación desbloquea un paso adicional: **generación automática de negativos difíciles** directamente desde tus bboxes GT, sin necesidad de ejecutar un detector.

```
bbox GT → búsqueda de candidatos basada en anillos → filtro de similitud DINOv2 → muestras "No"
```

Para cada bbox positivo, YoloGen escanea anillos concéntricos a su alrededor, inserta (embeds) cada región candidata con un codificador auto-supervisado preentrenado (DINOv2 por defecto) y conserva los candidatos cuya similitud con el positivo cae dentro de una ventana de "negativo difícil" configurable (predeterminado `0.25–0.50`). Estas son regiones que parecen positivas pero no lo son: exactamente el tipo de muestra que entrena a un VLM para rechazar falsos positivos (FP) de detectores en el mundo real.

Medidas de seguridad de múltiples capas protegen contra filtraciones de verdaderos positivos:

1. **Exclusión por IoU** frente a cada bbox GT en la imagen (seguro para múltiples objetos)
2. **Límite estricto de similitud** para rechazar candidatos que puedan contener realmente el objetivo
3. **Filtro de diversidad** entre las regiones retenidas
4. *Opcional:* doble verificación **VLM zero-shot** — preguntar a un VLM base si la región contiene la clase; descartar si responde "Yes"

Actívalo con un solo bloque de configuración:

```yaml
vlm_dataset:
  qa_format: binary_multiclass       # required
  class_prompts: { ... }             # one system prompt per class
  negative_mining:
    enabled: true
    embedding_model: facebook/dinov2-base
    rings: [3.0, 6.0]                # multiples of GT bbox min side
    similarity_range: [0.25, 0.50]
    max_per_image: 3
    exclude_iou_with_any_gt: 0.1
    diversity_iou_threshold: 0.3
    # optional 4th safeguard (slow, opt-in)
    vlm_verify:
      enabled: false
      model: Qwen/Qwen3-VL-4B-Instruct
```

El enfoque es agnóstico al dominio. El mismo patrón de configuración funciona para armas, defectos, imagenología médica, daños en vehículos o cualquier tarea de detección donde "parece, pero no es" sea un concepto significativo.

## Familias de VLM soportadas

Cualquiera de los siguientes puede pasarse como `--vlm-model` (o `vlm.model`
en una configuración). La fábrica selecciona el adaptador correcto automáticamente basándose
en el id de HuggingFace. Agregar una nueva familia es una adición de un solo archivo
contra la interfaz `VLMBase` en `yologen/models/vlm/`.

| Familia | Tamaños | IDs de HuggingFace |
|---|---|---|
| **Qwen 2.5-VL** | 3B, 7B | `Qwen/Qwen2.5-VL-{3B,7B}-Instruct` |
| **Qwen 3-VL** (predeterminado) | 2B, 4B, 8B | `Qwen/Qwen3-VL-{2B,4B,8B}-Instruct` |
| **InternVL 3.5** | 1B, 4B, 8B | `OpenGVLab/InternVL3_5-{1B,4B,8B}` |
| **GLM-4.6V-Flash** | 9B | `zai-org/GLM-4.6V-Flash` |

Todos los VLMs se entrenan con QLoRA + LoRA de 4 bits por defecto; el adaptador
por familia maneja su propio preprocesamiento de imágenes, plantilla de chat y módulos
objetivo de LoRA.

## Construido con

- [Ultralytics YOLOv8/v11](https://github.com/ultralytics/ultralytics) — implementación de YOLO líder en la industria
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) / [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) / [InternVL 3.5](https://huggingface.co/OpenGVLab/InternVL3_5-4B) / [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) — modelos de visión y lenguaje
- [PEFT / QLoRA](https://github.com/huggingface/peft) — afinamiento eficiente en parámetros
- [DINOv2](https://github.com/facebookresearch/dinov2) — características de visión auto-supervisadas para minería de negativos difíciles

## Inicio Rápido

### 1. Instalación

```bash
pip install -r requirements.txt
```

### 2. Preparar Conjunto de Datos

Formato estándar de YOLO:
```
data/my_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

Ejemplo de `dataset.yaml`:
```yaml
path: .  # Dataset root (relative to this file)
train: images/train
val: images/val

names:
  0: class_a
  1: class_b
```

### 3. Configurar

```bash
cp configs/default.yaml configs/my_run.yaml
# then edit `data:` inside my_run.yaml
```

`configs/default.yaml` es la única fuente de verdad. Los campos requeridos
no están comentados; las características avanzadas (modo de verificación, minería de
negativos difíciles) viven como bloques comentados — descomenta para habilitar.

### 4. Entrenar

```bash
python train.py --config configs/my_run.yaml
```

Esto ejecutará:
1. Entrenar YOLO (100 épocas)
2. Generar conjunto de datos VLM (pares P&R con cajas rojas)
3. Entrenar VLM con QLoRA (3 épocas)
4. Exportar a ONNX
5. Generar visualizaciones

#### Banderas de salto

Ejecuta solo una parte de la tubería:

```bash
# VLM dataset only (no YOLO training, no VLM training)
python train.py --config configs/default.yaml --skip-yolo --skip-vlm-training

# YOLO only
python train.py --config configs/default.yaml  # with vlm.enabled: false in config

# Reuse an existing VLM dataset, retrain VLM
python train.py --config configs/default.yaml --skip-yolo --skip-vlm-data
```

### 5. Predecir

```bash
# YOLO only
python predict.py --weights runs/exp_xxx/yolo/weights/best.pt --source image.jpg

# YOLO + VLM
python predict.py --weights runs/exp_xxx/yolo/weights/best.pt --source image.jpg \
    --vlm --vlm-adapter runs/exp_xxx/vlm/best
```

### 6. Evaluar (Comparar Base vs Afinado)

```bash
jupyter notebook examples/compare_vlm.ipynb
```

Compara tu VLM afinado contra el modelo base para medir las mejoras.

### API de Python

```python
from yologen.core.predictor import YOLOPredictor, VLMPredictor, UnifiedPredictor

# Solo YOLO
yolo = YOLOPredictor(weights="best.pt")
results = yolo.predict("image.jpg")

# Solo VLM (para imágenes con cajas delimitadoras existentes)
vlm = VLMPredictor(vlm_adapter="vlm/best")
answer = vlm.predict(image="image.jpg", bbox=[100, 100, 300, 300], question="What is this?")

# YOLO + VLM combinados
predictor = UnifiedPredictor(yolo_weights="best.pt", vlm_adapter="vlm/best")
results = predictor.predict(source="image.jpg", vlm_question="What is in the red box?")
```

**Modo de verificación** (adaptadores entrenados con `qa_format: binary_multiclass`):

```python
# Los metadatos del adaptador (class_prompts, qa_format) se cargan automáticamente desde
# el config.json del adaptador cuando se construye el predictor.
vlm = VLMPredictor(vlm_adapter="runs/exp_xxx/vlm/best")

# Preguntar al modelo sobre una clase
result = vlm.verify(
    image="frame.jpg",
    bbox=[120, 340, 280, 520],
    target_class="handgun",
)
# → {"label": "Yes" | "No" | "unknown", "raw": "...", "target": "handgun"}

# O ejecutar todas las clases que conoce el adaptador en una sola llamada
all_results = vlm.verify_all(image="frame.jpg", bbox=[120, 340, 280, 520])
```

**Minería de negativos difíciles** — uso independiente del minerador:

```python
from PIL import Image
from yologen.data import NegativeMiner, GTBox

miner = NegativeMiner({
    "enabled": True,
    "embedding_model": "facebook/dinov2-base",
    "rings": [3.0, 6.0],
    "similarity_range": [0.25, 0.50],
    "max_per_image": 3,
})

image = Image.open("frame.jpg")
gt_boxes = [GTBox(bbox=(900, 605, 987, 664), class_id=0, class_name="handgun")]

regions = miner.mine_image(image, gt_boxes)
for r in regions:
    print(r.bbox, r.similarity, r.ring_idx, r.source_gt_class)
```

El punto de entrada de conjunto de datos completo del minerador, `mine_dataset(pairs)`, devuelve
las regiones minadas más las estadísticas agregadas `MiningStats`, que es lo que usa
internamente la tubería impulsada por YAML.

## Configuración

Copia y edita `configs/default.yaml`. Dos configuraciones representativas:

### A. Modo descriptivo (predeterminado)

```yaml
yolo:
  model: yolov8n.pt
  epochs: 100
  batch: 16

vlm:
  enabled: true
  model: Qwen/Qwen3-VL-4B-Instruct     # Qwen2.5-VL / Qwen3-VL / InternVL 3.5 / GLM-4.6V-Flash supported
  epochs: 3
  precision: 4bit

vlm_dataset:
  qa_format: descriptive               # default
  box_color: [0, 0, 255]               # BGR red
  box_thickness: 3
  system_prompt: |
    You are an object detection assistant.
    Identify objects in red marked areas clearly.
  prompts:
    - question: "What is in the red marked area?"
      answer:   "The red marked area contains a {class}. {detail}"
```

### B. Modo de verificación + minería de negativos difíciles

```yaml
yolo:
  model: yolov8m.pt
  epochs: 150

vlm:
  enabled: true
  model: Qwen/Qwen3-VL-4B-Instruct
  epochs: 2
  precision: 4bit

vlm_dataset:
  qa_format: binary_multiclass
  box_color: [0, 0, 255]
  box_thickness: 3

  class_prompts:
    handgun: |
      You are a security analyst reviewing weapon-detection alerts.
      Decide if the red bounding box contains a handgun.
      Answer Yes or No only.
    rifle: |
      You are a security analyst reviewing weapon-detection alerts.
      Decide if the red bounding box contains a long gun (rifle or shotgun).
      Answer Yes or No only.

  negative_mining:
    enabled: true
    embedding_model: facebook/dinov2-base
    rings: [3.0, 6.0]
    similarity_range: [0.25, 0.50]
    max_per_image: 3
```

Consulta [`configs/default.yaml`](configs/default.yaml) para ver todos los campos disponibles.

## Estructura de Salida

```
runs/exp_20251217_xxx/
├── yolo/
│   └── weights/
│       ├── best.pt           # YOLO model
│       └── best.onnx         # ONNX export
├── vlm/
│   └── best/                 # VLM adapter (~150MB)
└── visualizations/
    ├── training_curves.png
    └── prediction_samples.png
```

## Características Principales

| Característica | Descripción |
|---------|-------------|
| Configuración Única | Un solo YAML controla todo |
| Entrenamiento Secuencial | YOLO → VLM automáticamente |
| Dos modos de VLM | Descripción basada en plantillas o verificación binaria Sí/No |
| **Minería de Negativos Difíciles** | Sin detector, generación espacial basada en anillos + similitud de embedding de muestras "No". Agnóstico al dominio. |
| **Minería con Salvaguardas** | Protección multicapa contra filtración de TP (IoU, límite de similitud, diversidad, verificación opcional de VLM) |
| QLoRA | Entrenamiento de VLM de 4B / 7B / 8B con cuantización de 4 bits |
| Anclaje Visual (Visual Grounding) | Las cajas rojas vinculan la detección al VLM |
| Banderas de Salto | `--skip-yolo`, `--skip-vlm-data`, `--skip-vlm-training` para ejecuciones modulares |
| Configurable | Colores, prompts, modelos y parámetros de minería, todo en YAML |

## Requisitos

- Python 3.10+

### Uso de Memoria de GPU

| Tarea | VRAM |
|------|------|
| Entrenamiento de YOLO | 4-12 GB |
| VLM 2B-3B | ~14-18 GB |
| VLM 4B | ~18-20 GB |
| VLM 7B-8B | ~24-28 GB |

*La memoria del VLM depende de la configuración `max_pixels`. Los valores anteriores son para QLoRA de 4 bits con la configuración predeterminada de píxeles.*

## Ejemplo de Resultados

**Entrada**: Imagen de producto de línea de ensamblaje

**Salida de YOLO**:
```
[defect] conf=0.92 bbox=[120, 340, 280, 520]
```

**Salida de VLM**:
```json
{"defect": true, "type": "scratch", "size": "3mm"}
```

## Preguntas Frecuentes (FAQ)

**¿Necesito escribir manualmente los datos de entrenamiento del VLM?**
No. YoloGen genera automáticamente pares de preguntas y respuestas a partir de tus etiquetas de YOLO. Solo prepara un conjunto de datos en formato estándar de YOLO.

**¿Cuántas imágenes necesito?**
Mínimo ~100 imágenes para YOLO, se recomiendan ~500+ para mejores resultados con el VLM. La minería de negativos difíciles escala con el tamaño del conjunto de datos: cada bbox positivo puede contribuir hasta `max_per_image` muestras "No".

**¿Puedo usar solo YOLO sin VLM?**
Sí. Establece `vlm.enabled: false` en la configuración, o simplemente usa `predict.py` sin la bandera `--vlm`.

**¿Cuándo debería usar el modo `binary_multiclass`?**
Cuando tu sistema aguas abajo solo necesite Sí / No por clase; por ejemplo, validando las salidas del detector para reducir falsos positivos. El modo emite automáticamente negativos interclase (bbox de Clase A preguntado como Clase B → `"No"`), y es un prerrequisito para la minería de negativos difíciles.

**¿Qué hace realmente la minería de negativos difíciles?**
Para cada bbox GT encuentra regiones de imagen que se parecen al positivo pero no contienen ningún objetivo, y las etiqueta como `"No"`. A diferencia del ciclo clásico de "ejecutar el detector para recopilar FP", esto no requiere un detector ni etiquetado adicional. Los valores predeterminados están ajustados para DINOv2 (`similarity_range: [0.25, 0.50]`); intercambia el modelo de embedding para otras arquitecturas base.

**¿La minería de negativos difíciles etiquetará erróneamente un positivo real como `"No"`?**
Tres salvaguardas están siempre activas: (1) se descarta cualquier candidato que se superpone a algún bbox GT por encima de `exclude_iou_with_any_gt`, (2) se descartan candidatos por encima de `similarity_range[1]` (podrían ser el objetivo), (3) las regiones retenidas se deduplican por IoU. Una cuarta salvaguarda, la verificación VLM zero-shot, es opcional y ejecuta un VLM base en cada candidato, descartando cualquier cosa que llame `"Yes"`. En la práctica, combinar las primeras tres da tasas de filtración de TP muy por debajo del 1% en conjuntos de datos típicos.

**¿Cuánta VRAM necesito?**
Consulta la tabla de Uso de Memoria de GPU arriba. Una RTX 4090 (24 GB) puede entrenar tanto modelos de 3B como de 7B con la configuración predeterminada. La minería de negativos difíciles con DINOv2-base necesita solo ~1 GB extra y se ejecuta en CPU / MPS / CUDA.

**¿Cómo personalizo las respuestas del VLM?**
En modo descriptivo, edita `system_prompt`, `prompts` y `details` bajo `vlm_dataset`. En modo `binary_multiclass`, edita `class_prompts` (un prompt de sistema por clase) y opcionalmente `class_questions`.

## Licencia

MIT

Nota: Este proyecto utiliza [Ultralytics](https://github.com/ultralytics/ultralytics), licenciado bajo AGPL-3.0. Consulta su licencia para más detalles.
