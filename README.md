# vid2model

CLI-пайплайн для конвертации видео в скелетную анимацию:

- `BVH`
- `JSON`
- `CSV`
- `NPZ`
- `TRC`
- `FBX` (через Blender CLI)

Также в проекте есть локальный браузерный viewer (`viewer/index.html`) для предпросмотра `.bvh`.

## Quick Start

Обычная конвертация:

```bash
./convert.sh think.mp4 output/think.bvh
```

Auto-mode с обученной моделью:

```bash
python3 tools/generate_auto_pose_dataset.py \
  --input think.mp4 \
  --label default \
  --output output/auto_pose_dataset.jsonl

python3 tools/train_auto_pose_model.py \
  --input output/auto_pose_dataset.jsonl \
  --output models/auto_pose_model.npz

./convert.sh --config config.auto.json think.mp4 output/think.bvh
```

One-liners:

```bash
./convert.sh think.mp4 output/think.bvh
python3 tools/generate_auto_pose_dataset.py --input think.mp4 --label default --output output/auto_pose_dataset.jsonl
python3 tools/train_auto_pose_model.py --input output/auto_pose_dataset.jsonl --output models/auto_pose_model.npz
./convert.sh --config config.auto.json think.mp4 output/think.bvh
```

## Структура

- `convert_video_to_bvh.py` - основной конвертер (MediaPipe Tasks API).
- `convert.sh` - удобный shell-раннер, умеет конвертировать во все форматы и опционально в FBX.
- `bvh_to_fbx.sh` - BVH -> FBX через Blender в headless-режиме.
- `export_bvh_to_fbx_blender.py` - скрипт экспорта внутри Blender.
- `viewer/index.html` - локальный BVH viewer (Three.js).

## Требования

- Python 3.10+ (рекомендуется 3.11+)
- `pip`
- (опционально) Blender для `FBX`

`convert.sh` автоматически:

1. создаёт `.venv` (если её нет),
2. обновляет `pip`,
3. ставит зависимости из `requirements.txt`.

## Быстрый старт

```bash
cd /Users/fedor/projects/personal/videoToModel/vid2model
./convert.sh /path/to/input.mp4 /path/to/output.bvh
```

## Форматы вывода через `convert.sh`

Сигнатура:

```bash
./convert.sh <input_video> <output_bvh> [output_json] [output_csv] [output_npz] [output_trc] [output_fbx]
```

Примеры:

Только BVH:

```bash
./convert.sh think.mp4 output/think.bvh
```

BVH + JSON:

```bash
./convert.sh think.mp4 output/think.bvh output/think.json
```

Все форматы сразу:

```bash
./convert.sh \
  think.mp4 \
  output/think.bvh \
  output/think.json \
  output/think.csv \
  output/think.npz \
  output/think.trc \
  output/think.fbx
```

## Прямой запуск Python-скрипта

```bash
source .venv/bin/activate
python3 convert_video_to_bvh.py \
  --input think.mp4 \
  --output-bvh output/think.bvh \
  --output-json output/think.json \
  --output-csv output/think.csv \
  --output-npz output/think.npz \
  --output-trc output/think.trc
```

Поддерживаемые флаги:

- `--output` (legacy-алиас для BVH, вместо `--output-bvh`)
- `--output-bvh`
- `--output-json`
- `--output-csv`
- `--output-npz`
- `--output-trc`
- `--model-complexity {0,1,2}` (по умолчанию `1`)
- `--min-detection-confidence` (по умолчанию `0.5`)
- `--min-tracking-confidence` (по умолчанию `0.5`)

## BVH -> FBX (Blender CLI)

```bash
./bvh_to_fbx.sh output/think.bvh output/think.fbx
```

Если Blender не найден в `PATH`, укажи путь явно:

```bash
BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender ./bvh_to_fbx.sh output/think.bvh output/think.fbx
```

## Локальный viewer

```bash
python3 -m http.server 8080
```

Открыть:

`http://localhost:8080/viewer/index.html`

Возможности viewer:

- загрузка `output/think.bvh` одной кнопкой,
- загрузка любого локального `.bvh`,
- `Zoom +/-`,
- `Play / Pause / Stop`,
- timeline/scrub,
- `Reset Camera`.

## Smoke-test

```bash
./convert.sh think.mp4 output/think.bvh output/think.json output/think.csv output/think.npz output/think.trc
python3 -m http.server 8080
```

Опционально FBX:

```bash
./bvh_to_fbx.sh output/think.bvh output/think.fbx
```

## Тесты

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## Датасет для auto-mode

Если хочешь обучать модель, которая сама выбирает `pose_corrections`, рабочий цикл такой:

1. собрать размеченные примеры из видео,
2. обучить `auto_pose_model.npz`,
3. подключить модель через `pose_corrections.model_path`,
4. запускать конвертацию с `pose_corrections.mode = auto`.

Сборка JSONL-датасета из видео:

```bash
python3 tools/generate_auto_pose_dataset.py \
  --input think.mp4 \
  --label default \
  --output output/auto_pose_dataset.jsonl
```

Можно передать несколько `--input` за один запуск, если у всех одинаковая метка. Каждая строка в JSONL - это один пример с:

- `label`
- `source`
- `sample_count`
- `features`
- `summary`
- `meta`

Дальше этот файл можно использовать для обучения простой классификационной модели под `pose_corrections.mode = auto`.

Если у тебя уже есть папка с разметкой, можно собрать датасет сразу по структуре:

```text
dataset/
  default/
    think_01.mp4
    think_02.mp4
  mirrored/
    clip_01.mp4
  crouched/
    squat_01.mp4
```

И прогнать её одной командой:

```bash
python3 tools/build_auto_pose_dataset_from_dir.py \
  --dataset-dir dataset \
  --output output/auto_pose_dataset.jsonl
```

Обучение модели:

```bash
python3 tools/train_auto_pose_model.py \
  --input output/auto_pose_dataset.jsonl \
  --output models/auto_pose_model.npz
```

По умолчанию это обучает маленькую `mlp`-модель. Если нужно, можно подкрутить:

- `--hidden-size`
- `--epochs`
- `--learning-rate`
- `--l2`
- `--seed`

Если хочешь просто включить auto-mode без своей модели, можно оставить `model_path` пустым, тогда пайплайн будет использовать эвристику. Но для нормальной автоматической подстройки лучше подключать обученный `.npz`.

Пример конфига:

```json
{
  "pose_corrections": {
    "mode": "auto",
    "model_path": "models/auto_pose_model.npz"
  }
}
```

Пример полного запуска:

```bash
python3 tools/generate_auto_pose_dataset.py \
  --input think.mp4 \
  --label default \
  --output output/auto_pose_dataset.jsonl

python3 tools/train_auto_pose_model.py \
  --input output/auto_pose_dataset.jsonl \
  --output models/auto_pose_model.npz

./convert.sh --config config.auto.json think.mp4 output/think.bvh
```

Пример `config.auto.json`:

```json
{
  "pose_corrections": {
    "mode": "auto",
    "model_path": "models/auto_pose_model.npz"
  }
}
```

## Как Это Работает

Поток выполнения:

1. CLI получает входное видео и набор целевых форматов.
2. `pipeline.convert_video_to_bvh()` читает кадры через OpenCV.
3. Для каждого кадра MediaPipe Pose (Tasks API) возвращает landmarks.
4. Landmarks переводятся в набор ключевых точек скелета (`pose_points`).
5. По стабильным кадрам вычисляется rest-поза (`rest_offsets`).
6. Для каждого кадра считаются BVH-каналы:
   root translation + ротации суставов в порядке `Zrotation/Xrotation/Yrotation`.
7. `writers` сохраняют данные в выбранные форматы.

Почему так удобнее:

- проще поддерживать и расширять (изменения локализованы),
- проще тестировать (math/writers можно проверять отдельно),
- сохраняется совместимость старого запуска `python convert_video_to_bvh.py ...`.

## Примечания

- MediaPipe-модель (`*.task`) скачивается в папку `models/` автоматически при первом запуске.
- Качество трекинга сильно зависит от видео (свет, окклюзии, ракурс, полный рост в кадре).
- `output/` и `models/` рекомендуются как локальные runtime-артефакты (в `.gitignore`).
