# vid2model

CLI-пайплайн для конвертации видео в скелетную анимацию:

- `BVH`
- `JSON`
- `CSV`
- `NPZ`
- `TRC`
- `FBX` (через Blender CLI)

Также в проекте есть локальный браузерный viewer (`viewer/index.html`) для предпросмотра `.bvh`.

## Структура

- `convert_video_to_bvh.py` - основной конвертер (MediaPipe Tasks API).
- `convert.sh` - удобный shell-раннер, умеет конвертировать во все форматы и опционально в FBX.
- `batch_convert.sh` - пакетная конвертация директории с видео.
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

## Запуск Проекта

Полный минимальный сценарий локального запуска:

1. Перейти в директорию проекта:

```bash
cd /Users/fedor/projects/personal/videoToModel/vid2model
```

2. Проверить окружение (опционально, но рекомендуется):

```bash
.venv/bin/python convert_video_to_bvh.py --check-tools
```

3. Запустить конвертацию (пример):

```bash
./convert.sh think.mp4 output/think.bvh output/think.json output/think.csv output/think.npz output/think.trc
```

4. Поднять локальный viewer:

```bash
python3 -m http.server 8080
```

5. Открыть в браузере:

`http://localhost:8080/viewer/index.html`

Опционально экспорт в FBX:

```bash
./bvh_to_fbx.sh output/think.bvh output/think.fbx
```

## Batch Конвертация

Конвертация всех видео в директории:

```bash
./batch_convert.sh ./videos ./output_batch
```

Рекурсивно и во все основные форматы:

```bash
./batch_convert.sh ./videos ./output_batch --recursive --all-formats
```

С FBX (если Blender доступен):

```bash
./batch_convert.sh ./videos ./output_batch --recursive --all-formats --with-fbx
```

Справка:

```bash
./batch_convert.sh --help
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
- `--config` (путь к профилю `.json/.yaml/.yml`, CLI-аргументы имеют приоритет)
- `--output-bvh`
- `--output-json`
- `--output-csv`
- `--output-npz`
- `--output-trc`
- `--model-complexity {0,1,2}` (по умолчанию `1`)
- `--min-detection-confidence` (по умолчанию `0.5`)
- `--min-tracking-confidence` (по умолчанию `0.5`)
- `--max-gap-interpolate` (по умолчанию `8`, интерполяция коротких пропусков детекции)
- `--progress-every` (по умолчанию `100`, период логирования прогресса; `0` отключает)
- `--check-tools` (проверка окружения без конвертации)

Проверка окружения:

```bash
.venv/bin/python convert_video_to_bvh.py --check-tools
```

Запуск с профилем:

```bash
.venv/bin/python convert_video_to_bvh.py --config ./config.example.yaml
```

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
mkdir -p .cache/matplotlib .cache/fontconfig
MPLCONFIGDIR="$PWD/.cache/matplotlib" XDG_CACHE_HOME="$PWD/.cache" \
  .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## CI

Автотесты запускаются в GitHub Actions на каждый `push` и `pull request`.

- workflow: `.github/workflows/tests.yml`
- matrix Python: `3.10`, `3.11`, `3.12`
- команда: `python -m unittest discover -s tests -p 'test_*.py' -v`

## Как Это Работает

После рефакторинга код разбит на модули по зонам ответственности:

- `convert_video_to_bvh.py` - тонкий entrypoint с обратной совместимостью (запуск как раньше).
- `vid2model_lib/cli.py` - разбор аргументов и orchestration пайплайна.
- `vid2model_lib/pipeline.py` - основной цикл: чтение видео, инференс позы, сбор motion-данных.
- `vid2model_lib/pose_model.py` - загрузка/кеширование MediaPipe `.task` модели.
- `vid2model_lib/pose_points.py` - извлечение landmark-точек и приведение к внутренней системе координат.
- `vid2model_lib/math3d.py` - математика поворотов (`rotation_align`, Euler ZXY).
- `vid2model_lib/skeleton.py` - описание суставов, иерархии и маппинга joint -> pose point.
- `vid2model_lib/writers.py` - экспорт в `BVH/JSON/CSV/NPZ/TRC`.

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
