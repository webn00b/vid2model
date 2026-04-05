# vid2model

`vid2model` конвертирует движение человека из видео в скелетную анимацию и помогает ретаргетить её на VRM-модели.

Проект состоит из двух частей:

- Python CLI-пайплайн для извлечения motion из видео и экспорта в `BVH`, `JSON`, `CSV`, `NPZ`, `TRC`
- локальный browser viewer для просмотра `BVH`, ретаргета на `VRM/.glb` и калибровки rig profiles

## Что умеет проект

- извлекать pose sequence из видео через MediaPipe
- чинить пропуски, side-swaps и шумные участки motion
- стабилизировать foot contact, pelvis/root motion и leg IK
- адаптивно сглаживать motion без потери резких акцентов
- анализировать loopability и при необходимости выделять loop
- ретаргетить результат на VRM в viewer
- сохранять, валидировать, экспортировать и переиспользовать rig profiles
- писать diagnostic JSON с quality summary и before/after cleanup metrics

## Quick Start

**MediaPipe (быстро, без GPU):**

```bash
cd /Users/fedor/projects/personal/videoToModel/vid2model
./convert.sh think.mp4 output/think.bvh
python3 -m http.server 8080
```

**4D-Humans / HMR2.0 (нейронка, точнее):**

```bash
./setup_smpl_backend.sh          # первый раз: ~2GB, только один раз
./convert_video_smpl.sh think.mp4 output/think.bvh
python3 -m http.server 8080
```

После этого открой:

`http://localhost:8080/viewer/index.html`

## Типичный workflow

1. Сконвертировать видео в `BVH` и diagnostic JSON.
2. Проверить quality summary и cleanup evaluation.
3. Открыть `viewer/index.html`.
4. Загрузить `BVH`.
5. Загрузить `VRM` или `GLB` модель.
6. Запустить retarget.
7. Если результат хороший, нажать `Validate Profile`.
8. При необходимости экспортировать и зарегистрировать rig profile в репозитории.

## Конвертация через нейронку (4D-Humans / HMR2.0)

Более точный результат за счёт SMPL body model и нейронного pose estimator.

Первый раз — настройка (нужен Python 3.10, скачает ~2GB):

```bash
./setup_smpl_backend.sh
```

Запуск:

```bash
./convert_video_smpl.sh think.mp4 output/think.bvh
```

С ретаргетом на VRM сразу:

```bash
./convert_video_smpl.sh think.mp4 output/think.bvh --vrm viewer/models/MoonGirl.vrm output/think.vrm
```

Схема: `Video → SMPL params (нейронка) → BVH → (опционально VRM)`

### Известные проблемы при setup

**4D-Humans не устанавливается** (ошибка chumpy в pip):

`setup_smpl_backend.sh` пытается поставить `chumpy` из git, но он сломан на Python 3.10+. Скрипт обрабатывает эту ошибку, но fallback не устанавливает сам `hmr2`. Фикс:

```bash
git clone --filter=blob:none https://github.com/shubham-goel/4D-Humans.git /tmp/4d-humans
sed -i '' "/'chumpy/d" /tmp/4d-humans/setup.py
.venv-smpl/bin/pip install /tmp/4d-humans
```

**`omegaconf` не установлен** — ставим вручную:

```bash
.venv-smpl/bin/pip install omegaconf
```

**PyTorch 2.6+ ломает загрузку чекпоинта** (`weights_only=True` по умолчанию) — уже пропатчено в `extract_smpl_from_video.py`, ничего делать не нужно.

## Генерация статических поз из изображений и видео

### Из одной картинки (poklon.jpg → BVH)

```bash
./image_to_bvh.sh poklon.jpg output/poklon.bvh 2
```

Скрипт:

1. Конвертирует JPG в видео (ffmpeg)
2. Запускает pose detection
3. Генерирует BVH со статической позой

Результат: 2-секундная анимация с одной повторяющейся позой.

### Извлечь финальный кадр из существующей анимации

```bash
./extract_bvh_frame.sh ted.bvh poklon.bvh -1 2
```

Аргументы:

- `ted.bvh` — исходная анимация
- `poklon.bvh` — выходной файл
- `-1` — последний кадр (или номер кадра: 0, 1, 120, etc.)
- `2` — длительность в секундах

Пример: извлечь финальную фазу поклона из видео, где он начинается на кадре 1000:

```bash
# Сначала посмотрите сколько кадров: grep "^Frames:" ted.bvh
# Извлеките финальный кадр
./extract_bvh_frame.sh ted.bvh poklon_final.bvh -1 3
```

## Конвертация через `convert.sh`

Простой запуск:

```bash
./convert.sh think.mp4 output/think.bvh
```

Все основные форматы сразу:

```bash
./convert.sh \
  think.mp4 \
  output/think.bvh \
  output/think.json \
  output/think.csv \
  output/think.npz \
  output/think.trc
```

`convert.sh` автоматически:

1. создаёт `.venv`, если её ещё нет
2. обновляет `pip`
3. ставит зависимости из `requirements.txt`

### Автоматическое определение FPS видео (рекомендуется)

Если OpenCV неправильно определяет frame rate (например, видео 60fps генерирует BVH в 2x медленнее), используйте `convert_auto_fps.sh`:

```bash
./convert_auto_fps.sh think.mp4 output/think.bvh
```

Скрипт автоматически:
1. детектирует FPS из метаданных видео (ffprobe)
2. передаёт его в `convert.sh`
3. генерирует BVH с правильной длительностью

Дополнительные форматы:

```bash
./convert_auto_fps.sh think.mp4 output/think.bvh --all --fbx
```

**Или вручную через флаг:**

```bash
OVERRIDE_FPS=60 ./convert.sh think.mp4 output/think.bvh
```

Полезно если автодетекция не сработала:

```bash
# Проверить FPS видео
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 video.mp4

# Использовать явно
OVERRIDE_FPS=60 ./convert.sh video.mp4 output/video.bvh
```

## Прямой запуск CLI

Пример с quality diagnostics:

```bash
python3 convert_video_to_bvh.py \
  --preset walk \
  --input think.mp4 \
  --output-bvh output/think.bvh \
  --output-diag-json output/think.diag.json
```

Пример для более сложного движения:

```bash
python3 convert_video_to_bvh.py \
  --preset dance \
  --input think.mp4 \
  --output-bvh output/think.bvh \
  --output-diag-json output/think.diag.json
```

Полный пример:

```bash
python3 convert_video_to_bvh.py \
  --input think.mp4 \
  --output-bvh output/think.bvh \
  --output-json output/think.json \
  --output-csv output/think.csv \
  --output-npz output/think.npz \
  --output-trc output/think.trc \
  --output-diag-json output/think.diag.json
```

Полезные флаги:

- `--preset {idle,walk,run,dance}`: baseline-настройки под тип движения
- `--output-diag-json`: расширенный diagnostic JSON
- `--loop-mode {off,auto,force}`: loop detection/extraction
- `--opencv-enhance {off,light,strong}`: preprocessing перед pose detection
- `--roi-crop {off,auto}`: adaptive crop вокруг человека
- `--max-gap-interpolate`: сколько кадров подряд можно интерполировать
- `--skeleton-profile-json`: override rest offsets под конкретную модель

## Motion presets

В проекте есть motion presets:

- `idle`
- `walk`
- `run`
- `dance`

Они меняют baseline для detection и cleanup. Практически:

- `idle` полезен для спокойной речи, жестов, стоячих поз
- `walk` подходит для обычной циклической ходьбы
- `run` даёт более жёсткие параметры под быстрые ноги
- `dance` меньше давит выразительное движение и отключает auto-loop по умолчанию

См. пример конфига: [config.example.yaml](/Users/fedor/projects/personal/videoToModel/vid2model/config.example.yaml)

## Diagnostic JSON

`--output-diag-json` пишет расширенный JSON с несколькими блоками:

- `input`: сколько кадров найдено, сколько интерполировано, какой backend использовался
- `cleanup`: что именно делал cleanup
- `evaluation`: before/after метрики после cleanup
- `root_yaw`: нормализация yaw и дополнительные rotation transforms
- `loop`: pre-cleanup loopability, итоговое loop detection и extraction
- `quality`: итоговый score и пользовательские флаги качества

Сейчас в `quality` есть:

- `score`
- `rating`
- `tracking_ok`
- `foot_contact_ok`
- `loop_candidate`
- `retarget_risk`
- `reasons`

Сейчас в `evaluation` есть before/after сравнение по:

- `root_position_jitter`
- `root_height_jitter`
- `left_foot_contact_spread`
- `right_foot_contact_spread`
- `left_wrist_motion_energy`
- `right_wrist_motion_energy`

Это удобно для регрессионного контроля: видно не только итоговую оценку, но и что именно cleanup улучшил или ухудшил.

## Viewer

Viewer живёт в [viewer/index.html](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/index.html).

Запуск:

```bash
python3 -m http.server 8080
```

Открыть:

`http://localhost:8080/viewer/index.html`

Что умеет viewer:

- загрузка локального `BVH`
- загрузка `VRM` и `GLB` моделей
- `Auto Setup` для обычного сценария: взять лучший доступный profile/seed и сразу наложить анимацию на модель
- `Save Model Setup` для обычного сценария: сохранить удачную настройку модели локально без ручного разбора rig-profile flow
- retarget source animation на модель
- `Export Model Analysis` для выгрузки параметров модели без исходной анимации
- локальное сохранение `draft` rig profile после удачного retarget
- `Validate Profile` для фиксации проверенного профиля
- `Export Profile` / `Import Profile`
- автозагрузка repo-backed rig profiles
- базовые viewer controls: play/pause/stop, zoom, scrub, reset camera

Подробности по retarget path и runtime helpers: [viewer/RETARGET_NOTES.md](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/RETARGET_NOTES.md)

Python-side карта стадий до viewer-retarget: [docs/python-source-pipeline-analysis.md](/Users/fedor/projects/personal/videoToModel/vid2model/docs/python-source-pipeline-analysis.md)

Viewer-side карта retarget, rig-profile priority и model-fit hooks: [docs/viewer-retarget-analysis.md](/Users/fedor/projects/personal/videoToModel/vid2model/docs/viewer-retarget-analysis.md)

Сводка гипотез и следующих шагов по `skeleton-to-model mismatch`: [docs/mismatch-hypotheses-and-next-steps.md](/Users/fedor/projects/personal/videoToModel/vid2model/docs/mismatch-hypotheses-and-next-steps.md)

План рефакторинга `viewer/js/modules` без изменения поведения: [docs/viewer-modules-refactor-plan.md](/Users/fedor/projects/personal/videoToModel/vid2model/docs/viewer-modules-refactor-plan.md)

## Headless Retarget Validation

Для автоматической проверки `BVH + GLB/VRM` без браузера есть headless runner:

- reusable runtime: [viewer/js/modules/headless-retarget-validation.js](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/headless-retarget-validation.js)
- CLI entrypoint: [tools/headless_retarget_validation.mjs](/Users/fedor/projects/personal/videoToModel/vid2model/tools/headless_retarget_validation.mjs)

Он переиспользует viewer-side retarget path, rig-profile resolution, calibration и runtime diagnostics, но запускается в Node. Для bare imports используются тонкие локальные ESM bridge-пакеты в `node_modules`, которые просто реэкспортируют bundled vendor-копии `three` и `@pixiv/three-vrm` из `viewer/vendor`.

Пример:

```bash
node tools/headless_retarget_validation.mjs \
  --model viewer/models/low_poly_humanoid_robot.glb \
  --bvh output/think.bvh \
  --stage body \
  --pretty
```

С записью результата в файл:

```bash
node tools/headless_retarget_validation.mjs \
  --model viewer/models/MoonGirl.vrm \
  --bvh output/think.bvh \
  --stage full \
  --out output/headless-retarget.json \
  --pretty
```

CLI печатает machine-readable JSON со стабильным верхнеуровневым контрактом:

- `input`: stage и исходные пути
- `model`: fingerprint, VRM metadata и список костей skinned mesh
- `source`: summary по `BVH`
- `rigProfile`: какой profile реально был выбран
- `mapping`: matched pairs, topology fallback и mirrored-side decisions
- `selection`: выбранный retarget attempt, mode, root yaw и pose-error probe
- `diagnostics`: viewer-like events, debug state и calibration summary

Это удобно для CI/regression-проверок, когда нужно сравнить headless retarget decisions с браузерным viewer flow без ручного открытия `viewer/index.html`.

## Regression Pass

Для повторяемой проверки проблемных сценариев есть единый runner:

```bash
python3 tools/run_regression_checks.py
```

Что он покрывает сейчас:

- `source_pipeline_diagnostics`: source-stage diagnostics и `quality.retarget_risk` для проблем в cleanup/finalize path
- `root_yaw_contract`: viewer-политику вокруг `retarget-root-yaw`, чтобы clip, уже центрированный Python-экспортом, не получал лишний large flip
- `atypical_model_mapping`: выбор canonical bone для нетипичных rig'ов, где helper/socket/end кости не должны выигрывать у основной кости
- `headless_retarget_validation`: machine-readable headless retarget contract для `BVH + GLB/VRM` вне browser viewer

Полезные режимы:

```bash
python3 tools/run_regression_checks.py --list
python3 tools/run_regression_checks.py --scenario root_yaw_contract
python3 tools/run_regression_checks.py --scenario atypical_model_mapping --dry-run
```

## Rig Profiles

Rig profile описывает, как конкретная модель лучше ретаргетится:

- mapping target bone -> source bone
- preferred retarget mode
- limb calibration flags
- yaw / scale / rotation adjustments
- cached calibration data

Внутри viewer есть несколько состояний профиля:

- `draft`: автосохранён после удачного retarget, но ещё не подтверждён
- `validated`: профиль проверен вручную и должен использоваться приоритетно

Приоритет загрузки:

1. local validated
2. repo validated
3. local draft
4. built-in fallback

## Model analysis

Если анимации ещё нет, но нужно собрать параметры модели, viewer теперь умеет экспортировать `model-analysis.json`.

Он содержит:

- `modelFingerprint`
- humanoid mapping
- иерархию костей
- local bind pose
- primary child directions
- segment lengths
- torso/arm/leg proportions
- foot hints

Это не готовый `rig profile`, но хороший seed для будущей автоматической калибровки.

Если для модели ещё нет сохранённого profile, viewer теперь может использовать такой seed автоматически как in-memory fallback по `modelFingerprint`.

## Export / Import / Register profile

## Quick flow

Для большинства случаев теперь достаточно такого порядка:

1. Загрузить `BVH`
2. Загрузить `VRM` / `GLB`
3. Нажать `Auto Setup`
4. Если результат хороший, нажать `Save Model Setup`

Advanced-кнопки `Validate Profile`, `Export Profile`, `Import Profile` и `Export Model Analysis` остаются для отладки, обмена профилями и repo-backed workflow.

Если profile уже проверен:

1. Нажми `Validate Profile`
2. Нажми `Export Profile`
3. Зарегистрируй JSON в репозитории:

```bash
python3 tools/register_rig_profile.py --input /path/to/exported.rig-profile.json
```

Скрипт:

- копирует JSON в `viewer/rig-profiles/`
- обновляет `viewer/rig-profiles/index.json`
- заменяет старую запись для того же `modelFingerprint + stage`

После экспорта viewer ещё и печатает готовую register-команду в console log.

Отдельная документация по формату и repo manifest: [viewer/rig-profiles/README.md](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/rig-profiles/README.md)

## Shared rig profiles

Repo-shared profiles лежат в:

- [viewer/rig-profiles/index.json](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/rig-profiles/index.json)
- [viewer/rig-profiles](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/rig-profiles)

Viewer автоматически подхватывает их по `modelFingerprint`.

## BVH -> FBX

Если нужен FBX через Blender CLI:

```bash
./bvh_to_fbx.sh output/think.bvh output/think.fbx
```

Если Blender не в `PATH`:

```bash
BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender ./bvh_to_fbx.sh output/think.bvh output/think.fbx
```

## Auto-pose dataset / training

Если хочешь обучать auto-mode, рабочий цикл такой:

1. собрать JSONL-датасет
2. обучить `auto_pose_model.npz`
3. подключить модель через config
4. запускать конвертацию с `pose_corrections.mode = auto`

Сборка датасета:

```bash
python3 tools/generate_auto_pose_dataset.py \
  --input think.mp4 \
  --label default \
  --output output/auto_pose_dataset.jsonl
```

Обучение:

```bash
python3 tools/train_auto_pose_model.py \
  --input output/auto_pose_dataset.jsonl \
  --output models/auto_pose_model.npz
```

## Архитектура

Ключевые модули в `vid2model_lib`:

- `pipeline.py`: публичный orchestration facade
- `pipeline_video_scan.py`: чтение видео и detector/ROI logic
- `pipeline_gap_fill.py`: заполнение пропусков
- `pipeline_cleanup.py`: smoothing, foot contacts, pelvis/root stabilization, leg IK
- `pipeline_auto_pose.py`: presets, features, classifier integration
- `pipeline_retarget.py`: canonicalization и pose corrections
- `pipeline_channels.py`: solving motion channels
- `pipeline_motion_transforms.py`: root yaw и rotation cleanup
- `pipeline_loop.py`: loop analysis / extraction / blend
- `pipeline_rest_offsets.py`: rest offsets и skeleton profile overrides
- `pipeline_mirror.py`: mirror / side-swap heuristics

## Требования

- Python `3.10+`
- `pip`
- Blender, если нужен `FBX`

Проверка toolchain:

```bash
python3 convert_video_to_bvh.py --check-tools
```

## Тесты

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```
