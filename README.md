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

Самый короткий путь:

```bash
cd /Users/fedor/projects/personal/videoToModel/vid2model
./convert.sh think.mp4 output/think.bvh
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
- retarget source animation на модель
- локальное сохранение `draft` rig profile после удачного retarget
- `Validate Profile` для фиксации проверенного профиля
- `Export Profile` / `Import Profile`
- автозагрузка repo-backed rig profiles
- базовые viewer controls: play/pause/stop, zoom, scrub, reset camera

Подробности по retarget path и runtime helpers: [viewer/RETARGET_NOTES.md](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/RETARGET_NOTES.md)

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

## Export / Import / Register profile

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
