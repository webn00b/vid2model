# Python Source Pipeline Analysis

Эта заметка фиксирует Python-цепочку до viewer-retarget для задачи `vid2model-e5c.1`.

## Entry Points

- [`convert_video_to_bvh.py`](/Users/fedor/projects/personal/videoToModel/vid2model/convert_video_to_bvh.py)
  Тонкий entry script. Реальный CLI живёт в `vid2model_lib.cli.main()`.
- [`vid2model_lib/cli.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/cli.py)
  Основной публичный вход:
  - `parse_args()` собирает CLI flags
  - `_build_cli_options()` мерджит CLI и config, валидирует диапазоны и собирает `CliOptions`
  - `main()` вызывает `convert_video_to_bvh(...)`, затем пишет `BVH/JSON/CSV/NPZ/TRC/diag JSON`
- [`vid2model_lib/pipeline.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline.py)
  Стабильный orchestration facade. Основной проход реализован в `convert_video_to_bvh(...)`.

## Stage Map

### 1. Video Scan

- Модуль: [`vid2model_lib/pipeline_video_scan.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_video_scan.py)
- Ключевые функции:
  - `collect_detected_pose_samples(...)`
  - `preprocess_video_frame(...)`
  - `extract_pose_bbox_pixels(...)`
  - `update_tracking_roi(...)`
- Что делает:
  - открывает видео
  - выбирает MediaPipe backend (`tasks`, fallback в `solutions`)
  - применяет optional preprocessing и resize
  - optional ROI crop вокруг человека
  - извлекает pose points по кадрам
- Основные сигналы:
  - `diagnostics.input.detected_frames`
  - `diagnostics.input.roi_used`
  - `diagnostics.input.roi_fallback`
  - `diagnostics.input.roi_resets`
  - `diagnostics.input.pose_backend`

### 2. Gap Fill

- Модуль: [`vid2model_lib/pipeline_gap_fill.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_gap_fill.py)
- Ключевые функции:
  - `fill_pose_gaps(...)`
  - `interpolate_pose_points(...)`
- Что делает:
  - интерполирует короткие провалы detections
  - carry-forward заполняет длинные пустые окна
  - заполняет leading/trailing пустые участки
- Основные сигналы:
  - `diagnostics.input.interpolated_frames`
  - `diagnostics.input.carried_frames`
  - `diagnostics.quality.detected_ratio`
  - `diagnostics.quality.carried_ratio`

### 3. Reference Basis And Canonicalization

- Модуль: [`vid2model_lib/pipeline_retarget.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_retarget.py)
- Ключевые функции:
  - `median_pose_sample(...)`
  - `build_reference_basis(...)`
  - `canonicalize_pose_points(...)`
- Что делает:
  - строит опорный sample из anchor frames
  - выбирает reference basis и origin
  - переводит все точки в каноническую локальную систему
- Риск:
  - плохой anchor sample или unstable shoulders/hips искажают всю дальнейшую motion semantics
- Диагностика:
  - при `include_source_stage_diagnostics=True` сохраняется стадия `source_stages.pose.stages.canonical`

### 4. Pose Corrections

- Модули:
  - [`vid2model_lib/pipeline_auto_pose.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_auto_pose.py)
  - [`vid2model_lib/pipeline_retarget.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_retarget.py)
- Ключевые функции:
  - `resolve_auto_pose_corrections(...)`
  - `apply_pose_corrections(...)`
- Что делает:
  - при `mode=auto` выбирает correction preset
  - применяет grounding, body bend reduction, arm/leg IK floors, collider logic и offsets
- Риск:
  - слишком агрессивные correction heuristics могут усреднить исходный скелет до generic humanoid
- Диагностика:
  - stderr: `pose_corrections auto ...`
  - `diagnostics.source_stages.pose.comparisons.canonical_to_corrected_pre_cleanup`
  - `diagnostics.source_stages.pose.flags.suspected_issue_stage`

### 5. Cleanup

- Модуль: [`vid2model_lib/pipeline_cleanup.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_cleanup.py)
- Ключевые функции:
  - `cleanup_pose_frames(...)`
  - `_smooth_pose_frames(...)`
  - `_apply_segment_length_constraints(...)`
- Что делает:
  - adaptive smoothing
  - target segment length constraints
  - side-swap fixing
  - foot-contact stabilization
  - pelvis/root stabilization
  - leg IK cleanup
- Риск:
  - cleanup может уменьшить jitter, но при плохих эвристиках смазать акценты, испортить root motion или переусреднить длины
- Диагностика:
  - `diagnostics.cleanup`
  - `diagnostics.evaluation`
  - `diagnostics.source_stages.pose.comparisons.corrected_pre_cleanup_to_post_cleanup_pre_loop`
  - `diagnostics.quality.reasons`

### 6. Loop Analysis And Extraction

- Модуль: [`vid2model_lib/pipeline_loop.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_loop.py)
- Ключевые функции:
  - `analyze_motion_loopability(...)`
  - `extract_motion_loop(...)`
  - `blend_motion_loop_edges(...)`
- Что делает:
  - определяет cyclic vs oneshot
  - optional extraction/blending loop window
- Риск:
  - wrong loop classification может обрезать реальный oneshot или оставить плохую cyclic window
- Диагностика:
  - `diagnostics.loop.pre_cleanup_detected`
  - `diagnostics.loop.detected`
  - `diagnostics.loop.extracted`

### 7. Rest Offsets

- Модуль: [`vid2model_lib/pipeline_rest_offsets.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_rest_offsets.py)
- Ключевые функции:
  - `build_rest_offsets(...)`
  - `apply_skeleton_profile_to_rest_offsets(...)`
  - `_apply_vrm_humanoid_baseline_to_rest_offsets(...)`
- Что делает:
  - строит median-based rest skeleton
  - optional blend/override под model skeleton profile
  - optional VRM humanoid baseline normalization
- Риск:
  - плохой source median или неудачный skeleton profile меняют пропорции до retarget ещё до viewer
- Диагностика:
  - `diagnostics.skeleton_profile`

### 8. Motion Channels And Final Transforms

- Модули:
  - [`vid2model_lib/pipeline_channels.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_channels.py)
  - [`vid2model_lib/pipeline_motion_transforms.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_motion_transforms.py)
- Ключевые функции:
  - `frame_channels(...)`
  - `normalize_motion_root_yaw(...)`
  - `apply_manual_root_yaw_offset(...)`
  - `apply_lower_body_rotation_mode(...)`
  - `apply_upper_body_rotation_scale(...)`
  - `unwrap_motion_rotation_channels(...)`
- Что делает:
  - переводит source pose frames в BVH channels
  - нормализует `root yaw`
  - применяет manual/source-side rotation transforms
  - unwrap rotation discontinuities
- Риск:
  - double ownership у `root yaw`
  - source-side lower-body or arm scaling transforms могут менять motion semantics уже после cleanup
- Диагностика:
  - `diagnostics.root_yaw`
  - `diagnostics.rotation_unwrap`
  - `diagnostics.source_stages.motion`

### 9. Quality Summary And Writers

- Модули:
  - [`vid2model_lib/pipeline.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline.py)
  - [`vid2model_lib/writers.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/writers.py)
- Ключевые функции:
  - `_build_quality_summary(...)`
  - `write_bvh(...)`
  - `write_json(...)`
  - `write_diagnostic_json(...)`
- Что делает:
  - собирает итоговую quality оценку
  - пишет output files
- Основные сигналы:
  - `diagnostics.quality.score`
  - `diagnostics.quality.rating`
  - `diagnostics.quality.retarget_risk`
  - `diagnostics.quality.reasons`

## Likely Failure Points Before Viewer

### Video Scan / Detection

- Низкий `detected_ratio`
- Слишком много `carried_frames`
- Частые ROI fallback/reset
- Fallback на legacy `solutions` backend

Как ловится:
- `diagnostics.input.*`
- `diagnostics.quality.reasons`

### Reference Basis / Canonicalization

- Ошибка в выборе anchor sample
- Нестабильная ориентация плеч/таза
- Неверный facing basis ещё до root-yaw normalization

Как ловится:
- `source_stages.pose`
- косвенно через downstream `root_yaw` и cleanup degradation

### Pose Corrections

- Агрессивный bend reduction
- Жёсткий IK/collider logic
- Неудачный auto preset

Как ловится:
- `source_stages.pose.comparisons`
- `source_stages.pose.flags.suspected_issue_stage == "pose_corrections"`

### Cleanup

- Переусреднение expressive motion
- Избыточная стабилизация pelvis/root
- Segment constraints подавляют живую позу

Как ловится:
- `diagnostics.cleanup`
- `diagnostics.evaluation`
- `quality.reasons`

### Motion Finalization

- Root yaw normalization с лишним flip
- Manual yaw offset конфликтует с viewer assumptions
- Rotation unwrap скрывает симптом, но не лечит исходную причину

Как ловится:
- `diagnostics.root_yaw`
- `source_stages.motion.flags.suspected_issue_stage`
- `source_stages.motion.flags.yaw_normalization_spike`

## Practical Debug Order

1. Сначала смотреть `diagnostics.input` и `quality.reasons`
2. Потом `source_stages.pose.flags` и `source_stages.motion.flags`
3. Затем `cleanup` и `evaluation`
4. Только после этого разбирать viewer-retarget path

Если source pipeline уже помечен как risky, viewer обычно лишь усиливает уже существующую проблему, а не создаёт её с нуля.
