# Viewer Retarget Notes

Этот файл нужен как короткая инженерная памятка по текущему состоянию viewer-retarget workflow.

## Текущий стабильный путь

- основной режим: `skeletonutils-skinnedmesh + live-delta`
- VRM direct retarget: выключен по умолчанию и считается экспериментальным
- успешный retarget автоматически сохраняет локальный `draft` rig profile
- `Validate Profile` повышает текущий профиль до `validated`
- `Export Profile` сохраняет `vid2model.rig-profile.v1`
- `Import Profile` поднимает импортированный профиль как `validated`
- repo-backed profiles подхватываются из `viewer/rig-profiles/index.json`

## Приоритет профилей

Viewer выбирает профиль в таком порядке:

1. local validated
2. repo validated
3. local draft
4. auto-generated in-memory seed from current model analysis
5. built-in fallback

## Что сейчас умеет rig profile

- names mapping target -> source
- preferred retarget mode
- force/prefer live-delta flags
- cached calibration data
- per-side limb calibration flags

Сейчас через профиль можно включать отдельные corrections для:

- shoulder direction
- upper-arm direction
- forearm direction
- elbow plane
- knee plane
- upper-leg direction
- shin direction
- foot direction
- foot plane
- foot mirror correction

## Практический workflow

1. Загрузить `BVH`
2. Загрузить `VRM` или `GLB`
3. Для обычного сценария нажать `Auto Setup`
4. Проверить результат на модели
5. Для обычного сценария нажать `Save Model Setup`
6. Для advanced workflow можно отдельно использовать `Validate Profile`
7. Нажать `Export Profile`, если профиль нужно перенести или положить в репозиторий
8. Зарегистрировать профиль через `tools/register_rig_profile.py`

Если source animation ещё нет, можно отдельно выгрузить `model-analysis.json` через `Export Model Analysis` и использовать его как основу для будущей model-specific calibration.

## Полезные runtime helpers

```js
window.__vid2modelForceLiveDelta = null
window.__vid2modelUseVrmDirect = false
window.__vid2modelSetDiagMode("minimal")

window.__vid2modelGetRigProfileState()
window.__vid2modelValidateRigProfile()
window.__vid2modelListRigProfiles()
window.__vid2modelListRepoRigProfiles()

window.__vid2modelExportRigProfile(true)
window.__vid2modelExportModelAnalysis(true)
window.__vid2modelBuildRigProfileSeed()
window.__vid2modelAutoSetupModel()
window.__vid2modelSaveModelSetup()
window.__vid2modelImportRigProfile(payload)
window.__vid2modelBuildRegisterRigProfileCommand()
window.__vid2modelBuildRegisterRigProfileCommand("/absolute/path/to/exported.rig-profile.json")
```

## Текущие замечания

- generic stable path по-прежнему лучше всего отлажен на `MoonGirl.vrm`
- `retarget-vrm.js` оставлен для будущих VRM-specific axis/sign calibration экспериментов
- built-in profiles уже содержат limb calibration flags для основных тестовых моделей
