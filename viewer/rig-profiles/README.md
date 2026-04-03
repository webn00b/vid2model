# Rig Profiles

В этой папке лежат repo-shared rig profiles для viewer.

## Что хранится здесь

- `index.json`: manifest со списком доступных профилей
- `*.rig-profile.json`: отдельные exported profiles в формате `vid2model.rig-profile.v1`

## Формат manifest

```json
{
  "format": "vid2model.rig-profile-manifest.v1",
  "profiles": [
    {
      "modelFingerprint": "rig:1234abcd",
      "modelLabel": "MoonGirl.vrm",
      "stage": "body",
      "path": "./moon-girl.body.rig-profile.json"
    }
  ]
}
```

Каждый `path` должен указывать на `vid2model.rig-profile.v1` JSON рядом с manifest.

## Как добавить новый профиль

Рекомендуемый workflow:

1. Проверь retarget в viewer.
2. Нажми `Validate Profile`.
3. Нажми `Export Profile`.
4. Зарегистрируй JSON в репозитории:

```bash
python3 tools/register_rig_profile.py --input /path/to/exported.rig-profile.json
```

Опционально можно переопределить имя файла:

```bash
python3 tools/register_rig_profile.py \
  --input /path/to/exported.rig-profile.json \
  --name moon-girl.body.rig-profile.json
```

Скрипт:

- валидирует формат экспорта
- копирует профиль в `viewer/rig-profiles/`
- обновляет `index.json`
- заменяет старую запись для того же `modelFingerprint + stage`

## Как viewer выбирает profile

Приоритет такой:

1. local validated
2. repo validated
3. local draft
4. built-in fallback

Repo profiles используются как team-shared baseline, но не перебивают вручную validated локальный профиль.
