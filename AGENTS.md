## Правила для AI Агентов (Claude/Cursor)

**Главное**: Прочитай [CLAUDE.md](CLAUDE.md) перед началом работы. Там вся архитектура, ключевые концепции и решения типичных проблем.

### Workflow

1. **Планирование**: Используй beads CLI (`bd`) для отслеживания задач
   ```bash
   bd ready                    # Найти доступную работу
   bd show <id>               # Посмотреть детали задачи
   bd create --title="..."    # Создать новую задачу
   bd update <id> --claim     # Взять задачу
   ```

2. **Перед изменениями**: Составь план, разбей на мелкие шаги, обсуди со мной

3. **После изменений**:
   ```bash
   node tests/headless-retarget-validation.test.mjs  # Запусти тесты
   git status                                          # Проверь изменения
   bd close <id>                                       # Закрой задачу
   ```

4. **На выходе из сессии**:
   ```bash
   bd dolt pull && git pull --rebase
   git push && git status  # MUST show "up to date with origin"
   ```

### Важные Концепции

- **VRM Rotation Compensation** - VRM модели загружаются с Y-rotation = π. Это ВСЕГДА нужно компенсировать в ретаргетинге. Подробнее в [CLAUDE.md#critical-knowledge-vrm-rotation-compensation](CLAUDE.md#critical-knowledge-vrm-rotation-compensation)
- **Retargeting Modes** - Разные кости используют разные режимы. Смотри [CLAUDE.md#retargeting-modes](CLAUDE.md#retargeting-modes)
- **Bone Mapping** - Все кости маппятся на canonical names. Смотри `bone-utils.js`

### Типичные Задачи

- **Исправить проблему с ориентацией VRM** → Смотри [CLAUDE.md](CLAUDE.md#common-tasks--how-to-handle-them)
- **Добавить поддержку новой кости** → Смотри [CLAUDE.md](CLAUDE.md#common-tasks--how-to-handle-them)
- **Улучшить качество ретаргетинга** → Смотри [CLAUDE.md](CLAUDE.md#common-tasks--how-to-handle-them)

### Quick Links

- **Архитектура**: [CLAUDE.md](CLAUDE.md#architecture)
- **VRM фиксы**: [CLAUDE.md](CLAUDE.md#critical-knowledge-vrm-rotation-compensation)
- **Тестирование**: [CLAUDE.md](CLAUDE.md#testing)
- **Файлы**: [CLAUDE.md](CLAUDE.md#quick-reference-file-locations)
- **Beads**: Запусти `bd prime` в терминале
