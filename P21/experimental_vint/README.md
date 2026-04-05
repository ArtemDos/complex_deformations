# Анализ по винтовой траектории — эксперимент П.21

Структура и порядок работ совпадают с `P29/experimental_vint`, но:

- в CSV **122 шага** нагружения (столбец `No` 1…122 в `ezz_data.csv` и остальных файлах);
- в макросах APDL задано `exp_num_points = 122`.

## Файлы

| Тип | Имена |
|-----|--------|
| Опыт | `ezz_data.csv`, `ett_data.csv`, `etz_data.csv`, `szz_data.csv`, `stt_data.csv`, `stz_data.csv` |
| Параметры | `nu.txt`, `filter_step.txt`, `chab_params.txt`, `voce_hardening_params.txt` |
| APDL | `bkin.mac`, `biso.mac`, `chab.mac`, `voce_hardening.mac` |
| Анализ | `chab.ipynb` (аналог `P29/experimental_vint/chab.ipynb`) |

## ANSYS

Рабочая директория — эта папка. По очереди:

```apdl
/INPUT, bkin, mac
/INPUT, biso, mac
/INPUT, chab, mac
/INPUT, voce_hardening, mac
```

Получите `bkin.csv`, `biso.csv`, `chab.csv`, `voce_hardening.csv`.

Для ноутбука `chab.ipynb` дополнительно нужны **`chab_incompressible.csv`** и сжимаемый `chab.csv` — второй расчёт с несжимаемым материалом по той же схеме, что и для П.29 (отдельная постановка в APDL; отдельного `.mac` в репозитории может не быть).

## Параметры Шабоша / Восе

Сейчас в `chab_params.txt` и `voce_hardening_params.txt` лежат **копии значений с П.29** как стартовая точка. Для П.21 при необходимости выполните идентификацию заново (по аналогии с `P29/experimental_vint/identification_*.ipynb`) и перезапишите файлы.

## Jupyter

Откройте `chab.ipynb`, cwd ядра — `P21/experimental_vint` (или укажите `real_experiment_data_folder = "P21/experimental_vint"` из корня репозитория).

Переменная `split_time_exp = 14.0` задаёт границу этапов по **номеру шага/времени** в таблице (как в П.29); при другой разметке опыта измените её в ноутбуке.
