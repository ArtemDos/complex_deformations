# Анализ по винтовой траектории — эксперимент П.26

Аналогично `P29/experimental_vint` и `P21/experimental_vint`:

- в CSV **55 шагов** нагружения (`No` 1…55);
- в макросах APDL: `exp_num_points = 55`.

## Файлы

| Тип | Имена |
|-----|--------|
| Опыт | `ezz_data.csv`, `ett_data.csv`, `etz_data.csv`, `szz_data.csv`, `stt_data.csv`, `stz_data.csv` |
| Параметры | `nu.txt`, `filter_step.txt`, `chab_params.txt`, `voce_hardening_params.txt` |
| APDL | `bkin.mac`, `biso.mac`, `chab.mac`, `voce_hardening.mac` |
| Анализ | `chab.ipynb` |

## ANSYS

Рабочая директория — эта папка:

```apdl
/INPUT, bkin, mac
/INPUT, biso, mac
/INPUT, chab, mac
/INPUT, voce_hardening, mac
```

Для `chab.ipynb` нужны также **`chab.csv`** и **`chab_incompressible.csv`** (второй расчёт — как в П.29).

## Параметры

`chab_params.txt` и `voce_hardening_params.txt` сейчас скопированы с **П.29**; при необходимости замените после идентификации под П.26.

## Jupyter

cwd ядра — `P26/experimental_vint` или `real_experiment_data_folder = "P26/experimental_vint"`. Граница этапов: `split_time_exp` в ноутбуке (по умолчанию как в П.29 — скорректируйте под опыт).
