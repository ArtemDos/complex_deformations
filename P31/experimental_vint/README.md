# Анализ по винтовой траектории — эксперимент П.31

Аналогично `P29/experimental_vint`:

- в CSV **133 шага** нагружения (`No` 1…133);
- в макросах APDL: `exp_num_points = 133`.

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

Для `chab.ipynb` нужны **`chab.csv`** и **`chab_incompressible.csv`** (второй расчёт — как в П.29).

## Параметры

`chab_params.txt` и `voce_hardening_params.txt` скопированы с **П.29**; при необходимости замените после идентификации под П.31.

## Jupyter

cwd — `P31/experimental_vint` или `real_experiment_data_folder = "P31/experimental_vint"`. Граница этапов: `split_time_exp` в ноутбуке.
