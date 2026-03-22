# Анализ процессов деформирования по винтовым траекториям

Этот проект предназначен для анализа и визуализации данных, полученных в ходе численного моделирования в Ansys Mechanical APDL и сравнения их с экспериментальными данными.

Основной инструментарий:
- **Ansys Mechanical APDL** для численного моделирования.
- **Python** для обработки данных и визуализации.
- **Jupyter Notebook** как основная среда для анализа.

## Структура проекта

- `src/`: Исходный код Python.
  - `core/`: Пакет с основной логикой.
    - `reader.py`: Модуль для загрузки и фильтрации данных Ansys.
    - `visualizer.py`: Модуль для визуализации данных.
    - `calculations.py`: Модуль для вспомогательных вычислений.
- `pyproject.toml`: Метаданные пакета и список зависимостей.
- `requirements.txt`: Установка проекта в режиме редактирования одной командой.
- `P29/`: Анализ экперимента П29.
  - `experimental_vint/`: Реализация численного экперимента реальной траектории деформирования.
  - `ideal_vint/`: Реализация численного экперимента идеальной траектории деформирования.

## Инструкция по установке и запуску

Для работы с проектом необходимо настроить изолированное виртуальное окружение Python (Python 3.9+).

### Шаг 1: Создание и активация виртуального окружения

1. **Откройте терминал** в корневой папке проекта.

2. **Создайте виртуальное окружение**:
   ```bash
   python3 -m venv .venv
   ```
   (на Windows вместо `python3` часто используют `python`.)

3. **Активируйте окружение** (в начале строки появится префикс `(.venv)`):

   - **macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```
   - **Windows (cmd):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

### Шаг 2: Установка зависимостей и пакета

Находясь в **активированном** окружении и в **корне репозитория**:

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Если `pip install -e .` или `pip install -r requirements.txt` ругается на отсутствие `setup.py` или не поддерживает editable из `pyproject.toml`, обновите инструменты в виртуальном окружении (команда выше) и повторите установку.

Это установит проект в режиме редактирования (`-e .`) и все зависимости из `pyproject.toml` (pandas, numpy, matplotlib, scipy, scikit-learn, ipykernel, psutil).

**Дополнительно (по необходимости):**

- Ноутбуки с PyTorch (`P29/ideal_vint/research_nn_optimization.ipynb`, модуль `core/derivatives/neural.py`):
  ```bash
  pip install -e ".[torch]"
  ```
- Полноценный Jupyter Notebook / JupyterLab:
  ```bash
  pip install -e ".[jupyter]"
  ```
- Всё опциональное сразу:
  ```bash
  pip install -e ".[all]"
  ```

Альтернатива без `requirements.txt`:

```bash
pip install -e .
```

### Шаг 3: Регистрация окружения в Jupyter

```bash
python -m ipykernel install --user --name=complex_deformations_env
```

Будет доступно ядро с именем `complex_deformations_env`.

### Шаг 4: Запуск и работа в Jupyter Notebook

1. Откройте VS Code или Jupyter Lab и нужный `.ipynb` (например, `analysis.ipynb`).
2. Выберите ядро **complex_deformations_env** (или другое, где выполнен `pip install -e .`).
3. Импорты вида `from core.reader import ...` работают, если ядро использует это виртуальное окружение.
