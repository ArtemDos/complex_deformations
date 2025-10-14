import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_xy(data_pairs, x_col, y_cols, title='', exp_data=None):
    """
    Строит сравнительный 2D-график с разделением по цвету (для Y-величин) 
    и стилю линии (для наборов данных).

    Args:
        data_pairs (list of tuples): Список кортежей вида [('model_name', DataFrame), ...].
        x_col (str): Название столбца для оси X.
        y_cols (list or str): Список названий столбцов для оси Y или одна строка.
        title (str, optional): Заголовок графика.
        exp_data (tuple, optional): Кортеж для ЭКСПЕРИМЕНТАЛЬНЫХ данных вида ('exp_name', DataFrame).
    """

    if isinstance(y_cols, str):
        y_cols = [y_cols]

    if not isinstance(data_pairs, list) or not all(isinstance(i, tuple) and len(i) == 2 for i in data_pairs):
        raise TypeError("Аргумент 'data_pairs' должен быть списком кортежей вида [('model_name', DataFrame), ...]")
    if not isinstance(y_cols, list) or not y_cols:
        raise TypeError("Аргумент 'y_cols' должен быть непустым списком строк.")
    if exp_data and (not isinstance(exp_data, tuple) or len(exp_data) != 2):
        raise TypeError("Аргумент 'exp_data' должен быть кортежем вида ('exp_name', DataFrame).")

    plt.figure(figsize=(12, 7))
    
    is_multi_y = len(y_cols) > 1
    if is_multi_y:
        y_colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(y_cols)))
    else:
        data_colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(data_pairs) + (1 if exp_data else 0)))
    styles = ['-', '--', '-.', ':']
    line_styles = {name: styles[i % len(styles)] for i, (name, df) in enumerate(data_pairs)}
    for i, (name, df) in enumerate(data_pairs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Объект для '{name}' не является DataFrame.")

        line_style = line_styles[name]
        for j, y_col in enumerate(y_cols):
            if x_col not in df.columns or y_col not in df.columns:
                raise KeyError(f"В DataFrame для '{name}' отсутствует один из столбцов: '{x_col}' или '{y_col}'.")
            label = f'{name} - {y_col}' if len(data_pairs) > 1 or len(y_cols) > 1 else y_col
            color = y_colors[j] if is_multi_y else data_colors[i]
            linewidth = 2.5 if line_style == '-' else 2.0
            plt.plot(df[x_col], df[y_col], color=color, linestyle=line_style, label=label, lw=linewidth)
        
    if exp_data:
        exp_name, df_exp = exp_data
        if not isinstance(df_exp, pd.DataFrame):
            raise TypeError(f"Объект для '{exp_name}' не является DataFrame.")
        
        for j, y_col in enumerate(y_cols):
            if x_col not in df_exp.columns or y_col not in df_exp.columns:
                raise KeyError(f"В DataFrame для эксперимента '{exp_name}' отсутствует один из столбцов: '{x_col}' или '{y_col}'.")
            color = y_colors[j] if is_multi_y else data_colors[len(data_pairs)]
            label = f'{exp_name} - {y_col}' if len(data_pairs) > 0 or len(y_cols) > 1 else exp_name
            plt.scatter(
                df_exp[x_col],
                df_exp[y_col], 
                label=label,
                color=color,
                marker='x',
                s=20, alpha=0.9
            )

    y_labels = ', '.join(y_cols)
    if not title:
        title = f'Зависимость {y_labels} от {x_col}'
        
    plt.title(title, fontsize=16)
    plt.xlabel(x_col)
    plt.ylabel(y_labels if is_multi_y else y_cols[0])
    plt.grid(True)
    plt.legend()
    plt.show()
