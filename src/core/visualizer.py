import pandas as pd
import matplotlib.pyplot as plt

def plot_deformation_comparison(calc_data, exp_data, model_name=''):
    """
    Строит сравнительные графики полных деформаций (расчет vs эксперимент)
    в зависимости от времени.

    Args:
        calc_data (pd.DataFrame): DataFrame с данными численного расчета.
                                  Должен содержать столбцы 'Time', 'EPTO_TT', 'EPTO_ZZ', 'EPTO_TZ'.
        exp_data (pd.DataFrame): DataFrame с данными эксперимента.
                                 Должен содержать столбцы 'Time', 'EPTO_TT', 'EPTO_ZZ', 'EPTO_TZ'.
        model_name (str, optional): Название модели для заголовка графика.
                                    По умолчанию пустая строка.
    """

    required_calc_cols = ['Time', 'EPTO_TT', 'EPTO_ZZ', 'EPTO_TZ']
    required_exp_cols = ['Time', 'EPTO_TT', 'EPTO_ZZ', 'EPTO_TZ']
    
    if not all(col in calc_data.columns for col in required_calc_cols):
        raise ValueError("DataFrame с расчетными данными не содержит всех необходимых столбцов.")
    if not all(col in exp_data.columns for col in required_exp_cols):
        raise ValueError("DataFrame с экспериментальными данными не содержит всех необходимых столбцов.")

    plt.figure(figsize=(14, 8))
    plt.plot(calc_data['Time'], calc_data['EPTO_TT'], label='EPTO_TT (Расчет)', color='blue')
    plt.plot(calc_data['Time'], calc_data['EPTO_ZZ'], label='EPTO_ZZ (Расчет)', color='red')
    plt.plot(calc_data['Time'], calc_data['EPTO_TZ'], label='EPTO_XY (Расчет)', color='green')
    plt.scatter(exp_data['Time'], exp_data['EPTO_TT'], 
                label='EPTO_TT (Эксперимент)', color='cyan', marker='x', s=20, alpha=0.9)
    plt.scatter(exp_data['Time'], exp_data['EPTO_ZZ'], 
                label='EPTO_ZZ (Эксперимент)', color='magenta', marker='x', s=20, alpha=0.9)
    plt.scatter(exp_data['Time'], exp_data['EPTO_TZ'], 
                label='EPTO_XY (Эксперимент)', color='lime', marker='x', s=20, alpha=0.9)

    title = f'Сравнение расчетных (модель {model_name}) и экспериментальных деформаций от времени'
    plt.title(title, fontsize=16)
    plt.xlabel('Время (номер шага)')
    plt.ylabel('Полные деформации')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=0)
    plt.show()

def plot_stress_comparison(calc_data, exp_data, model_name=''):
    """
    Строит сравнительные графики напряжений (расчет vs эксперимент)
    в зависимости от времени.

    Args:
        calc_data (pd.DataFrame): DataFrame с данными численного расчета.
                                  Должен содержать столбцы 'Time', 'S_TT', 'S_ZZ', 'S_TZ'.
        exp_data (pd.DataFrame): DataFrame с данными эксперимента.
                                 Должен содержать столбцы 'Time', 'S_TT', 'S_ZZ', 'S_TZ'.
        model_name (str, optional): Название модели для заголовка графика.
                                    По умолчанию пустая строка.
    """

    required_calc_cols = ['Time', 'S_TT', 'S_ZZ', 'S_TZ']
    required_exp_cols = ['Time', 'S_TT', 'S_ZZ', 'S_TZ']
    
    if not all(col in calc_data.columns for col in required_calc_cols):
        raise ValueError("DataFrame с расчетными данными не содержит всех необходимых столбцов для напряжений.")
    if not all(col in exp_data.columns for col in required_exp_cols):
        raise ValueError("DataFrame с экспериментальными данными не содержит всех необходимых столбцов для напряжений.")

    plt.figure(figsize=(14, 8))
    plt.plot(calc_data['Time'], calc_data['S_TT'], label='S_TT (Расчет)', color='blue')
    plt.plot(calc_data['Time'], calc_data['S_ZZ'], label='S_ZZ (Расчет)', color='red')
    plt.plot(calc_data['Time'], calc_data['S_TZ'], label='S_TZ (Расчет)', color='green')
    plt.scatter(exp_data['Time'], exp_data['S_TT'], 
                label='S_TT (Эксперимент)', color='cyan', marker='x', s=20, alpha=0.9)
    
    plt.scatter(exp_data['Time'], exp_data['S_ZZ'], 
                label='S_ZZ (Эксперимент)', color='magenta', marker='x', s=20, alpha=0.9)

    plt.scatter(exp_data['Time'], exp_data['S_TZ'], 
                label='S_TZ (Эксперимент)', color='lime', marker='x', s=20, alpha=0.9)

    title = f'Сравнение расчетных (модель {model_name}) и экспериментальных напряжений от времени'
    plt.title(title, fontsize=16)
    plt.xlabel('Время (номер шага)')
    plt.ylabel('Напряжения (МПа)')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=0)
    plt.show()

def plot_equivalent_stress_comparison(calc_data, exp_data, model_name='', yield_stress=None):
    """
    Строит сравнительный график интенсивностей напряжений (S_EQV) 
    (расчет vs эксперимент) в зависимости от времени.

    Args:
        calc_data (pd.DataFrame): DataFrame с данными численного расчета.
                                  Должен содержать столбцы 'Time' и 'S_EQV'.
        exp_data (pd.DataFrame): DataFrame с данными эксперимента.
                                 Должен содержать столбцы 'Time' и 'S_EQV'.
        model_name (str, optional): Название модели для заголовка графика.
                                    По умолчанию пустая строка.
        yield_stress (float, optional): Значение предела текучести.
    """

    required_cols = ['Time', 'S_EQV']
    
    if not all(col in calc_data.columns for col in required_cols):
        raise ValueError("DataFrame с расчетными данными не содержит столбцов 'Time' и 'S_EQV'.")
    if not all(col in exp_data.columns for col in required_cols):
        raise ValueError("DataFrame с экспериментальными данными не содержит столбцов 'Time' и 'S_EQV'.")

    plt.figure(figsize=(14, 8))
    plt.plot(calc_data['Time'], calc_data['S_EQV'], label='Интенсивность S_EQV (Расчет)', color='blue')
    plt.scatter(exp_data['Time'], exp_data['S_EQV'], 
                label='Интенсивность S_EQV (Эксперимент)', color='red', marker='x', s=30, alpha=0.9)

    if yield_stress is not None:
        plt.axhline(y=yield_stress, color='gray', linestyle='--', label=f'Предел текучести ({yield_stress} МПа)')

    title = f'Сравнение расчетных (модель {model_name}) и экспериментальных интенсивностей напряжени от времени'
    plt.title(title, fontsize=16)
    plt.xlabel('Время (номер шага)')
    plt.ylabel('Интенсивность напряжений (S_EQV, МПа)')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

def plot_xy(data_pairs, x_col, y_col, title=''):
    """
    Строит сравнительный 2D-график для нескольких наборов данных.

    Args:
        data_pairs (list of tuples): Список кортежей, где каждый кортеж 
                                     содержит (имя_кривой, DataFrame).
                                     Пример: [('Расчет', df1), ('Эксперимент', df2)]
        x_col (str): Название столбца для оси X (должно быть во всех DataFrame).
        y_col (str): Название столбца для оси Y (должно быть во всех DataFrame).
        title (str, optional): Заголовок графика.
    """
    
    if not isinstance(data_pairs, list) or not all(isinstance(i, tuple) and len(i) == 2 for i in data_pairs):
        raise TypeError("Аргумент 'data_pairs' должен быть списком кортежей вида [('имя1', df1), ('имя2', df2), ...]")

    plt.figure(figsize=(12, 7))

    for name, df in data_pairs:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Объект для '{name}' не является DataFrame.")
        if x_col not in df.columns or y_col not in df.columns:
            raise KeyError(f"В DataFrame для '{name}' отсутствует один из столбцов: '{x_col}' или '{y_col}'.")

        plt.plot(df[x_col], df[y_col], marker='.', linestyle='-', label=name)

    plt.title(title if title else f'Зависимость {y_col} от {x_col}', fontsize=16)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.legend()
    plt.show()
