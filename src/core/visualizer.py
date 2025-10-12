import pandas as pd
import matplotlib.pyplot as plt

def plot_deformation_comparison(calc_data, exp_data):
    """
    Строит сравнительные графики полных деформаций (расчет vs эксперимент)
    в зависимости от времени.

    Args:
        calc_data (pd.DataFrame): DataFrame с данными численного расчета.
                                  Должен содержать столбцы 'Time', 'EPTO_TT', 'EPTO_ZZ', 'EPTO_TZ'.
        exp_data (pd.DataFrame): DataFrame с данными эксперимента.
                                 Должен содержать столбцы 'Time', 'EPTO_TT', 'EPTO_ZZ', 'EPTO_TZ'.
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

    plt.title('Сравнение расчетных и экспериментальных деформаций от времени', fontsize=16)
    plt.xlabel('Время (номер шага)')
    plt.ylabel('Полные деформации')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=0)
    plt.show()
