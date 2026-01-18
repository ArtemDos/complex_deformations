import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

def compute_spline_derivatives(df, col_name, time_col='Time', s=0.0, k = 5):
    """
    Вычисляет 1-ю, 2-ю и 3-ю производные для указанного столбца с помощью сглаживающего сплайна.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        col_name (str): Название столбца значений (Y).
        time_col (str, optional): Название столбца времени (X). По умолчанию 'Time'.
        s (float, optional): Параметр сглаживания (smoothing factor). По умолчанию 0.0 (интерполяция).
        k (int, optional): Степень многочлена сглаживания. По умолчанию 5.

    Returns:
        tuple: (d1, d2, d3) - массивы numpy с значениями 1-й, 2-й и 3-й производных.
    """
    if col_name not in df.columns or time_col not in df.columns:
        raise KeyError(f"Столбцы '{col_name}' или '{time_col}' не найдены.")

    df_sorted = df.sort_values(by=time_col)
    
    x = df_sorted[time_col].values
    y = df_sorted[col_name].values

    tck = splrep(x, y, s=s, k=k)

    # Вычисляем производные
    d1 = splev(x, tck, der=1)
    d2 = splev(x, tck, der=2)
    d3 = splev(x, tck, der=3)
    
    return d1, d2, d3

def find_optimal_spline_params(df, col_name, time_col, analytic_obj, comp_idx, 
                               s_range=None, k_range=None):
    """
    Ищет оптимальный параметр сглаживания s и степень полинома аппроксимации k путем перебора (Grid Search),
    минимизирует ошибку MSE для первой, второй, и третьей производных.

    Args:
        df: DataFrame с численными данными ANSYS.
        col_name: Имя столбца с деформацией (например, 'Eps_3').
        analytic_obj: Экземпляр (объект) класса AnalyticalDerivatives с аналитическими производными.
        comp_idx: Индекс аналитической компоненты вектора Ильюшина из AnalyticalDerivatives.
        s_range: Список или массив значений s для перебора. 
                 Если None, создается логарифмическая сетка.
        k_range: Список или массив значений k для перебора. 
                 Если None, создается [3, 4, 5, 6, 7].
    
    Returns:
        best_s: Лучшее найденное значение s.
        best_k: Лучшее найденное значение k.
        results_df: DataFrame с результатами поиска.
    """

    df_sorted = df.sort_values(by=time_col)
    t = df_sorted[time_col].values
    y = df_sorted[col_name].values

    d1_true_all, d2_true_all, d3_true_all = analytic_obj.get_derivatives(t)
    true_d1 = d1_true_all[comp_idx]
    true_d2 = d2_true_all[comp_idx]
    true_d3 = d3_true_all[comp_idx]

    if s_range is None:
        s_range = np.logspace(-16, -4, 50)
    if k_range is None:
        k_range = [3, 4, 5]

    results = []

    scale_d1 = np.max(np.abs(true_d1)) if np.max(np.abs(true_d1)) > 0 else 1.0
    scale_d2 = np.max(np.abs(true_d2)) if np.max(np.abs(true_d2)) > 0 else 1.0
    scale_d3 = np.max(np.abs(true_d3)) if np.max(np.abs(true_d3)) > 0 else 1.0

    for k in k_range:
        for s in s_range:
            try:
                # Строим сплайн степени k
                tck = splrep(t, y, s=s, k=k)
                
                # Считаем производные
                d1_calc = splev(t, tck, der=1)
                d2_calc = splev(t, tck, der=2)
                d3_calc = splev(t, tck, der=3)
                
                # Считаем MSE (без краев, чтобы убрать краевые эффекты)
                sl = slice(10, -10) 
                
                mse1 = mean_squared_error(true_d1[sl], d1_calc[sl])
                mse2 = mean_squared_error(true_d2[sl], d2_calc[sl])
                mse3 = mean_squared_error(true_d3[sl], d3_calc[sl])
                
                # Взвешенная ошибка (Score)
                score = 1.0 * (mse1/scale_d1**2) +  1.0 * (mse2/scale_d2**2) +  1.0 * (mse3/scale_d3**2)
                
                results.append({
                    'k': k,
                    's': s,
                    'score': score
                })
            except Exception as e:
                pass

    res_df = pd.DataFrame(results)

    best_idx = res_df['score'].idxmin()
    best_k = int(res_df.loc[best_idx, 'k'])
    best_s = res_df.loc[best_idx, 's']
    
    return best_s, best_k, res_df
