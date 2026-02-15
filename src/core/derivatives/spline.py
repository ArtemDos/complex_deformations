import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from sklearn.metrics import mean_squared_error

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

def find_optimal_spline_params_mse(df, col_name, time_col, analytic_obj, comp_idx, 
                               s_range=None, k_range=None,
                               weights=(1.0, 1.0, 1.0)):
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

    d1_true_all, d2_true_all, d3_true_all = analytic_obj.get_all_derivatives(t)
    true_d1 = d1_true_all[comp_idx]
    true_d2 = d2_true_all[comp_idx]
    true_d3 = d3_true_all[comp_idx]

    if s_range is None:
        s_range = np.logspace(-16, -4, 50)
    if k_range is None:
        k_range = [3, 4, 5]

    results = []

    for k in k_range:
        for s in s_range:
            try:
                tck = splrep(t, y, s=s, k=k)
                
                d1_calc = splev(t, tck, der=1)
                d2_calc = splev(t, tck, der=2)
                d3_calc = splev(t, tck, der=3)

                mse1 = mean_squared_error(true_d1, d1_calc)
                mse2 = mean_squared_error(true_d2, d2_calc)
                mse3 = mean_squared_error(true_d3, d3_calc)

                results.append({
                    'k': k,
                    's': s,
                    'mse1': mse1,
                    'mse2': mse2,
                    'mse3': mse3
                })
            except Exception:
                pass

    res_df = pd.DataFrame(results)

    for col in ['mse1', 'mse2', 'mse3']:
        res_df[f'{col}_norm'] = res_df[col] / res_df[col].max()
    
    w1, w2, w3 = weights
    res_df['score'] = (w1 * res_df['mse1_norm'] + 
                       w2 * res_df['mse2_norm'] +
                       w3 * res_df['mse3_norm'])

    best_idx = res_df['score'].idxmin()
    best_k = int(res_df.loc[best_idx, 'k'])
    best_s = res_df.loc[best_idx, 's']
    
    return best_s, best_k, res_df


def find_optimal_spline_params_rough(df, col_name, time_col, 
                                            s_range=None, k_range=None, 
                                            weights=(1.0, 0.0, 0.0, 1.0)):
    """
    Ищет оптимальные s и k БЕЗ аналитического решения.
    Критерий: Баланс между точностью аппроксимации (MSE самой функции)
    и гладкостью производных (Roughness/Total Variation).
    
    Args:
        weights: Кортеж весов (w_mse_func, w_rough_d1, w_rough_d2, w_rough_d3).
    """

    df_sorted = df.sort_values(by=time_col)
    t = df_sorted[time_col].values
    y = df_sorted[col_name].values
    
    if s_range is None:
        s_range = np.logspace(-15, -2, 50)
    if k_range is None:
        k_range = [3, 4, 5]

    results = []
    
    for k in k_range:
        for s in s_range:
            try:
                tck = splrep(t, y, s=s, k=k)
                y_pred = splev(t, tck, der=0)
                d1 = splev(t, tck, der=1)
                d2 = splev(t, tck, der=2)
                d3 = splev(t, tck, der=3)

                mse_func = mean_squared_error(y, y_pred)
                r1 = np.sum(np.abs(np.diff(d1))) / (len(t) - 1)
                r2 = np.sum(np.abs(np.diff(d2))) / (len(t) - 1)
                r3 = np.sum(np.abs(np.diff(d3))) / (len(t) - 1)
                
                results.append({
                    'k': k,
                    's': s,
                    'mse_func': mse_func,
                    'rough_d1': r1,
                    'rough_d2': r2,
                    'rough_d3': r3
                })
            except Exception:
                pass

    res_df = pd.DataFrame(results)

    for col in ['mse_func', 'rough_d1', 'rough_d2', 'rough_d3']:
        min_v = res_df[col].min()
        max_v = res_df[col].max()
        if max_v > min_v:
            res_df[f'{col}_norm'] = (res_df[col] - min_v) / (max_v - min_v)
        else:
            res_df[f'{col}_norm'] = 0.0

    w0, w1, w2, w3 = weights
    
    res_df['score'] = (w0 * res_df['mse_func_norm'] + 
                       w1 * res_df['rough_d1_norm'] +
                       w2 * res_df['rough_d2_norm'] +
                       w3 * res_df['rough_d3_norm'])
    
    best_idx = res_df['score'].idxmin()
    best_k = int(res_df.loc[best_idx, 'k'])
    best_s = res_df.loc[best_idx, 's']
    
    return best_s, best_k, res_df
