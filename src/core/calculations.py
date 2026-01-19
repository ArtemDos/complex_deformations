import pandas as pd
import numpy as np
from scipy import integrate


def calculate_von_mises_stress(df, 
                               s_x_col='S_TT', 
                               s_y_col='S_ZZ', 
                               s_xy_col='S_TZ', 
                               new_col_name='S_EQV'):
    """
    Рассчитывает эквивалентное напряжение по Мизесу для плоского напряженного состояния
    и добавляет его в виде нового столбца в DataFrame.

    Args:
        df (pd.DataFrame): DataFrame, содержащий компоненты напряжений.
        s_x_col (str): Название столбца с напряжением σ_x.
        s_y_col (str): Название столбца с напряжением σ_y.
        s_xy_col (str): Название столбца с напряжением τ_xy.
        new_col_name (str): Название нового столбца для записи результата.

    Returns:
        pd.DataFrame: Исходный DataFrame с добавленным новым столбцом.
        
    Raises:
        KeyError: Если один из необходимых столбцов отсутствует в DataFrame.
    """

    required_cols = [s_x_col, s_y_col, s_xy_col]
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"Один из необходимых столбцов {required_cols} отсутствует в DataFrame.")

    s_x = df[s_x_col]
    s_y = df[s_y_col]
    s_xy = df[s_xy_col]
    
    # Формула Мизеса для плоского напряженного состояния S_eqv = sqrt(s_x² - s_x*s_y + s_y² + 3*s_xy²)
    df[new_col_name] = np.sqrt(
        s_x**2 - s_x * s_y + s_y**2 + 3 * s_xy**2
    )
    return df

def calculate_ilushin_strain_vector(df, 
                                    epto_x_col='EPTO_TT', 
                                    epto_y_col='EPTO_ZZ',
                                    epto_z_col='EPTO_RR',
                                    epto_xy_col='EPTO_TZ',
                                    compressible=True):
    """
    Рассчитывает компоненты вектора деформаций Ильюшина на основе полных деформаций.
    Возвращает новый DataFrame с результатами.

    Args:
        df (pd.DataFrame): DataFrame, содержащий столбцы 'Time' и полные деформации.
        epto_x_col (str): Название столбца с полной деформацией по X.
        epto_y_col (str): Название столбца с полной деформацией по Y.
        epto_z_col (str): Название столбца с полной деформацией по Z.
        epto_xy_col (str): Название столбца с тензорной сдвиговой деформацией.
        compressible (bool): Флаг, указывающий, как считать.
                             True (по умолч.): Рассчитывает девиатор, учитывая сжимаемость.
                             False: Игнорирует сжимаемость (e_ij = ε_ij).

    Returns:
        pd.DataFrame: Новый DataFrame со столбцами 'Time', 'Eps_1', 'Eps_2', 'Eps_3'.
        
    Raises:
        KeyError: Если один из необходимых столбцов отсутствует в DataFrame.
    """

    required_cols = ['Time', epto_x_col, epto_y_col, epto_xy_col]
    if compressible:
        required_cols.append(epto_z_col)
        
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"Один из необходимых столбцов {required_cols} отсутствует в DataFrame.")

    if compressible:
        mean_deformation = (1/3) * (df[epto_x_col] + df[epto_y_col] + df[epto_z_col])
        e_xx = df[epto_x_col] - mean_deformation
        e_yy = df[epto_y_col] - mean_deformation
    else:
        e_xx = df[epto_x_col]
        e_yy = df[epto_y_col]

    e_xy = df[epto_xy_col] / 2.0

    Eps_1 = e_yy
    Eps_2 = (1 / np.sqrt(3)) * (e_yy + 2 * e_xx)
    Eps_3 = (2 / np.sqrt(3)) * e_xy

    ilushin_df = pd.DataFrame({
        'Time': df['Time'],
        'Eps_1': Eps_1,
        'Eps_2': Eps_2,
        'Eps_3': Eps_3
    })

    return ilushin_df

def calculate_geometry(df, suffix, time_col='Time_Local'):
    """
    Рассчитывает геометрические характеристики траектории (s, kappa, tau).
    
    Args:
        df: DataFrame с данными.
        suffix: Суффикс набора данных (например, '_analytic' или '_spline').
                Функция будет искать колонки: 'dEps_1_analytic', 'd2Eps_1_analytic' и т.д.
        time_col: Колонка времени.
        
    Returns:
        tuple: (s, kappa, tau) - три массива numpy.
    """

    v = df[[f'dEps_1{suffix}', f'dEps_2{suffix}', f'dEps_3{suffix}']].values
    a = df[[f'd2Eps_1{suffix}', f'd2Eps_2{suffix}', f'd2Eps_3{suffix}']].values
    j = df[[f'd3Eps_1{suffix}', f'd3Eps_2{suffix}', f'd3Eps_3{suffix}']].values
    
    # Длина дуги s (интеграл от модуля скорости)
    # |v| = sqrt(v1^2 + v2^2 + v3^2)
    v_norm = np.linalg.norm(v, axis=1)
    t = df[time_col].values
    s = integrate.cumulative_trapezoid(v_norm, t, initial=0)
    
    # Кривизна (Kappa)
    # k = |v x a| / |v|^3
    cross_va = np.cross(v, a)
    cross_va_norm = np.linalg.norm(cross_va, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = cross_va_norm / (v_norm**3)
        kappa[v_norm < 1e-30] = 0.0
        
    # Кручение (Tau)
    # tau = ((v x a) . j) / |v x a|^2
    numerator = np.sum(cross_va * j, axis=1)
    denominator = cross_va_norm**2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = numerator / denominator
        tau[denominator < 1e-30] = 0.0
        
    return s, kappa, tau

def calculate_stress_vector(df):
    """
    Рассчитывает вектор напряжений Ильюшина (Sig_1, Sig_2, Sig_3)
    на основе компонент тензора напряжений из ANSYS.
    """
    df = df.copy()    
    sigma_z = df['S_ZZ']
    sigma_t = df['S_TT']
    sigma_tz = df['S_TZ']
    sigma_r = 0.0

    # sigma_mean = (sz + st + sr) / 3
    sigma_mean = (sigma_z + sigma_t + sigma_r) / 3.0
    
    # Вычисляем девиаторы напряжений (S_ij = sigma_ij - sigma_mean)
    S_zz = sigma_z - sigma_mean
    S_tt = sigma_t - sigma_mean
    S_tz = sigma_tz

    # Sig_1 = 1.5 * S_zz
    sig_1 = 1.5 * S_zz
    
    # Sig_2 = (sqrt(3)/2) * (S_zz + 2*S_tt)
    sig_2 = (np.sqrt(3) / 2.0) * (S_zz + 2 * S_tt)
    
    # Sig_3 = sqrt(3) * S_tz
    sig_3 = np.sqrt(3) * S_tz
    
    return sig_1, sig_2, sig_3

def calculate_theta(df, deriv_suffix='_spline'):
    """
    Рассчитывает угол сближения Theta (в градусах) между 
    вектором напряжений (Sig) и вектором скорости деформаций (dEps).
    
    Args:
        df: DataFrame с напряжениями (Sig_1..3) и производными (dEps_1..3).
        deriv_suffix: Суффикс колонок производных ('_spline' или '_analytic').
    """
    # Вектор напряжений
    sig = df[['Sig_1', 'Sig_2', 'Sig_3']].values
    
    # Вектор скоростей деформаций
    deps_cols = [f'dEps_1{deriv_suffix}', f'dEps_2{deriv_suffix}', f'dEps_3{deriv_suffix}']
    deps = df[deps_cols].values

    dot_prod = np.sum(sig * deps, axis=1)
    sig_norm = np.linalg.norm(sig, axis=1)
    deps_norm = np.linalg.norm(deps, axis=1)
    denom = sig_norm * deps_norm
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = dot_prod / denom
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta))
        theta[denom < 1e-30] = 0.0
        
    return theta
