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
                                    use_deviatoric_strains=True):
    """
    Рассчитывает компоненты вектора деформаций Ильюшина на основе полных деформаций.
    Возвращает новый DataFrame с результатами.

    Args:
        df (pd.DataFrame): DataFrame, содержащий столбцы 'Time' и полные деформации.
        epto_x_col (str): Название столбца с полной деформацией по X.
        epto_y_col (str): Название столбца с полной деформацией по Y.
        epto_z_col (str): Название столбца с полной деформацией по Z.
        epto_xy_col (str): Название столбца с тензорной сдвиговой деформацией.
        use_deviatoric_strains (bool): Флаг, указывающий, учитывая девиатор деформаций при расчете.

    Returns:
        pd.DataFrame: Новый DataFrame со столбцами 'Time', 'Eps_1', 'Eps_2', 'Eps_3'.
        
    Raises:
        KeyError: Если один из необходимых столбцов отсутствует в DataFrame.
    """

    required_cols = ['Time', epto_x_col, epto_y_col, epto_xy_col]
    if use_deviatoric_strains:
        required_cols.append(epto_z_col)
        
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"Один из необходимых столбцов {required_cols} отсутствует в DataFrame.")

    if use_deviatoric_strains:
        mean_deformation = (1/3) * (df[epto_x_col] + df[epto_y_col] + df[epto_z_col])
        e_xx = df[epto_x_col] - mean_deformation
        e_yy = df[epto_y_col] - mean_deformation
    else:
        e_xx = df[epto_x_col]
        e_yy = df[epto_y_col]

    e_xy = df[epto_xy_col]

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

def ilushin_strain_path_length(
    df,
    time_col='Time',
    *,
    use_existing_ilushin_strains=False,
    use_deviatoric_strains=False,
):
    """
    Длина дуги s в пространстве деформаций Ильюшина: кумулятивная сумма евклидовых
    приращений между соседними точками (по возрастанию времени):
    ds_k = sqrt(ΔEps_1² + ΔEps_2² + ΔEps_3²), s_0 = 0.

    Args:
        df: Исходный кадр данных.
        time_col: Столбец времени для сортировки и в выходном кадре.
        use_existing_ilushin_strains: Если False — внутри вызывается
            ``calculate_ilushin_strain_vector(df, use_deviatoric_strains=...)``.
            Если True — из ``df`` берутся ``Eps_1``, ``Eps_2``, ``Eps_3`` и ``time_col``;
            ``use_deviatoric_strains`` не используется.

    Returns:
        pd.DataFrame: столбцы `time_col` и ``s``, отсортировано по времени.
    """
    if use_existing_ilushin_strains:
        required = [time_col, 'Eps_1', 'Eps_2', 'Eps_3']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"При use_existing_ilushin_strains=True в df должны быть столбцы {required}; "
                f"нет: {missing}"
            )
        il = df[required].copy()
    else:
        il = calculate_ilushin_strain_vector(df, use_deviatoric_strains=use_deviatoric_strains)
        if time_col not in il.columns:
            if time_col in df.columns and len(df) == len(il):
                il = il.copy()
                il[time_col] = df[time_col].to_numpy()
            else:
                raise KeyError(
                    f"После calculate_ilushin_strain_vector в кадре нет '{time_col}'. "
                    f"Задайте time_col='Time' либо добавьте в df столбец '{time_col}' той же длины."
                )
    il = il.sort_values(by=time_col).reset_index(drop=True)
    d1 = np.diff(il['Eps_1'].to_numpy())
    d2 = np.diff(il['Eps_2'].to_numpy())
    d3 = np.diff(il['Eps_3'].to_numpy())
    ds = np.sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    out = il[[time_col]].copy()
    out['s'] = s
    return out


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


def prepare_ilushin_path_dataset(df, use_deviatoric_strains=False, time_col='Time'):
    """
    Один кадр (эксперимент или расчёт): добавляет столбцы деформаций Ильюшина ``Eps_*``,
    длину дуги ``s`` вдоль этой траектории и напряжения Ильюшина ``Sig_*``.
    """
    out = df.copy()
    il = calculate_ilushin_strain_vector(out, use_deviatoric_strains=use_deviatoric_strains)
    out['Eps_1'] = il['Eps_1'].to_numpy()
    out['Eps_2'] = il['Eps_2'].to_numpy()
    out['Eps_3'] = il['Eps_3'].to_numpy()
    s_df = ilushin_strain_path_length(
      out,
      time_col=time_col,
      use_existing_ilushin_strains=True,
  )
    out = out.merge(s_df, on=time_col, how='inner')
    sig1, sig2, sig3 = calculate_stress_vector(out)
    out['Sig_1'] = sig1.to_numpy()
    out['Sig_2'] = sig2.to_numpy()
    out['Sig_3'] = sig3.to_numpy()
    return out


def ilushin_stress_metrics_at_experiment(
    experimental_df,
    numerical_df,
    time_col='Time',
    direction='nearest',
    angle_col='phi',
    relative_modulus_col='relative_error',
    degrees=False,
):
    """
    Сравнение напряжений Ильюшина «расчёт / эксперимент» в шагах эксперимента.

    - Сопоставление по времени: к каждой строке эксперимента подставляется ближайший по ``time_col``
      шаг расчёта (``pd.merge_asof``).
    - Добавляются ``phi`` — угол между векторами (MC и EXP) и ``relative_error`` = (|σ_mc|-|σ_exp|)/|σ_exp|.
    """
    exp = experimental_df.copy()
    num = numerical_df.copy()

    se1, se2, se3 = calculate_stress_vector(exp)
    exp['Sig_1_exp'] = se1.to_numpy()
    exp['Sig_2_exp'] = se2.to_numpy()
    exp['Sig_3_exp'] = se3.to_numpy()

    sn1, sn2, sn3 = calculate_stress_vector(num)
    num['Sig_1_mc'] = sn1.to_numpy()
    num['Sig_2_mc'] = sn2.to_numpy()
    num['Sig_3_mc'] = sn3.to_numpy()

    left = exp.sort_values(time_col).reset_index(drop=True)
    right = num[[time_col, 'Sig_1_mc', 'Sig_2_mc', 'Sig_3_mc']].sort_values(time_col)
    merged = pd.merge_asof(left, right, on=time_col, direction=direction)

    mc1 = merged['Sig_1_mc'].to_numpy(dtype=float)
    mc2 = merged['Sig_2_mc'].to_numpy(dtype=float)
    mc3 = merged['Sig_3_mc'].to_numpy(dtype=float)
    ex1 = merged['Sig_1_exp'].to_numpy(dtype=float)
    ex2 = merged['Sig_2_exp'].to_numpy(dtype=float)
    ex3 = merged['Sig_3_exp'].to_numpy(dtype=float)
    dot = mc1 * ex1 + mc2 * ex2 + mc3 * ex3
    norm_mc = np.sqrt(mc1 * mc1 + mc2 * mc2 + mc3 * mc3)
    norm_exp = np.sqrt(ex1 * ex1 + ex2 * ex2 + ex3 * ex3)
    denom = norm_mc * norm_exp
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_t = np.clip(dot / denom, -1.0, 1.0)
        phi = np.arccos(cos_t)
    if degrees:
        phi = np.degrees(phi)
    merged[angle_col] = phi
    merged[relative_modulus_col] = (norm_mc - norm_exp) / norm_exp
    return merged


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

def mean_hydrostatic_stress(
    df,
    s_zz_col='S_ZZ',
    s_tt_col='S_TT',
    s_rr_col='S_RR',
):
    """
    Среднее гидростатическое напряжение σ_mean = (σ_zz + σ_θθ + σ_rr) / 3.

    Если столбца ``s_rr_col`` в ``df`` нет, принимается σ_rr = 0 (тогда
    σ_mean = (σ_zz + σ_θθ) / 3.

    Returns:
        pd.Series: значения σ_mean по строкам ``df``.
    """
    required = [s_zz_col, s_tt_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Для среднего напряжения нужны столбцы {required}; отсутствуют: {missing}"
        )
    if s_rr_col in df.columns:
        return (df[s_zz_col] + df[s_tt_col] + df[s_rr_col]) / 3.0
    return (df[s_zz_col] + df[s_tt_col]) / 3.0


def calculate_mean_stress(
    df,
    new_col_name='Sigma_Mean',
    s_zz_col='S_ZZ',
    s_tt_col='S_TT',
    s_rr_col='S_RR',
):
    """
    Добавляет в ``df`` столбец со средним напряжением.
    При наличии ``S_RR`` он входит в среднее; иначе используется (S_ZZ + S_TT) / 3.
    """
    df[new_col_name] = mean_hydrostatic_stress(
        df,
        s_zz_col=s_zz_col,
        s_tt_col=s_tt_col,
        s_rr_col=s_rr_col,
    )
    return df
