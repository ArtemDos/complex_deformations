import pandas as pd
import numpy as np

from scipy.interpolate import splrep, splev
from statsmodels.tsa.seasonal import seasonal_decompose

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

def add_derivatives(df, col_name, time_split, time_col='Time', d1_name=None, d2_name=None, d3_name=None, 
                            s1=0.0, s2=0.0):
    """
    Вычисляет производные по частям (сегментам), строя отдельный сплайн для
    каждого этапа деформирования.

    Args:
        df (pd.DataFrame): DataFrame, который будет изменен.
        col_name (str): Название столбца для дифференцирования.
        time_col (str, optional): Название столбца времени.
        d1_name, d2_name, d3_name (str, optional): Названия новых столбцов.
        time_split (float, optional): Момент времени, разделяющий 1-й и 2-й этапы.
        s1 (float, optional): Параметр сглаживания для 1-го этапа (линейного).
        s2 (float, optional): Параметр сглаживания для 2-го этапа (колебательного).
    """
    if col_name not in df.columns or time_col not in df.columns:
        raise KeyError(f"Один из столбцов '{col_name}' или '{time_col}' не найден.")

    df_stage1 = df[df[time_col] <= time_split].copy()
    df_stage2 = df[df[time_col] > time_split].copy()

    if not df_stage1.empty:
        x1, y1 = df_stage1[time_col].values, df_stage1[col_name].values
        tck1 = splrep(x1, y1, s=s1)
        if d1_name: df_stage1[d1_name] = splev(x1, tck1, der=1)
        if d2_name: df_stage1[d2_name] = splev(x1, tck1, der=2)
        if d3_name: df_stage1[d3_name] = splev(x1, tck1, der=3)

    if not df_stage2.empty:
        x2, y2 = df_stage2[time_col].values, df_stage2[col_name].values
        tck2 = splrep(x2, y2, s=s2)
        if d1_name: df_stage2[d1_name] = splev(x2, tck2, der=1)
        if d2_name: df_stage2[d2_name] = splev(x2, tck2, der=2)
        if d3_name: df_stage2[d3_name] = splev(x2, tck2, der=3)

    if d1_name: df.loc[df_stage1.index, d1_name] = df_stage1[d1_name]
    if d1_name: df.loc[df_stage2.index, d1_name] = df_stage2[d1_name]
    
    if d2_name: df.loc[df_stage1.index, d2_name] = df_stage1[d2_name]
    if d2_name: df.loc[df_stage2.index, d2_name] = df_stage2[d2_name]

    if d3_name: df.loc[df_stage1.index, d3_name] = df_stage1[d3_name]
    if d3_name: df.loc[df_stage2.index, d3_name] = df_stage2[d3_name]

# def add_derivatives(df, col_name, time_col='Time', d1_name=None, d2_name=None, d3_name=None, s = 0.0):
#     """
#     Вычисляет производные для столбца DataFrame и добавляет их в переданный DataFrame.

#     Args:
#         df (pd.DataFrame): DataFrame, который будет изменен.
#         col_name (str): Название столбца, для которого нужно вычислить производные.
#         time_col (str, optional): Название столбца со временем. По умолчанию 'Time'.
#         d1_name (str, optional): Название нового столбца для первой производной.
#         d2_name (str, optional): Название нового столбца для второй производной.
#         d3_name (str, optional): Название нового столбца для третьей производной.
#         s (double, optional): Параметр сглаживания сплайна
    
#     Raises:
#         KeyError: Если col_name или time_col отсутствуют в DataFrame.
#         ValueError: Если шаг по времени в DataFrame не является постоянным.
#     """
#     print("USE spline")
#     if col_name not in df.columns or time_col not in df.columns:
#         raise KeyError(f"Один из столбцов '{col_name}' или '{time_col}' не найден в DataFrame.")

#     x = df[time_col].values
#     y = df[col_name].values
#     tck = splrep(x, y, s=s)
#     print(s)
    
#     if d1_name:
#         d1y = splev(x, tck, der=1)
#         df[d1_name] = d1y
        
#     if d2_name:
#         d2y = splev(x, tck, der=2)
#         df[d2_name] = d2y

#     if d3_name:
#         d3y = splev(x, tck, der=3)
#         df[d3_name] = d3y

def decompose_signal(df, col_name, time_split, time_col='Time', 
                     num_turns=1.0, time_step=1.0):
    """
    Изолирует вторую фазу сигнала, выполняет декомпозицию на тренд,
    сезонность и остатки, и возвращает результат.

    Args:
        df (pd.DataFrame): DataFrame с полными данными.
        col_name (str): Название столбца с сигналом для анализа (например, 'Eps_1').
        time_col (str, optional): Название столбца времени. По умолчанию 'Time'.
        time_split (float, optional): Момент времени, разделяющий этапы. По умолчанию 20.0.
        num_turns (float, optional): Количество витков/циклов на втором этапе. По умолчанию 1.0.
        time_step (float, optional): Временной шаг данных. По умолчанию 1.0.

    Returns:
        statsmodels.tsa.seasonal.DecomposeResult: Объект с результатами декомпозиции
                                                  (содержит .trend, .seasonal, .resid).
        
    Raises:
        KeyError: Если один из необходимых столбцов отсутствует.
        ValueError: Если данных для анализа недостаточно.
    """
    
    if col_name not in df.columns or time_col not in df.columns:
        raise KeyError(f"Один из столбцов '{col_name}' или '{time_col}' не найден в DataFrame.")

    stage_data = df[df[time_col] > time_split].copy()
    
    if len(stage_data) < 10:
        raise ValueError("Слишком мало данных на втором этапе для проведения анализа.")

    signal_to_analyze = stage_data[col_name]
    time_axis = stage_data[time_col]
    duration_stage = time_axis.max() - time_axis.min()
    duration_one_turn = duration_stage / num_turns
    period = int(round(duration_one_turn / time_step))

    ts_data = signal_to_analyze.set_axis(pd.to_datetime(time_axis, unit='s'))

    if len(ts_data) < 2 * period:
        raise ValueError(f"Длина данных ({len(ts_data)}) меньше двух периодов ({2*period}). "
                         "Результаты декомпозиции будут неточными.")
    
    decomposition = seasonal_decompose(ts_data, model='additive', period=period)
    return decomposition

def analyze_noise(decomposition_result):
    """
    Анализирует остатки (шум) из результата декомпозиции временного ряда.

    Args:
        decomposition_result (statsmodels.tsa.seasonal.DecomposeResult): 
            Объект, возвращенный функцией seasonal_decompose.

    Returns:
        tuple: (Среднее абсолютное значение шума, Стандартное отклонение шума)
    """

    residuals = decomposition_result.resid.dropna()
    
    if residuals.empty:
        print("Не удалось извлечь шум из декомпозиции.")
        return None
    
    return (residuals.abs().mean(),  residuals.std())
