import pandas as pd
import numpy as np

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
