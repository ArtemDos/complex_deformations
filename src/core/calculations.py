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
