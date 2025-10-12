import pandas as pd
import numpy as np
import os

def load_and_filter_ansys_csv(file_path, time_step=1.0):
    """
    Загружает CSV-файл с результатами Ansys, выполняет фильтрацию по заданному
    шагу времени и возвращает очищенный DataFrame.

    Args:
        file_path (str): Полный путь к CSV-файлу.
        time_step (float, optional): Шаг времени для фильтрации. 
                                     Оставляет точки, кратные этому шагу. 
                                     По умолчанию 1.0 (только целые шаги).

    Returns:
        pandas.DataFrame: Отфильтрованный и очищенный DataFrame с результатами.

    Raises:
        FileNotFoundError: Если указанный файл не найден.
        ValueError: Если в файле некорректные данные или произошла ошибка чтения.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден по пути: '{file_path}'")

    try:
        data = pd.read_csv(file_path, sep=',')
        data.columns = data.columns.str.strip()
        data = data.sort_values(by='Time').reset_index(drop=True)
        max_time = data['Time'].max()
        target_times = np.arange(0, max_time + time_step, time_step)
        target_df = pd.DataFrame({'Time': target_times})
        df_filtered = pd.merge_asof(
            target_df, 
            data, 
            on='Time', 
            direction='nearest'
        )
        df_filtered = df_filtered.drop_duplicates(subset=['Time'], keep='first').reset_index(drop=True)
        return df_filtered

    except Exception as e:
        raise ValueError(f"Произошла ошибка при чтении или обработке файла: {e}") from e
