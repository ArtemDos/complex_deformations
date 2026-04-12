import pandas as pd
import numpy as np
import csv
import os


def load_experimental_data(folder_path,
                           szz_file='szz_data.csv',
                           stt_file='stt_data.csv',
                           stz_file='stz_data.csv',
                           ezz_file='ezz_data.csv',
                           ett_file='ett_data.csv',
                           etz_file='etz_data.csv'):
    """
    Загружает и объединяет 6 отдельных CSV-файлов с экспериментальными данными 
    в один DataFrame.

    Предполагается, что каждый файл имеет 2 колонки: 'No' и значение переменной.

    Args:
        folder_path (str): Путь к папке, где лежат все 6 CSV-файлов.
        szz_file (str): Имя файла для SZZ (осевое напряжение).
        stt_file (str): Имя файла для STT (окружное напряжение).
        stz_file (str): Имя файла для STZ (касательное напряжение).
        ezz_file (str): Имя файла для EZZ (осевая деформация).
        ett_file (str): Имя файла для ETT (окружная деформация).
        etz_file (str): Имя файла для ETZ (сдвиговая деформация).

    Returns:
        pandas.DataFrame: Объединенный DataFrame со всеми экспериментальными данными,
                          либо None в случае ошибки.
                          
    Raises:
        FileNotFoundError: Если папка или один из файлов не найден.
        ValueError: Если произошла ошибка при чтении или объединении файлов.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Папка не найдена по пути: '{folder_path}'")

    try:
        files_to_load = {
            'S_ZZ': szz_file, 'S_TT': stt_file, 'S_TZ': stz_file,
            'EPTO_ZZ': ezz_file, 'EPTO_TT': ett_file, 'EPTO_TZ': etz_file
        }
        
        df_list = []
        
        for var_name, file_name in files_to_load.items():
            full_path = os.path.join(folder_path, file_name)
            
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Файл '{file_name}' не найден.")
            
            temp_df = pd.read_csv(full_path, index_col='No')
            if var_name in ('EPTO_TT', 'EPTO_ZZ', 'EPTO_TZ'):
                temp_df = temp_df / 100.0

            df_list.append(temp_df)

        exp_df = pd.concat(df_list, axis=1)
        exp_df.reset_index(inplace=True)
        exp_df['Time'] = exp_df['No'].astype(float)
        return exp_df

    except Exception as e:
        raise ValueError(f"Произошла ошибка при загрузке или объединении: {e}") from e
    

def read_ansys_csv(file_path):
    """
    Загружает CSV-файл с отфильтрованными результатами Ansys и возвращает очищенный DataFrame.

    Args:
        file_path (str): Путь к файлу.

    Returns:
        pd.DataFrame: DataFrame с данными.
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден по пути: '{file_path}'")

    try:
        df = pd.read_csv(file_path, sep=',')
        df.columns = df.columns.str.strip()
        df = df.sort_values(by='Time').reset_index(drop=True)
        df['Time'] = df['Time'] - 1.0
        
        return df

    except Exception as e:
        raise ValueError(f"Ошибка при чтении файла: {e}") from e
