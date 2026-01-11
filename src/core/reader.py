import polars as pl
import pandas as pd
import numpy as np
import csv
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
        min_time = data['Time'].min()
        max_time = data['Time'].max()
        target_times = np.arange(np.floor(min_time), max_time + time_step, time_step)
        target_df = pd.DataFrame({'Time': target_times})
        df_filtered = pd.merge_asof(
            target_df, 
            data, 
            on='Time', 
            direction='nearest'
        )
        df_filtered = df_filtered.drop_duplicates(subset=['Time'], keep='first').reset_index(drop=True)
        df_filtered['Time'] = df_filtered['Time'] - 1.0
        return df_filtered

    except Exception as e:
        raise ValueError(f"Произошла ошибка при чтении или обработке файла: {e}") from e

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
    

def load_and_filter_ansys_large(file_path, time_step=1.0):
    """
    Оптимизированная функция для чтения огромных файлов ANSYS (100Гб+) с помощью Polars.
    Автоматически исправляет имена колонок (удаляет пробелы).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден по пути: '{file_path}'")

    try:
        lazy_df = pl.scan_csv(file_path, separator=',')
        old_columns = lazy_df.collect_schema().names()
        rename_map = {col: col.strip() for col in old_columns}
        lazy_df = lazy_df.rename(rename_map)
        time_stats = lazy_df.select([
            pl.col("Time").min().alias("min"),
            pl.col("Time").max().alias("max")
        ]).collect()
        
        min_time = time_stats["min"][0]
        max_time = time_stats["max"][0]

        target_times = np.arange(np.floor(min_time), max_time + time_step, time_step)
        target_df = pl.DataFrame({'Time': target_times})
        lazy_df_sorted = lazy_df.sort("Time") 

        result_lazy = target_df.lazy().join_asof(
            lazy_df_sorted, 
            on='Time', 
            strategy='nearest'
        )

        df_filtered = result_lazy.collect()
        df_pandas = df_filtered.to_pandas()
        df_pandas = df_pandas.drop_duplicates(subset=['Time'], keep='first').reset_index(drop=True)
        df_pandas['Time'] = df_pandas['Time'] - 1.0
        
        return df_pandas

    except Exception as e:
        raise ValueError(f"Ошибка при обработке файла: {e}") from e


def filter_ansys_manual(input_path, output_path, time_step=1.0, tolerance=0.05):
    print("Запуск построчной обработки (самый надежный метод)...")
    
    with open(input_path, 'r', buffering=1024*1024) as f_in, \
         open(output_path, 'w', newline='') as f_out:
        
        # Читаем первую строку (заголовки)
        header_line = f_in.readline()
        headers = [h.strip() for h in header_line.split(',')]
        
        # Находим индекс колонки Time
        try:
            time_idx = headers.index("Time")
        except ValueError:
            # Если не нашли "Time", пробуем найти что-то похожее (Ansys часто делает "Time   ")
            for i, h in enumerate(headers):
                if "Time" in h:
                    time_idx = i
                    break
            else:
                raise ValueError("Колонка Time не найдена!")

        writer = csv.writer(f_out)
        writer.writerow(headers) # Пишем чистые заголовки

        last_saved_step = -999.0
        
        reader = csv.reader(f_in)
        
        for i, row in enumerate(reader):
            if not row: continue
            
            try:
                current_time = float(row[time_idx])
            except ValueError:
                continue # Пропускаем битые строки
                
            # Проверяем, близко ли это время к нужному шагу
            # Например, если time_step=1.0, ищем числа близкие к 1, 2, 3...
            # И также проверяем, чтобы не сохранять дубликаты для одного шага
            
            # Округляем до ближайшего теоретического шага
            target_step = round(current_time / time_step) * time_step
            
            if abs(current_time - target_step) < tolerance:
                # Если мы еще не сохраняли этот шаг (или это новый шаг)
                if abs(target_step - last_saved_step) > (time_step * 0.5):
                    
                    # Корректируем время (ваше условие -1.0)
                    row[time_idx] = str(current_time - 1.0)
                    
                    writer.writerow(row)
                    last_saved_step = target_step
            
            if i % 1_000_000 == 0:
                print(f"Обработано {i} строк...", end='\r')

    print("\nГотово! Файл сохранен.")
