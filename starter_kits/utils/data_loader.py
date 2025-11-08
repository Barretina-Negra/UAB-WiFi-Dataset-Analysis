"""
Utilidades para cargar y procesar el dataset WiFi UAB
======================================================

Este módulo proporciona funciones auxiliares para facilitar
la carga y exploración del dataset WiFi de la UAB.

Autor: Albert Gil López
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def load_json_file(file_path: Union[str, Path]) -> List[dict]:
    """
    Carga un archivo JSON y retorna la lista de registros.

    Args:
        file_path: Ruta al archivo JSON

    Returns:
        Lista de diccionarios con los registros
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_multiple_files(
    directory: Union[str, Path],
    pattern: str = "*.json",
    max_files: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Carga múltiples archivos JSON de un directorio y los combina en un DataFrame.

    Args:
        directory: Directorio con los archivos JSON
        pattern: Patrón de archivos a buscar (default: "*.json")
        max_files: Número máximo de archivos a cargar (None = todos)
        verbose: Mostrar progreso de carga

    Returns:
        DataFrame de pandas con todos los registros combinados

    Ejemplo:
        >>> df = load_multiple_files("anonymized_data/aps", max_files=10)
        >>> print(f"Cargados {len(df)} registros")
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))

    if max_files:
        files = files[:max_files]

    if verbose:
        print(f"📁 Encontrados {len(files)} archivos en {directory}")
        print(f"📊 Cargando {'todos' if not max_files else max_files} archivos...")

    all_records = []
    for i, file in enumerate(files):
        try:
            records = load_json_file(file)
            all_records.extend(records)

            if verbose and (i + 1) % 10 == 0:
                print(f"   Procesados {i + 1}/{len(files)} archivos... ({len(all_records)} registros)")
        except Exception as e:
            print(f"⚠️  Error en {file.name}: {e}")
            continue

    df = pd.DataFrame(all_records)

    if verbose:
        print(f"✅ Cargados {len(df)} registros de {len(files)} archivos")
        print(f"💾 Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def load_aps(
    data_dir: Union[str, Path] = "../anonymized_data/aps",
    max_files: Optional[int] = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Carga archivos de Access Points.

    Args:
        data_dir: Directorio con archivos de APs
        max_files: Número máximo de archivos (None = todos)
        verbose: Mostrar progreso

    Returns:
        DataFrame con datos de APs
    """
    df = load_multiple_files(data_dir, max_files=max_files, verbose=verbose)

    # Convertir timestamp a datetime
    if 'last_modified' in df.columns:
        df['timestamp'] = pd.to_datetime(df['last_modified'], unit='s')

    return df


def load_clients(
    data_dir: Union[str, Path] = "../anonymized_data/clients",
    max_files: Optional[int] = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Carga archivos de Clientes/Dispositivos.

    Args:
        data_dir: Directorio con archivos de clientes
        max_files: Número máximo de archivos (None = todos)
        verbose: Mostrar progreso

    Returns:
        DataFrame con datos de clientes
    """
    df = load_multiple_files(data_dir, max_files=max_files, verbose=verbose)

    # Convertir timestamp a datetime
    if 'last_connection_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['last_connection_time'], unit='ms')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Retorna información básica del DataFrame.

    Args:
        df: DataFrame a analizar

    Returns:
        Diccionario con información del dataset
    """
    info = {
        'total_records': len(df),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns': list(df.columns),
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }

    if 'timestamp' in df.columns:
        info['date_range'] = {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max(),
            'days': (df['timestamp'].max() - df['timestamp'].min()).days
        }

    return info


def print_dataset_summary(df: pd.DataFrame, name: str = "Dataset"):
    """
    Imprime un resumen legible del dataset.

    Args:
        df: DataFrame a resumir
        name: Nombre del dataset
    """
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN: {name}")
    print(f"{'='*60}")

    print(f"\n🔢 Registros totales: {len(df):,}")
    print(f"📝 Columnas: {len(df.columns)}")
    print(f"💾 Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    if 'timestamp' in df.columns:
        print(f"\n📅 Rango temporal:")
        print(f"   Inicio: {df['timestamp'].min()}")
        print(f"   Fin:    {df['timestamp'].max()}")
        print(f"   Días:   {(df['timestamp'].max() - df['timestamp'].min()).days}")

    print(f"\n📋 Columnas disponibles:")
    for col in df.columns[:10]:
        non_null = df[col].notna().sum()
        pct = (non_null / len(df)) * 100
        print(f"   • {col:30s} ({pct:5.1f}% completo)")

    if len(df.columns) > 10:
        print(f"   ... y {len(df.columns) - 10} columnas más")

    print(f"\n{'='*60}\n")


def get_top_aps(df_clients: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Retorna los APs más utilizados.

    Args:
        df_clients: DataFrame de clientes
        top_n: Número de APs a retornar

    Returns:
        DataFrame con los top APs y número de conexiones
    """
    return (df_clients['associated_device_name']
            .value_counts()
            .head(top_n)
            .reset_index()
            .rename(columns={'index': 'AP', 'associated_device_name': 'connections'}))


def filter_by_time(
    df: pd.DataFrame,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    hour_range: Optional[tuple] = None
) -> pd.DataFrame:
    """
    Filtra registros por tiempo.

    Args:
        df: DataFrame con columna 'timestamp'
        start_time: Fecha/hora inicio (formato: "2025-04-03")
        end_time: Fecha/hora fin
        hour_range: Tupla (hora_inicio, hora_fin) ej: (8, 18) para 8am-6pm

    Returns:
        DataFrame filtrado
    """
    result = df.copy()

    if start_time:
        result = result[result['timestamp'] >= pd.to_datetime(start_time)]

    if end_time:
        result = result[result['timestamp'] <= pd.to_datetime(end_time)]

    if hour_range:
        start_hour, end_hour = hour_range
        result = result[
            (result['timestamp'].dt.hour >= start_hour) &
            (result['timestamp'].dt.hour < end_hour)
        ]

    return result


def get_device_history(
    df: pd.DataFrame,
    device_id: str,
    sort_by_time: bool = True
) -> pd.DataFrame:
    """
    Obtiene el historial completo de un dispositivo.

    Args:
        df: DataFrame de clientes
        device_id: ID del dispositivo (ej: "CLIENT_87e3ddea248c")
        sort_by_time: Ordenar por timestamp

    Returns:
        DataFrame con historial del dispositivo
    """
    history = df[df['macaddr'] == device_id].copy()

    if sort_by_time and 'timestamp' in history.columns:
        history = history.sort_values('timestamp')

    return history


def calculate_signal_quality_stats(df_clients: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas de calidad de señal por AP.

    Args:
        df_clients: DataFrame de clientes

    Returns:
        DataFrame con estadísticas por AP
    """
    return (df_clients.groupby('associated_device_name')
            .agg({
                'signal_db': ['mean', 'std', 'min', 'max'],
                'signal_strength': 'mean',
                'snr': 'mean',
                'speed': 'mean',
                'macaddr': 'count'
            })
            .round(2)
            .rename(columns={'macaddr': 'total_connections'}))


def get_hourly_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula actividad por hora del día.

    Args:
        df: DataFrame con columna 'timestamp'

    Returns:
        DataFrame con conteo por hora
    """
    if 'hour' not in df.columns and 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour

    return (df.groupby('hour')
            .size()
            .reset_index(name='count')
            .sort_values('hour'))


# Constantes útiles
AP_NAME_PATTERN = r'AP-([A-Z]+)(\d+)'  # Patrón para extraer edificio del nombre

SIGNAL_STRENGTH_LABELS = {
    1: 'Muy Mala',
    2: 'Mala',
    3: 'Regular',
    4: 'Buena',
    5: 'Excelente'
}

NETWORK_TYPES = ['UAB', 'eduroam']

BANDS = {
    2.4: '2.4 GHz',
    5: '5 GHz',
    6: '6 GHz'
}


if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Testing data_loader.py...")
    print("\nEste módulo proporciona utilidades para cargar el dataset.")
    print("Importa las funciones en tu notebook con:")
    print("  from utils.data_loader import load_aps, load_clients")
