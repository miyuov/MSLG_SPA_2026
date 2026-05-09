# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def parse_mslg_data(file_path):
    """
    Lee el archivo TXT y extrae las columnas respetando las anotaciones especiales
    como #, dm-, + y -.
    """
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8', on_bad_lines='skip')
    return df

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Iniciando procesamiento de glosas y español...')

    # 1. Cargar datos
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)
    
    # Procesamos el archivo de entrenamiento (MSLG_SPA_train.txt)
    train_file = input_path / "MSLG_SPA_train.txt"
    if train_file.exists():
        df_train = parse_mslg_data(train_file)
        # Guardamos en processed como CSV 
        df_train.to_csv(output_path / "train_cleaned.csv", index=False)
        logger.info(f'Set de entrenamiento guardado en {output_path} con {len(df_train)} registros.')
    
    # Procesamos los archivos de test (SPA2MSLG_test.txt y MSLG2SPA_test.txt)
    for test_name in ["SPA2MSLG_test.txt", "MSLG2SPA_test.txt"]:
        test_file = input_path / test_name
        if test_file.exists():
            df_test = parse_mslg_data(test_file)
            df_test.to_csv(output_path / f"{test_name.replace('.txt', '.csv')}", index=False)
            logger.info(f'Archivo de prueba {test_name} procesado.')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()