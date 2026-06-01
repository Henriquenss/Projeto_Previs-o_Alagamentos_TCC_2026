import pandas as pd
import os
from tqdm import tqdm

# Defina o caminho para o diretório
caminho_diretorio = 'Projeto_HJV\SAO_PAULO\dados_inmet'

SAVE_PATH = os.path.join("Projeto_HJV","SAO_PAULO","outputs","datasets")

# Liste os arquivos no diretório
arquivos = os.listdir(caminho_diretorio)

# Filtrar apenas arquivos (não pastas)
somente_arquivos = [f for f in arquivos if os.path.isfile(os.path.join(caminho_diretorio, f))]

chuva_data = pd.DataFrame()
max_total_month_data = pd.DataFrame()

for arquivo in tqdm(somente_arquivos, desc="Processando Arquivos", unit="arq"):
    caminho_arquivo = os.path.join(caminho_diretorio, arquivo)

    chuvas =  pd.read_csv(caminho_arquivo, skiprows = 10, sep=';').iloc[:,:-1]
    prefixo  = pd.read_csv(caminho_arquivo, nrows =  1, sep=':')
    prefixo = prefixo.iloc[0,1]

    chuvas['PREFIXO'] = prefixo

    # Uni todas as informações
    chuva_data = pd.concat([chuva_data,chuvas])

for column in chuva_data.columns[1:-1]:
    chuva_data[column] = chuva_data[column].astype(str).replace(",", ".", regex=True).astype(float)
chuva_data.to_csv(SAVE_PATH + "/rain_measurements_series.csv")