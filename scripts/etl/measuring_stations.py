import pandas as pd
import os
from tqdm import tqdm

# Defina o caminho para o diretório
caminho_diretorio = "Projeto_HJV\SAO_PAULO\dados_inmet"

SAVE_PATH = os.path.join("Projeto_HJV","SAO_PAULO","outputs","datasets")

# Liste os arquivos no diretório
arquivos = os.listdir(caminho_diretorio)

# Filtrar apenas arquivos (não pastas)
somente_arquivos = [f for f in arquivos if os.path.isfile(os.path.join(caminho_diretorio, f))]

colunas_concat = []

# Adicionando o índice

colunas = ['NOME DO POSTO', 'PREFIXO', 'LATITUDE','LONGITUDE', 'ALTITUDE','SITUACAO', 'DATA_INICIAL', 'DATA_FINAL', 'PERIODICIDADE']

# Iterar sobre os arquivos e extrair a segunda coluna
for arquivo in tqdm(somente_arquivos, desc="Processando Arquivos", unit="arq"):

    caminho_arquivo = os.path.join(caminho_diretorio, arquivo)

    df = pd.read_csv(caminho_arquivo, nrows=8, sep=':').transpose().reset_index()

    df = df[1:]  
    df.columns = colunas  
    
    # Adicionar a coluna à lista
    colunas_concat.append(df)

# Concatenar todas as colunas em um único DataFrame
data_pluv_posto = pd.concat(colunas_concat, axis=0)

# Transformando os dados transpondo o dataframe
data_pluv_posto = data_pluv_posto
data_pluv_posto.reset_index(inplace = True, drop = True)

print("\n")
print(data_pluv_posto)

# Saving...
data_pluv_posto.to_csv(SAVE_PATH + "/measuring_stations_data.csv")