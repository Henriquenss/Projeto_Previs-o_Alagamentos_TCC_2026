from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from time import sleep
from tqdm import tqdm  # Biblioteca para exibir uma barra de progresso nos loops

def download_alag(lista_dias: list, exist: bool = False, path: str = "") -> None:

    if exist:
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["date", "current_zone", "neighbor", "status", "way"])
        df.to_csv(path, index=False)  # Cria o CSV com cabeçalho
    
    grouped_data = []

    for days, date in enumerate(tqdm(lista_dias, desc="Processando dias", unit="dia"), start=1):
        data = date.split("-")
        url = f"https://www.cgesp.org/v3/alagamentos.jsp?dataBusca={data[0]}%2F{data[1]}%2F{data[2]}&enviaBusca=Buscar"
        response = requests.get(url)
        soup = bs(response.text, 'html.parser')
        content_div = soup.find('div', class_='content')

        # Crie um dicionário para armazenar os dados agrupados
        current_zone = None

        h1_table = content_div.find_all(['h1', 'table'])

        if h1_table:  # Verifica se o HTML está completo
            # Iterar sobre todos os elementos
            for element in h1_table:
                if element.name == 'h1':
                    # Novo título de zona, então inicialize uma nova chave no dicionário
                    current_zone = element.get_text(strip=True)

                elif element.name == 'table' and current_zone:
                    save = element.find_all('div', class_='ponto-de-alagamento')

                    neighbor = element.find('td', class_='bairro arial-bairros-alag linha-pontilhada')
                    if neighbor:
                        neighbor_extract = neighbor.get_text(strip=True)

                    for ponto in save:
                        neighbor = ponto.find('td', class_='bairro arial-bairros-alag linha-pontilhado')

                        inativo_transitavel = ponto.find('li', class_='inativo-transitavel')
                        inativo_intransitavel = ponto.find('li', class_='inativo-intransitavel')
                        ativo_transitavel = ponto.find('li', class_='ativo-transitavel')
                        ativo_intransitavel = ponto.find('li', class_='ativo-intransitavel')

                        status_list = [inativo_transitavel, inativo_intransitavel, ativo_transitavel, ativo_intransitavel]
                        status = next((s for s in status_list if s is not None), None)
                        status_title = status['title'] if status else None

                        way = ponto.find('li', class_='arial-descr-alag col-local')
                        way_title = way.decode_contents().split('<br/>')[-1].strip() if way else None

                        grouped_data.append([date, current_zone, neighbor_extract, status_title, way_title])
        else:
            grouped_data.append([date, None, None, None, None])

        if days % 101 == 0:  # Salva a cada 100 dias processados
            lim_data = pd.DataFrame(grouped_data, columns=["date", "current_zone", "neighbor", "status", "way"])
            save_data = pd.concat([df, lim_data])
            save_data.to_csv(path, index=False)  # Salva os dados
            df = save_data
            grouped_data.clear()  # Limpa os dados agrupados após salvar
        
        sleep(4)

    # Salva o restante, se houver
    if grouped_data:
        lim_data = pd.DataFrame(grouped_data, columns=["date", "current_zone", "neighbor", "status", "way"])
        save_data = pd.concat([df, lim_data])
        save_data.to_csv(path, index=False)

    print("Download e salvamento concluídos.")