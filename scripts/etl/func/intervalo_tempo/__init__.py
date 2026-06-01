from datetime import datetime, timedelta

__version__ = "0.0.1"

class intervalo_tempo:

    def __init__(self, inicio:str, fim:str):
        self.inicio = inicio
        self.fim = fim
        pass

    def gerar_lista_dias(self):
        # Converter as strings para objetos datetime
        data_inicio = datetime.strptime(self.inicio, '%d-%m-%Y')
        data_fim = datetime.strptime(self.fim,'%d-%m-%Y')
        
        # Lista para armazenar as datas
        lista_dias = []
        
        # Usar um loop para iterar sobre os dias no intervalo
        data_atual = data_inicio
        while data_atual <= data_fim:
            lista_dias.append(data_atual.strftime('%d-%m-%Y'))
            data_atual += timedelta(days=1)
        
        return lista_dias