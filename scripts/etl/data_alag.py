import os
from func.prompt import PromptYN, DataframeProperties
from func.intervalo_tempo import intervalo_tempo
from func.functions import download_alag

# Vamos verificar se o arquivo existe
ARQUIVO = "final_data.csv"

PATH = os.path.join("data","outputs","datasets")
os.makedirs(PATH, exist_ok = True)

PATH_ARQUIVO = PATH + f"\{ARQUIVO}"

date_inteval = ["01-01-2015","31-05-2026"] # FORMATO ["DATA_INICIAL","DATA_FINAL"]

arq_exist = os.path.exists(PATH_ARQUIVO)

if(arq_exist):

    print(F"\nJá existe um arquivo com o nome '{ARQUIVO}.'")

    # Nos apresenta a data máxima e mínima que possui no arquivo.
    df = DataframeProperties(PATH_ARQUIVO)
    max_date, min_date = df.max_min_date()
    tes = PromptYN(f"Deseja continar utilizando o arquivo selecionado? (Histórico: {min_date} - {max_date} )")

    # Nos informa se ele deseja ou não continuar
    if(tes.response()):
        print("\nO arquivo será agregado a partir da data final.\n")   

        # Vamos aplicar em valores que não existam na lista do dataframe já salvo
        lista_dias = intervalo_tempo(date_inteval[0],date_inteval[1]).gerar_lista_dias()

        diferents = [item for item in lista_dias if item not in df.date_list()]

        download_alag(diferents, exist = True, path = PATH_ARQUIVO)    
    
    else:
        print("\nO arquivo existente não será utilizado. O novo arquivo será criado.")
        task = PromptYN(f"Para continuar o arquivo anterior será excluido, deseja continuar?")
        if task.response():
            arq_exist = False
        else:
            print("Finalizando o programa...")

if not arq_exist:
    print("\n O arquivo será criado o arquivo.\n")


    # Vamos aplicar em valores do intervalo de tempo
    lista_dias = intervalo_tempo(date_inteval[0],date_inteval[1]).gerar_lista_dias()

    download_alag(lista_dias, exist = False, path = PATH_ARQUIVO)    
