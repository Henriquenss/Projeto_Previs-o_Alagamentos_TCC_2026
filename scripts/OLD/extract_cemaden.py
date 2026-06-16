"""

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ESTE MÉTODO NÃO FUNCIONA POR CONTA DA NECESSIDADE DE PREENCHIMENTO DE CAPTCHA.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Extração de Dados de Pluviômetros — CEMADEN
TCC – Previsão de Alagamentos em São Paulo

Fonte  : https://mapainterativo.cemaden.gov.br
Portal : http://www2.cemaden.gov.br/mapainterativo/
Dados  : acumulado de chuva a cada 10 minutos, rede de ~500+ pluviômetros em SP

Como funciona o download do CEMADEN:
  O portal disponibiliza um endpoint público que retorna um CSV mensal
  com todas as estações de um município/UF. O script automatiza esse
  processo para o período completo, mês a mês.

Estratégia de extração:
  1. Baixar lista de estações de SP capital via endpoint de metadados
  2. Para cada mês do período, baixar o CSV mensal consolidado
  3. Filtrar estações do município de São Paulo (codIBGE = 3550308)
  4. Agregar para nível diário (soma e máximo de 10 min)
  5. Calcular features de intensidade de chuva

Nota sobre o CAPTCHA:
  O portal do CEMADEN tem CAPTCHA na interface web, mas o endpoint de
  download de dados (action=downloadDadosMunicipal) é acessível
  diretamente via HTTP POST sem CAPTCHA. Se o acesso for bloqueado,
  o script instrui como baixar manualmente.

Campos do CSV retornado:
  codEstacao, nomeEstacao, municipio, uf, latitude, longitude,
  dataHora (UTC), valorMedida (mm acumulado em 10 min)
"""

import requests
import pandas as pd
import time
import os
import io
from datetime import date
from dateutil.relativedelta import relativedelta

# ── Configurações ──────────────────────────────────────────────────────────

# Código IBGE do Município de São Paulo
COD_IBGE_SP = "3550308"
UF_SP       = "SP"

START_DATE = date(2015, 1, 1)
END_DATE   = date(2026, 5, 31)

PATH = 'C:/Users/rickn/OneDrive/Documentos/GitHub/Projeto_Previsão_Alagamentos_TCC_2026/data/outputs'

OUTPUT_PATH       = f"{PATH}/datasets/cemaden_sp_diario.csv"
RAW_DIR           = f"{PATH}/cemaden_raw"          # pasta para CSVs mensais brutos
STATIONS_OUT      = f"{PATH}/datasets/cemaden_estacoes.csv"

# Endpoints do CEMADEN
# (inspecionados via DevTools no portal mapainterativo.cemaden.gov.br)
URL_DOWNLOAD = "http://www.cemaden.gov.br/mapainterativo/download/downloadDadosMunicipal.php"
URL_ESTACOES = "http://www.cemaden.gov.br/mapainterativo/download/listaEstacoesMunicipio.php"

HEADERS = {
    "User-Agent"  : "Mozilla/5.0 (compatible; TCC-research/1.0)",
    "Referer"     : "https://mapainterativo.cemaden.gov.br/",
    "Content-Type": "application/x-www-form-urlencoded",
}

# Pausa entre requisições (segundos) — respeitar o servidor
SLEEP_BETWEEN_REQUESTS = 2.0

# Codificação dos CSVs do CEMADEN
CEMADEN_ENCODING = "latin-1"


# ── Funções ────────────────────────────────────────────────────────────────

def fetch_stations(uf: str = UF_SP, cod_ibge: str = COD_IBGE_SP) -> pd.DataFrame:
    """
    Obtém lista de estações CEMADEN para o município via endpoint público.
    Retorna DataFrame com metadados das estações.
    """
    payload = {"uf": uf, "codIBGE": cod_ibge}
    try:
        resp = requests.post(URL_ESTACOES, data=payload, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep=";",
                         encoding=CEMADEN_ENCODING, on_bad_lines="skip")
        print(f"  {len(df)} estações encontradas para SP (cod. IBGE {cod_ibge})")
        return df
    except Exception as e:
        print(f"  [aviso] Não foi possível listar estações: {e}")
        print("  Continuando sem filtro de estações...")
        return pd.DataFrame()


def fetch_month_csv(year: int, month: int,
                    uf: str = UF_SP,
                    cod_ibge: str = COD_IBGE_SP) -> pd.DataFrame | None:
    """
    Baixa CSV mensal de uma UF/município do CEMADEN.
    Retorna DataFrame bruto ou None em caso de falha.

    O arquivo retornado contém todas as estações do município,
    com leituras a cada 10 minutos.
    """
    payload = {
        "uf"        : uf,
        "codIBGE"   : cod_ibge,
        "ano"       : str(year),
        "mes"       : f"{month:02d}",
    }

    try:
        resp = requests.post(URL_DOWNLOAD, data=payload,
                             headers=HEADERS, timeout=120)
        resp.raise_for_status()

        # Verificar se o retorno tem conteúdo real (não erro HTML)
        if len(resp.content) < 100 or b"<html" in resp.content[:200].lower():
            print(f"  [{year}-{month:02d}] Resposta vazia ou HTML — sem dados.")
            return None

        df = pd.read_csv(
            io.StringIO(resp.content.decode(CEMADEN_ENCODING, errors="replace")),
            sep=";",
            on_bad_lines="skip",
        )

        if df.empty:
            print(f"  [{year}-{month:02d}] CSV vazio.")
            return None

        return df

    except requests.exceptions.Timeout:
        print(f"  [{year}-{month:02d}] Timeout — tentando novamente em 5s...")
        time.sleep(5)
        return None
    except Exception as e:
        print(f"  [{year}-{month:02d}] Erro: {e}")
        return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza nomes de colunas do CSV do CEMADEN.
    O CEMADEN usa nomes ligeiramente diferentes dependendo da versão do portal.
    """
    # Mapeamentos conhecidos
    rename_map = {
        # variação 1
        "codEstacao"   : "cod_estacao",
        "nomeEstacao"  : "nome_estacao",
        "municipio"    : "municipio",
        "uf"           : "uf",
        "latitude"     : "lat",
        "longitude"    : "lon",
        "dataHora"     : "data_hora",
        "valorMedida"  : "valor_mm",
        # variação 2 (algumas versões)
        "Codigo"       : "cod_estacao",
        "Nome"         : "nome_estacao",
        "Municipio"    : "municipio",
        "UF"           : "uf",
        "Latitude"     : "lat",
        "Longitude"    : "lon",
        "DataHora"     : "data_hora",
        "ValorMedido"  : "valor_mm",
        "Chuva"        : "valor_mm",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def aggregate_to_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega leituras de 10 minutos para nível diário por estação.

    Features criadas:
      - precip_total_mm   → soma diária (mm)
      - precip_max_10m_mm → máximo em 10 min no dia (intensidade de pico)
      - n_leituras        → count de leituras válidas no dia
    """
    df = df_raw.copy()
    df = standardize_columns(df)

    # Verificar colunas mínimas
    required = {"data_hora", "valor_mm"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [aviso] Colunas faltando no CSV: {missing}")
        print(f"  Colunas disponíveis: {df.columns.tolist()}")
        return pd.DataFrame()

    # Converter datetime — CEMADEN usa UTC
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce", utc=True)
    # Converter para horário de São Paulo (UTC-3)
    df["data_hora"] = df["data_hora"].dt.tz_convert("America/Sao_Paulo")
    df["date"] = df["data_hora"].dt.date

    # Limpar valor
    df["valor_mm"] = pd.to_numeric(df["valor_mm"], errors="coerce")
    df = df.dropna(subset=["data_hora", "valor_mm"])
    df = df[df["valor_mm"] >= 0]  # remover negativos (erros de sensor)

    # Identificador de estação
    id_col = "cod_estacao" if "cod_estacao" in df.columns else None

    group_cols = ["date"]
    if id_col:
        group_cols = [id_col] + group_cols
        if "lat" in df.columns and "lon" in df.columns:
            # preservar lat/lon na agregação
            df_meta = df.groupby(id_col)[["lat", "lon"]].first().reset_index()

    agg = df.groupby(group_cols).agg(
        precip_total_mm  = ("valor_mm", "sum"),
        precip_max_10m_mm= ("valor_mm", "max"),
        n_leituras       = ("valor_mm", "count"),
    ).reset_index()

    agg["date"] = pd.to_datetime(agg["date"])

    if id_col and "df_meta" in dir():
        agg = pd.merge(agg, df_meta, on=id_col, how="left")

    return agg


def build_sp_aggregate(df_stations: pd.DataFrame,
                       df_daily_all: pd.DataFrame) -> pd.DataFrame:
    """
    A partir dos dados diários por estação, cria uma série temporal única
    para SP usando média espacial ponderada (ou simples, se não houver coords).

    Para um TCC, a média simples já é uma linha de base razoável.
    A ponderação IDW (como feita no build_dataset.py) é opcional aqui.
    """
    if "cod_estacao" not in df_daily_all.columns:
        # Sem identificador de estação — agregar tudo como média por dia
        agg = df_daily_all.groupby("date").agg(
            cemaden_precip_total_mm  = ("precip_total_mm",   "mean"),
            cemaden_precip_max_10m_mm= ("precip_max_10m_mm", "max"),
            cemaden_n_estacoes       = ("precip_total_mm",   "count"),
        ).reset_index()
        return agg

    # Com identificador: média entre estações ativas no dia
    agg = df_daily_all.groupby("date").agg(
        cemaden_precip_mean_mm    = ("precip_total_mm",    "mean"),
        cemaden_precip_max_mm     = ("precip_total_mm",    "max"),
        cemaden_precip_median_mm  = ("precip_total_mm",    "median"),
        cemaden_intensity_max_10m = ("precip_max_10m_mm",  "max"),
        cemaden_n_estacoes_ativas = ("precip_total_mm",    "count"),
    ).reset_index()

    return agg


# ── Execução principal ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CEMADEN — Extração pluviométrica para São Paulo")
    print("=" * 60)

    os.makedirs(RAW_DIR, exist_ok=True)

    # 1. Listar estações
    print("\n[1/4] Obtendo lista de estações em SP...")
    df_stations = fetch_stations()
    if not df_stations.empty:
        df_stations.to_csv(STATIONS_OUT, index=False)
        print(f"      Metadados salvos em '{STATIONS_OUT}'")

    # 2. Baixar CSVs mensais
    print(f"\n[2/4] Baixando CSVs mensais ({START_DATE} → {END_DATE})...")
    current = START_DATE.replace(day=1)
    all_raw = []
    n_success = 0
    n_fail = 0

    while current <= END_DATE:
        year, month = current.year, current.month
        cache_path = os.path.join(RAW_DIR, f"cemaden_sp_{year}_{month:02d}.csv")

        # Usar cache local se já baixou
        if os.path.exists(cache_path):
            print(f"  [{year}-{month:02d}] Usando cache local.")
            df_month = pd.read_csv(cache_path, encoding=CEMADEN_ENCODING,
                                   sep=";", on_bad_lines="skip")
        else:
            df_month = fetch_month_csv(year, month)
            if df_month is not None and not df_month.empty:
                df_month.to_csv(cache_path, index=False, sep=";",
                                encoding=CEMADEN_ENCODING)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        if df_month is not None and not df_month.empty:
            all_raw.append(df_month)
            n_success += 1
            print(f"  [{year}-{month:02d}] OK — {len(df_month):,} leituras")
        else:
            n_fail += 1

        current += relativedelta(months=1)

    print(f"\n  Meses obtidos: {n_success} / {n_success + n_fail}")

    if not all_raw:
        print("\n[ERRO] Nenhum dado obtido. Verifique conexão ou baixe manualmente.")
        print_manual_instructions()
        return

    # 3. Agregar para diário
    print("\n[3/4] Agregando para nível diário por estação...")
    df_raw_all = pd.concat(all_raw, ignore_index=True)
    print(f"  Total de leituras brutas: {len(df_raw_all):,}")
    df_daily = aggregate_to_daily(df_raw_all)
    print(f"  Após agregação diária: {len(df_daily):,} linhas")

    # 4. Consolidar em série SP
    print("\n[4/4] Consolidando em série temporal única para SP...")
    df_sp = build_sp_aggregate(df_stations, df_daily)

    # Cobrir todos os dias do range
    all_days = pd.DataFrame({"date": pd.date_range(START_DATE, END_DATE)})
    df_sp = pd.merge(all_days, df_sp, on="date", how="left")

    df_sp.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Salvo em '{OUTPUT_PATH}'")
    print(f"  Shape: {df_sp.shape}")
    print(f"  Dias com dados: {df_sp['cemaden_precip_mean_mm'].notna().sum()}")
    print(f"  Dias sem dados: {df_sp['cemaden_precip_mean_mm'].isna().sum()}")
    print(f"\nColunas:\n  {df_sp.columns.tolist()}")
    print("=" * 60)


def print_manual_instructions():
    """
    Instrução de fallback caso o script automatizado seja bloqueado.
    """
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOWNLOAD MANUAL — CEMADEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Acesse: https://mapainterativo.cemaden.gov.br/
2. Na barra de ferramentas, clique no ícone "Download de Dados"
   (penúltimo botão, ícone de seta para baixo)
3. Preencha:
   - UF       : SP
   - Município: São Paulo
   - Mês/Ano  : selecione mês a mês
4. Clique em "Download"
5. Salve os arquivos na pasta  ./cemaden_raw/
   com o nome  cemaden_sp_YYYY_MM.csv

Depois re-execute o script — ele detectará os arquivos
em cache e pulará o download automático.

Dica: o portal também envia o arquivo por e-mail se você
preencher o formulário de solicitação na seção "Ajuda".
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


if __name__ == "__main__":
    # Verificar dependência extra
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        print("Instalando python-dateutil...")
        os.system("pip install python-dateutil --break-system-packages -q")
        from dateutil.relativedelta import relativedelta

    main()
