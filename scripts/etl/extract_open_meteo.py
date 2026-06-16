"""
Extração de Dados Históricos — Open-Meteo (ERA5 / ERA5-Land)
TCC – Previsão de Alagamentos em São Paulo

Fonte  : https://open-meteo.com/en/docs/historical-weather-api
API    : archive-api.open-meteo.com/v1/archive
Auth   : sem autenticação (gratuito, CC BY 4.0)
Limite : ~10.000 req/dia no tier gratuito; este script faz ~1 req/ano

Variáveis extraídas (diárias, ERA5-Land + ERA5-Seamless):
  - precipitation_sum          → total de chuva diário (mm)
  - et0_fao_evapotranspiration → evapotranspiração ref. (mm) — proxy de demanda hídrica
  - soil_moisture_0_to_7cm     → umidade superficial do solo (m³/m³)
  - soil_moisture_7_to_28cm    → umidade camada intermediária
  - temperature_2m_mean/max/min
  - relative_humidity_2m_mean
  - wind_speed_10m_mean
  - cloud_cover_mean           → cobertura de nuvens (%)
  - shortwave_radiation_sum    → radiação solar (MJ/m²)

Por que ERA5-Land para solo e ERA5 para o resto?
  ERA5-Land tem resolução ~9 km e foco em superfície (solo, temperatura, umidade).
  ERA5 cobre precipitação e vento com mais consistência histórica.
  O parâmetro models="best_match" combina ambos automaticamente.
"""

import requests
import pandas as pd
import time
import os
from datetime import date

# ── Configurações ──────────────────────────────────────────────────────────

# Coordenadas do centro de São Paulo
LAT = -23.5505
LON = -46.6333

START_DATE = "2015-01-01"
END_DATE   = "2026-05-31"

OUTPUT_PATH = "C:/Users/rickn/OneDrive/Documentos/GitHub/Projeto_Previsão_Alagamentos_TCC_2026/data/outputs/datasets/open_meteo_sp.csv"

# Variáveis diárias desejadas
# Referência completa: https://open-meteo.com/en/docs/historical-weather-api
DAILY_VARS = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "wind_gusts_10m_max",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "vapour_pressure_deficit_max",
]

# Variáveis horárias (para calcular acumulados personalizados)
# Deixe vazio se não precisar de dados horários
HOURLY_VARS = [
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
]

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


# ── Funções ────────────────────────────────────────────────────────────────

def fetch_daily(lat: float, lon: float,
                start: str, end: str,
                variables: list,
                model: str = "best_match",
                timezone: str = "America/Sao_Paulo") -> pd.DataFrame:
    """
    Busca variáveis diárias da Historical Weather API do Open-Meteo.
    Retorna DataFrame com coluna 'date' + uma coluna por variável.
    """
    params = {
        "latitude"   : lat,
        "longitude"  : lon,
        "start_date" : start,
        "end_date"   : end,
        "daily"      : ",".join(variables),
        "models"     : model,
        "timezone"   : timezone,
    }

    print(f"  GET {BASE_URL}")
    print(f"  Período : {start} → {end}")
    print(f"  Modelo  : {model}")
    print(f"  Vars    : {len(variables)} variáveis diárias")

    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"Open-Meteo retornou erro: {data['reason']}")

    df = pd.DataFrame({"date": data["daily"]["time"]})
    df["date"] = pd.to_datetime(df["date"])

    for var in variables:
        if var in data["daily"]:
            df[var] = data["daily"][var]
        else:
            print(f"  [aviso] variável '{var}' não retornada pelo modelo '{model}'")
            df[var] = float("nan")

    return df


def fetch_hourly_soil(lat: float, lon: float,
                      start: str, end: str,
                      variables: list,
                      model: str = "era5_land",
                      timezone: str = "America/Sao_Paulo") -> pd.DataFrame:
    """
    Busca variáveis horárias (ex: umidade do solo) e agrega para nível diário.
    ERA5-Land é o modelo recomendado para solo — resolução ~9 km.
    """
    # A API limita a ~1 ano por requisição para dados horários volumosos.
    # Aqui fazemos uma requisição por ano e concatenamos.
    start_year = pd.to_datetime(start).year
    end_year   = pd.to_datetime(end).year
    all_dfs    = []

    for year in range(start_year, end_year + 1):
        y_start = max(start, f"{year}-01-01")
        y_end   = min(end,   f"{year}-12-31")

        params = {
            "latitude"   : lat,
            "longitude"  : lon,
            "start_date" : y_start,
            "end_date"   : y_end,
            "hourly"     : ",".join(variables),
            "models"     : model,
            "timezone"   : timezone,
        }

        print(f"  [{year}] Buscando dados horários de solo...")
        resp = requests.get(BASE_URL, params=params, timeout=90)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            print(f"  [erro] {year}: {data.get('reason', '?')}")
            continue

        df_h = pd.DataFrame({"datetime": pd.to_datetime(data["hourly"]["time"])})
        for var in variables:
            if var in data["hourly"]:
                df_h[var] = data["hourly"][var]

        # Agregar para diário: média da umidade do solo
        df_h["date"] = df_h["datetime"].dt.date
        agg = df_h.groupby("date")[variables].mean().reset_index()
        agg["date"] = pd.to_datetime(agg["date"])
        agg.columns = ["date"] + [f"{v}_mean" for v in variables]
        all_dfs.append(agg)

        time.sleep(0.5)  # pausa entre requisições

    if not all_dfs:
        return pd.DataFrame(columns=["date"])

    return pd.concat(all_dfs, ignore_index=True)


def add_lag_features(df: pd.DataFrame,
                     col: str = "precipitation_sum",
                     lags: list = [1, 2, 3],
                     windows: list = [3, 7, 14]) -> pd.DataFrame:
    """
    Adiciona lags e acumulados de precipitação.
    Mantém consistência com o pipeline principal (build_dataset.py).
    """
    df = df.sort_values("date").reset_index(drop=True)
    for lag in lags:
        df[f"{col}_lag{lag}d"] = df[col].shift(lag)
    for w in windows:
        df[f"{col}_acc{w}d"] = df[col].rolling(w).sum()
    return df


# ── Execução principal ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Open-Meteo — Extração histórica para São Paulo")
    print("=" * 60)

    # 1. Dados diários (rápido — 1 requisição única)
    print("\n[1/3] Buscando variáveis diárias (ERA5 best_match)...")
    df_daily = fetch_daily(
        lat=LAT, lon=LON,
        start=START_DATE, end=END_DATE,
        variables=DAILY_VARS,
        model="best_match",
    )
    print(f"      {len(df_daily)} dias obtidos. Shape: {df_daily.shape}")

    # 2. Umidade do solo (horário → diário, ERA5-Land)
    if HOURLY_VARS:
        print("\n[2/3] Buscando umidade do solo horária (ERA5-Land) por ano...")
        df_soil = fetch_hourly_soil(
            lat=LAT, lon=LON,
            start=START_DATE, end=END_DATE,
            variables=HOURLY_VARS,
            model="era5_land",
        )
        print(f"      {len(df_soil)} dias obtidos. Shape: {df_soil.shape}")

        # Join com dados diários
        df = pd.merge(df_daily, df_soil, on="date", how="left")
    else:
        df = df_daily

    # 3. Features de lag e acumulado
    print("\n[3/3] Calculando lags e acumulados de precipitação...")
    df = add_lag_features(df, col="precipitation_sum")
    print(f"      Novas colunas: {[c for c in df.columns if 'lag' in c or 'acc' in c]}")

    # Salvar
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Salvo em '{OUTPUT_PATH}'")
    print(f"  Shape final: {df.shape}")
    print(f"  Colunas: {df.columns.tolist()}")

    # Preview de nulos
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        print(f"\nColunas com nulos:\n{nulos}")
    else:
        print("\nNenhum nulo encontrado.")

    print("=" * 60)


if __name__ == "__main__":
    main()
