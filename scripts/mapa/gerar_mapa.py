# -*- coding: utf-8 -*-
"""
Geração de Mapas — Estações CEMADEN e Locais de Alagamento em SP
TCC – Previsão de Alagamentos em São Paulo

Roda 100% local, sem geopandas (evita dor de cabeça de instalação no
Windows com GDAL). Usa só matplotlib + json + csv, que você já tem.

ESTRUTURA DE PASTAS ESPERADA (conforme o projeto):

  PROJETO_PREVISAO_ALAGAMENTOS_TCC_2026/
    data/
      outputs/
        datasets/
          cemaden_estacoes.csv     ← gerado pelo consolidate_cemaden.py
          final_data.csv           ← registros de alagamento do CGE
    scripts/
      mapa/
        distritos-sp.geojson       ← polígonos dos 96 distritos de SP
        gerar_mapas.py             ← este script (rode a partir daqui)

SAÍDAS (salvas em scripts/mapa/, junto deste script):
  mapa_estacoes_sp.png         → pluviômetros CEMADEN sobre o mapa de SP
  mapa_alagamentos_sp.png      → top locais de alagamento por distrito,
                                  com tamanho do ponto proporcional à
                                  frequência de eventos

Se algum arquivo não existir, o script avisa e segue gerando o que for
possível com os dados disponíveis.
"""

import json
import os
import csv
import unicodedata
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import numpy as np


# ── Configurações — ajuste os caminhos se necessário ───────────────────────
#
# Este script foi pensado para rodar de dentro de:
#   scripts/mapa/gerar_mapas.py
#
# Estrutura do projeto (conforme pasta do TCC):
#   PROJETO_PREVISAO_ALAGAMENTOS_TCC_2026/
#     data/outputs/datasets/cemaden_estacoes.csv
#     data/outputs/datasets/final_data.csv
#     scripts/mapa/distritos-sp.geojson
#     scripts/mapa/gerar_mapas.py   ← este script
#     scripts/mapa/mapa_estacoes_sp.png      (saída)
#     scripts/mapa/mapa_alagamentos_sp.png   (saída)

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))  # scripts/mapa/
RAIZ_PROJETO = os.path.abspath(os.path.join(PASTA_SCRIPT, "..", ".."))  # raiz do projeto
PASTA_DATASETS = os.path.join(RAIZ_PROJETO, "data", "outputs", "datasets")

PATH_GEOJSON      = os.path.join(PASTA_SCRIPT, "distritos-sp.geojson")
PATH_CEMADEN      = os.path.join(PASTA_DATASETS, "cemaden_estacoes.csv")
PATH_ALAGAMENTOS  = os.path.join(PASTA_DATASETS, "final_data.csv")

OUT_ESTACOES    = os.path.join(PASTA_SCRIPT, "mapa_estacoes_sp.png")
OUT_ALAGAMENTOS = os.path.join(PASTA_SCRIPT, "mapa_alagamentos_sp.png")

DPI = 200  # resolução boa para inserir no Word sem pixelizar

# Paleta consistente com os gráficos já usados no TCC
COR_DISTRITO_FILL   = "#F1EFE8"
COR_DISTRITO_EDGE   = "#B4B2A9"
COR_CEMADEN         = "#378ADD"
COR_ALAGAMENTO      = "#E05252"


# ── Funções utilitárias ──────────────────────────────────────────────────

def normalizar(texto: str) -> str:
    """Remove acentos, deixa maiúsculo e tira espaços extras — para
    comparar nomes de bairro/distrito de fontes diferentes."""
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().upper()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    # Normalizações específicas comuns entre CGE e os nomes oficiais do geojson
    texto = texto.replace("FREGUESIA DO O", "FREGUESIA DO O")
    texto = texto.replace("SE", "SE") if texto == "SE" else texto
    return texto


def carregar_geojson(path: str):
    if not os.path.exists(path):
        print(f"[ERRO] Arquivo não encontrado: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extrair_poligonos(geojson: dict):
    """
    Retorna lista de (patches_matplotlib, nome_distrito, centroide_aprox)
    para cada feature do geojson.
    """
    resultado = []
    for feature in geojson["features"]:
        nome = feature["properties"].get("ds_nome", "")
        geom = feature["geometry"]

        if geom["type"] == "Polygon":
            aneis = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            aneis = geom["coordinates"]
        else:
            continue

        patches_distrito = []
        todos_pontos = []
        for polygon in aneis:
            exterior = polygon[0]
            pts = np.array(exterior)
            patches_distrito.append(MplPolygon(pts, closed=True))
            todos_pontos.append(pts)

        # Centroide aproximado: média dos pontos do maior anel (suficiente
        # para posicionar um marcador, não precisa ser o centroide exato)
        maior_anel = max(todos_pontos, key=len)
        centroide = maior_anel.mean(axis=0)  # [lon, lat]

        resultado.append((patches_distrito, nome, centroide))

    return resultado


def desenhar_base_sp(ax, poligonos):
    """Desenha o contorno de todos os distritos como camada de fundo."""
    all_patches = []
    for patches_distrito, _, _ in poligonos:
        all_patches.extend(patches_distrito)

    pc = PatchCollection(
        all_patches,
        facecolor=COR_DISTRITO_FILL,
        edgecolor=COR_DISTRITO_EDGE,
        linewidths=0.5,
    )
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")


def ler_csv_generico(path: str):
    """Lê CSV com csv.DictReader (evita depender de pandas, embora pandas
    provavelmente já esteja instalado no seu ambiente)."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(val):
    if val is None or val == "":
        return None
    try:
        return float(str(val).replace(",", "."))
    except ValueError:
        return None


# ── Mapa 1: Estações meteorológicas ────────────────────────────────────────

def gerar_mapa_estacoes(poligonos):
    print("\n[Mapa 1] Estações meteorológicas (CEMADEN)")

    fig, ax = plt.subplots(figsize=(9, 11))
    desenhar_base_sp(ax, poligonos)

    legend_handles = []

    cemaden_rows = ler_csv_generico(PATH_CEMADEN)
    if cemaden_rows:
        lats, lons = [], []
        for row in cemaden_rows:
            lat = to_float(row.get("lat"))
            lon = to_float(row.get("lon"))
            if lat is not None and lon is not None:
                lats.append(lat)
                lons.append(lon)
        if lons:
            ax.scatter(lons, lats, c=COR_CEMADEN, s=28, alpha=0.85,
                      edgecolors="white", linewidths=0.6, zorder=5)
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=COR_CEMADEN, markersize=9,
                      label=f"Pluviômetros CEMADEN (n={len(lons)})")
            )
            print(f"  ✓ {len(lons)} estações CEMADEN plotadas.")
        else:
            print("  [aviso] Nenhuma coordenada válida em cemaden_estacoes.csv")
    else:
        print(f"  [ERRO] Arquivo não encontrado: {PATH_CEMADEN}")
        print("         Não é possível gerar este mapa sem cemaden_estacoes.csv.")
        plt.close(fig)
        return

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
                 frameon=True, facecolor="white", edgecolor=COR_DISTRITO_EDGE)

    ax.set_title("Rede de Pluviômetros CEMADEN — São Paulo",
                 fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(OUT_ESTACOES, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Salvo em '{OUT_ESTACOES}'")


# ── Mapa 2: Locais de alagamento ───────────────────────────────────────────

def gerar_mapa_alagamentos(poligonos):
    print("\n[Mapa 2] Principais locais de alagamento (por distrito)")

    rows = ler_csv_generico(PATH_ALAGAMENTOS)
    if not rows:
        print(f"  [ERRO] Arquivo não encontrado: {PATH_ALAGAMENTOS}")
        print("         Não é possível gerar este mapa sem o final_data.csv.")
        return

    # Contar eventos por bairro (coluna 'neighbor' no final_data.csv)
    contagem = Counter()
    for row in rows:
        bairro = row.get("neighbor", "")
        if bairro and bairro.strip():
            contagem[normalizar(bairro)] += 1

    if not contagem:
        print("  [ERRO] Nenhum bairro encontrado na coluna 'neighbor'.")
        return

    print(f"  {len(contagem)} bairros únicos encontrados no CSV de alagamentos.")

    # Cruzar com os centroides dos distritos do geojson
    centroides_por_nome = {normalizar(nome): centroide for _, nome, centroide in poligonos}

    pontos_lon, pontos_lat, pontos_n, pontos_nome = [], [], [], []
    nao_encontrados = []

    for bairro_norm, n_eventos in contagem.items():
        if bairro_norm in centroides_por_nome:
            lon, lat = centroides_por_nome[bairro_norm]
            pontos_lon.append(lon)
            pontos_lat.append(lat)
            pontos_n.append(n_eventos)
            pontos_nome.append(bairro_norm)
        else:
            nao_encontrados.append((bairro_norm, n_eventos))

    print(f"  ✓ {len(pontos_lon)} bairros casados com distritos do geojson.")
    if nao_encontrados:
        print(f"  [aviso] {len(nao_encontrados)} bairros não encontrados no geojson "
              f"(provavelmente nomes de zona, não de distrito):")
        for nome, n in sorted(nao_encontrados, key=lambda x: -x[1])[:10]:
            print(f"      {nome}: {n} eventos")

    if not pontos_lon:
        print("  [ERRO] Nenhum bairro pôde ser georreferenciado. Abortando mapa.")
        return

    # ── Plotagem ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 11))
    desenhar_base_sp(ax, poligonos)

    pontos_n = np.array(pontos_n)
    # Escala de tamanho: raiz quadrada para não deixar os maiores
    # excessivamente dominantes visualmente
    tamanhos = 30 + 800 * np.sqrt(pontos_n / pontos_n.max())

    sc = ax.scatter(pontos_lon, pontos_lat, s=tamanhos, c=COR_ALAGAMENTO,
                    alpha=0.55, edgecolors=COR_ALAGAMENTO, linewidths=1.2, zorder=5)

    # Anotar os top 10 bairros com mais eventos
    top_idx = np.argsort(pontos_n)[::-1][:10]
    for i in top_idx:
        nome_exibicao = pontos_nome[i].title()
        ax.annotate(
            f"{nome_exibicao} ({pontos_n[i]})",
            (pontos_lon[i], pontos_lat[i]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, fontweight="bold", color="#791F1F",
            zorder=6,
        )

    # Legenda de tamanho (círculos de referência)
    ref_vals = [int(pontos_n.max()), int(pontos_n.max()/3), int(pontos_n.max()/10)]
    ref_sizes = 30 + 800 * np.sqrt(np.array(ref_vals) / pontos_n.max())
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COR_ALAGAMENTO,
              alpha=0.55, markeredgecolor=COR_ALAGAMENTO,
              markersize=np.sqrt(s), label=f"{v} eventos")
        for v, s in zip(ref_vals, ref_sizes)
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
             title="Nº de eventos registrados", title_fontsize=9,
             frameon=True, facecolor="white", edgecolor=COR_DISTRITO_EDGE)

    ax.set_title("Principais Locais de Alagamento por Distrito — São Paulo\n"
                 "(tamanho proporcional ao nº de eventos registrados pelo CGE)",
                 fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(OUT_ALAGAMENTOS, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Salvo em '{OUT_ALAGAMENTOS}'")


# ── Execução principal ─────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Geração de Mapas — TCC Previsão de Alagamentos SP")
    print("=" * 65)
    print(f"Pasta do script: {PASTA_SCRIPT}")
    print(f"Pasta de datasets: {PASTA_DATASETS}")

    geojson = carregar_geojson(PATH_GEOJSON)
    if geojson is None:
        print("\n[ERRO FATAL] Não é possível continuar sem o geojson dos distritos.")
        return

    poligonos = extrair_poligonos(geojson)
    print(f"✓ {len(poligonos)} distritos carregados do geojson.")

    gerar_mapa_estacoes(poligonos)
    gerar_mapa_alagamentos(poligonos)

    print("\n" + "=" * 65)
    print("Concluído. Verifique os arquivos .png gerados nesta pasta.")
    print("=" * 65)


if __name__ == "__main__":
    main()