"""TCC – regionalização do CEMADEN e ablação regional de alagamentos D+1.

Executar na raiz do repositório:
    python scripts/etl/regionalizar_cemaden_e_modelar.py --root .

Usa as observações reais de data/outputs/cemaden_raw/ ou um
`data/outputs/datasets/cemaden_estacao_dia.csv` preexistente. Não cria
precipitação por estação a partir de estatísticas municipais.

Somente dados observados até o fim de t predizem registros no dia t+1.
Dados urbanos de 2017/2026 são apenas análise retrospectiva ex post.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve)

SEMENTE = 42
CENTRO_MIN_KM = 0.25
RAIO_KM = 30.0
K_ESTACOES = 5
MIN_ESTACOES = 2
POTENCIA_IDW = 2.0
CHUVA_COL = 'cemaden_local_idw_mm'
CHUVA_LAGS = [1, 2, 3]
CHUVA_ACUM = [3, 7, 14]
URB = ['pct_area_mapeada_vegetacao','densidade_rede_km_por_km2','area_km2']
MUNICIPAL = ['cemaden_precip_idw_mm','cemaden_precip_lag1d',
  'cemaden_precip_lag2d','cemaden_precip_lag3d','cemaden_precip_acc3d',
  'cemaden_precip_acc7d','cemaden_precip_acc14d','sp_precip_mm',
  'mes_sin','mes_cos']
LOCAL = [CHUVA_COL, *[f'cemaden_local_lag{n}d' for n in CHUVA_LAGS],
  *[f'cemaden_local_acc{n}d' for n in CHUVA_ACUM],
  'sp_precip_mm','mes_sin','mes_cos']


def localizar_arquivos(raiz: Path):
    raiz = Path(raiz).resolve()
    if (raiz/'data/outputs/datasets/dataset_final_integrado.csv').exists():
        return (raiz/'data/outputs/datasets', raiz/'data/outputs/geosampa',
                raiz/'data/outputs/cemaden_raw', raiz/'scripts/etl')
    # Somente para trabalhar com os anexos desta conversa.
    if (raiz/'dataset_final_integrado.csv').exists():
        return raiz, raiz, raiz/'cemaden_raw', raiz
    raise FileNotFoundError('A raiz precisa conter data/outputs/datasets/dataset_final_integrado.csv')


def sha256(caminho):
    h=hashlib.sha256()
    with Path(caminho).open('rb') as f:
        for bloco in iter(lambda: f.read(1024*1024), b''):h.update(bloco)
    return h.hexdigest()


def _carrega_etl(caminho):
    spec=importlib.util.spec_from_file_location('etl_cemaden_original',caminho)
    if spec is None or spec.loader is None:raise ImportError(f'Não consegui carregar {caminho}')
    modulo=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def ler_estacao_dia(raiz, dados, raw, scripts, saida):
    """Lê dados estação-dia verdadeiros ou reconstitui a partir do RAW local.

    Não sobrescreve arquivos meteorológicos do projeto. Preserva CSV com
    estação-dia limpa como intermediário reutilizável (novo arquivo).
    """
    cache = dados/'cemaden_estacao_dia.csv'
    excl_path = (scripts/'cemaden_exclusoes.csv' if
        (scripts/'cemaden_exclusoes.csv').exists() else raiz/'cemaden_exclusoes.csv')
    if not excl_path.exists():
        raise FileNotFoundError('Faltou cemaden_exclusoes.csv: sem auditoria de exclusões não prosseguir.')
    exc=pd.read_csv(excl_path, dtype={'cod_estacao':str}, parse_dates=['date'])
    assert not exc.duplicated(['cod_estacao','date']).any()
    rel={'arquivo_exclusoes':str(excl_path), 'hash_exclusoes':sha256(excl_path)}
    if cache.exists():
        df=pd.read_csv(cache, dtype={'cod_estacao':str},parse_dates=['date'])
        rel['proveniencia']='cemaden_estacao_dia.csv previamente exportado (verificar origem)'
        rel['arquivos_raw_lidos']=0
        rel['hash_estacao_dia']=sha256(cache)
    else:
        raw_files=sorted(raw.rglob('*.csv')) if raw.exists() else []
        if not raw_files:
            raise FileNotFoundError(
              'Faltam as observações individuais CEMADEN. No seu computador elas aparecem em '
              'data/outputs/cemaden_raw/. Rode este script na raiz do projeto (onde a pasta existe) '
              'ou exporte data/outputs/datasets/cemaden_estacao_dia.csv com colunas '
              'cod_estacao,date,precip_total_mm,n_leituras. O arquivo cemaden_sp_diario.csv '
              'NÃO permite recuperar as chuvas individuais.')
        src=scripts/'transforming_cemaden_data.py'
        if not src.exists():
            src=raiz/'transforming_cemaden_data.py'
        if not src.exists():
            raise FileNotFoundError('Copie o script original transforming_cemaden_data.py para scripts/etl/')
        etl=_carrega_etl(src)
        partes=[];falhas=[]
        for i,arq in enumerate(raw_files):
            tab=etl.read_cemaden_file(str(arq))
            if tab is None:
                falhas.append({'arquivo':str(arq),'motivo':'sem colunas CEMADEN reconhecidas'})
                continue
            if 'cod_estacao' not in tab or tab.cod_estacao.isna().any():
                raise ValueError(f'Arquivo sem código da estação: {arq}')
            tab['cod_estacao']=tab.cod_estacao.astype(str).str.strip()
            diario=etl.aggregate_daily(tab)
            diario['_ordem_arquivo']=i
            diario['_origem']=str(arq)
            partes.append(diario)
        pd.DataFrame(falhas,columns=['arquivo','motivo']).to_csv(
            saida/'auditoria_arquivos_raw_nao_lidos.csv',index=False,encoding='utf-8-sig')
        if falhas:
            raise ValueError(f'{len(falhas)} arquivo(s) raw não reconhecidos. '
                             'Revise auditoria_arquivos_raw_nao_lidos.csv antes da modelagem.')
        if not partes:raise ValueError('Nenhuma estação/dia válida nos arquivos raw.')
        df=pd.concat(partes, ignore_index=True)
        df['cod_estacao']=df.cod_estacao.astype(str)
        df['date']=pd.to_datetime(df.date).dt.tz_localize(None).dt.normalize()
        duplicados=df.loc[df.duplicated(['cod_estacao','date'],keep=False)].copy()
        duplicados.to_csv(saida/'auditoria_sobreposicao_arquivos_raw.csv',index=False,encoding='utf-8-sig')
        if not duplicados.empty:
            varia=duplicados.groupby(['cod_estacao','date']).precip_total_mm.agg(
                lambda x: float(x.max()-x.min()))
            n_conf=int((varia>1e-5).sum())
            if n_conf:
                raise ValueError(f'{n_conf} estação-dia com totais conflitantes entre arquivos raw. '
                    'Audite a sobreposição antes de escolher qual leitura usar. '
                    'Arquivo: auditoria_sobreposicao_arquivos_raw.csv')
        df=df.sort_values(['cod_estacao','date','_ordem_arquivo']).drop_duplicates(
            ['cod_estacao','date'],keep='last').drop(columns=['_ordem_arquivo','_origem'])
        rel['proveniencia']='séries horárias/10min originais agregadas por estação-dia'
        rel['arquivos_raw_lidos']=len(raw_files)
        rel['etapa_tempo_original']='timezone conforme função aggregate_daily do ETL legado; auditar UTC x horário local'
    requeridas={'cod_estacao','date','precip_total_mm'}
    if not requeridas.issubset(df.columns):
        raise ValueError(f'Estação-dia requer {sorted(requeridas)}. Obtidas: {list(df.columns)}')
    df['cod_estacao']=df.cod_estacao.astype(str).str.strip()
    df['date']=pd.to_datetime(df.date).dt.tz_localize(None).dt.normalize()
    df['precip_total_mm']=pd.to_numeric(df.precip_total_mm,errors='raise')
    if df.duplicated(['cod_estacao','date']).any():
        raise ValueError('Estação-dia duplicada. Precisa consolidar a origem antes de interpolar.')
    if not np.isfinite(df.precip_total_mm).all() or (df.precip_total_mm<0).any():
        raise ValueError('Precipitação negativa/não-finita na estação-dia.')
    marcados=df.merge(exc,on=['cod_estacao','date'],how='left',validate='one_to_one',indicator=True)
    removidos=marcados.loc[marcados['_merge'].eq('both')].copy()
    removidos.to_csv(saida/'auditoria_estacao_dia_excluida.csv',index=False,encoding='utf-8-sig')
    df=marcados.loc[marcados['_merge'].eq('left_only'),df.columns].copy()
    rel['exclusoes_configuradas']=len(exc)
    rel['exclusoes_presentes_na_entrada']=len(removidos)
    rel['exclusoes_ja_ausentes']=len(exc)-len(removidos)
    rel['registros_estacao_dia_finais']=len(df)
    rel['estacoes_distintas']=int(df.cod_estacao.nunique())
    # Mesmo que um outro total anômalo exista, não descartá-lo automaticamente.
    df.loc[df.precip_total_mm>500].to_csv(
       saida/'extremos_estacao_dia_para_revisar.csv',index=False,encoding='utf-8-sig')
    rel['extremos_acima_500mm_para_revisar']=int((df.precip_total_mm>500).sum())
    if 'n_leituras' in df.columns:
        # Diagnóstico, não filtro: a frequência de amostragem pode mudar no tempo.
        # Uma razão pequena sinaliza possível dia parcial; não significa chuva zero.
        qa=df[['cod_estacao','date','n_leituras','precip_total_mm']].copy()
        qa['mes']=qa.date.dt.to_period('M').astype(str)
        qa['n_ref_p90']=qa.groupby(['cod_estacao','mes']).n_leituras.transform(
            lambda x: max(1.0,float(x.quantile(.9))))
        qa['fracao_leituras_referencia']=qa.n_leituras/qa.n_ref_p90
        suspeitos=qa.loc[qa.fracao_leituras_referencia<.75]
        suspeitos.to_csv(saida/'dias_com_possivel_coleta_parcial.csv',
                        index=False,encoding='utf-8-sig')
        rel['estacao_dias_possivelmente_parciais_sem_exclusao']=len(suspeitos)
    if not cache.exists():
        cache.parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(cache,index=False,encoding='utf-8-sig')
        rel['arquivo_intermediario_gerado']=str(cache)
        rel['hash_estacao_dia']=sha256(cache)
    return df,rel



def auditar_reproducao_municipal(df, estacoes, dados, saida):
    """Compara agregação dos registros individuais ao CSV municipal atual.

    Diferenças não são automaticamente apagadas; reportadas para revisão.
    O índice atual usa 1 / distância ao centro, conforme o ETL anterior.
    """
    salvo=dados/'cemaden_sp_diario.csv'
    if not salvo.exists():return {'comparacao_disponivel':False}
    meta=estacoes[['cod_estacao','lat','lon']].copy()
    meta.cod_estacao=meta.cod_estacao.astype(str).str.strip()
    rad=np.pi/180
    lat1, lon1=meta.lat.to_numpy(float)*rad,meta.lon.to_numpy(float)*rad
    lat2,lon2=np.deg2rad(-23.5505),np.deg2rad(-46.6333)
    a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    dist=2*6371*np.arcsin(np.sqrt(a))
    pesos=pd.Series(1/np.maximum(dist,.1),index=meta.cod_estacao.values)
    mat=df.pivot(index='date',columns='cod_estacao',values='precip_total_mm')
    pesos=pesos.reindex(mat.columns)
    num=mat.mul(pesos,axis=1).sum(axis=1,min_count=1)
    den=mat.notna().mul(pesos,axis=1).sum(axis=1)
    rec=(num/den.replace(0,np.nan)).rename('reconstituido_mm').reset_index()
    original=pd.read_csv(salvo,parse_dates=['date'])[['date','cemaden_precip_idw_mm']]
    tab=original.merge(rec,on='date',how='outer',validate='one_to_one')
    tab['delta_mm']=tab.reconstituido_mm-tab.cemaden_precip_idw_mm
    tab['divergente_001_mm']=(tab.delta_mm.abs()>.01) & tab.delta_mm.notna()
    tab.to_csv(saida/'auditoria_confronto_chuva_municipal.csv',
               index=False,encoding='utf-8-sig')
    rel={'comparacao_disponivel':True,
      'n_datas_comparaveis':int(tab.delta_mm.notna().sum()),
      'n_datas_com_delta_absoluto_maior_001mm':int(tab.divergente_001_mm.sum()),
      'maior_delta_absoluto_mm':float(tab.delta_mm.abs().max())}
    return rel


def localizacoes(estacoes, limites):
    """Distâncias fixas em km entre pontos representativos e estações (EPSG:31983)."""
    e=estacoes.copy()
    e['cod_estacao']=e.cod_estacao.astype(str).str.strip()
    if e.cod_estacao.duplicated().any():raise ValueError('Estação com metadados de localização duplicados')
    if e[['lat','lon']].isna().any().any():raise ValueError('Estação sem coordenada')
    if not (e.lat.between(-24.5,-22.0)&e.lon.between(-48,-45)).all():
        raise ValueError('Coordenadas das estações fora da região plausível')
    reg=limites[['codigo_subprefeitura','nome_subprefeitura','geometry']].copy()
    reg['codigo_subprefeitura']=reg.codigo_subprefeitura.astype(str).str.zfill(2)
    if reg.crs is None:raise ValueError('Limites sem sistema de referência explícito')
    reg=reg.to_crs(epsg=31983)
    reg['ponto']=reg.geometry.representative_point()
    local=gpd.GeoDataFrame(e[['cod_estacao']],geometry=gpd.points_from_xy(e.lon,e.lat),
                           crs='EPSG:4326').to_crs(epsg=31983)
    d=np.hypot(reg.ponto.x.to_numpy()[:,None]-local.geometry.x.to_numpy()[None,:],
               reg.ponto.y.to_numpy()[:,None]-local.geometry.y.to_numpy()[None,:])/1000
    return reg,d


def idw_por_regiao(estacao_dia, estacoes, limites, datas,
                   *, raio_km=RAIO_KM, k=K_ESTACOES, min_estacoes=MIN_ESTACOES,
                   potencia=POTENCIA_IDW):
    """Usa até k estações válidas MAIS PRÓXIMAS em cada região-dia.

    Não usa nenhuma leitura posterior ao dia. Sem >=min_estacoes no raio, retorna
    NaN (não inventa zero, não preenche com climatologia municipal).
    """
    if not (k>=min_estacoes>=1 and raio_km>0 and potencia>0):
        raise ValueError('Parâmetros IDW inválidos')
    reg, dist = localizacoes(estacoes,limites)
    if len(reg)!=32 or reg.codigo_subprefeitura.nunique()!=32:
        raise ValueError('Esperadas as 32 subprefeituras GeoSampa')
    dados=estacao_dia.copy()
    dados['cod_estacao']=dados.cod_estacao.astype(str).str.strip()
    nao_mapeadas=set(dados.cod_estacao)-set(estacoes.cod_estacao.astype(str))
    if nao_mapeadas:
        raise ValueError(f'Estações sem coordenadas oficiais: {sorted(nao_mapeadas)[:15]}')
    datas=pd.DatetimeIndex(pd.to_datetime(datas)).normalize().drop_duplicates().sort_values()
    matriz=(dados.pivot(index='date',columns='cod_estacao',values='precip_total_mm')
            .reindex(index=datas,columns=estacoes.cod_estacao.astype(str)))
    v=matriz.to_numpy(dtype=float)
    resultados=[];audit=[]
    for i,item in enumerate(reg.itertuples()):
        candidatos=np.argsort(dist[i])
        candidatos=candidatos[dist[i,candidatos]<=raio_km]
        if len(candidatos)==0:
            bloco=pd.DataFrame({'date':datas, 'codigo_subprefeitura':item.codigo_subprefeitura,
                'nome_subprefeitura':item.nome_subprefeitura, CHUVA_COL:np.nan,
                'cemaden_local_n_estacoes':np.zeros(len(datas),dtype=int),
                'cemaden_local_dist_max_km':np.nan})
            audit.append({'codigo_subprefeitura':item.codigo_subprefeitura,
                          'nome_subprefeitura':item.nome_subprefeitura,
                          'estacoes_no_raio':0,'dist_proxima_km':np.nan})
        else:
            # Reavalia as mais próximas para CADA dia (sem estação válida -> próxima disponível).
            sub=v[:,candidatos]
            valido=np.isfinite(sub) & (sub>=0)
            usar=valido & (np.cumsum(valido,axis=1)<=k)
            cont=usar.sum(axis=1)
            pesos=1/np.maximum(dist[i,candidatos],CENTRO_MIN_KM)**potencia
            w=np.where(usar,pesos[None,:],0)
            den=w.sum(axis=1)
            numer=(np.where(usar,sub,0)*w).sum(axis=1)
            rain=np.divide(numer,den,out=np.full(len(datas),np.nan),where=den>0)
            rain[cont<min_estacoes]=np.nan
            dist_max=np.where(usar,dist[i,candidatos][None,:],-np.inf).max(axis=1)
            dist_max[(cont<min_estacoes)]=np.nan
            bloco=pd.DataFrame({'date':datas,'codigo_subprefeitura':item.codigo_subprefeitura,
               'nome_subprefeitura':item.nome_subprefeitura,CHUVA_COL:rain,
               'cemaden_local_n_estacoes':cont,'cemaden_local_dist_max_km':dist_max})
            audit.append({'codigo_subprefeitura':item.codigo_subprefeitura,
                'nome_subprefeitura':item.nome_subprefeitura,
                'estacoes_no_raio':len(candidatos),
                'dist_proxima_km':float(dist[i,candidatos[0]])})
        for lag in CHUVA_LAGS:
            bloco[f'cemaden_local_lag{lag}d']=bloco[CHUVA_COL].shift(lag)
        for janela in CHUVA_ACUM:
            bloco[f'cemaden_local_acc{janela}d']=bloco[CHUVA_COL].rolling(
                        janela,min_periods=janela).sum()
        resultados.append(bloco)
    saida=pd.concat(resultados,ignore_index=True).sort_values(
        ['date','codigo_subprefeitura']).reset_index(drop=True)
    assert not saida.duplicated(['date','codigo_subprefeitura']).any()
    assert len(saida)==len(datas)*32
    qualidade=(saida.groupby(['codigo_subprefeitura','nome_subprefeitura'],as_index=False)
        .agg(dias=('date','size'),dias_cobertos=(CHUVA_COL,'count'),
             n_estacoes_mediana=('cemaden_local_n_estacoes','median'),
             n_estacoes_min=('cemaden_local_n_estacoes','min')))
    qualidade['pct_dias_cobertos']=100*qualidade.dias_cobertos/qualidade.dias
    qualidade=qualidade.merge(pd.DataFrame(audit),on=['codigo_subprefeitura','nome_subprefeitura'])
    return saida, qualidade


def construir_painel(dados,geo,chuva):
    cge=pd.read_csv(geo/'cge_subprefeitura_dia.csv',parse_dates=['date'],
                    dtype={'codigo_subprefeitura':str})
    ur=pd.read_csv(geo/'indicadores_subprefeituras.csv',
                  dtype={'codigo_subprefeitura':str})
    mun=pd.read_csv(dados/'dataset_final_integrado.csv',parse_dates=['date'])
    inm=pd.read_csv(dados/'consolidated_dataset.csv',parse_dates=['date'])
    cge['codigo_subprefeitura']=cge.codigo_subprefeitura.str.zfill(2)
    ur['codigo_subprefeitura']=ur.codigo_subprefeitura.str.zfill(2)
    if cge.duplicated(['date','codigo_subprefeitura']).any():
        raise ValueError('Duplicação CGE região-dia')
    for base,nome in [(mun,'municipal'),(inm,'INMET')]:
        if base.date.duplicated().any():raise ValueError(f'Data duplicada {nome}')
    for c in MUNICIPAL:
        if c not in mun and c not in ('sp_precip_mm','mes_sin','mes_cos'):
            raise ValueError(f'Coluna municipal ausente: {c}')
    # Não unir o alvo do dia futuro à matriz de atributos.
    cols_mun=[x for x in MUNICIPAL if x not in ('sp_precip_mm','mes_sin','mes_cos')]
    panel=(cge[['date','codigo_subprefeitura','dia_presente_base_cge','tem_registro_cge']]
      .merge(mun[['date',*cols_mun]],on='date',validate='many_to_one')
      .merge(inm[['date','sp_precip_mm']],on='date',validate='many_to_one')
      .merge(ur[['codigo_subprefeitura','nome_subprefeitura',*URB]],
          on='codigo_subprefeitura',validate='many_to_one')
      .merge(chuva.drop(columns=['nome_subprefeitura']),
          on=['date','codigo_subprefeitura'],validate='one_to_one')
      .sort_values(['codigo_subprefeitura','date']).reset_index(drop=True))
    panel['data_alvo']=panel.date+pd.Timedelta(days=1)
    g=panel.groupby('codigo_subprefeitura')
    panel['alvo_amanha']=g.tem_registro_cge.shift(-1)
    obs_amanha=g.dia_presente_base_cge.shift(-1)
    data_proxima=g.date.shift(-1)
    panel.loc[~(obs_amanha.eq(True) & data_proxima.eq(panel.data_alvo)),
              'alvo_amanha']=np.nan
    panel['mes_sin']=np.sin(2*np.pi*(panel.data_alvo.dt.month-1)/12)
    panel['mes_cos']=np.cos(2*np.pi*(panel.data_alvo.dt.month-1)/12)
    panel=panel.dropna(subset=['alvo_amanha']).copy()
    panel['alvo_amanha']=panel.alvo_amanha.astype(int)
    regs=pd.get_dummies(panel.codigo_subprefeitura,prefix='regiao',dtype=np.int8)
    panel=panel.join(regs)
    if (panel.data_alvo-panel.date).ne(pd.Timedelta(days=1)).any():
        raise ValueError('Alvo desalinhado temporalmente')
    assert not panel.duplicated(['date','codigo_subprefeitura']).any()
    return panel,list(regs.columns)


def recortes_mesma_amostra(painel):
    # Uma mesma amostra para todos os experimentos: só quando a chuva local de t
    # foi observada com suporte >=2. Outras ausências são tratadas pelo HGB.
    elegivel=painel.loc[painel[CHUVA_COL].notna()].copy()
    treino=elegivel.loc[elegivel.data_alvo<'2022-01-01'].copy()
    val=elegivel.loc[elegivel.data_alvo.between('2022-01-01','2023-12-31')].copy()
    teste=elegivel.loc[elegivel.data_alvo>='2024-01-01'].copy()
    if min(len(treino),len(val),len(teste))==0:
        raise ValueError('Uma partição temporal ficou vazia: examine cobertura do CEMADEN.')
    # Embargo de data-alvo / data das variáveis na fronteira dos períodos.
    treino=treino.loc[treino.data_alvo<val.date.min()]
    val=val.loc[val.data_alvo<teste.date.min()]
    if min(len(treino),len(val),len(teste))==0:
        raise ValueError('Partição vazia após embargo.')
    assert treino.data_alvo.max() < val.date.min()
    assert val.data_alvo.max() < teste.date.min()
    for nome,df in [('treino',treino),('validacao',val),('teste',teste)]:
        if df.alvo_amanha.nunique()!=2:
            raise ValueError(f'Partição {nome} sem ambas as classes')
    return treino,val,teste


def classificador():
    return HistGradientBoostingClassifier(max_iter=115,max_leaf_nodes=15,
            learning_rate=.065,min_samples_leaf=80,l2_regularization=1,
            early_stopping=False,random_state=SEMENTE)


def limiar_validado(y,p):
    limiares=np.arange(.01,.501,.01)
    f1=np.array([f1_score(y,p>=t,zero_division=0) for t in limiares])
    rec=np.array([recall_score(y,p>=t,zero_division=0) for t in limiares])
    ix=np.lexsort((-limiares,-rec,-f1))[0]
    return float(limiares[ix])


def metricas(y,p,limiar):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    pred=p>=limiar
    return {'n_observacoes':len(y),'positivos':int(y.sum()),
      'prevalencia':float(y.mean()),
      'AP':float(average_precision_score(y,p)) if np.unique(y).size==2 else np.nan,
      'ROC_AUC':float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan,
      'Brier':float(brier_score_loss(y,p)),
      'precisao':float(precision_score(y,pred,zero_division=0)),
      'recall':float(recall_score(y,pred,zero_division=0)),
      'F1':float(f1_score(y,pred,zero_division=0)),
      'fracao_alertas':float(pred.mean()),'limiar':float(limiar)}


def modelos_contraste(painel,regs,saida):
    treino,val,teste=recortes_mesma_amostra(painel)
    ensaios={
      'Municipal + região (referência)':MUNICIPAL+regs,
      'Chuva local + região':LOCAL+regs,
      'Municipal + local + região':MUNICIPAL+[x for x in LOCAL if x not in MUNICIPAL]+regs,
      'Chuva local + urbano':LOCAL+URB,
      'Chuva local + região + urbano':LOCAL+regs+URB,
    }
    linhas=[];objetos={};p_valid={};limiares={}
    for nome,cols in ensaios.items():
        if len(cols)!=len(set(cols)):
            raise ValueError('Variáveis duplicadas no experimento '+nome)
        m=classificador()
        m.fit(treino[cols],treino.alvo_amanha)
        p=m.predict_proba(val[cols])[:,1]
        lv=limiar_validado(val.alvo_amanha,p)
        objetos[nome]=m;p_valid[nome]=p;limiares[nome]=lv
        linhas.append({'experimento':nome,**metricas(val.alvo_amanha,p,lv)})
    valid=pd.DataFrame(linhas).set_index('experimento')
    selecionado=valid.AP.idxmax()  # decisão AP apenas na validação
    valid.to_csv(saida/'metricas_validacao_chuva_regional.csv',encoding='utf-8-sig')
    saidas=[];probs_teste={}
    for nome,cols in ensaios.items():
        p=objetos[nome].predict_proba(teste[cols])[:,1]
        probs_teste[nome]=p
        saidas.append({'experimento':nome,**metricas(teste.alvo_amanha,p,limiares[nome])})
    teste_resultado=pd.DataFrame(saidas).set_index('experimento')
    teste_resultado.to_csv(saida/'metricas_teste_chuva_regional.csv',encoding='utf-8-sig')
    diagnostico=pd.DataFrame({'codigo_subprefeitura':teste.codigo_subprefeitura.values,
          'nome_subprefeitura':teste.nome_subprefeitura.values,
          'date':teste.date.values,'data_alvo':teste.data_alvo.values,
          'alvo_amanha':teste.alvo_amanha.values,
          'probabilidade':probs_teste[selecionado],
          'alerta':(probs_teste[selecionado]>=limiares[selecionado]).astype(int)})
    diagnostico.to_csv(saida/'previsoes_regionais_teste.csv',index=False,encoding='utf-8-sig')
    reg=[]
    for codigo,g in diagnostico.groupby('codigo_subprefeitura'):
        n=int(g.alvo_amanha.sum())
        med=metricas(g.alvo_amanha,g.probabilidade,limiares[selecionado])
        med.update({'codigo_subprefeitura':codigo,
            'nome_subprefeitura':g.nome_subprefeitura.iloc[0],
            'avaliacao_exploratoria_min_5_positivos':n>=5})
        if n<5:
            for coluna in ['AP','ROC_AUC','recall','precisao','F1']:
                med[coluna]=np.nan
        reg.append(med)
    regionais=pd.DataFrame(reg).sort_values('positivos',ascending=False)
    regionais.to_csv(saida/'metricas_por_subprefeitura.csv',index=False,encoding='utf-8-sig')
    anual=[]
    for ano,g in diagnostico.groupby(diagnostico.data_alvo.dt.year):
        anual.append({'ano_alvo':int(ano),**metricas(g.alvo_amanha,g.probabilidade,
                                                       limiares[selecionado])})
    pd.DataFrame(anual).to_csv(saida/'metricas_anuais_chuva_regional.csv',index=False,encoding='utf-8-sig')
    # Importância agrupada do modelo que usa chuva local + região, sempre na validação.
    # Executar mesmo que outra variante seja selecionada, para não esconder
    # o diagnóstico regional quando a chuva local não vencer a referência.
    # Não é efeito causal da chuva, vegetação ou drenagem.
    modelo_permutacao = "Chuva local + região"
    if CHUVA_COL in ensaios[modelo_permutacao]:
        rng=np.random.default_rng(SEMENTE)
        base=pd.DataFrame({'codigo':val.codigo_subprefeitura.values,
           'nome':val.nome_subprefeitura.values,'y':val.alvo_amanha.values,
           'p':p_valid[modelo_permutacao]})
        cols=ensaios[modelo_permutacao]
        imp=[]
        alvo_local=[x for x in LOCAL if x.startswith('cemaden_local_')]
        for codigo,idx in base.groupby('codigo').indices.items():
            yy=base.y.iloc[idx].to_numpy()
            if yy.sum()<10:continue
            ref=average_precision_score(yy,base.p.iloc[idx].to_numpy())
            xp=val.iloc[idx][cols].copy()
            # Deslocar conjuntamente todas as séries de chuva local entre datas da MESMA região.
            perm=rng.permutation(len(xp))
            xp.loc[:,alvo_local]=xp[alvo_local].to_numpy()[perm,:]
            pp=objetos[modelo_permutacao].predict_proba(xp)[:,1]
            imp.append({'codigo_subprefeitura':codigo,
              'nome_subprefeitura':base.nome.iloc[idx[0]],'positivos_validacao':int(yy.sum()),
              'AP_validacao_original':float(ref),
              'AP_validacao_chuva_local_permutada':float(average_precision_score(yy,pp)),
              'queda_AP_descritiva':float(ref-average_precision_score(yy,pp))})
        pd.DataFrame(imp,columns=['codigo_subprefeitura','nome_subprefeitura',
            'positivos_validacao','AP_validacao_original',
            'AP_validacao_chuva_local_permutada','queda_AP_descritiva']).to_csv(
              saida/'permutacao_chuva_local_por_regiao_validacao.csv',index=False,encoding='utf-8-sig')
    # Gráficos em arquivos, não ajustam o limiar após olhar teste.
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    cores=plt.get_cmap('tab10')
    for i,nome in enumerate(ensaios):
        pr,rc,_=precision_recall_curve(teste.alvo_amanha,probs_teste[nome])
        axes[0].step(rc,pr,label=f'{nome}: AP {teste_resultado.at[nome,"AP"]:.3f}',
                     color=cores(i),linewidth=1.5)
    axes[0].axhline(teste.alvo_amanha.mean(),color='gray',linestyle='--',label='Prevalência')
    axes[0].set(xlabel='Recall',ylabel='Precisão',title='Teste região-dia · mesmo subconjunto',xlim=(0,1),ylim=(0,1))
    axes[0].legend(fontsize=7)
    tabel=valid[['AP']].rename(columns={'AP':'Validação'}).join(
        teste_resultado[['AP']].rename(columns={'AP':'Teste'}))
    tabel.plot(kind='barh',ax=axes[1]);axes[1].set(xlabel='Average Precision',
        title='Comparação de chuva municipal e regional')
    fig.tight_layout();fig.savefig(saida/'comparacao_modelos_chuva_regional.png',dpi=150,bbox_inches='tight');plt.close(fig)
    return valid,teste_resultado,selecionado,regionais,{
       'treino':len(treino),'validacao':len(val),'teste':len(teste),
       'prevalencia_teste':float(teste.alvo_amanha.mean()),
       'limiar_escolhido_validacao':limiares[selecionado]}


def executar(raiz:Path):
    raiz=Path(raiz).resolve()
    dados,geo,raw,scripts=localizar_arquivos(raiz)
    saida=(raiz/'resultados_chuva_regional' if dados==raiz else
           raiz/'data/outputs/modelagem/chuva_regional_subprefeituras')
    saida.mkdir(parents=True,exist_ok=True)
    estacao_dia,origem=ler_estacao_dia(raiz,dados,raw,scripts,saida)
    metadados=pd.read_csv(dados/'cemaden_estacoes.csv',dtype={'cod_estacao':str})
    confr=auditar_reproducao_municipal(estacao_dia,metadados,dados,saida)
    origem['auditoria_reproducao_municipal']=confr
    if confr.get('n_datas_com_delta_absoluto_maior_001mm',0):
        print('ATENÇÃO: a chuva municipal recomposta difere da série original em',
              confr['n_datas_com_delta_absoluto_maior_001mm'],'dias. Confira auditoria_confronto_chuva_municipal.csv.')
    limites=gpd.read_file(geo/'subprefeituras_tratadas.gpkg')
    cge=pd.read_csv(geo/'cge_subprefeitura_dia.csv',usecols=['date'],parse_dates=['date'])
    chuva,qualidade=idw_por_regiao(estacao_dia,metadados,limites,cge.date.unique())
    chuva.to_csv(saida/'cemaden_subprefeitura_dia.csv',index=False,encoding='utf-8-sig')
    qualidade.to_csv(saida/'cobertura_chuva_por_subprefeitura.csv',index=False,encoding='utf-8-sig')
    panel,regs=construir_painel(dados,geo,chuva)
    panel.to_csv(saida/'painel_modelagem_chuva_regional.csv',index=False,encoding='utf-8-sig')
    valid,teste,escolhido,regionais,tamanhos=modelos_contraste(panel,regs,saida)
    resumo={'origem':origem,'parametros_interpolacao':{
      'CRS':'EPSG:31983','ponto':'representative_point da subprefeitura',
      'raio_km':RAIO_KM,'k_proximas':K_ESTACOES,'min_estacoes':MIN_ESTACOES,
      'potencia_idw':POTENCIA_IDW,'distancia_minima_km':CENTRO_MIN_KM},
      'cobertura_local_pct_min':float(qualidade.pct_dias_cobertos.min()),
      'cobertura_local_pct_max':float(qualidade.pct_dias_cobertos.max()),
      'regioes_com_menos_5_positivos_teste':int((regionais.positivos<5).sum()),
      'experimento_escolhido_validacao':escolhido,
      'divisao':tamanhos,
      'interpretacao':'Registro CGE, não ausência física confirmada; vegetação 2017/drenagem 2026 ex post; chuva local observada em t, não previsão de chuva t+1.'}
    (saida/'manifesto_regionalizacao.json').write_text(
      json.dumps(resumo,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('Chuva por subprefeitura calculada:',len(chuva),'linhas; fonte:',origem['proveniencia'])
    print('Cobertura por subprefeitura (%):',round(resumo['cobertura_local_pct_min'],2),
          'a',round(resumo['cobertura_local_pct_max'],2))
    print('VALIDAÇÃO\n',valid[['AP','precisao','recall']].round(3))
    print('TESTE reservado\n',teste[['AP','precisao','recall','prevalencia']].round(3))
    print('Selecionado pela validação:',escolhido)
    print('Resultados:',saida)
    return resumo,saida


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd(),help='Raiz do repositório')
    arg=parser.parse_args()
    try:executar(arg.root)
    except (ValueError,FileNotFoundError) as exc:
        print('ERRO DE ENTRADA / QUALIDADE:',exc,file=sys.stderr)
        sys.exit(2)
