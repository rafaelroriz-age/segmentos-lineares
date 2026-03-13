"""
Pipeline de Segmentação Homogênea — Etapa 1 e 2
Leitura de dados, agregação 40m → 120m e engenharia de features.
"""

import pandas as pd
import numpy as np


def carregar_dados(filepath):
    """Carrega dados de deflexão a partir de arquivo Excel ou CSV."""
    if hasattr(filepath, 'name'):
        # Streamlit UploadedFile
        nome = filepath.name
    else:
        nome = str(filepath)

    if nome.endswith('.xlsx') or nome.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        # Tenta detectar separador automaticamente
        try:
            df = pd.read_csv(filepath, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(filepath)
    return df


def agregar_120m(df, col_est='Estacao_m', col_def='Deflexao', passo=40, janela=3):
    """
    Une grupos de 3 pontos consecutivos (3 x 40m = 120m).
    A deflexão do segmento de 120m é a MÉDIA das deflexões dos 3 pontos.

    Parâmetros
    ----------
    df : DataFrame com dados brutos (leitura a cada 40m)
    col_est : nome da coluna de estação/posição (m)
    col_def : nome da coluna de deflexão
    passo : espaçamento entre medições (padrão 40m)
    janela : número de pontos por segmento (padrão 3 → 120m)

    Retorna
    -------
    DataFrame de segmentos de 120m com estatísticas descritivas.
    """
    df = df.copy()
    df[col_est] = pd.to_numeric(df[col_est], errors='coerce')
    df[col_def] = pd.to_numeric(df[col_def], errors='coerce')
    df = df.dropna(subset=[col_est, col_def]).sort_values(col_est).reset_index(drop=True)

    registros = []
    n = len(df)
    for i in range(0, n - janela + 1, janela):
        grupo = df.iloc[i:i + janela]
        reg = {
            'seg_id': i // janela + 1,
            'km_ini': grupo[col_est].min(),
            'km_fim': grupo[col_est].max() + passo,
            'comprimento_m': grupo[col_est].max() - grupo[col_est].min() + passo,
            'Deflexao': grupo[col_def].mean(),
            'Deflexao_std': grupo[col_def].std(),
            'Deflexao_caract': grupo[col_def].mean() + grupo[col_def].std(),
            'Deflexao_max': grupo[col_def].max(),
            'Deflexao_min': grupo[col_def].min(),
            'n_pontos': len(grupo),
        }
        # Incluir demais colunas numéricas com média
        for col in df.select_dtypes(include=np.number).columns:
            if col not in [col_est, col_def]:
                reg[f'{col}_med'] = grupo[col].mean()
        registros.append(reg)

    seg120 = pd.DataFrame(registros)
    return seg120


def construir_features(seg120, vars_clustering=None):
    """
    Constrói features derivadas para clustering.

    Parâmetros
    ----------
    seg120 : DataFrame de segmentos de 120m
    vars_clustering : lista de colunas a usar. Se None, usa deflexão + extras.

    Retorna
    -------
    seg : DataFrame enriquecido
    X : DataFrame com as features selecionadas para clustering
    vars_ok : lista de nomes das features efetivamente usadas
    """
    seg = seg120.copy()

    # Feature derivada: Coeficiente de Variação (CV)
    seg['CV_deflexao'] = seg['Deflexao_std'] / seg['Deflexao'].replace(0, np.nan)

    # Feature de posição relativa (normalizada 0-1)
    range_km = seg['km_ini'].max() - seg['km_ini'].min()
    if range_km > 0:
        seg['posicao_rel'] = (seg['km_ini'] - seg['km_ini'].min()) / range_km
    else:
        seg['posicao_rel'] = 0.0

    if vars_clustering is None:
        vars_clustering = ['Deflexao_caract', 'Deflexao_std', 'CV_deflexao']
        extras = [c for c in seg.columns if c.endswith('_med')]
        vars_clustering.extend(extras)

    # Remover colunas com muitos nulos
    vars_ok = [v for v in vars_clustering if v in seg.columns and seg[v].isnull().mean() < 0.3]
    X = seg[vars_ok].fillna(seg[vars_ok].median())

    return seg, X, vars_ok
