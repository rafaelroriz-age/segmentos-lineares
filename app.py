"""
🛣️ Segmentação Homogênea LINEAR de Rodovias — Pipeline Científico
Todos os segmentos são CONTÍGUOS ESPACIALMENTE (lineares).

15 métodos:
  A. Tradicionais: CDA, MCV, SHS
  B. Clustering + linearização: K-Means, Ward 1D, DBSCAN, HDBSCAN,
     Affinity Propagation, GMM, SOM, Spectral, UMAP+KM
  C. Change-point: PELT
  D. ML supervisionado: KNN, Random Forest

Referências:
  - Abdelaty & Jeong (2017) — framework de 3 estágios
  - Mukhtarli (2020) — SOM e avaliação extrínseca com ML
  - AASHTO (1993) — Cumulative Difference Approach (CDA)
  - Truong et al. (2020) — ruptures (PELT)
"""

import streamlit as st
import pandas as pd
import numpy as np

from pipeline.preprocessing import carregar_dados, agregar_120m, construir_features
from pipeline.clustering import (
    padronizar, linearizar_labels,
    cda_segmentation, mcv_segmentation, shs_segmentation,
    kmeans_linear, ward1d_linear, dbscan_linear, hdbscan_linear,
    affinity_propagation_linear, gmm_linear, som_linear,
    spectral_linear, pelt_segmentation, umap_kmeans_linear,
    knn_segmentation, random_forest_segmentation,
    aplicar_regras_espaciais,
)
from pipeline.evaluation import (
    avaliar_intrinseco, avaliar_extrinseco_xgb, variancia_intra,
    tabela_comparativa, perfil_clusters, interpretar_resultado,
)
from pipeline.visualization import (
    plot_clusters_rodovia, plot_boxplot_clusters,
    plot_perfil_longitudinal, plot_heatmap_metricas,
    plot_segmentos_mapa, plot_radar_clusters,
)
from pipeline.audit import (
    gap_statistic, estabilidade_bootstrap,
    teste_kruskal_wallis, teste_anova, teste_normalidade,
    homogeneidade_intra, matriz_ari_cruzado,
    feature_importance_xgb, diagnostico_metodo,
    auditoria_completa,
)
from pipeline.export_excel import exportar_excel

# ── Configuração da página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Segmentação Homogênea LINEAR",
    page_icon="🛣️",
    layout="wide",
)
st.title("🛣️ Segmentação Homogênea LINEAR de Rodovias")
st.caption(
    "Pipeline completo com 15 métodos. **Todos os segmentos são contíguos "
    "espacialmente (lineares).** Inclui métodos tradicionais (CDA, MCV, SHS), "
    "clustering com linearização, e métodos supervisionados (KNN, RF)."
)

# ── Mapeamento método → coluna ────────────────────────────────────────────────

METODO_COL = {
    # Tradicionais
    'CDA': 'cluster_cda',
    'MCV': 'cluster_mcv',
    'SHS': 'cluster_shs',
    # Clustering + linearização
    'K-Means': 'cluster_kmeans',
    'Ward 1D': 'cluster_ward1d',
    'DBSCAN': 'cluster_dbscan',
    'HDBSCAN': 'cluster_hdbscan',
    'Affinity Prop.': 'cluster_ap',
    'GMM': 'cluster_gmm',
    'SOM': 'cluster_som',
    'Spectral': 'cluster_spectral',
    'UMAP+KM': 'cluster_umap_km',
    # Change-point
    'PELT': 'cluster_pelt',
    # ML supervisionado
    'KNN': 'cluster_knn',
    'Random Forest': 'cluster_rf',
}

TABELA_METODOS = pd.DataFrame([
    {'Método': 'CDA', 'Tipo': '🏗️ Tradicional',
     'Descrição': 'Cumulative Difference Approach (AASHTO). Detecta mudanças na tendência média.',
     'Linear': '✅ Nativo'},
    {'Método': 'MCV', 'Tipo': '🏗️ Tradicional',
     'Descrição': 'Method of Concomitant Variations. Segmenta por variabilidade local (CV).',
     'Linear': '✅ Nativo'},
    {'Método': 'SHS', 'Tipo': '🏗️ Tradicional',
     'Descrição': 'Spatial Homogeneity Segmentation. Teste t sequencial de homogeneidade.',
     'Linear': '✅ Nativo'},
    {'Método': 'K-Means', 'Tipo': '🔬 Clustering',
     'Descrição': 'Centroide-based. Rápido, assume clusters esféricos.',
     'Linear': '🔄 Linearizado'},
    {'Método': 'Ward 1D', 'Tipo': '🔬 Clustering',
     'Descrição': 'Hierárquico com conectividade 1D. Contiguidade garantida.',
     'Linear': '✅ Nativo'},
    {'Método': 'DBSCAN', 'Tipo': '🔬 Clustering',
     'Descrição': 'Baseado em densidade. Detecta outliers automaticamente.',
     'Linear': '🔄 Linearizado'},
    {'Método': 'HDBSCAN', 'Tipo': '🔬 Clustering',
     'Descrição': 'DBSCAN hierárquico. Múltiplas densidades.',
     'Linear': '🔄 Linearizado'},
    {'Método': 'Affinity Prop.', 'Tipo': '🔬 Clustering',
     'Descrição': 'k automático via troca de mensagens. Identifica protótipos.',
     'Linear': '🔄 Linearizado'},
    {'Método': 'GMM', 'Tipo': '🔬 Clustering',
     'Descrição': 'Gaussianas: fronteiras suaves, clustering probabilístico.',
     'Linear': '🔄 Linearizado'},
    {'Método': 'SOM', 'Tipo': '🔬 Clustering',
     'Descrição': 'Rede neural: projeta em grade 2D. Bom p/ muitas features.',
     'Linear': '🔄 Linearizado'},
    {'Método': 'Spectral', 'Tipo': '🔬 Clustering',
     'Descrição': 'Autovetores do grafo de similaridade. Clusters irregulares.',
     'Linear': '🔄 Linearizado'},
    {'Método': 'UMAP+KM', 'Tipo': '🔬 Clustering',
     'Descrição': 'Redução dimensional UMAP + K-Means. Alta dimensionalidade.',
     'Linear': '🔄 Linearizado'},
    {'Método': 'PELT', 'Tipo': '📊 Change-Point',
     'Descrição': 'Detecção exata de rupturas. Segmentação direta da série.',
     'Linear': '✅ Nativo'},
    {'Método': 'KNN', 'Tipo': '🤖 ML Supervisionado',
     'Descrição': 'K-Nearest Neighbors refina clustering via vizinhos próximos.',
     'Linear': '🔄 Linearizado'},
    {'Método': 'Random Forest', 'Tipo': '🤖 ML Supervisionado',
     'Descrição': 'Ensemble de árvores refina clustering. Gera feature importance.',
     'Linear': '🔄 Linearizado'},
])

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configurações")
    uploaded = st.file_uploader(
        "Carregar dados (Excel / CSV)", type=['xlsx', 'xls', 'csv']
    )
    col_est = st.text_input("Coluna de Estação/km", value="Estação (m)")
    col_def = st.text_input("Coluna de Deflexão", value="Deflexão (0,01mm)")

    st.subheader("Agregação")
    passo_m = st.number_input("Espaçamento entre medições (m)", value=40, min_value=1)
    janela = st.number_input("Pontos por segmento", value=3, min_value=2, max_value=10)
    comp_seg = passo_m * janela
    st.info(f"Segmentos de **{comp_seg} m** ({janela} × {passo_m} m)")

    st.subheader("Clustering")
    n_clust = st.slider("Número de clusters (para métodos com k)", 2, 15, 5)
    min_comp = st.slider("Comprimento mínimo de segmento (m)", 120, 2000, 400, 120)

    st.subheader("Hiperparâmetros avançados")
    with st.expander("CDA"):
        cda_conf = st.slider("Fator de confiança CDA", 0.5, 3.0, 1.0, 0.1)
    with st.expander("MCV"):
        mcv_janela = st.slider("Janela MCV", 3, 15, 5)
        mcv_limiar = st.slider("Limiar CV", 0.1, 1.0, 0.3, 0.05)
    with st.expander("SHS"):
        shs_alpha = st.slider("Nível de significância (α)", 0.01, 0.20, 0.05, 0.01)
        shs_min = st.slider("Tamanho mínimo (SHS)", 2, 10, 3)
    with st.expander("DBSCAN"):
        eps_db = st.slider("eps", 0.1, 3.0, 0.5, 0.1)
        min_samp_db = st.slider("min_samples", 2, 10, 3)
    with st.expander("HDBSCAN"):
        min_cs_hdb = st.slider("min_cluster_size", 3, 30, 5)
    with st.expander("Affinity Propagation"):
        damping_ap = st.slider("damping", 0.5, 1.0, 0.8, 0.05)
    with st.expander("SOM"):
        som_rows = st.slider("Grid linhas", 2, 6, 3)
        som_cols = st.slider("Grid colunas", 2, 6, 3)
        som_epochs = st.slider("Époques", 200, 5000, 1000, 100)
    with st.expander("PELT"):
        pelt_pen = st.slider("Penalidade", 1, 100, 30)
        pelt_model = st.selectbox("Modelo de custo", ['rbf', 'l2', 'l1', 'linear'])
    with st.expander("UMAP + K-Means"):
        umap_nn = st.slider("n_neighbors (UMAP)", 5, 50, 15)
    with st.expander("KNN"):
        knn_k = st.slider("k vizinhos (KNN)", 3, 15, 5)
    with st.expander("Random Forest"):
        rf_trees = st.slider("Número de árvores", 50, 500, 100, 50)

    st.header("🔬 Métodos a executar")
    st.markdown("**Tradicionais (engenharia)**")
    met_cda     = st.checkbox("CDA (Cumulative Difference)", value=True)
    met_mcv     = st.checkbox("MCV (Concomitant Variations)", value=True)
    met_shs     = st.checkbox("SHS (Spatial Homogeneity)", value=True)

    st.markdown("**Clustering + linearização**")
    met_kmeans  = st.checkbox("K-Means", value=True)
    met_ward    = st.checkbox("Agglomerative Ward 1D", value=True)
    met_dbscan  = st.checkbox("DBSCAN", value=True)
    met_hdbscan = st.checkbox("HDBSCAN", value=True)
    met_ap      = st.checkbox("Affinity Propagation", value=False)
    met_gmm     = st.checkbox("GMM (Gaussian Mixture)", value=True)
    met_som     = st.checkbox("SOM (Self-Organizing Maps)", value=False)
    met_spec    = st.checkbox("Spectral Clustering", value=False)
    met_umap    = st.checkbox("UMAP + K-Means", value=False)

    st.markdown("**Change-point detection**")
    met_pelt    = st.checkbox("PELT (Change-Point)", value=True)

    st.markdown("**ML Supervisionado**")
    met_knn     = st.checkbox("KNN (K-Nearest Neighbors)", value=True)
    met_rf      = st.checkbox("Random Forest", value=True)

    executar = st.button("▶️ Executar Pipeline", type="primary", use_container_width=True)

# ── Verificação de upload ─────────────────────────────────────────────────────

if uploaded is None:
    st.info("⬆️ Faça upload dos dados na barra lateral para começar.")

    with st.expander("📚 Referência: métodos disponíveis", expanded=True):
        st.dataframe(TABELA_METODOS, use_container_width=True, hide_index=True)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — CARREGAMENTO E VISUALIZAÇÃO DOS DADOS BRUTOS
# ══════════════════════════════════════════════════════════════════════════════

df_raw = carregar_dados(uploaded)

tab_dados, tab_pipeline, tab_relatorio, tab_auditoria = st.tabs([
    "📋 Dados", "🔬 Pipeline & Resultados", "📄 Relatório", "📊 Auditoria"
])

with tab_dados:
    st.subheader("1. Dados brutos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", len(df_raw))
    c2.metric("Colunas", len(df_raw.columns))
    if col_def in df_raw.columns:
        c3.metric("Deflexão média", f"{df_raw[col_def].mean():.1f}")
    st.dataframe(df_raw.head(20), use_container_width=True)

    # ── ETAPA 2 — AGREGAÇÃO ──────────────────────────────────────────────────
    st.subheader(f"2. Segmentos de {comp_seg} m (média de deflexão)")
    seg120 = agregar_120m(df_raw, col_est=col_est, col_def=col_def,
                          passo=passo_m, janela=janela)
    c1, c2, c3 = st.columns(3)
    c1.metric("Segmentos", len(seg120))
    if len(seg120) > 0:
        c2.metric("Extensão (m)",
                  f"{seg120['km_fim'].max() - seg120['km_ini'].min():.0f}")
        c3.metric("Deflexão média (seg)",
                  f"{seg120['Deflexao'].mean():.1f}")
    st.dataframe(seg120.head(20), use_container_width=True)

    # ── Seleção de features ──────────────────────────────────────────────────
    _feature_candidates = ['Deflexao', 'Deflexao_caract', 'Deflexao_std',
                           'Deflexao_max', 'Deflexao_min']
    _feature_candidates += [c for c in seg120.columns if c.endswith('_med')]
    _feature_candidates += ['CV_deflexao', 'posicao_rel']
    _available = list(dict.fromkeys(_feature_candidates))

    features_selecionadas = st.sidebar.multiselect(
        "🎯 Features para clustering",
        options=_available,
        default=[f for f in ['Deflexao_caract', 'Deflexao_std', 'CV_deflexao']
                 if f in _available],
        help="Features usadas pelos métodos de clustering e ML."
    )
    if not features_selecionadas:
        st.sidebar.warning("⚠️ Selecione pelo menos uma feature.")
        features_selecionadas = ['Deflexao_caract', 'Deflexao_std', 'CV_deflexao']

    st.subheader("3. Features para clustering")
    seg120, X_df, vars_ok = construir_features(seg120, vars_clustering=features_selecionadas)
    st.write(f"Features selecionadas: **{', '.join(vars_ok)}**")
    st.dataframe(X_df.describe().round(3), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — EXECUÇÃO DO PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

with tab_pipeline:
    if not executar:
        st.info("Configure os parâmetros e clique em **▶️ Executar Pipeline** "
                "na barra lateral.")
        st.stop()

    # Preparar dados
    seg120, X_df, vars_ok = construir_features(seg120, vars_clustering=features_selecionadas)
    X = X_df.values
    X_scaled, scaler = padronizar(X)

    # Vetor de deflexão para métodos tradicionais (1D)
    defl_vals = seg120['Deflexao'].values

    resultados = {}
    progress = st.progress(0, text="Iniciando pipeline...")

    metodos_selecionados = []
    if met_cda:     metodos_selecionados.append('CDA')
    if met_mcv:     metodos_selecionados.append('MCV')
    if met_shs:     metodos_selecionados.append('SHS')
    if met_kmeans:  metodos_selecionados.append('K-Means')
    if met_ward:    metodos_selecionados.append('Ward 1D')
    if met_dbscan:  metodos_selecionados.append('DBSCAN')
    if met_hdbscan: metodos_selecionados.append('HDBSCAN')
    if met_ap:      metodos_selecionados.append('Affinity Prop.')
    if met_gmm:     metodos_selecionados.append('GMM')
    if met_som:     metodos_selecionados.append('SOM')
    if met_spec:    metodos_selecionados.append('Spectral')
    if met_pelt:    metodos_selecionados.append('PELT')
    if met_umap:    metodos_selecionados.append('UMAP+KM')
    if met_knn:     metodos_selecionados.append('KNN')
    if met_rf:      metodos_selecionados.append('Random Forest')

    total = len(metodos_selecionados)
    if total == 0:
        st.warning("Selecione pelo menos um método na barra lateral.")
        st.stop()

    for step, nome in enumerate(metodos_selecionados):
        progress.progress(step / total, text=f"Executando {nome}...")

        try:
            # ── A. Tradicionais ──────────────────────────────────────────────
            if nome == 'CDA':
                lbls, _ = cda_segmentation(defl_vals, confianca=cda_conf)
                seg120['cluster_cda'] = lbls

            elif nome == 'MCV':
                lbls, _ = mcv_segmentation(defl_vals, janela=mcv_janela,
                                           limiar_cv=mcv_limiar)
                seg120['cluster_mcv'] = lbls

            elif nome == 'SHS':
                lbls, _ = shs_segmentation(defl_vals,
                                           nivel_significancia=shs_alpha,
                                           min_tamanho=shs_min)
                seg120['cluster_shs'] = lbls

            # ── B. Clustering + linearização ─────────────────────────────────
            elif nome == 'K-Means':
                lbls, _, _ = kmeans_linear(X_scaled, n_clust)
                seg120['cluster_kmeans'] = lbls

            elif nome == 'Ward 1D':
                lbls, _, _ = ward1d_linear(X_scaled, n_clust)
                seg120['cluster_ward1d'] = lbls

            elif nome == 'DBSCAN':
                lbls, _, _ = dbscan_linear(X_scaled, eps=eps_db,
                                           min_samples=min_samp_db)
                seg120['cluster_dbscan'] = lbls

            elif nome == 'HDBSCAN':
                lbls, _, _ = hdbscan_linear(X_scaled,
                                            min_cluster_size=min_cs_hdb)
                seg120['cluster_hdbscan'] = lbls

            elif nome == 'Affinity Prop.':
                lbls, _, _ = affinity_propagation_linear(X_scaled,
                                                         damping=damping_ap)
                seg120['cluster_ap'] = lbls

            elif nome == 'GMM':
                lbls, _, _ = gmm_linear(X_scaled, n_clust)
                seg120['cluster_gmm'] = lbls

            elif nome == 'SOM':
                lbls, _, _ = som_linear(X_scaled,
                                        grid_size=(som_rows, som_cols),
                                        epochs=som_epochs)
                seg120['cluster_som'] = lbls

            elif nome == 'Spectral':
                lbls, _, _ = spectral_linear(X_scaled, n_clust)
                seg120['cluster_spectral'] = lbls

            elif nome == 'UMAP+KM':
                lbls, _, _, _, _ = umap_kmeans_linear(X_scaled, n_clust,
                                                       n_neighbors=umap_nn)
                seg120['cluster_umap_km'] = lbls

            # ── C. Change-point ──────────────────────────────────────────────
            elif nome == 'PELT':
                lbls, _, _ = pelt_segmentation(X_scaled, model=pelt_model,
                                               pen=pelt_pen)
                seg120['cluster_pelt'] = lbls

            # ── D. ML Supervisionado ─────────────────────────────────────────
            elif nome == 'KNN':
                lbls, _, _ = knn_segmentation(X_scaled, n_clust,
                                              k_neighbors=knn_k)
                seg120['cluster_knn'] = lbls

            elif nome == 'Random Forest':
                lbls, _, _ = random_forest_segmentation(X_scaled, n_clust,
                                                         n_estimators=rf_trees)
                seg120['cluster_rf'] = lbls

            resultados[nome] = {'labels': lbls, 'X_scaled': X_scaled}
        except Exception as e:
            st.warning(f"⚠️ Erro em {nome}: {e}")

    progress.progress(1.0, text="Pipeline concluído ✅")

    # ══════════════════════════════════════════════════════════════════════════
    # ETAPA 4 — TABELA COMPARATIVA DE MÉTRICAS
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader("📊 Comparação de Métodos")
    df_comp = tabela_comparativa(resultados)

    if not df_comp.empty:
        cols_max = [c for c in ['silhouette', 'calinski', 'XGB_Accuracy_CV']
                    if c in df_comp.columns]
        cols_min = [c for c in ['davies_bouldin', 'Std_Intra']
                    if c in df_comp.columns]

        styler = df_comp.style
        if cols_max:
            styler = styler.highlight_max(subset=cols_max, color='#90EE90')
        if cols_min:
            styler = styler.highlight_min(subset=cols_min, color='#90EE90')

        st.dataframe(styler, use_container_width=True, hide_index=True)
        st.caption(
            "**Silhouette** ↑ | **Calinski-Harabász** ↑ | "
            "**Davies-Bouldin** ↓ | **XGB Accuracy** ↑ | "
            "**Std Intra** ↓"
        )

        fig_heat = plot_heatmap_metricas(df_comp)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Nenhum método produziu clusters válidos.")

    # ══════════════════════════════════════════════════════════════════════════
    # VISUALIZAÇÃO COMPARATIVA — TODOS OS MÉTODOS
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader("🗺️ Visão geral — Segmentos lineares por método")
    nomes_exec = list(resultados.keys())

    for i in range(0, len(nomes_exec), 2):
        cols_grid = st.columns(2)
        for j, col_st in enumerate(cols_grid):
            idx = i + j
            if idx >= len(nomes_exec):
                break
            nome = nomes_exec[idx]
            col_cluster = METODO_COL.get(nome, 'cluster_kmeans')
            if col_cluster in seg120.columns:
                with col_st:
                    fig = plot_clusters_rodovia(
                        seg120, col_cluster, titulo=f"{nome}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SEGMENTOS FINAIS — TODOS (após regras espaciais)
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader(f"🔧 Segmentos finais (≥ {min_comp} m) — todos os métodos")
    expand_all = len(nomes_exec) <= 4
    all_downloads = {}

    for nome in nomes_exec:
        col_cluster = METODO_COL.get(nome, 'cluster_kmeans')
        if col_cluster not in seg120.columns:
            continue

        with st.expander(f"📌 {nome}", expanded=expand_all):
            seg_final = aplicar_regras_espaciais(
                seg120.copy(), col_cluster=col_cluster,
                col_deflexao='Deflexao', min_comprimento_m=min_comp
            )
            col_final = f'{col_cluster}_espacial'

            c1, c2 = st.columns(2)
            with c1:
                fig2 = plot_clusters_rodovia(
                    seg_final, col_final,
                    titulo=f"Segmentos ≥{min_comp}m — {nome}"
                )
                st.plotly_chart(fig2, use_container_width=True)
            with c2:
                fig_m2 = plot_segmentos_mapa(seg_final, col_final)
                st.plotly_chart(fig_m2, use_container_width=True)

            tbl = seg_final.groupby(col_final).agg(
                Inicio_m=('km_ini', 'min'),
                Fim_m=('km_fim', 'max'),
                Comprimento_m=('comprimento_m', 'sum'),
                Deflexao_med=('Deflexao', 'mean'),
                Deflexao_std=('Deflexao', 'std'),
                Deflexao_caract_med=('Deflexao_caract', 'mean'),
                N_segmentos_120m=('seg_id', 'count'),
            ).reset_index()
            tbl['Deflexao_caract'] = tbl['Deflexao_med'] + tbl['Deflexao_std']
            tbl = tbl.rename(columns={col_final: 'Cluster'})
            st.dataframe(tbl, use_container_width=True, hide_index=True)

            perfil = perfil_clusters(seg120, col_cluster, vars_ok)
            if not perfil.empty:
                st.dataframe(perfil, use_container_width=True)

            all_downloads[nome] = (seg_final, tbl)

    # ── Visualização detalhada ────────────────────────────────────────────────
    st.subheader("🔍 Visualização detalhada")
    metodo_viz = st.selectbox("Método para detalhamento", nomes_exec)
    col_cluster = METODO_COL.get(metodo_viz, 'cluster_kmeans')

    if col_cluster in seg120.columns:
        viz1, viz2, viz3, viz4 = st.tabs([
            "📊 Boxplot", "📉 Perfil longitudinal", "🕸️ Radar", "🗺️ Mapa"
        ])
        with viz1:
            st.plotly_chart(plot_boxplot_clusters(seg120, col_cluster),
                            use_container_width=True)
        with viz2:
            st.plotly_chart(plot_perfil_longitudinal(seg120, col_cluster),
                            use_container_width=True)
        with viz3:
            st.plotly_chart(plot_radar_clusters(seg120, col_cluster, vars_ok),
                            use_container_width=True)
        with viz4:
            st.plotly_chart(plot_segmentos_mapa(seg120, col_cluster),
                            use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    if all_downloads:
        st.divider()
        st.subheader("⬇️ Downloads")

        # ── Excel completo (relatório robusto) ────────────────────────────
        st.markdown("**📊 Relatório Excel Completo** — inclui todas as abas, "
                    "gráficos nativos do Excel, formatação condicional e auditoria.")

        # Armazenar dados de auditoria para o Excel (se disponíveis)
        _audit_diag = st.session_state.get('audit_df_diag', None)
        _audit_gap = st.session_state.get('audit_gap_df', None)
        _audit_k = st.session_state.get('audit_k_otimo', None)
        _audit_ari = st.session_state.get('audit_ari_matrix', None)

        parametros_info = {
            'Espaçamento medições (m)': passo_m,
            'Pontos por segmento': janela,
            'Comprimento segmento (m)': comp_seg,
            'Número de clusters (k)': n_clust,
            'Comprimento mínimo (m)': min_comp,
        }

        xlsx_bytes = exportar_excel(
            seg120=seg120,
            resultados=resultados,
            df_comp=df_comp,
            all_downloads=all_downloads,
            vars_ok=vars_ok,
            METODO_COL=METODO_COL,
            df_diag=_audit_diag,
            gap_df=_audit_gap,
            k_otimo=_audit_k,
            ari_matrix=_audit_ari,
            parametros=parametros_info,
        )

        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        st.download_button(
            "📥 Baixar Relatório Excel Completo",
            data=xlsx_bytes,
            file_name=f"relatorio_segmentacao_linear_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
            key="dl_excel_completo",
        )

        st.divider()
        st.markdown("**CSVs individuais por método:**")
        for nome, (seg_f, tbl) in all_downloads.items():
            safe = nome.replace(' ', '_').replace('.', '').replace('+', '_')
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    f"Segmentos — {nome}",
                    data=seg_f.to_csv(index=False).encode('utf-8'),
                    file_name=f"segmentos_{safe}.csv",
                    mime="text/csv",
                    key=f"dl_seg_{safe}",
                )
            with c2:
                st.download_button(
                    f"Resumo — {nome}",
                    data=tbl.to_csv(index=False).encode('utf-8'),
                    file_name=f"resumo_{safe}.csv",
                    mime="text/csv",
                    key=f"dl_tbl_{safe}",
                )

# ══════════════════════════════════════════════════════════════════════════════
# ABA RELATÓRIO
# ══════════════════════════════════════════════════════════════════════════════

with tab_relatorio:
    if 'resultados' not in dir() or not resultados:
        st.info("Execute o pipeline para gerar o relatório.")
        st.stop()

    st.subheader("📄 Relatório Científico — Segmentação Linear")

    st.markdown("""
    ### Metodologia

    Este pipeline implementa segmentação homogênea de rodovias com **restrição de
    linearidade**: todos os segmentos resultantes são **contíguos espacialmente**.

    **3 categorias de métodos:**

    1. **Tradicionais (CDA, MCV, SHS)**: métodos clássicos da engenharia rodoviária
       que já produzem segmentos lineares por natureza.
    2. **Clustering + linearização**: métodos de clustering livre (K-Means, DBSCAN, etc.)
       seguidos de linearização automática — cada trecho contíguo com o mesmo label
       torna-se um segmento independente.
    3. **ML supervisionado (KNN, RF)**: usam K-Means como inicialização e depois
       refinam fronteiras com classificadores, seguido de linearização.

    **Regras espaciais (Stage III)**: segmentos curtos são fundidos ao vizinho
    que minimize a variância interna.

    ### Referências
    - **AASHTO (1993)** — Cumulative Difference Approach (CDA)
    - **Abdelaty & Jeong (2017)** — Framework de 3 estágios
    - **Mukhtarli (2020)** — SOM e avaliação extrínseca com ML
    """)

    st.markdown("### Resultados por Método")

    for nome, res in resultados.items():
        metricas = avaliar_intrinseco(res['X_scaled'], res['labels'])
        if metricas is None:
            continue

        extr = avaliar_extrinseco_xgb(res['X_scaled'], res['labels'])
        metricas_completas = {**metricas, 'XGB_Accuracy_CV': extr.get('xgb_accuracy_cv')}

        col_c = METODO_COL.get(nome)
        perfil = None
        if col_c and col_c in seg120.columns:
            perfil = perfil_clusters(seg120, col_c, vars_ok)

        with st.expander(f"📌 {nome}", expanded=False):
            texto = interpretar_resultado(nome, metricas_completas, perfil)
            st.markdown(texto)

    st.markdown("### Tabela Comparativa")
    if not df_comp.empty:
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("### Métodos Disponíveis")
    st.dataframe(TABELA_METODOS, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA AUDITORIA ESTATÍSTICA
# ══════════════════════════════════════════════════════════════════════════════

with tab_auditoria:
    if 'resultados' not in dir() or not resultados:
        st.info("Execute o pipeline para gerar a auditoria estatística.")
        st.stop()

    st.subheader("📊 Auditoria Estatística da Clusterização")
    st.markdown("""
    Valida se os clusters lineares são **estatisticamente significativos**.
    Inclui Gap Statistic, estabilidade bootstrap, testes de hipótese,
    concordância entre métodos e feature importance.
    """)

    with st.spinner("Executando auditoria completa..."):
        df_diag, gap_df, k_otimo, ari_matrix = auditoria_completa(
            resultados, seg120, vars_ok, METODO_COL,
            max_k=min(n_clust + 5, 15)
        )

    # Salvar em session_state para o export Excel acessar
    st.session_state['audit_df_diag'] = df_diag
    st.session_state['audit_gap_df'] = gap_df
    st.session_state['audit_k_otimo'] = k_otimo
    st.session_state['audit_ari_matrix'] = ari_matrix

    # ── 1. Diagnóstico consolidado ────────────────────────────────────────────
    st.subheader("1. Parecer Diagnóstico por Método")
    if not df_diag.empty:
        st.dataframe(df_diag, use_container_width=True, hide_index=True)

        aprovados = df_diag['Parecer'].str.contains('✅').sum() if 'Parecer' in df_diag.columns else 0
        aceitos = df_diag['Parecer'].str.contains('⚠️').sum() if 'Parecer' in df_diag.columns else 0
        reprovados = df_diag['Parecer'].str.contains('❌').sum() if 'Parecer' in df_diag.columns else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Aprovados", aprovados)
        c2.metric("⚠️ Aceitáveis", aceitos)
        c3.metric("❌ Reprovados", reprovados)

    # ── 2. Gap Statistic ─────────────────────────────────────────────────────
    st.subheader(f"2. Gap Statistic — k ótimo sugerido: {k_otimo}")
    import plotly.graph_objects as go
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(
        x=gap_df['k'], y=gap_df['Gap'], mode='lines+markers',
        name='Gap', error_y=dict(type='data', array=gap_df['s_k'])
    ))
    fig_gap.add_vline(x=k_otimo, line_dash='dash', line_color='red',
                      annotation_text=f'k_ótimo={k_otimo}')
    fig_gap.update_layout(title='Gap Statistic', xaxis_title='k',
                          yaxis_title='Gap(k)', template='plotly_white')
    st.plotly_chart(fig_gap, use_container_width=True)
    st.dataframe(gap_df, use_container_width=True, hide_index=True)

    if n_clust != k_otimo:
        st.warning(f"⚠️ k={n_clust} mas Gap sugere k={k_otimo}.")
    else:
        st.success(f"✅ k={n_clust} coincide com Gap ótimo.")

    # ── 3. ARI cruzado ───────────────────────────────────────────────────────
    st.subheader("3. Concordância entre Métodos (ARI)")
    import plotly.express as px
    fig_ari = px.imshow(
        ari_matrix.values,
        x=ari_matrix.columns.tolist(),
        y=ari_matrix.index.tolist(),
        text_auto='.2f', color_continuous_scale='RdYlGn',
        zmin=0, zmax=1, labels=dict(color='ARI'),
    )
    fig_ari.update_layout(title='Matriz ARI Cruzado', template='plotly_white')
    st.plotly_chart(fig_ari, use_container_width=True)

    # ── 4. Testes por método ─────────────────────────────────────────────────
    st.subheader("4. Testes Estatísticos Detalhados")
    metodo_audit = st.selectbox(
        "Método para detalhar", list(resultados.keys()), key='audit_metodo'
    )
    col_audit = METODO_COL.get(metodo_audit, 'cluster_kmeans')

    if col_audit in seg120.columns:
        a1, a2, a3, a4 = st.tabs([
            "📊 Kruskal-Wallis", "📊 ANOVA", "📉 Normalidade", "🎯 Feature Importance"
        ])
        with a1:
            kw = teste_kruskal_wallis(seg120, col_audit, vars_ok)
            if not kw.empty:
                st.dataframe(kw, use_container_width=True, hide_index=True)
            else:
                st.info("Dados insuficientes.")
        with a2:
            an = teste_anova(seg120, col_audit, vars_ok)
            if not an.empty:
                st.dataframe(an, use_container_width=True, hide_index=True)
            else:
                st.info("Dados insuficientes.")
        with a3:
            norm = teste_normalidade(seg120, col_audit, vars_ok)
            if not norm.empty:
                st.dataframe(norm, use_container_width=True, hide_index=True)
            else:
                st.info("Dados insuficientes.")
        with a4:
            res = resultados[metodo_audit]
            fi = feature_importance_xgb(res['X_scaled'], res['labels'], vars_ok)
            if not fi.empty:
                st.dataframe(fi, use_container_width=True, hide_index=True)
                fig_fi = px.bar(fi, x='Feature', y='Importância (%)',
                                title=f'Feature Importance — {metodo_audit}',
                                template='plotly_white')
                st.plotly_chart(fig_fi, use_container_width=True)

        st.subheader(f"5. Homogeneidade Intra-Cluster — {metodo_audit}")
        homo = homogeneidade_intra(seg120, col_audit, vars_ok)
        if not homo.empty:
            st.dataframe(homo, use_container_width=True, hide_index=True)
