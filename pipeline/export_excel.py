"""
Exportação robusta de resultados em Excel (.xlsx) com múltiplas abas,
gráficos embutidos, formatação condicional, cabeçalhos estilizados e
resumos estatísticos.

Usa openpyxl para gráficos nativos do Excel (abrem em qualquer leitor).
"""

import io
import numpy as np
import pandas as pd
from datetime import datetime

from openpyxl import Workbook
from openpyxl.chart import (
    BarChart, LineChart, Reference, BarChart3D, ScatterChart, Series
)
from openpyxl.chart.label import DataLabelList
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule


# ── Estilos globais ───────────────────────────────────────────────────────

FONT_HEADER = Font(name='Calibri', bold=True, size=11, color='FFFFFF')
FONT_TITLE  = Font(name='Calibri', bold=True, size=14, color='1F4E79')
FONT_SUB    = Font(name='Calibri', bold=True, size=12, color='2E75B6')
FONT_BODY   = Font(name='Calibri', size=10)
FONT_SMALL  = Font(name='Calibri', size=9, italic=True, color='808080')

FILL_HEADER = PatternFill(start_color='2E75B6', end_color='2E75B6',
                          fill_type='solid')
FILL_ALT    = PatternFill(start_color='D6E4F0', end_color='D6E4F0',
                          fill_type='solid')
FILL_GREEN  = PatternFill(start_color='C6EFCE', end_color='C6EFCE',
                          fill_type='solid')
FILL_YELLOW = PatternFill(start_color='FFEB9C', end_color='FFEB9C',
                          fill_type='solid')
FILL_RED    = PatternFill(start_color='FFC7CE', end_color='FFC7CE',
                          fill_type='solid')

ALIGN_CENTER = Alignment(horizontal='center', vertical='center',
                         wrap_text=True)
ALIGN_LEFT   = Alignment(horizontal='left', vertical='center',
                         wrap_text=True)

BORDER_THIN = Border(
    left=Side(style='thin', color='B0B0B0'),
    right=Side(style='thin', color='B0B0B0'),
    top=Side(style='thin', color='B0B0B0'),
    bottom=Side(style='thin', color='B0B0B0'),
)


def _write_df(ws, df, start_row=1, start_col=1, with_header=True,
              alternating=True):
    """Escreve DataFrame em worksheet com formatação."""
    rows = list(dataframe_to_rows(df, index=False, header=with_header))
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = ws.cell(row=start_row + r_idx,
                           column=start_col + c_idx, value=val)
            cell.font = FONT_BODY
            cell.border = BORDER_THIN
            cell.alignment = ALIGN_CENTER

            if r_idx == 0 and with_header:
                cell.font = FONT_HEADER
                cell.fill = FILL_HEADER
                cell.alignment = ALIGN_CENTER
            elif alternating and r_idx % 2 == 0 and r_idx > 0:
                cell.fill = FILL_ALT

            # Formatar números
            if isinstance(val, float):
                cell.number_format = '#,##0.000'

    # Ajustar largura das colunas
    for c_idx in range(len(df.columns)):
        col_letter = get_column_letter(start_col + c_idx)
        max_len = max(
            len(str(df.columns[c_idx])),
            df.iloc[:, c_idx].astype(str).str.len().max() if len(df) > 0 else 0
        )
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 30)

    return start_row + len(rows)


def _add_title(ws, title, row=1, col=1, merge_cols=6):
    """Adiciona título estilizado com merge."""
    ws.merge_cells(
        start_row=row, start_column=col,
        end_row=row, end_column=col + merge_cols - 1
    )
    cell = ws.cell(row=row, column=col, value=title)
    cell.font = FONT_TITLE
    cell.alignment = ALIGN_LEFT
    return row + 2


def _add_subtitle(ws, title, row=1, col=1):
    """Adiciona subtítulo."""
    cell = ws.cell(row=row, column=col, value=title)
    cell.font = FONT_SUB
    cell.alignment = ALIGN_LEFT
    return row + 1


def _add_bar_chart(ws, title, categories_col, values_cols, labels,
                   start_row, data_start_row, data_end_row,
                   anchor_cell='A1', width=18, height=12):
    """Adiciona gráfico de barras ao worksheet."""
    chart = BarChart()
    chart.type = 'col'
    chart.title = title
    chart.style = 10
    chart.width = width
    chart.height = height

    cats = Reference(ws, min_col=categories_col,
                     min_row=data_start_row + 1,
                     max_row=data_end_row)
    chart.set_categories(cats)

    for i, (col_idx, label) in enumerate(zip(values_cols, labels)):
        data = Reference(ws, min_col=col_idx,
                         min_row=data_start_row,
                         max_row=data_end_row)
        chart.add_data(data, titles_from_data=True)

    ws.add_chart(chart, anchor_cell)


def _add_line_chart(ws, title, cat_col, val_cols, labels,
                    data_start_row, data_end_row,
                    anchor_cell='A1', width=18, height=12):
    """Adiciona gráfico de linhas."""
    chart = LineChart()
    chart.title = title
    chart.style = 10
    chart.width = width
    chart.height = height

    cats = Reference(ws, min_col=cat_col,
                     min_row=data_start_row + 1,
                     max_row=data_end_row)
    chart.set_categories(cats)

    for col_idx in val_cols:
        data = Reference(ws, min_col=col_idx,
                         min_row=data_start_row,
                         max_row=data_end_row)
        chart.add_data(data, titles_from_data=True)

    ws.add_chart(chart, anchor_cell)


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL DE EXPORTAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def exportar_excel(
    seg120,
    resultados,
    df_comp,
    all_downloads,
    vars_ok,
    METODO_COL,
    df_diag=None,
    gap_df=None,
    k_otimo=None,
    ari_matrix=None,
    parametros=None,
):
    """
    Gera arquivo Excel completo com múltiplas abas:

    1. Resumo Geral       — parâmetros, data, estatísticas gerais
    2. Dados Agregados     — seg120 completo
    3. Comparação Métricas — tabela comparativa + gráficos
    4. Segmentos_{método}  — uma aba por método com segmentos finais
    5. Auditoria           — diagnóstico, Gap Statistic, ARI
    6. Perfil Longitudinal — dados para gráfico de perfil

    Retorna
    -------
    bytes : conteúdo do arquivo .xlsx pronto para download
    """
    wb = Workbook()

    # ══════════════════════════════════════════════════════════════════════
    # ABA 1 — RESUMO GERAL
    # ══════════════════════════════════════════════════════════════════════
    ws_resumo = wb.active
    ws_resumo.title = 'Resumo Geral'
    ws_resumo.sheet_properties.tabColor = '1F4E79'

    row = _add_title(ws_resumo, '🛣️ Segmentação Homogênea LINEAR — Relatório',
                     merge_cols=8)
    row = _add_subtitle(ws_resumo, 'Gerado automaticamente pelo pipeline', row)

    # Info geral
    info = {
        'Data de geração': datetime.now().strftime('%d/%m/%Y %H:%M'),
        'Total de segmentos (120m)': len(seg120),
        'Extensão total (m)': f"{seg120['km_fim'].max() - seg120['km_ini'].min():.0f}" if len(seg120) > 0 else 'N/A',
        'Deflexão média': f"{seg120['Deflexao'].mean():.2f}" if 'Deflexao' in seg120.columns else 'N/A',
        'Deflexão caract. média': f"{seg120['Deflexao_caract'].mean():.2f}" if 'Deflexao_caract' in seg120.columns else 'N/A',
        'Features usadas': ', '.join(vars_ok),
        'Métodos executados': ', '.join(resultados.keys()),
        'Total de métodos': len(resultados),
    }
    if parametros:
        info.update(parametros)

    for key, val in info.items():
        ws_resumo.cell(row=row, column=1, value=key).font = Font(bold=True, name='Calibri', size=10)
        ws_resumo.cell(row=row, column=1).border = BORDER_THIN
        ws_resumo.cell(row=row, column=2, value=str(val)).font = FONT_BODY
        ws_resumo.cell(row=row, column=2).border = BORDER_THIN
        ws_resumo.cell(row=row, column=2).alignment = ALIGN_LEFT
        row += 1

    ws_resumo.column_dimensions['A'].width = 28
    ws_resumo.column_dimensions['B'].width = 50

    # Tabela de métodos
    row += 1
    row = _add_subtitle(ws_resumo, 'Métodos Disponíveis', row)
    metodos_info = pd.DataFrame([
        {'Método': m, 'Tipo': t, 'Segmentos': len(set(r['labels']))}
        for m, r in resultados.items()
        for t in [('Tradicional' if m in ['CDA','MCV','SHS']
                    else 'Change-Point' if m == 'PELT'
                    else 'ML Supervisionado' if m in ['KNN','Random Forest']
                    else 'Clustering')]
    ])
    row = _write_df(ws_resumo, metodos_info, start_row=row)

    # ══════════════════════════════════════════════════════════════════════
    # ABA 2 — DADOS AGREGADOS
    # ══════════════════════════════════════════════════════════════════════
    ws_dados = wb.create_sheet('Dados Agregados')
    ws_dados.sheet_properties.tabColor = '548235'

    row = _add_title(ws_dados, 'Dados Agregados (segmentos)', merge_cols=len(seg120.columns))
    row = _write_df(ws_dados, seg120.round(4), start_row=row)

    # Gráfico de perfil de deflexão
    data_start = 3  # header row (row 3 = título row2 + header row)
    data_end = data_start + len(seg120)
    km_col = list(seg120.columns).index('km_ini') + 1
    defl_col = list(seg120.columns).index('Deflexao') + 1

    chart = LineChart()
    chart.title = 'Perfil Longitudinal de Deflexão'
    chart.style = 10
    chart.width = 24
    chart.height = 12
    chart.x_axis.title = 'Estação (m)'
    chart.y_axis.title = 'Deflexão (0,01mm)'

    cats = Reference(ws_dados, min_col=km_col, min_row=data_start + 1,
                     max_row=data_end)
    vals = Reference(ws_dados, min_col=defl_col, min_row=data_start,
                     max_row=data_end)
    chart.add_data(vals, titles_from_data=True)
    chart.set_categories(cats)
    chart.series[0].graphicalProperties.line.width = 15000

    ws_dados.add_chart(chart, f'{get_column_letter(len(seg120.columns) + 2)}3')

    # Deflexão caract se existir
    if 'Deflexao_caract' in seg120.columns:
        dc_col = list(seg120.columns).index('Deflexao_caract') + 1
        vals2 = Reference(ws_dados, min_col=dc_col, min_row=data_start,
                          max_row=data_end)
        chart.add_data(vals2, titles_from_data=True)

    # ══════════════════════════════════════════════════════════════════════
    # ABA 3 — COMPARAÇÃO DE MÉTRICAS
    # ══════════════════════════════════════════════════════════════════════
    if df_comp is not None and not df_comp.empty:
        ws_comp = wb.create_sheet('Comparação Métricas')
        ws_comp.sheet_properties.tabColor = 'BF8F00'

        row = _add_title(ws_comp, 'Comparação de Métricas entre Métodos',
                         merge_cols=len(df_comp.columns))
        data_start_row = row
        row = _write_df(ws_comp, df_comp.round(4), start_row=row)
        data_end_row = row - 1

        # Formatação condicional: Silhouette (mais alto = melhor)
        if 'silhouette' in df_comp.columns:
            sil_col = list(df_comp.columns).index('silhouette') + 1
            sil_letter = get_column_letter(sil_col)
            ws_comp.conditional_formatting.add(
                f'{sil_letter}{data_start_row + 1}:{sil_letter}{data_end_row}',
                ColorScaleRule(
                    start_type='min', start_color='FFC7CE',
                    mid_type='percentile', mid_value=50, mid_color='FFEB9C',
                    end_type='max', end_color='C6EFCE'
                )
            )

        # Formatação condicional: Davies-Bouldin (mais baixo = melhor)
        if 'davies_bouldin' in df_comp.columns:
            db_col = list(df_comp.columns).index('davies_bouldin') + 1
            db_letter = get_column_letter(db_col)
            ws_comp.conditional_formatting.add(
                f'{db_letter}{data_start_row + 1}:{db_letter}{data_end_row}',
                ColorScaleRule(
                    start_type='min', start_color='C6EFCE',
                    mid_type='percentile', mid_value=50, mid_color='FFEB9C',
                    end_type='max', end_color='FFC7CE'
                )
            )

        # Gráfico de barras — Silhouette por método
        metodo_col_idx = 1  # primeira coluna = método
        chart_row = data_end_row + 2

        if 'silhouette' in df_comp.columns:
            chart1 = BarChart()
            chart1.type = 'col'
            chart1.title = 'Silhouette Score por Método'
            chart1.style = 10
            chart1.width = 18
            chart1.height = 10

            cats = Reference(ws_comp, min_col=metodo_col_idx,
                             min_row=data_start_row + 1,
                             max_row=data_end_row)
            chart1.set_categories(cats)

            data = Reference(ws_comp, min_col=sil_col,
                             min_row=data_start_row,
                             max_row=data_end_row)
            chart1.add_data(data, titles_from_data=True)
            chart1.series[0].graphicalProperties.solidFill = '2E75B6'

            dl = DataLabelList()
            dl.showVal = True
            dl.numFmt = '0.000'
            chart1.series[0].dLbls = dl

            ws_comp.add_chart(chart1, f'A{chart_row}')

        # Gráfico — Calinski-Harabász
        if 'calinski' in df_comp.columns:
            cal_col = list(df_comp.columns).index('calinski') + 1
            chart2 = BarChart()
            chart2.type = 'col'
            chart2.title = 'Calinski-Harabász por Método'
            chart2.style = 10
            chart2.width = 18
            chart2.height = 10

            chart2.set_categories(cats)
            data2 = Reference(ws_comp, min_col=cal_col,
                              min_row=data_start_row,
                              max_row=data_end_row)
            chart2.add_data(data2, titles_from_data=True)
            chart2.series[0].graphicalProperties.solidFill = '548235'

            dl2 = DataLabelList()
            dl2.showVal = True
            dl2.numFmt = '#,##0.0'
            chart2.series[0].dLbls = dl2

            ws_comp.add_chart(chart2, f'J{chart_row}')

        # Gráfico — Davies-Bouldin
        if 'davies_bouldin' in df_comp.columns:
            chart3 = BarChart()
            chart3.type = 'col'
            chart3.title = 'Davies-Bouldin por Método (menor = melhor)'
            chart3.style = 10
            chart3.width = 18
            chart3.height = 10

            chart3.set_categories(cats)
            data3 = Reference(ws_comp, min_col=db_col,
                              min_row=data_start_row,
                              max_row=data_end_row)
            chart3.add_data(data3, titles_from_data=True)
            chart3.series[0].graphicalProperties.solidFill = 'BF8F00'

            dl3 = DataLabelList()
            dl3.showVal = True
            dl3.numFmt = '0.000'
            chart3.series[0].dLbls = dl3

            ws_comp.add_chart(chart3, f'A{chart_row + 16}')

        # Gráfico agrupado — todas as métricas normalizadas
        # Normalizar para 0-1 para comparação visual
        numeric_cols = df_comp.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            df_norm = df_comp.copy()
            for c in numeric_cols:
                rng = df_norm[c].max() - df_norm[c].min()
                if rng > 0:
                    df_norm[c] = (df_norm[c] - df_norm[c].min()) / rng
                else:
                    df_norm[c] = 0.5

            norm_start = data_end_row + 35
            ws_comp.cell(row=norm_start, column=1,
                         value='Métricas Normalizadas (0–1)').font = FONT_SUB
            norm_data_start = norm_start + 1
            norm_end = _write_df(ws_comp, df_norm.round(3),
                                 start_row=norm_data_start) - 1

            chart4 = BarChart()
            chart4.type = 'col'
            chart4.title = 'Métricas Normalizadas (0–1) — Comparação Visual'
            chart4.style = 10
            chart4.width = 24
            chart4.height = 12

            cats_norm = Reference(ws_comp, min_col=1,
                                  min_row=norm_data_start + 1,
                                  max_row=norm_end)
            chart4.set_categories(cats_norm)

            for col_idx in range(len(numeric_cols)):
                ref = Reference(ws_comp,
                                min_col=list(df_norm.columns).index(numeric_cols[col_idx]) + 1,
                                min_row=norm_data_start,
                                max_row=norm_end)
                chart4.add_data(ref, titles_from_data=True)

            ws_comp.add_chart(chart4, f'J{chart_row + 16}')

    # ══════════════════════════════════════════════════════════════════════
    # ABAS 4..N — SEGMENTOS POR MÉTODO
    # ══════════════════════════════════════════════════════════════════════
    for nome, (seg_final, tbl) in all_downloads.items():
        safe = nome.replace(' ', '_').replace('.', '').replace('+', '_')[:28]
        ws_met = wb.create_sheet(f'Seg_{safe}')
        ws_met.sheet_properties.tabColor = '2E75B6'

        col_cluster = METODO_COL.get(nome, 'cluster_kmeans')
        col_final = f'{col_cluster}_espacial'

        # Título
        row = _add_title(ws_met, f'Segmentos Finais — {nome}', merge_cols=10)

        # Info do método
        tipo = ('Tradicional' if nome in ['CDA', 'MCV', 'SHS']
                else 'Change-Point' if nome == 'PELT'
                else 'ML Supervisionado' if nome in ['KNN', 'Random Forest']
                else 'Clustering + Linearização')
        ws_met.cell(row=row, column=1, value='Tipo:').font = Font(bold=True, name='Calibri')
        ws_met.cell(row=row, column=2, value=tipo).font = FONT_BODY
        row += 1
        ws_met.cell(row=row, column=1, value='Segmentos:').font = Font(bold=True, name='Calibri')
        ws_met.cell(row=row, column=2, value=len(tbl)).font = FONT_BODY
        row += 1
        if 'Comprimento_m' in tbl.columns:
            ws_met.cell(row=row, column=1, value='Extensão total (m):').font = Font(bold=True, name='Calibri')
            ws_met.cell(row=row, column=2, value=f"{tbl['Comprimento_m'].sum():.0f}").font = FONT_BODY
            row += 1

        row += 1

        # Tabela resumo dos segmentos
        row = _add_subtitle(ws_met, 'Resumo dos Segmentos', row)
        tbl_start = row
        row = _write_df(ws_met, tbl.round(2), start_row=row)
        tbl_end = row - 1

        # Formatação condicional na deflexão
        if 'Deflexao_med' in tbl.columns:
            defl_idx = list(tbl.columns).index('Deflexao_med') + 1
            defl_letter = get_column_letter(defl_idx)
            ws_met.conditional_formatting.add(
                f'{defl_letter}{tbl_start + 1}:{defl_letter}{tbl_end}',
                ColorScaleRule(
                    start_type='min', start_color='C6EFCE',
                    mid_type='percentile', mid_value=50, mid_color='FFEB9C',
                    end_type='max', end_color='FFC7CE'
                )
            )

        # Gráfico de barras — Deflexão média por segmento
        if 'Deflexao_med' in tbl.columns and len(tbl) > 0:
            chart_seg = BarChart()
            chart_seg.type = 'col'
            chart_seg.title = f'Deflexão Média por Segmento — {nome}'
            chart_seg.style = 10
            chart_seg.width = 20
            chart_seg.height = 10
            chart_seg.y_axis.title = 'Deflexão (0,01mm)'

            cluster_col_idx = 1
            cats = Reference(ws_met, min_col=cluster_col_idx,
                             min_row=tbl_start + 1, max_row=tbl_end)
            chart_seg.set_categories(cats)

            data = Reference(ws_met, min_col=defl_idx,
                             min_row=tbl_start, max_row=tbl_end)
            chart_seg.add_data(data, titles_from_data=True)
            chart_seg.series[0].graphicalProperties.solidFill = '2E75B6'

            dl = DataLabelList()
            dl.showVal = True
            dl.numFmt = '0.00'
            chart_seg.series[0].dLbls = dl

            ws_met.add_chart(chart_seg, f'A{tbl_end + 2}')

        # Gráfico de comprimento dos segmentos
        if 'Comprimento_m' in tbl.columns and len(tbl) > 0:
            comp_idx = list(tbl.columns).index('Comprimento_m') + 1
            chart_comp = BarChart()
            chart_comp.type = 'col'
            chart_comp.title = f'Comprimento dos Segmentos — {nome}'
            chart_comp.style = 10
            chart_comp.width = 20
            chart_comp.height = 10
            chart_comp.y_axis.title = 'Comprimento (m)'

            chart_comp.set_categories(cats)
            data_c = Reference(ws_met, min_col=comp_idx,
                               min_row=tbl_start, max_row=tbl_end)
            chart_comp.add_data(data_c, titles_from_data=True)
            chart_comp.series[0].graphicalProperties.solidFill = '548235'

            dl_c = DataLabelList()
            dl_c.showVal = True
            dl_c.numFmt = '#,##0'
            chart_comp.series[0].dLbls = dl_c

            ws_met.add_chart(chart_comp, f'L{tbl_end + 2}')

        # Gráfico de perfil — deflexão ao longo da rodovia com cor por segmento
        if col_final in seg_final.columns and 'km_ini' in seg_final.columns:
            # Tabela auxiliar para perfil
            perf_start = tbl_end + 20
            perf_start = _add_subtitle(ws_met, 'Perfil Longitudinal', perf_start)

            perf_df = seg_final[['km_ini', 'Deflexao', col_final]].copy()
            perf_df = perf_df.rename(columns={col_final: 'Segmento'})
            perf_data_start = perf_start
            perf_end = _write_df(ws_met, perf_df.round(2),
                                 start_row=perf_start)
            perf_data_end = perf_end - 1

            chart_perf = LineChart()
            chart_perf.title = f'Perfil Longitudinal — {nome}'
            chart_perf.style = 10
            chart_perf.width = 24
            chart_perf.height = 12
            chart_perf.x_axis.title = 'Estação (m)'
            chart_perf.y_axis.title = 'Deflexão (0,01mm)'

            cats_p = Reference(ws_met, min_col=1,
                               min_row=perf_data_start + 1,
                               max_row=perf_data_end)
            vals_p = Reference(ws_met, min_col=2,
                               min_row=perf_data_start,
                               max_row=perf_data_end)
            chart_perf.add_data(vals_p, titles_from_data=True)
            chart_perf.set_categories(cats_p)

            ws_met.add_chart(chart_perf,
                             f'{get_column_letter(len(perf_df.columns) + 2)}{perf_data_start}')

        # Dados completos 120m deste método
        row_det = perf_end + 18 if 'perf_end' in dir() else tbl_end + 20
        row_det = _add_subtitle(ws_met, 'Dados Completos (120m)', row_det)

        cols_export = ['seg_id', 'km_ini', 'km_fim', 'comprimento_m',
                       'Deflexao', 'Deflexao_std', 'Deflexao_caract']
        if col_cluster in seg_final.columns:
            cols_export.append(col_cluster)
        if col_final in seg_final.columns:
            cols_export.append(col_final)
        cols_export = [c for c in cols_export if c in seg_final.columns]

        _write_df(ws_met, seg_final[cols_export].round(4), start_row=row_det)

    # ══════════════════════════════════════════════════════════════════════
    # ABA — AUDITORIA ESTATÍSTICA
    # ══════════════════════════════════════════════════════════════════════
    if df_diag is not None and not df_diag.empty:
        ws_audit = wb.create_sheet('Auditoria')
        ws_audit.sheet_properties.tabColor = 'C00000'

        row = _add_title(ws_audit, 'Auditoria Estatística da Clusterização',
                         merge_cols=len(df_diag.columns))

        # Diagnóstico
        row = _add_subtitle(ws_audit, 'Diagnóstico por Método', row)
        diag_start = row
        row = _write_df(ws_audit, df_diag, start_row=row)
        diag_end = row - 1

        # Colorir parecer
        if 'Parecer' in df_diag.columns:
            parecer_idx = list(df_diag.columns).index('Parecer') + 1
            for r in range(diag_start + 1, diag_end + 1):
                cell = ws_audit.cell(row=r, column=parecer_idx)
                val = str(cell.value) if cell.value else ''
                if '✅' in val:
                    cell.fill = FILL_GREEN
                elif '⚠️' in val or '⚠' in val:
                    cell.fill = FILL_YELLOW
                elif '❌' in val:
                    cell.fill = FILL_RED

        # Gap Statistic
        if gap_df is not None and not gap_df.empty:
            row += 2
            row = _add_subtitle(ws_audit,
                                f'Gap Statistic — k ótimo = {k_otimo}', row)
            gap_data_start = row
            row = _write_df(ws_audit, gap_df.round(4), start_row=row)
            gap_data_end = row - 1

            # Gráfico Gap
            chart_gap = LineChart()
            chart_gap.title = f'Gap Statistic (k ótimo = {k_otimo})'
            chart_gap.style = 10
            chart_gap.width = 18
            chart_gap.height = 10
            chart_gap.x_axis.title = 'k'
            chart_gap.y_axis.title = 'Gap(k)'

            k_col = list(gap_df.columns).index('k') + 1
            gap_col = list(gap_df.columns).index('Gap') + 1

            cats_g = Reference(ws_audit, min_col=k_col,
                               min_row=gap_data_start + 1,
                               max_row=gap_data_end)
            vals_g = Reference(ws_audit, min_col=gap_col,
                               min_row=gap_data_start,
                               max_row=gap_data_end)
            chart_gap.add_data(vals_g, titles_from_data=True)
            chart_gap.set_categories(cats_g)

            ws_audit.add_chart(chart_gap, f'F{gap_data_start}')

        # ARI matrix
        if ari_matrix is not None and not ari_matrix.empty:
            row += 18
            row = _add_subtitle(ws_audit, 'Concordância entre Métodos (ARI)', row)
            ari_export = ari_matrix.reset_index()
            ari_export = ari_export.rename(columns={'index': 'Método'})
            ari_start = row
            row = _write_df(ws_audit, ari_export.round(3), start_row=row)
            ari_end = row - 1

            # Formatação condicional na matriz ARI
            n_met = len(ari_matrix.columns)
            for c in range(2, n_met + 2):
                col_letter = get_column_letter(c)
                ws_audit.conditional_formatting.add(
                    f'{col_letter}{ari_start + 1}:{col_letter}{ari_end}',
                    ColorScaleRule(
                        start_type='num', start_value=0, start_color='FFC7CE',
                        mid_type='num', mid_value=0.5, mid_color='FFEB9C',
                        end_type='num', end_value=1.0, end_color='C6EFCE'
                    )
                )

    # ══════════════════════════════════════════════════════════════════════
    # ABA — ESTATÍSTICAS DESCRITIVAS
    # ══════════════════════════════════════════════════════════════════════
    ws_stats = wb.create_sheet('Estatísticas')
    ws_stats.sheet_properties.tabColor = '7030A0'

    row = _add_title(ws_stats, 'Estatísticas Descritivas', merge_cols=8)

    # Estatísticas gerais dos dados
    desc = seg120[['Deflexao', 'Deflexao_std', 'Deflexao_caract']].describe()
    desc_cols = [c for c in desc.columns if c in seg120.columns]
    if desc_cols:
        desc = desc[desc_cols].round(4)
        row = _add_subtitle(ws_stats, 'Deflexão — Estatísticas Gerais', row)
        desc_reset = desc.reset_index().rename(columns={'index': 'Estatística'})
        row = _write_df(ws_stats, desc_reset, start_row=row)

    # Estatísticas por método — resumo de segmentos
    row += 2
    row = _add_subtitle(ws_stats, 'Resumo por Método', row)

    resumo_metodos = []
    for nome, (seg_f, tbl) in all_downloads.items():
        r = {
            'Método': nome,
            'N_segmentos': len(tbl),
            'Comp_medio_m': tbl['Comprimento_m'].mean() if 'Comprimento_m' in tbl.columns else None,
            'Comp_min_m': tbl['Comprimento_m'].min() if 'Comprimento_m' in tbl.columns else None,
            'Comp_max_m': tbl['Comprimento_m'].max() if 'Comprimento_m' in tbl.columns else None,
            'Defl_media': tbl['Deflexao_med'].mean() if 'Deflexao_med' in tbl.columns else None,
            'Defl_std_media': tbl['Deflexao_std'].mean() if 'Deflexao_std' in tbl.columns else None,
        }
        resumo_metodos.append(r)

    if resumo_metodos:
        df_resumo = pd.DataFrame(resumo_metodos)
        res_start = row
        row = _write_df(ws_stats, df_resumo.round(2), start_row=row)
        res_end = row - 1

        # Gráfico — Número de segmentos por método
        chart_nseg = BarChart()
        chart_nseg.type = 'col'
        chart_nseg.title = 'Número de Segmentos por Método'
        chart_nseg.style = 10
        chart_nseg.width = 18
        chart_nseg.height = 10

        cats_r = Reference(ws_stats, min_col=1, min_row=res_start + 1,
                           max_row=res_end)
        chart_nseg.set_categories(cats_r)

        nseg_col = list(df_resumo.columns).index('N_segmentos') + 1
        data_r = Reference(ws_stats, min_col=nseg_col, min_row=res_start,
                           max_row=res_end)
        chart_nseg.add_data(data_r, titles_from_data=True)
        chart_nseg.series[0].graphicalProperties.solidFill = '7030A0'

        dl_r = DataLabelList()
        dl_r.showVal = True
        chart_nseg.series[0].dLbls = dl_r

        ws_stats.add_chart(chart_nseg, f'A{res_end + 2}')

        # Gráfico — Comprimento médio por método
        if 'Comp_medio_m' in df_resumo.columns:
            chart_comp = BarChart()
            chart_comp.type = 'col'
            chart_comp.title = 'Comprimento Médio dos Segmentos (m)'
            chart_comp.style = 10
            chart_comp.width = 18
            chart_comp.height = 10

            chart_comp.set_categories(cats_r)

            comp_col = list(df_resumo.columns).index('Comp_medio_m') + 1
            data_comp = Reference(ws_stats, min_col=comp_col,
                                  min_row=res_start, max_row=res_end)
            chart_comp.add_data(data_comp, titles_from_data=True)
            chart_comp.series[0].graphicalProperties.solidFill = '548235'

            dl_comp = DataLabelList()
            dl_comp.showVal = True
            dl_comp.numFmt = '#,##0'
            chart_comp.series[0].dLbls = dl_comp

            ws_stats.add_chart(chart_comp, f'J{res_end + 2}')

    # ══════════════════════════════════════════════════════════════════════
    # SALVAR EM BYTES
    # ══════════════════════════════════════════════════════════════════════
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()
