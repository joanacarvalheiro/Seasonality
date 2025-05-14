import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Carregar os dados
df = pd.read_csv('../all_bike_counts.csv', parse_dates=['detected'], dtype='int32')
df = df.set_index('detected')
df_loc = pd.read_csv('../all_counter_locations.csv')

st.title("Análise de Contagem de Bicicletas")

# --- Colocar controles na barra lateral ---
with st.sidebar:
    st.header("Controles")
    
    # Seletor de Location ID
    location_ids = [col.replace("count_", "") for col in df.columns if col.startswith("count_")]
    locationId_select = st.selectbox("Seleciona o contador (locationId):", location_ids)
    count_col = f"count_{locationId_select}"

    # Seletor de Intervalo de Datas
    start_date = st.date_input("Data de início", df.index.min().date())
    end_date = st.date_input("Data de fim", df.index.max().date())

# Filtrar com base no intervalo de datas escolhido
df_filtered = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

df_filtered_daily = df_filtered.resample('D').sum()

# Cronograma
if st.sidebar.checkbox("Mostrar gráfico de linha do contador selecionado"):
    fig_line = go.Figure()

    # Filtra os dados para o contador selecionado
    df_line_resampled = df_filtered_daily[[count_col]].dropna()

    # Adiciona a linha do contador selecionado
    fig_line.add_trace(go.Scatter(
        x=df_line_resampled.index,
        y=df_line_resampled[count_col],
        mode='lines',
        name=f'Contador {locationId_select}'
    ))

    fig_line.update_layout(
        title=f'Contador {locationId_select}',
        xaxis_title='Data',
        yaxis_title='Contagem',
        template='plotly_dark'
    )

    st.plotly_chart(fig_line, use_container_width=True)

# --- Processar colunas temporais ---
df_filtered_daily['day_of_week'] = df_filtered_daily.index.dayofweek
df_filtered_daily['month'] = df_filtered_daily.index.month

def get_season(month):
    if month in [3, 4, 5]:
        return 'Primavera'
    elif month in [6, 7, 8]:
        return 'Verão'
    elif month in [9, 10, 11]:
        return 'Outono'
    else:
        return 'Inverno'

# Certifique-se de que a coluna 'season' está sendo calculada corretamente
df_filtered_daily['season'] = df_filtered_daily['month'].apply(get_season)

# --- Boxplot por Dia da Semana ---
if st.sidebar.checkbox("Mostrar Boxplot por Dia da Semana"):
    fig_day_of_week = go.Figure()

    dias_semana = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}

    for day in sorted(df_filtered_daily['day_of_week'].unique()):
        fig_day_of_week.add_trace(go.Box(
            y=df_filtered_daily[df_filtered_daily['day_of_week'] == day][count_col],
            name=dias_semana[day],
            boxmean=True
        ))

    fig_day_of_week.update_layout(
        title=f'Boxplot por Dia da Semana - Contador {locationId_select}',
        xaxis_title='Dia da Semana',
        yaxis_title='Contagem',
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_day_of_week, use_container_width=True)

# --- Boxplot por Mês ---
if st.sidebar.checkbox("Mostrar Boxplot por Mês"):
    fig_month = go.Figure()

    meses = {1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Junho', 7: 'Julho', 
             8: 'Agosto', 9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'}

    for month in df_filtered_daily['month'].unique():
        fig_month.add_trace(go.Box(
            y=df_filtered_daily[df_filtered_daily['month'] == month][count_col],
            name=meses[month],
            boxmean=True
        ))

    fig_month.update_layout(
        title=f'Boxplot por Mês - Contador {locationId_select}',
        xaxis_title='Mês',
        yaxis_title='Contagem',
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_month, use_container_width=True)

# --- Boxplot por Estação do Ano ---
if st.sidebar.checkbox("Mostrar Boxplot por Estação do Ano"):
    fig_season = go.Figure()

    for season in df_filtered_daily['season'].unique():
        fig_season.add_trace(go.Box(
            y=df_filtered_daily[df_filtered_daily['season'] == season][count_col],
            name=season,
            boxmean=True
        ))

    fig_season.update_layout(
        title=f'Boxplot por Estação do Ano - Contador {locationId_select}',
        xaxis_title='Estação do Ano',
        yaxis_title='Contagem',
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_season, use_container_width=True)
    










