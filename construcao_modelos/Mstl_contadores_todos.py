import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import plotly.express as px
from sqlalchemy import create_engine
from statsmodels.tsa.seasonal import STL
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA
from utilsforecast.plotting import plot_series
from datetime import date, timedelta
import calendar
from coreforecast.scalers import boxcox_lambda, boxcox, inv_boxcox
from scipy.stats import entropy
import os
import glob
import pickle

import os
import glob
import pickle
import pandas as pd

import os
import glob
import pickle
import pandas as pd

def GetMSTL():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pasta_resultados = os.path.join(base_dir, "resultados_corrigidos")

    if not os.path.exists(pasta_resultados):
        raise FileNotFoundError(f"Pasta não encontrada: {pasta_resultados}")

    ficheiros = glob.glob(os.path.join(pasta_resultados, "resultado_count_*.pkl"))
    if len(ficheiros) == 0:
        raise FileNotFoundError("Nenhum ficheiro resultado_count_*.pkl encontrado na pasta.")

    lista_dfs_corrigidos = []
    lista_dfs_anomalias = []

    for ficheiro in ficheiros:
        nome = os.path.basename(ficheiro)
        contador_id = nome.replace("resultado_", "").replace(".pkl", "")

        with open(ficheiro, "rb") as f:
            dados = pickle.load(f)

        # Parte corrigida
        lista_resultados_corrigidos = dados.get("corrigido", [])
        if not lista_resultados_corrigidos:
            continue
        df_corrigido = pd.concat(lista_resultados_corrigidos, axis=0)
        if contador_id not in df_corrigido.columns:
            # Se não tiver a coluna com o nome do contador, tenta só manter a 1ª coluna
            df_corrigido = df_corrigido.iloc[:, [0]]
            df_corrigido.columns = [contador_id]
        else:
            df_corrigido = df_corrigido[[contador_id]]
        lista_dfs_corrigidos.append(df_corrigido)

        # Parte das anomalias
        lista_anomalias = dados.get("anomalias", [])
        if lista_anomalias:
            df_anomalias = pd.concat(lista_anomalias, ignore_index=True)
            df_anomalias["contador"] = contador_id  # identificar o contador
            lista_dfs_anomalias.append(df_anomalias)

    # Juntar tudo num único DataFrame (corrigido)
    df_corrigido_total = pd.concat(lista_dfs_corrigidos, axis=1)
    df_corrigido_total = df_corrigido_total[~df_corrigido_total.index.duplicated(keep="last")]

    # Juntar anomalias (se houver)
    if lista_dfs_anomalias:
        df_anomalias_total = pd.concat(lista_dfs_anomalias, ignore_index=True)
    else:
        df_anomalias_total = pd.DataFrame()

    return df_corrigido_total, df_anomalias_total




