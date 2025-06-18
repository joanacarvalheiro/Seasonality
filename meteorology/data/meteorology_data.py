import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import plotly.express as px
import seaborn as sns
import matplotlib.dates as mdates
from dateutil import parser

#user = "avnadmin"
#password = "AVNS_9fZb3BkX9qGXxKpxsrZ"
#host = "postgresql-iscac.f.aivencloud.com"
#port = "25674"
#bucket = "Seasonality"

bucket = 'Seasonality'
user = 'postgres'
password = 'postgres'
host = 'localhost'
port = 5432


# URL de conexão com PostgreSQL
engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{bucket}')

#Nome da tabela 
meteorology ='meteorology'

def GetMeteorologyData():
    try:
    # Ler os dados da tabela no DataFrame
        df = pd.read_sql(f"SELECT * FROM {meteorology}", con=engine)   
    # Exibir as duas primeiras colunas
        return df 
    except Exception as e:
        print(f"Erro ao importar os dados: {e}")

