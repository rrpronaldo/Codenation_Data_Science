import streamlit as st
import pydeck as pdk

import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

import random

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

def create_map_deck(leads):
  leads = leads.head(30)
  vermelho = "https://img.icons8.com/plasticine/100/000000/marker.png"
  azul = "https://img.icons8.com/ultraviolet/40/000000/marker.png"

  icon_data = {
    "url": azul,
    "width": 128,
    "height":128,
    "anchorY": 128
  }
 
  leads['icon_data']= None
  for i in leads.index:
    leads['lat'][i] = leads['lat'][i] + random.uniform(-0.1, 0.1)
    leads['lng'][i] = leads['lng'][i] + random.uniform(-0.1, 0.1)
    leads['icon_data'][i] = icon_data
  
  for i in range(0, leads.shape[0], 5):
    leads['icon_data'][i] = {"url": vermelho,"width": 128,"height":128,"anchorY": 128}

  icon_layer = pdk.Layer(
      type='IconLayer',
      data=leads,
      get_position='[lng, lat]',
      get_icon='icon_data',
      get_size=4,
      pickable=True,
      size_scale=15
  )
  
  mapa = pdk.Deck(
      map_style='mapbox://styles/mapbox/light-v9',
          initial_view_state=pdk.ViewState(
              latitude=-16.1237611,
              longitude=-59.9219642,
              zoom=3,
          ),
          layers=[icon_layer]
      )
  return mapa

def main():
    st.image('logo.png', width=400)
    st.title('AceleraDev Data Science - Projeto Final')
    st.subheader('Ronaldo Regis Posser - Sistema de recomendação de novos clientes')
    
    
    label_enc = LabelEncoder()

    base = pd.read_csv('base.zip')
    base['id'] = label_enc.fit_transform(base['id'])
    
    st.subheader('Visualização base de dados')

    st.markdown('A base possui ' + str(base.shape[0]) + ' empresas, com ' + str(base.shape[1]) + ' variáveis que diferenciam as empresas por faturamento, ramo de atividade e localização.')
    st.dataframe(base.head())

    
    base.groupby('sg_uf')['setor'].apply(pd.Series.value_counts).unstack().plot.bar(figsize = (10,5))
    plt.title('Distribuição dos setores nos estados')
    plt.xticks(rotation='horizontal')
    plt.xlabel('Estados')
    st.pyplot()
    
    dict_porte = {'DE R$ 1.500.000,01 A R$ 4.800.000,00' : 'Pequena',
       'DE R$ 81.000,01 A R$ 360.000,00': 'Micro',
       'ATE R$ 81.000,00': 'Micro',
       'SEM INFORMACAO': 'Micro',
       'DE R$ 360.000,01 A R$ 1.500.000,00': 'Pequena',
       'DE R$ 10.000.000,01 A R$ 30.000.000,00': 'Média',
       'DE R$ 4.800.000,01 A R$ 10.000.000,00': 'Média',
       'DE R$ 30.000.000,01 A R$ 100.000.000,00': 'Grande',
       'DE R$ 300.000.000,01 A R$ 500.000.000,00': 'Grande',
       'DE R$ 100.000.000,01 A R$ 300.000.000,00': 'Grande',
       'ACIMA DE 1 BILHAO DE REAIS': 'Grande',
       'DE R$ 500.000.000,01 A 1 BILHAO DE REAIS': 'Grande'}
    
    base_aux = base[['setor', 'de_faixa_faturamento_estimado']]
    base_aux['porte'] = base_aux['de_faixa_faturamento_estimado'].map(dict_porte)

    base_aux.groupby('setor')['porte'].apply(pd.Series.value_counts).unstack().plot.bar(figsize = (10,5), log=True)
    plt.title('Distribuição dos portes de empresa por setores')
    plt.xticks(rotation='horizontal')
    plt.xlabel('Estados')
    st.pyplot()

    #Engenharia de features
    base.set_index(['id'], inplace=True)

    features_cat = ['sg_uf',	'setor', 'idade_emp_cat', 'nm_divisao', 'de_saude_tributaria',
                'de_saude_rescencia', 'de_nivel_atividade', 'nm_meso_regiao',
                'nm_micro_regiao', 'de_faixa_faturamento_estimado']
    base_dummies = pd.get_dummies(base, columns=features_cat)

    #Treinamento modelo
    qtd_neighbors = 5
    model = NearestNeighbors(n_neighbors=qtd_neighbors, metric = 'cosine')
    model.fit(base_dummies)

    #Gerando sugestões com base em uma portfólio
    st.subheader('Recomendação de novos clientes')
    st.markdown('Escolha o arquivo com o portfólio que deseja analisar (.csv)')
    file  = st.file_uploader(' ',type = 'csv')
    
    if file is not None:
        portfolio = pd.read_csv(file)
        portfolio['id'] = label_enc.transform(portfolio['id'])
        portfolio.set_index(['id'], inplace=True)
        portfolio = base.loc[portfolio.index.to_list()]

        st.markdown('A portfólio possui ' + str(portfolio.shape[0]) + ' empresas.')
        
        #Visualização dados Portfólio
        portfolio.groupby('sg_uf')['setor'].apply(pd.Series.value_counts).unstack().plot.bar(figsize = (10,5))
        plt.title('Portfólio - Distribuição dos setores nos estados')
        plt.xticks(rotation='horizontal')
        plt.xlabel('Estados')
        st.pyplot()

        port_aux = portfolio[['setor', 'de_faixa_faturamento_estimado']]
        port_aux['porte'] = port_aux['de_faixa_faturamento_estimado'].map(dict_porte)

        port_aux.groupby('setor')['porte'].apply(pd.Series.value_counts).unstack().plot.bar(figsize = (10,5), log=True)
        plt.title('Portfólio - Distribuição dos portes de empresa por setores')
        plt.xticks(rotation='horizontal')
        plt.xlabel('Estados')
        st.pyplot()

        portfolio_train = base_dummies.loc[portfolio.index.to_list()]

        previsao_port = model.kneighbors(portfolio_train, return_distance=False)

        leads_port = base.iloc[previsao_port.reshape(-1)]

        leads_port.reset_index(inplace=True)

        st.markdown('Primeiras 20 sugestões geradas pelo modelo.\n As marcadas em azul são clientes atuais e as quatro seguintes as sugestões de novos clientes.')
        st.dataframe(leads_port.head(20).style.apply(lambda x: ['background: #5cade2' if (x.name % 5 == 0) else ' ' for i in x], 
                    axis=1))

        
        #Adicionando Latitude e Longitude ao conjunto de dados previstos para visualização no mapa
        lat_long = pd.read_csv('lat_long_micro.csv')
        leads_port['lat'] = leads_port['nm_micro_regiao'].apply(lambda micro: lat_long[lat_long['nm_micro'] == micro]['lat'].values[0])
        leads_port['lng'] = leads_port['nm_micro_regiao'].apply(lambda micro: lat_long[lat_long['nm_micro'] == micro]['lng'].values[0])
        leads_port.reset_index(inplace=True)

        st.markdown('Mapa 30 primeiras recomendações')
        st.markdown('Marcadores vermelhos são atuais clientes. Marcadores azuis são recomendações.')
        st.pydeck_chart(create_map_deck(leads_port))
        

        

if __name__ == '__main__':
    main()
