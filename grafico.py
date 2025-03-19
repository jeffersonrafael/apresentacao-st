import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Análise - identificação de vídeos legendados", page_icon=":bar_chart:", layout="wide")

st.markdown(
    """
    <style>
        header {
            display: none!important;
        }
        .st-emotion-cache-1jicfl2 {
            padding: 3rem 5rem 3rem;
        }
        h1 {
            text-decoration: underline;
        }
        h3 {
            margin-top:20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns([6, 2, 2, 2])

with col1:
    st.write("# Análise - identificação de vídeos legendados")


with col4:
    st.image("LOGO.png", width=180)



st.write("## Interpretando o modelo de classificação de vídeos do tiktok")
st.image("best_tree.png")

# st.markdown(
#     """
#     <div style="text-align: center;">
#         <img src="./best_tree.png" width="800"/>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# col1, col2 = st.columns([1,4])
# with col1:
#     escolha = st.selectbox("Selecione a IA", ("GPT-4o", "Gemini-Pro", "Analista Humano"))

# #Gráfico de pizza
# col1, col2 = st.columns(2)
# with col1:
#     # Dados de exemplo
#     labels = ['Não Utilizados', 'Utilizados']
#     if escolha == "GPT-4o":
#         values = [7, 14]
#     elif escolha == "Gemini-Pro":
#         values = [9, 12]
#     else:
#         values = [12, 9]

#     # Criando o gráfico de pizza
#     fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    
#     fig.update_traces(marker=dict(colors=['gold', 'lightgreen'], 
#                               line=dict(color='#000000', width=2)))

#     fig.update_layout(title='Temas Disponíveis x Utilizados')
    
#     fig.update_layout(legend=dict(
#         orientation='v',
#         y=1,
#         xanchor='right',
#         x=0
#     ))
#     # Mostrando o gráfico
#     st.plotly_chart(fig)
    

# with col2:
#      # Dados de exemplo
#     labels = ['Utilizadas', 'Não Utilizadas']
#     if escolha == "GPT-4o":
#         values = [30, 160]
#     elif escolha == "Gemini-Pro":
#         values = [29, 161]
#     else:
#         values = [35, 155]

#     # Criando o gráfico de pizza
#     fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    
#     fig.update_traces(marker=dict(colors=['lightgreen','gold'], 
#                               line=dict(color='#000000', width=2)))

#     fig.update_layout(title='Tags Disponíveis x Utilizadas')
    
#     fig.update_layout(legend=dict(
#         orientation='v',
#         y=1,
#         xanchor='right',
#         x=1
#     ))
#     # Mostrando o gráfico
#     st.plotly_chart(fig)


# if escolha == "GPT-4o":

#     df_tags = pd.DataFrame(columns=['Redes Sociais', 'Quantidade de acertos', 'Quantidade de erros'])
#     df_tags.loc[0] = ['Instagram', 6, 13]
#     df_tags.loc[1] = ['Facebook', 4, 9]
#     df_tags.loc[2] = ['Twitter', 18, 32]
#     df_tags.loc[3] = ['Youtube', 1, 2]
#     df_tags.loc[4] = ['TikTok', 19, 28]
#     df_tags.loc[5] = ['Bluesky', 4, 10]

    

# elif escolha == "Gemini-Pro":

#     df_tags = pd.DataFrame(columns=['Redes Sociais', 'Quantidade de acertos', 'Quantidade de erros'])
#     df_tags.loc[0] = ['Instagram', 4, 13]
#     df_tags.loc[1] = ['Facebook', 5, 10]
#     df_tags.loc[2] = ['Twitter', 20, 33]
#     df_tags.loc[3] = ['Youtube', 0, 2]
#     df_tags.loc[4] = ['TikTok', 19, 27]
#     df_tags.loc[5] = ['Bluesky', 6, 7]

# else:
#     df_tags = pd.DataFrame(columns=['Redes Sociais', 'Quantidade de acertos', 'Quantidade de erros'])
#     df_tags.loc[0] = ['Instagram', 17, 0]
#     df_tags.loc[1] = ['Facebook', 15, 0]
#     df_tags.loc[2] = ['Twitter', 53, 0]
#     df_tags.loc[3] = ['Youtube', 2, 0]
#     df_tags.loc[4] = ['TikTok', 46, 0]
#     df_tags.loc[5] = ['Bluesky', 13, 0]


# fig2 = go.Figure(
#         data=[
#             go.Bar(x=df_tags['Redes Sociais'], y=df_tags['Quantidade de acertos'], name="Acertos"),
#             go.Bar(x=df_tags['Redes Sociais'], y=df_tags['Quantidade de erros'], name="Erros"),
#         ],
#         layout=dict(
#             # barcornerradius=15,
#             title='Quantidade de acertos e erros por rede social [Tags]',
#             colorway=['#90EE90','#FFD700'],
#         ),
#     )

# fig2.update_traces(marker_line_width=1, marker_line_color='black')

# st.plotly_chart(fig2)
#Principais acertos
#EXP - Situação de consumo
#EXP - Desejo de consumo
#COMP - Comparação de concorrente


#principal erro
#LIQ - Sabor
#LIQ - Qualidade genérica






# GRAFICO FEATURE IMPORTANCE
st.write("## Relevância das Features")
# Cria um DataFrame com as features e suas importâncias
df_importance = pd.read_csv("grafico_importance.csv")
# print(df_importance.columns)
# Remove a coluna "Unnamed: 0", se existir
if "Unnamed: 0" in df_importance.columns:
    df_importance.drop(columns=["Unnamed: 0"], inplace=True)

# Cria o gráfico de barras com Plotly Express
fig3 = px.bar(
    df_importance,
    y="Feature",
    x="Importance",
    # title="Importância das Features (Random Forest)",
    color="Importance",
    color_continuous_scale='Blues',
    orientation='h'
)

# Atualiza o layout para ajustar rótulos e tamanho da figura
fig3.update_layout(
    title=dict(
        text="Importância das Features (Random Forest)",
        x=0.3,           # centraliza o título
        font=dict(size=36)
    ),
    xaxis_title="Importância",
    yaxis_title="Features",
    # xaxis_tickangle=-90,
    width=500,
    height=600
)

st.plotly_chart(fig3)








# col1,col2,col3 = st.columns([1,2,1])
# with col2:
#     # TABELA
#     fig = go.Figure(data=[go.Table(
#         header=dict(values=['<b>Tags</b>', '<b>Acertos</b>', '<b>Erros</b>'],
#                     line_color='darkslategray',
#                     fill_color='black',
#                     align='left',
#                     font=dict(color='white', size=16)               
#         ),
#         cells=dict(values=[['EXP - Desejo de consumo', 'EXP - Situação de consumo', 'COMP - Comparação de concorrente', 'HMA - Outras bebidas'], # 1st column
#                         [24, 7, 15, 0],
#                         [5, 16, 3, 25]                      
#                         ],
#                 line_color='darkslategray',
#                 fill_color='black',
#                 align='left',
#                 font=dict(size=15),
#                 height=30
#         ),
#                 columnwidth = [150, 50, 50],
#     )])

#     fig.update_layout(width=900, height=400, title='Tabela de acertos e erros das principais tags')

#     st.plotly_chart(fig)

#Acuracy
st.write("## Acuracidade")
# metodos = ['Analista Humano', 'GPT-4o', 'Gemini-Pro']
# acuracidade = [100, 33, 37]  # 100% para o analista humano e 40% para o GPT

# # Criar gráfico interativo de barras
# fig = go.Figure([go.Bar(x=metodos, y=acuracidade, text=acuracidade, textposition='auto', marker_color=['white', 'green', 'blue'])])

# # Título e rótulos
# fig.update_layout(
#     title='Comparação de Acuracidade: Analista Humano vs IAs',
#     xaxis_title='',
#     yaxis_title='Acuracidade (%)',
#     template='plotly_white'
# )

# # Exibir gráfico com Streamlit
# st.plotly_chart(fig)



# Gerando a matriz de confusão
dados_matrix = pd.read_csv("confusion_matrix.csv")

if "Unnamed: 0" in dados_matrix.columns:
    dados_matrix.drop(columns=["Unnamed: 0"], inplace=True)

cm = confusion_matrix(dados_matrix['y_teste'], dados_matrix['pred'], labels=dados_matrix['y_teste'].unique())

# Cria o heatmap com Plotly Express
fig4 = px.imshow(
    cm,
    text_auto=True,  # Exibe os valores dentro de cada célula
    labels=dict(x="Previsto", y="Real", color="Contagem"),
    x=['Não', 'Sim'],  # Define os rótulos do eixo x
    y=['Não', 'Sim'],  # Define os rótulos do eixo y
    color_continuous_scale="Blues"
)

# Atualiza o layout para ajustar título e dimensões
fig4.update_layout(
    title=dict(
        text="Acertos e Erros do modelo",
        x=0.3,           # centraliza o título
        font=dict(size=36)
    ),
    width=600,
    height=500
)

st.plotly_chart(fig4)




col1, col2 = st.columns(2)
with col1:
    #Conclusão
    st.write("## Conclusão do estudo")    
    st.write("""
    O modelo apresentou um alto desempenho, alcançando uma acurácia de 98%, demonstrando sua capacidade eficiente de classificar vídeos com base na presença e sincronização das legendas. Esse resultado indica que o modelo foi altamente preciso na diferenciação entre vídeos com e sem legendas corretamente sincronizadas.

    A característica mais importante para a tomada de decisão foi a presença de texto na tela que transcreve exatamente o que está sendo dito. Esse critério foi determinante para classificar um vídeo como contendo legendas sincronizadas. Caso o texto exibido não correspondesse fielmente ao áudio, o vídeo era classificado como sem legendas sincronizadas, independentemente de outros fatores.

    Outro fator essencial para a decisão foi a sincronização do texto com a fala. Se o texto não aparecesse de forma sincronizada, o modelo classificava o vídeo como sem legendas. No entanto, se o texto estivesse alinhado com o áudio, a decisão final dependia da exatidão da transcrição. Com essa abordagem estruturada, o modelo demonstrou uma alta capacidade de tomada de decisão baseada em critérios bem definidos, garantindo uma classificação robusta e confiável dos vídeos.
    """)

with col2:
    st.write("## Problemas encontrados")  
    st.write("""
    Os problemas encontrados na base de dados:
    - A base de dados possui uma amostra pequena, o que pode limitar a capacidade de generalização do modelo e aumentar o risco de overfitting, sendo recomendável a aquisição de mais dados para aprimorar a precisão e a confiabilidade dos resultados. 
    """)