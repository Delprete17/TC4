import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Triagem Preventiva de Obesidade", layout="centered")
st.title("Sistema de Triagem Preventiva")
st.write("Insira os hábitos do paciente para prever a tendência de desenvolvimento de obesidade.")


# CLASSES DO PIPELINE

class RemoverAlturaPeso(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop=['Weight', 'Height']):
        self.features_to_drop = features_to_drop
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        cols_present = [col for col in self.features_to_drop if col in X_copy.columns]
        return X_copy.drop(columns=cols_present)


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_mappings = {
            'Gender': {'Masculino': 0, 'Feminino': 1, 'Male': 0, 'Female': 1},
            'family_history': {'Não': 0, 'Sim': 1, 'no': 0, 'yes': 1, 'No': 0},
            'FAVC': {'Não': 0, 'Sim': 1, 'no': 0, 'yes': 1, 'No': 0},
            'SMOKE': {'Não': 0, 'Sim': 1, 'no': 0, 'yes': 1, 'No': 0},
            'SCC': {'Não': 0, 'Sim': 1, 'no': 0, 'yes': 1, 'No': 0}
        }
        self.ordinal_scale = {
            'Não': 0, 'Às vezes': 1, 'Frequentemente': 2, 'Sempre': 3,
            'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3,
            'No': 0
        }
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        
        for col, mapping in self.binary_mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping).fillna(0).astype(int)
                
        for col in ['CAEC', 'CALC']:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(self.ordinal_scale).fillna(0).astype(int)
                
        return X_copy


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categories = ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        
        if 'MTRANS' in X_copy.columns:
            for cat in self.categories[1:]:
                col_name = f'MTRANS_{cat}'
                X_copy[col_name] = (X_copy['MTRANS'] == cat).astype(int)
            X_copy = X_copy.drop(columns=['MTRANS'])
            
        return X_copy


# CARREGAMENTO DOS ARTEFATOS E DADOS 

@st.cache_resource
def load_model():
    return joblib.load('modelo_obesidade.pkl')

modelo_pipeline = load_model()

@st.cache_data
def load_data():
    return pd.read_csv('Obesity.csv')

df = load_data()

# EXTRAÇÃO DE IMPORTÂNCIA DAS VARIÁVEIS VIA PIPELINE
def get_feature_importances(pipeline):
    try:
        model = pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            feature_names = [
                'Gender', 'Age', 'family_history', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE',
             'CH2O', 'SCC', 'FAF', 'TUE', 'CALC',
             'MTRANS_Walking', 'MTRANS_Automobile', 'MTRANS_Motorbike', 'MTRANS_Bike'
            ]
            
            if len(importances) == len(feature_names):
                return pd.DataFrame({'feature': feature_names, 'importance': importances})
    except:
        pass
        
    df_corr = df.copy()
    if 'Obesity' in df_corr.columns:
        df_corr['ObesityNumeric'] = df_corr['Obesity'].astype('category').cat.codes
        numeric = df_corr.select_dtypes(include=['number']).drop(columns=['ObesityNumeric'], errors='ignore')
        corr = numeric.corrwith(df_corr['ObesityNumeric']).abs().sort_values(ascending=False)
        return corr.reset_index().rename(columns={'index': 'feature', 0: 'importance'})
    return None

# NAVEGAÇÃO
aba = st.sidebar.selectbox("Escolha a página", ["Formulário", "Painel Analítico"])

if aba == "Formulário":
    st.header("Previsão Individual")

    st.subheader("Dados Demográficos")
    gender = st.selectbox("Gênero", ['Feminino', 'Masculino'])
    age = st.slider("Idade", 14, 65, 25)
    family_history = st.selectbox("Histórico Familiar de Obesidade?", ['Sim', 'Não'])

    st.subheader("Hábitos Alimentares")
    favc = st.selectbox("Consome alimentos calóricos com frequência (FAVC)?", ['Sim', 'Não'])
    fcvc = st.slider("Frequência de consumo de vegetais (1 = Nunca, 3 = Sempre)", 1, 3, 2)
    ncp = st.slider("Número de refeições principais por dia", 1, 4, 3)
    caec = st.selectbox("Come entre as refeições (CAEC)?", ['Não', 'Às vezes', 'Frequentemente', 'Sempre'])

    st.subheader("Estilo de Vida")
    smoke = st.selectbox("Fumante?", ['Sim', 'Não'])
    ch2o = st.slider("Consumo de água diário (Litros: 1 a 3)", 1, 3, 2)
    scc = st.selectbox("Monitora as calorias que consome (SCC)?", ['Sim', 'Não'])
    faf = st.slider("Dias de atividade física por semana (0 a 3)", 0, 3, 1)
    tue = st.slider("Horas de uso de eletrônicos por dia (0 a 2)", 0, 2, 1)
    calc = st.selectbox("Consumo de álcool (CALC)?", ['Não', 'Às vezes', 'Frequentemente', 'Sempre'])
    mtrans = st.selectbox("Principal meio de transporte", ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'])

    if st.button("Gerar Diagnóstico Preventivo"):
        dados_input = pd.DataFrame([{
            'Gender': gender, 
            'Age': age, 
            'family_history': family_history,
            'FAVC': favc, 
            'FCVC': fcvc, 
            'NCP': ncp, 
            'CAEC': caec, 
            'SMOKE': smoke,
            'CH2O': ch2o, 
            'SCC': scc, 
            'FAF': faf, 
            'TUE': tue, 
            'CALC': calc, 
            'MTRANS': mtrans
        }])

        predicao = modelo_pipeline.predict(dados_input)[0]
        st.success(f"**Diagnóstico Previsto:** O paciente apresenta perfil comportamental compatível com **{predicao}**.")

else:
    st.header("Painel Analítico de Saúde Populacional")
    st.write("Visão macro e epidemiológica dos fatores de risco de obesidade no hospital.")

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Pacientes", len(df))
    col2.metric("Idade Média", f"{df['Age'].mean():.1f} anos")
    col3.metric("% com Histórico Familiar", f"{(df['family_history'] == 'yes').mean()*100:.1f}%")

    st.markdown("---")

    # --- ANÁLISE DE IMPORTÂNCIA ---
    st.subheader("Análise de Relevância Estatística")
    importance_df = get_feature_importances(modelo_pipeline)
    if importance_df is not None:
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        
        fig_imp = px.bar(
            importance_df.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Variáveis por Relevância Estatística",
            labels={"importance": "Peso do Fator", "feature": "Fator Clínico"},
            color_discrete_sequence=["#084594"]
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.write("O modelo não expõe importâncias diretas.")

    st.markdown("---")
    st.subheader("Painel de Diagnóstico Epidemiológico")
    st.write("Distribuição proporcional e comportamento dos fatores de risco por classe de peso.")

    df_painel = df.copy()
    
    df_painel['Consumo de Vegetais'] = df_painel['FCVC'].round().map({
        1: '1. Raramente', 
        2: '2. Às vezes', 
        3: '3. Sempre'
    })
    
    df_painel['Exercício Semanal'] = df_painel['FAF'].round().map({
        0: '0. Inexistente', 
        1: '1. 1-2 dias/sem', 
        2: '2. 3-4 dias/sem', 
        3: '3. 5+ dias/sem'
    })

    ordem_classes = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]

    # --- GRÁFICO 1 ---
    st.markdown("#### Fator 1: Distribuição de Idade por Categoria")
    fig1 = px.box(
        df_painel, x="Age", y="Obesity",
        category_orders={"Obesity": ordem_classes},
        color_discrete_sequence=["#084594"],
        title="Dispersão de Idade (Mediana e Quartis) por Nível de Obesidade"
    )
    fig1.update_layout(yaxis_title="Categoria de Obesidade", xaxis_title="Idade (Anos)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # --- GRÁFICO 2 ---
    st.markdown("#### Fator 2: Histórico Familiar de Obesidade")
    fig2 = px.histogram(
        df_painel, y="Obesity", color="family_history",
        barmode="relative", barnorm="percent",
        orientation="h",
        category_orders={"Obesity": ordem_classes, "family_history": ['no', 'yes']},
        color_discrete_sequence=["#2c7bb6", "#fdae61"],
        text_auto='.1f',
        title="Proporção do Histórico Familiar de Obesidade dentro de cada Categoria"
    )
    fig2.update_layout(yaxis_title="Categoria de Obesidade", xaxis_title="Porcentagem (%)", template="plotly_white", legend_title="Histórico Familiar")
    fig2.update_traces(textposition="inside", insidetextanchor="middle")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # --- GRÁFICO 3 ---
    st.markdown("#### Fator 3: Atividade Física (FAF)")
    fig3 = px.histogram(
        df_painel, y="Obesity", color="Exercício Semanal",
        barmode="relative", barnorm="percent",
        orientation="h",
        category_orders={"Obesity": ordem_classes, "Exercício Semanal": ['0. Inexistente', '1. 1-2 dias/sem', '2. 3-4 dias/sem', '3. 5+ dias/sem']},
        color_discrete_sequence=["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"],
        text_auto='.1f',
        title="Proporção de Atividade Física dentro de cada Categoria"
    )
    fig3.update_layout(yaxis_title="Categoria de Obesidade", xaxis_title="Porcentagem (%)", template="plotly_white", legend_title="Exercícios")
    fig3.update_traces(textposition="inside", insidetextanchor="middle")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # --- GRÁFICO 4 ---
    st.markdown("#### Fator 4: Consumo de Vegetais (FCVC)")
    fig4 = px.histogram(
        df_painel, y="Obesity", color="Consumo de Vegetais",
        barmode="relative", barnorm="percent",
        orientation="h",
        category_orders={"Obesity": ordem_classes, "Consumo de Vegetais": ['1. Raramente', '2. Às vezes', '3. Sempre']},
        color_discrete_sequence=["#d7191c", "#fdae61", "#2c7bb6"],
        text_auto='.1f',
        title="Proporção do Consumo de Vegetais dentro de cada Categoria"
    )
    fig4.update_layout(yaxis_title="Categoria de Obesidade", xaxis_title="Porcentagem (%)", template="plotly_white", legend_title="Vegetais")
    fig4.update_traces(textposition="inside", insidetextanchor="middle")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")