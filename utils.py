import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from translate import Translator
from flair.data import Sentence
from flair.models import TextClassifier
from textblob import TextBlob
import seaborn as sns

def count_na(df):
    return df.isna().sum()

# Normality test to detect if any column of a dataframe have normal-distribution
def normality_test(df, colunas):
    results = []
    for col in colunas:  # Exclua a coluna 'class'
        stat, p_value = stats.shapiro(df[col])
        is_normal = p_value > 0.05  # Verifica se os dados são considerados normais (p-valor > 0.05)
        results.append([col, p_value, is_normal])
    
    return pd.DataFrame(results, columns=["Coluna", "p_valor", "Normal?"])

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

# t-student test for differences:
def differences_test_num(grupo1, grupo2, colunas):
    results = []
    for col in colunas:  # Exclua a coluna 'class'
        stat, p_value = stats.ttest_ind(grupo1[col], grupo2[col])
        is_different = p_value > 0.05  # Verifica se os dados são considerados normais (p-valor > 0.05)
        results.append([col, p_value, is_different])
    
    return pd.DataFrame(results, columns=["Coluna", "p_valor", "Diferente?"])

def differences_test_cat(df, colunas, binario):

    resultados_chi2 = []

    # Iterar sobre as variáveis categóricas
    for variavel in df[colunas].columns:  # Excluímos a última coluna (Churn)
        # Criar uma tabela de contingência
        tabela_contingencia = pd.crosstab(df[variavel], df[binario])
        
        # Realizar o teste qui-quadrado
        chi2, p_valor, _, _ = chi2_contingency(tabela_contingencia)
        
        # Armazenar os resultados
        resultados_chi2.append({
            'Variavel_Categorica': variavel,
            'Estatistica_Chi2': chi2,
            'Valor_p': p_valor,
            'Significativo (0.05)?': "Sim" if p_valor < 0.05 else "Não"
        })

        df_resultados = pd.DataFrame(resultados_chi2)

    return df_resultados
    


    for resultado in resultados_chi2:
        print(f"Variável Categórica: {resultado['Variavel_Categorica']}")
        print(f"Estatística qui-quadrado: {resultado['Estatistica_Chi2']}")
        print(f"Valor p: {resultado['Valor_p']}")
        print(f"Significativo (0.05)? : {resultado['Significativo (0.05)?']}")
        print('\n')

def translate_to_english(comment):
    translator = Translator(provider='mymemory', to_lang='en', from_lang='pt')
    translation = translator.translate(comment)
    return translation

def predict_sentiment(comment):
    classifier = TextClassifier.load('sentiment')
    sentence = Sentence(comment)
    classifier.predict(sentence)
    predicted_sentiment = sentence.labels[0].value
    sentiment_score = sentence.labels[0].score
    return predicted_sentiment, sentiment_score

def textblob_sentiment_polarity(comment):
    analysis = TextBlob(comment)
    sentiment_score = analysis.sentiment.polarity
    return sentiment_score

def textblob_classify_polarity(sentiment):
    if sentiment > 0:
        return 'POSITIVE'
    elif sentiment < 0:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def textblob_subjectivity(comment):
    analysis = TextBlob(comment)
    subjectivity = analysis.sentiment.subjectivity
    return subjectivity

def bar_plot(binary, column, df):
    # Calcular percentuais
    tabela_contingencia = pd.crosstab(df[column], df[binary], margins=True, margins_name='Total')

    tabela_percentual = tabela_contingencia.div(tabela_contingencia['Total'], axis=0) * 100

    # Plotar um gráfico empilhado com percentuais
    ax = tabela_percentual.drop('Total', axis=1).plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))

    # Adicionar rótulos e título
    plt.title(f'Percentual de Comentários Positivos e Negativos por {column}')
    plt.xlabel(f'{column}')
    plt.ylabel('Percentual de Comentários')
    plt.legend(title='Sentimento', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adicionar percentuais nas barras
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2), ha='center', va='center')

    # Remover rótulo 'Total'
    plt.xticks(ticks=range(len(tabela_percentual.index)), labels=tabela_percentual.index, rotation=0)
    #ax.get_legend().remove()

    # Exibir o gráfico
    plt.show()

def boxplot_plot(binary, column, df):

    # Filtrar comentários negativos
    df_negativos = df[df['flair_sentiment_analysis'] == 'NEGATIVE']
    # Filtrar comentários positivos
    df_positivos = df[df['flair_sentiment_analysis'] == 'POSITIVE']

    # Criar subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), sharey=True)

    # Boxplot para comentários negativos
    sns.boxplot(ax=axes[0], x=binary, y=column, data=df_negativos, palette='viridis')
    axes[0].set_title('Comentários Negativos')
    axes[0].set_xlabel('Sentimento')
    axes[0].set_ylabel(f'{column}')

    # Boxplot para comentários positivos
    sns.boxplot(ax=axes[1], x=binary, y=column, data=df_positivos, palette='viridis')
    axes[1].set_title('Comentários Positivos')
    axes[1].set_xlabel('Sentimento')
    axes[1].set_ylabel(f'{column}')

    # Exibir os subplots
    plt.tight_layout()
    plt.show()

def simple_bar_plot(df, column):

    plt.figure(figsize=(6, 4))

    count_values = df[column].value_counts()

    ax = count_values.plot(kind='bar', color='skyblue', edgecolor='black')

    total = count_values.sum()
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height/total:.1%}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.title(f'{column} dos clientes')
    plt.xlabel(f'{column}')
    plt.ylabel('Contagem')

    plt.show()