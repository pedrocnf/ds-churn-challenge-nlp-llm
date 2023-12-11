from flask import Flask, render_template, request
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from config import analyzed_path

app = Flask(__name__)

df = pd.read_parquet(analyzed_path)

join_text = '|'+df['Idade'].astype(str) + ' anos '+ '-' + df['Estado Civil'] + '-' + df['Tipo de Serviço'] + '-' + df['Comentários']+'-'+df['cidade']+'|'

contexto = ' '.join(join_text)

# Carregar o modelo T5
tokenizer = T5Tokenizer.from_pretrained('t5-base')
modelo = T5ForConditionalGeneration.from_pretrained('t5-base')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pergunta = request.form['pergunta']

        # Tokenizar o contexto e a pergunta
        tokens = tokenizer.encode("question: " + pergunta + " context: " + contexto, return_tensors='pt', max_length=512, truncation=True)

        # Obter a resposta do modelo
        resposta_ids = modelo.generate(tokens)
        resposta = tokenizer.decode(resposta_ids[0], skip_special_tokens=True)

        return render_template('index.html', pergunta=pergunta, resposta=resposta)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

