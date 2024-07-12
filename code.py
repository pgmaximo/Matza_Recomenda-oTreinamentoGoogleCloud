!pip install -U -q google-generativeai pypdf2 faiss-cpu gdown

import os
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
import faiss
import google.generativeai as genai
from google.colab import userdata

from google.colab import drive
drive.mount('/content/drive')

# Configurando a chave API
api_key = api_key=userdata.get('api_key')
genai.configure(api_key=api_key)

# Definindo caminho da pasta com os PDFs
pasta_pdf = '/content/drive/MyDrive/PDFs para IA'


def extrair_texto_pdf(caminho_arquivo, tamanho_parte=8000):
    with open(caminho_arquivo, 'rb') as f:
        reader = PdfReader(f)
        texto_completo = ""
        partes_texto = []
        for pagina in reader.pages:
            texto_completo += pagina.extract_text()
            if len(texto_completo) >= tamanho_parte:
                partes_texto.append(texto_completo[:tamanho_parte])
                texto_completo = texto_completo[tamanho_parte:]
        if texto_completo:
            partes_texto.append(texto_completo)
        return partes_texto

embeddings = []
textos_completos = []
nomes_arquivos = []

for nome_arquivo in os.listdir(pasta_pdf):
    if nome_arquivo.endswith(".pdf"):
        caminho_arquivo = os.path.join(pasta_pdf, nome_arquivo)
        partes_texto = extrair_texto_pdf(caminho_arquivo)
        embeddings_arquivo = []
        texto_completo_arquivo = ""

        for parte_texto in partes_texto:
            texto_completo_arquivo += parte_texto
            embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=parte_texto,
                task_type="RETRIEVAL_DOCUMENT"
            )["embedding"]
            embeddings_arquivo.append(embedding)

        embedding_media = np.mean(embeddings_arquivo, axis=0)
        nomes_arquivos.append(nome_arquivo)
        textos_completos.append(texto_completo_arquivo)
        embeddings.append(embedding_media)

# Criar DataFrame
df_treinamentos = pd.DataFrame({
    "Nome do Arquivo": nomes_arquivos,
    "Texto Completo": textos_completos,
    "Embeddings": embeddings
})

# Salvar embeddings em um índice FAISS
d = len(embeddings[0])  # Dimensão dos embeddings
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings))


generation_config = {
    "temperature": 0.2,
    "top_p": 0.25,
    "top_k": 20,
}

safety_settings = [
    {"category": "hate", "threshold": "medium"},
    {"category": "harassment", "threshold": "medium"},
    {"category": "sexual", "threshold": "medium"},
    {"category": "dangerous", "threshold": "medium"},
]

model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

chat = model.start_chat(history=[])


contexto = '''
  Aja como um assistente de vendas, onde seu objeito é ajudar a Matza Education a oferecer informações sobre Google Cloud produtos e seus treinamentos, recomendar um treinamento
  ideal de acordo com as nescessidades do cliente, exlicar o porque ele deve escolher este treinamento, responder suas duvidas sobre ferramentas da Google e duvidas gerais sobre
  os treinamentos e tentar convencer de maneira sutil o cliente a fazer este treinamento.
  Caso o usuario esteja pedindo ajuda para escolher o treinamento, deve decidir qual dos treinamentos que foram indicados pelo documentos relevantes faz mais sentido ao cliente,
  de acordo com o sua nescessidade no prompt do usuario.
  Depois do usuario dizer o que ele precisa, precisa exlicar o porque ele deve escolher este treinamento, lhe mostrando com alguns dados sobre ele, seu objetivo e como deve ser
  contato dizendo que o usuario precisar contatar a Matza Education para dizer mais sobre o treinamento;
'''

estrutura_respostas = '''
  Estrutura da resposta caso o usuario queira saber quais treinamentos são melhores para o prompt dele: '
  Para esta sua nescessidade [informações relevantes do prompt do usurio], posso te aconselhar estes 3 treinamentos:
  [treinamento 1 do [Documentos Relevante]], onde é aconselhavel para iniciantes,
  [treinamento 2 do [Documentos Relevante]], onde é aconselhavel [decidir para qual caso é o mais adequado este treinamento]
  [treinamento 3 do [Documentos Relevante]], onde é aconselhavel [decidir para qual caso é o mais adequado este treinamento]
  ';

  Estrutura de resposta caso o usuario queira saber mais sobre algum treinamento: '
  O treinamento [nome do treinamento] é focado em [objetivo do treinamento].

  [Oque voce aprenderá dentro do treinamento]

  Duração: [Duração],
  Level: [Level],
  Modulos: [modulos],
  Produtos:[Products],
  ';
'''

def recomendar_treinamento(prompt_usuario, historico_mensagens):
    # Adicionar a mensagem do usuário ao histórico
    historico_mensagens.append(f"Usuário: {prompt_usuario}")

    # Criar embedding da consulta do usuário
    embedding_consulta = genai.embed_content(
        model="models/text-embedding-004",
        content=prompt_usuario,
        task_type="RETRIEVAL_QUERY"
    )["embedding"]

    # Encontrar os 3 documentos mais similares
    _, indices_treinamentos = index.search(np.array([embedding_consulta]), k=3)
    indices_treinamentos = indices_treinamentos[0]

    # Combinar consulta do usuário com documentos relevantes
    documentos_relevantes = "\n".join(df_treinamentos.iloc[indices_treinamentos]["Texto Completo"])

    # Criar consulta aprimorada com histórico
    consulta_aprimorada = f'''
    Contexto: {contexto};

    Alguns exemplos de estrutura da resposta: {estrutura_respostas};

    Documentos relevantes: {documentos_relevantes};

    Histórico de mensagens com voce:
    {historico_mensagens}

    '''

    # Enviar consulta aprimorada para o LLM
    resposta = chat.send_message(consulta_aprimorada)

    # Adicionar resposta ao histórico
    historico_mensagens.append(f"Assistente: {resposta.text}")

    return resposta.text

def loop_chatbot():
    historico_mensagens = []
    prompt_usuario = input("Descreva suas necessidades de treinamento: ")
    while prompt_usuario.lower() != "desligar":
        resposta = recomendar_treinamento(prompt_usuario, historico_mensagens)
        print(f"Recomendação:\n{resposta}")

        prompt_usuario = input("Descreva suas necessidades de treinamento ou digite 'desligar' para encerrar: ")

loop_chatbot()
