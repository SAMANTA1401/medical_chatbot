from flask import Flask, render_template, request, redirect, url_for
from src.helper import download_hugging_face_embeddings
# from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

index_name = "medical-chatbot" 
pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = download_hugging_face_embeddings()

try:
    # Pinecone operation
    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
except Exception as e:
    print(f"An error occurred: {e}")


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_1.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)


@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods = ["POST", "GET"])
def chat():
   
   if request.method == "POST":
      msg = request.form["msg"]
      input = msg

      result=qa({"query":input})
      print(result["result"])
   
      return str(result["result"])

if __name__ == '__main__':
    app.run(port=5000, debug=True)