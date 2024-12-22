from src.helper import load_pdf_dir, text_split, download_hugging_face_embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore # To connect with the Vectorstore
from dotenv import load_dotenv
import os
load_dotenv()
import time


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf_dir("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


index_name = "medical-chatbot" #give the name to your index, or you can use an index which you created previously and load that.
#here we are using the new fresh index name
pc = Pinecone(api_key=PINECONE_API_KEY)
#Get your Pinecone API key to connect after successful login and put it here.

if index_name in pc.list_indexes().names():
  print("index already exists" , index_name)
  index= pc.Index(index_name) #your index which is already existing and is ready to use
  print(index.describe_index_stats())

else: #crate a new index with specs
  pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
   )
while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
        index= pc.Index(index_name)
        print("index created")
        print(index.describe_index_stats())


try:
    # Pinecone operation
    docsearch = PineconeVectorStore.from_texts([chunk.page_content for chunk in text_chunks], embeddings, index_name=index_name)
except Exception as e:
    print(f"An error occurred: {e}")
