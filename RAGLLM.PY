import os
from typing import Any
from dotenv import load_dotenv
import qdrant_client
import openai
import requests
import cohere

import warnings
warnings.filterwarnings('ignore')

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

from qdrant_client import QdrantClient
import docx2txt

# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CohereRerank

from langchain.chains import RetrievalQA


from flask import Flask, jsonify, request

load_dotenv()



client = QdrantClient(
    url= os.getenv("QDRANT_HOST"), 
    api_key= os.getenv("QDRANT_API"))

gpt_api_key = os.environ.get("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=gpt_api_key)


def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


def load_websites(urls):
  from langchain_community.document_loaders import UnstructuredURLLoader
  print(f'Loading {urls}')
  loader = UnstructuredURLLoader(urls=urls)

  data = loader.load()

  return data


def chunk_data(data, chunk_size=500):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n\n+", "\n", ". ", " ", ""], 
                                                   chunk_size=chunk_size, chunk_overlap= 50)
    chunks = text_splitter.split_documents(data)
   
    return chunks


def list_all_collections(client):
    response = client.get_collections()
    collections = [collection.name for collection in response.collections]
    return collections
    

def delete_collection(client, collection_name):
    client.delete_collection(collection_name=collection_name)
    print(f"Collection {collection_name} deleted.")


def insert_or_fetch_embeddings(collection_name, chunks):
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    from qdrant_client.http import models
    
    existing_collections = list_all_collections(client)
    if collection_name in existing_collections:
        print(f"Collection {collection_name} already exists.")
    else:
        print(f"Attempting to create collection: {collection_name}...")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            print(f"Collection {collection_name} created successfully.")
        except Exception as e:
            print(f"Failed to create collection: {e}")

    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    qdrant_vectorstore = Qdrant(
        client=client,
        collection_name= collection_name,
        embeddings=embeddings
    )
    qdrant_vectorstore.add_documents(documents=chunks)

    return qdrant_vectorstore 

def ask_and_get_answer(collection_name, query, model_name, temperature):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CohereRerank

    personality_prompt = '''Role: friendly and helpful assistant. 
                            ########
                            Objective: Assistant thinks logically, Focuses on the Keywords asked in a question.
                            ########
                            Instructions: Please respond to the user’s question based on the given information. If you cannot find the answer, kindly state that Answer cannot be found.
                            Also respond in the same language user asks the question.'''

    query = personality_prompt + query

    qdrant_vectorstore = Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings
            )
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    retriever = qdrant_vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':3})
    co_client = cohere.Client(os.getenv("COHERE_API_KEY"))
    compressor = CohereRerank(client=co_client, model='rerank-multilingual-v2.0', top_n=3)

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever, return_source_documents=True)

    answer = chain.invoke({"query": query})
    result = answer['result']
    return result


def process_query(collection_name: str, file_url: str, query: str, model_name: str, temperature: int):
    # Load the document
    if file_url.startswith("http"):
        data = load_websites([file_url])
    else:
        data = load_document(file_url)
    print(f'You have {len(data)} pages in your data')

    # Chunk the data
    chunks = chunk_data(data)
    print(f'Number of chunks created: {len(chunks)}')

    # Insert or fetch embeddings
    qdrant_vectorstore = insert_or_fetch_embeddings(collection_name, chunks)

    # Get the answer
    answer = ask_and_get_answer(qdrant_vectorstore, query, model_name=model_name, temperature=temperature)
    return answer



 
app = Flask(__name__)

@app.route('/api/v1/createAiPorject', methods=['POST'])            # project
def create_ai_porject():
    try:
        # created variables
        request_data = request.get_json()
        type = request_data.get('type')
        fileLink = request_data.get('fileLink')
        urls = request_data.get('urls')
        collection_name = request_data.get('collectionName')
        temprature = request_data.get('bestGuess')
        data_anomiyzer = request_data.get('dataAnomiyzer')
        model = request_data.get('model')
        source_chat_gpt = request_data.get('sourceChatGpt')
        language = request_data.get('language')
        noOfPages = request_data.get('noOfPages')
        
        if type == "url":
            data = load_websites(urls)
            newProjectPages = len(data)
            if noOfPages+newProjectPages>500:
                return jsonify({'success': False,'message':'Assistant page limit reached'}),412
            chunks = chunk_data(data)
            qdrant_vectorstore = insert_or_fetch_embeddings(collection_name, chunks)
            return jsonify({'success':True
                    ,'message': 'Project created successfully','noOfPages':newProjectPages}),200
        else:           
            res= requests.get(fileLink)
            
            var_content = res.content
            filename = ''
            if 'pdf' in res.headers['Content-Type']:
                filename = 'test.pdf'
                with open(filename,'wb') as f:
                    f.write(var_content)
            if 'text' in res.headers['Content-Type']:
                filename = 'test.txt'
                with open(filename,'wb') as f:
                    f.write(var_content)
            if 'doc' in res.headers['Content-Type']:
                filename = 'test.docx'
                with open(filename,'wb') as f:
                    f.write(var_content)
            


            data = load_document(filename)
            newProjectPages = len(data)
            if noOfPages+newProjectPages>500:
                return jsonify({'success': False,'message':'Assistant page limit reached'}),412
            chunks = chunk_data(data)
            
            qdrant_vectorstore = insert_or_fetch_embeddings(collection_name, chunks)
            return jsonify({'success':True
                    ,'message': 'Project created successfully','noOfPages':newProjectPages}),200
       
           

       
    except Exception as e:
        print(e)
        return jsonify({'success': False,'message':'Something went wrong'}),500
    

@app.route('/api/v1/answerQuery', methods=['POST'])
def get_answer():
    try:
        request_data = request.get_json()
        type = request_data.get('type')
        filename = request_data.get('filename')
        file_index =  request_data.get('fileIndex')
        urls = request_data.get('urls')
        collection_name = request_data.get('collectionName')
        temprature = request_data.get('bestGuess')
        data_anomiyzer = request_data.get('dataAnomiyzer')
        model = request_data.get('model')
        source_chat_gpt = request_data.get('sourceChatGpt')
        language = request_data.get('language')
        query = request_data.get('query')
        

        answer = ask_and_get_answer(collection_name,query, model, 0)
        

        return jsonify({'success':True
                        ,'answer':answer})
    except Exception as e:
        print(e)
        return jsonify({'success': False})
    
@app.route('/api/v1/collection/delete', methods=['POST'])
def delete_collection_api():
    try:
        request_data = request.get_json()
        
        collection_name = request_data.get('collectionName')
        delete_collection(client,collection_name)

        
        return jsonify({'success':True
                        ,'message':'Collection deleted successfully'})
    except Exception as e:
        print(e)
        return jsonify({'success': False,'message':'Something went wrong'})
    

@app.route('/api/v1/collection/edit', methods=['POST'])
def edit_collection_api():
    try:
        request_data = request.get_json()
        
        old_collection_name = request_data.get('oldCollectionName')
        new_collection_name = request_data.get('newCollectionName')

        print(old_collection_name,new_collection_name)
        
        
        return jsonify({'success':True
                        ,'message':'Collection deleted successfully'})
    except Exception as e:
        print(e)
        return jsonify({'success': False,'message':'Something went wrong'})
    




if __name__ == '__main__':
    app.run()








