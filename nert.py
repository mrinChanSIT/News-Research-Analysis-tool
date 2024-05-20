import os
import langchain
import streamlit as st
import openai
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_objectbox.vectorstores import ObjectBox
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from secretKey import openAi_NERT_key

os.environ["OPENAI_API_KEY"] = openAi_NERT_key

EMBEDDING_MODEL = "text-embedding-ada-002"
llm = OpenAI(temperature=0.9, max_tokens=500)

loaders = UnstructuredURLLoader(urls=[
    "https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html"
])

try:
    data = loaders.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
except Exception as e:
    print(f"Error processing documents: {e}")

vector = ObjectBox.from_documents(docs, OpenAIEmbeddings(model=EMBEDDING_MODEL),embedding_dimensions=1536)

# !pip install langchainhub



qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector.as_retriever())

while True:
    question = input("Enter query (or type 'exit' to quit): ").strip()
    if question.lower() == "exit":
        print("Exiting the program. Goodbye!")
        break
    
    try:
        result = qa_chain({"question": question}, return_only_outputs=True)
        print(result['answer'])
    except Exception as e:
        print(f"Error processing query: {e}")

