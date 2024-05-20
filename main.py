import os
import streamlit as st
import time
import langchain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_objectbox.vectorstores import ObjectBox

os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"
import openai
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

EMBEDDING_MODEL = "text-embedding-ada-002"
llm = OpenAI(temperature=0.9, max_tokens=500)
st.title("News Reaserch Tool ♾")

st.sidebar.title("News Article URLs")
URLs = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    URLs.append(url)
print(URLs)

Analysze_url_clicked = st.sidebar.button("Analyze URLs")
main_placeholder = st.empty()

if Analysze_url_clicked:
    loaders = UnstructuredURLLoader(urls = URLs)
    main_placeholder.text("Data Loading Started . . . ⇣ ⇣ ⇣")
    data = loaders.load()

    # split the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    main_placeholder.text("Splitting Documents . . . ⇣ ⇣ ⇣")

    docs = text_splitter.split_documents(data)

    # Create Embeddings and store in vector DB
    main_placeholder.text("Embedding Vector Started Building . . . ⇣ ⇣ ⇣")
    st.session_state.vector = ObjectBox.from_documents(docs, OpenAIEmbeddings(model=EMBEDDING_MODEL),embedding_dimensions=1536)


query = main_placeholder.text_input("Question")
if query:
    qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=st.session_state.vector.as_retriever())
    result = qa_chain({"question":query}, return_only_outputs=True)
    print(result)
    st.header("Answer")
    st.write(result['answer'])
    sources = result.get("sources",'')
    if sources:
        st.subheader("Source")
        source_list = sources.split('\n')
        for s in source_list:
            st.write(s)








