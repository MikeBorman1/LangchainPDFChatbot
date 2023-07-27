
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st
import pandas as pd
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt
# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from transformers import GPT2TokenizerFast
from langchain.text_splitter import CharacterTextSplitter

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = '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'

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True, max_tokens=3000)
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader
#loader = PyPDFLoader('Polar.pdf')
# Split pages from pdf 
#pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB

# Advanced method - Split by chunk

# importing required modules
from PyPDF2 import PdfReader
  
reader = PdfReader('Polar.pdf')

raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text




text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)



store = Chroma.from_texts(texts, embeddings, collection_name='polarlounge')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="Polar Lounge",
    description="Polar Lounge Bot",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    reduce_k_below_max_tokens=True
)
st.title('Polar Lounge Bot')
# Create a text input box for the user
prompt = st.text_input('Input your question here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 