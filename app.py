import streamlit as st

st.title("Chat with Your PDF")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

from langchain_community.document_loaders import PyPDFLoader

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)


from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

pipe = pipeline(
    "text2text-generation",
    #model="google/flan-t5-base",
    model="google/flan-t5-large",
    
    max_length=512
)

llm = HuggingFacePipeline(pipeline=pipe)


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
    """Answer based only on the context:
    
    {context}
    
    Question: {question}
    """
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

query = st.text_input("Ask a question")

if query:
    result = rag_chain.invoke(query)

    if isinstance(result, dict) and "text" in result:
        st.write(result["text"])
    else:
        st.write(result)