import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf_doc in pdf_docs:
        pdf = PdfReader(pdf_doc)
        for page in pdf.pages:
            pdf_text += page.extract_text() or ""  # Ensure text is extracted
    return pdf_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Fixed method
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant that answers questions based on the given context.
    If the answer is not in the context, say: "Answer is not available in the context."
    Do not provide wrong answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Correctly structure the function call
    return create_stuff_documents_chain(prompt=prompt, llm=model)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain.invoke({"context": docs, "question": user_question})

    # If response is a string, just display it
    if isinstance(response, str):
        st.write("Reply:", response)
    elif isinstance(response, dict) and "output_text" in response:
        st.write("Reply:", response["output_text"])
    else:
        st.write("Reply: Unexpected response format")


def main():
    st.set_page_config(page_title="Chat With Multiple PDFs")
    st.header("Chat With Multiple PDFs using Gemini")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.image("./profilepic.jpeg", width=170)
        st.markdown("Developed by ***Basudev Das***")
        st.markdown(
        """
        <style>
        .social-links {
            display: flex;
            gap: 10px;
        }
        .social-links a {
            text-decoration: none;
            font-size: 18px;
        }
        .social-links img {
            width: 15px;
            height: 15px;
            vertical-align: middle;
        }
        </style>

        <div class="social-links">
            <a href="https://github.com/basudev77" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png"> GitHub
            </a> 
            <a href="https://www.linkedin.com/in/basudev-das-6568b0274/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/61/61109.png"> LinkedIn
            </a> 
            <a href="https://basudevdas.netlify.app/" target="_blank">
                <img src="https://img.icons8.com/?size=100&id=87836&format=png&color=000000"> Portfolio
            </a>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")  # Fixed spelling

if __name__ == "__main__":
    main()
