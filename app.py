


import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
   text = ""
   for pdf in pdf_docs:
       pdf_reader = PdfReader(pdf)
       for page in pdf_reader.pages:
           text += page.extract_text()
   return text


def get_text_chunks(text):
   text_splitter = CharacterTextSplitter(
       separator="\n",
       chunk_size=1000,
       chunk_overlap=200,
       length_function=len
   )
   chunks = text_splitter.split_text(text)
   return chunks


def get_vectorstore(text_chunks, index_path="faiss_index"):
   embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
   vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
   os.makedirs(index_path, exist_ok=True)
   vector_store.save_local(index_path)


def get_conversation_chain():
   prompt_template = """
   You are an assistant that provides detailed answers based on the content of a PDF document. Answer the following
   questions using the provided context from the PDF, based on extracted text summarize the pdf if user asks and
   you should be able to answer if the question is related to text based on pdf. Basically,
   you should be able to answer any question if the pdf has an answer. If the answer is not available in the
    context, respond with
   "The answer is not available in the provided context." \n\n
   Context:\n {context}?\n
   Question: \n{question}\n

   Answer:
   """

   model = ChatGoogleGenerativeAI(model="gemini-pro",
                                  temperature=0.3)

   prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
   chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

   return chain


def user_input(user_question, index_path="faiss_index"):
   embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

   new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
   docs = new_db.similarity_search(user_question)

   chain = get_conversation_chain()

   response = chain(
       {"input_documents": docs, "question": user_question}
       , return_only_outputs=True)

   print(response)
   st.write("Reply: ", response["output_text"])


def main():
   load_dotenv()
   st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

   st.header("Chat with PDFs :books:")
   user_question = st.text_input("Ask a question about your documents:")
   if user_question:
       user_input(user_question)

   with st.sidebar:
       st.subheader("Your documents")
       pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
       if st.button("Process"):
           with st.spinner("Processing..."):
               raw_text = get_pdf_text(pdf_docs)

               text_chunks = get_text_chunks(raw_text)

               get_vectorstore(text_chunks)

               st.success("Done")


if __name__ == "__main__":
   main()




