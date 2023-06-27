import streamlit as st
from dotenv import load_dotenv
import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import tempfile
import subprocess


def clone_github_repo(github_url, local_path):
    try:
        subprocess.run(["git", "clone", github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to clone repository: {e}")
        return False


def load_github_repo(local_path):
    extensions = {".js", "jsx", "ts", "tsx", ".css", ".html", ".md", ".txt"}
    for ext in extensions:
        glob_pattern = f"**/*{ext}"
        try:
            loader = DirectoryLoader(local_path, glob=glob_pattern)
            loaded_documents = loader.load() if callable(loader.load) else []
            if loaded_documents:
                st.write(f"Loaded {len(loaded_documents)} documents of type {ext}")

        except Exception as e:
            st.write(f"Error processing file type: {ext}, Exception: {str(e)}")


def load_and_split_text(local_path):
    # root_dir = "./fake-repo"
    root_dir = local_path
    file_types = [".html", ".css", ".js", ".md"]

    docs = []
    for file_type in file_types:
        try:
            # Create a loader for the current file type
            loader = DirectoryLoader(
                root_dir, glob=f"**/*{file_type}", loader_cls=TextLoader
            )

            # Load and split the documents
            docs.extend(loader.load_and_split())
        except Exception as e:
            st.write(f"Error processing file type: {file_type}, Exception: {str(e)}")
            pass

    return docs


def text_split(docs):
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)
    # return [doc.content for doc in chunks]
    return chunks


def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_input(user_input):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({"input": user_input})
        st.write(response)
    else:
        st.write("Please click 'Start Processing' to initialize the conversation.")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with repo")
    st.write(css, unsafe_allow_html=True)

    # create an input field for the GitHub repo URL
    github_url = st.text_input("Enter GitHub repo URL")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with repo")
    user_input = st.text_input("Ask question to repo")
    if user_input:
        handle_user_input(user_input)

    # st.write(user_template, unsafe_allow_html=True)
    # st.write(bot_template, unsafe_allow_html=True)

    if st.button("Start Processing"):
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as local_path:
                if clone_github_repo(github_url, local_path):
                    # get code files
                    # docs = load_and_split_text(local_path)
                    load_github_repo(local_path)
                    # get code chunks
                    # chunks = text_split(docs)
                    # create vector store
                    # vectorstore = get_vectorstore(chunks)
                    # create conversation chain
                    # st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()


# def load_text():
#     root_dir = "./fake-repo"

#     docs = []
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         for file in filenames:
#             if file.endswith((".html", ".css", ".js", ".md")):
#                 try:
#                     loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
#                     docs.extend(loader.load_and_split())
#                 except Exception as e:
#                     st.write(f"Error processing file: {file}, Exception: {str(e)}")
#                     pass
#     return docs


# def text_split(docs):
#     text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
#     chunks = text_splitter.split_documents(docs)
#     return [doc.content for doc in chunks]
