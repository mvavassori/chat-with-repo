import streamlit as st
from dotenv import load_dotenv
import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import tempfile
import subprocess
import uuid


def clone_github_repo(github_url, local_path):
    try:
        subprocess.run(["git", "clone", github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to clone repository: {e}")
        return False


def load_github_repo(local_path):
    # Language to extension mapping
    lang_to_ext = {
        "cpp": [".cpp"],
        "go": [".go"],
        "java": [".java"],
        "js": [".js", ".jsx", ".ts", ".tsx"],
        "php": [".php"],
        "proto": [".proto"],
        "python": [".py"],
        "rst": [".rst"],
        "ruby": [".rb"],
        "rust": [".rs"],
        "scala": [".scala"],
        "swift": [".swift"],
        "markdown": [".md"],
        "latex": [".tex"],
        "html": [".html", ".htm", ".css", "scss"],
        "sol": [".sol"],
    }

    text_extensions = {".txt", ".json"}

    documents_dict = {}
    file_type_counts = {}

    # Process code files
    for lang, exts in lang_to_ext.items():
        for ext in exts:
            glob_pattern = f"**/*{ext}"
            try:
                loader = DirectoryLoader(local_path, glob=glob_pattern)
                loaded_documents = loader.load() if callable(loader.load) else []
                if loaded_documents:
                    file_type_counts[ext] = len(loaded_documents)
                    for doc in loaded_documents:
                        file_path = doc.metadata["source"]
                        relative_path = os.path.relpath(file_path, local_path)
                        doc.metadata["source"] = relative_path
                        doc.metadata["file_id"] = str(uuid.uuid4())
                        documents_dict[doc.metadata["file_id"]] = doc

                    # Display the results for this file type
                    # st.write(
                    #     f"Processed {len(loaded_documents)} documents with extension {ext}."
                    # )

            except Exception as e:
                st.write(f"Error processing file type: {ext}, Exception: {str(e)}")

    # Process text files
    for ext in text_extensions:
        glob_pattern = f"**/*{ext}"
        try:
            loader = DirectoryLoader(
                local_path,
                glob=glob_pattern,
                loader_cls=TextLoader,
            )
            loaded_documents = loader.load() if callable(loader.load) else []
            if loaded_documents:
                # Filter out package-lock.json after loading if dealing with json files
                if ext == ".json":
                    loaded_documents = [
                        doc
                        for doc in loaded_documents
                        if "package-lock.json" not in doc.metadata["source"]
                    ]

                file_type_counts[ext] = len(loaded_documents)
                for doc in loaded_documents:
                    file_path = doc.metadata["source"]
                    relative_path = os.path.relpath(file_path, local_path)
                    doc.metadata["source"] = relative_path
                    doc.metadata["file_id"] = str(uuid.uuid4())
                    documents_dict[doc.metadata["file_id"]] = doc

                # Display the results for this file type
                # st.write(
                #     f"Processed {len(loaded_documents)} documents with extension {ext}."
                # )

        except Exception as e:
            st.write(f"Error processing file type: {ext}, Exception: {str(e)}")

    # st.write(dict(documents_dict))
    return documents_dict


def split_documents(documents_dict):
    split_documents_dict = {}

    for file_id, doc in documents_dict.items():
        try:
            ext = os.path.splitext(doc.metadata["source"])[1]
            lang = get_language_from_extension(ext)
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=1000, chunk_overlap=0
            )
            split_docs = splitter.create_documents([doc.page_content])
            for split_doc in split_docs:
                split_doc.metadata.update(
                    doc.metadata
                )  # Copy metadata from original doc
                split_documents_dict[
                    str(uuid.uuid4())
                ] = split_doc  # Store split documents with unique IDs

        except Exception as e:
            st.write(
                f"Error splitting document: {doc.metadata['source']}, Exception: {str(e)}"
            )
    st.write(split_documents_dict)
    return split_documents_dict


def get_language_from_extension(ext):
    # Simplified mapping from file extension to LangChain Language enum
    ext_to_lang = {
        ".cpp": Language.CPP,
        ".go": Language.GO,
        ".java": Language.JAVA,
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.JS,
        ".tsx": Language.JS,
        ".php": Language.PHP,
        ".proto": Language.PROTO,
        ".py": Language.PYTHON,
        ".rst": Language.RST,
        ".rb": Language.RUBY,
        ".rs": Language.RUST,
        ".scala": Language.SCALA,
        ".swift": Language.SWIFT,
        ".md": Language.MARKDOWN,
        ".tex": Language.LATEX,
        ".html": Language.HTML,
        ".htm": Language.HTML,
        ".sol": Language.SOL,
        ".css": Language.HTML,
        ".txt": Language.MARKDOWN,
        ".json": Language.MARKDOWN,
    }
    return ext_to_lang.get(ext, Language.MARKDOWN)


# def get_vectorstore(chunks):
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
#     return vectorstore


def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    # vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    dataset_path = "hub://mvavassori/text_embedding"
    vectorstore = DeepLake.from_documents(
        chunks, dataset_path=dataset_path, embedding=embeddings
    )
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
        response = st.session_state.conversation({"question": user_input})
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
                    docs = load_github_repo(local_path)
                    # get code chunks
                    chunks = split_documents(docs)
                    # create vector store
                    # vectorstore = get_vectorstore(chunks)
                    # create conversation chain
                    # st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
