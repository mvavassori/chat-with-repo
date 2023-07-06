import streamlit as st
from dotenv import load_dotenv
import os
from htmlTemplates import css, bot_template, user_template
import langchain
from langchain.document_loaders import GitLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import tempfile
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def load_github_repo(github_url, local_path, repo_branch):
    loader = GitLoader(
        clone_url=github_url,
        repo_path=local_path,
        branch=repo_branch,
    )
    docs = loader.load()
    return docs


def split_documents(documents_list):
    split_documents_list = []

    for doc in documents_list:
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
                split_documents_list.append(
                    split_doc
                )  # Store split documents in a list

        except Exception as e:
            st.write(
                f"Error splitting document: {doc.metadata['source']}, Exception: {str(e)}"
            )
    return split_documents_list


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


def create_vectorstore(chunks, dataset_path):
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    db.add_documents(chunks)
    return db


def load_vectorstore(dataset_path):
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = DeepLake(
        dataset_path=dataset_path,
        read_only=True,
        embedding_function=embeddings,
    )
    return db


def get_conversation_chain(vectorstore, gpt_model):
    langchain.verbose = False
    llm = ChatOpenAI(model=gpt_model, temperature=0.5)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Define your system message template
    general_system_template = """You are a superintelligent AI that answers questions about codebases.
    You are:
    - helpful & friendly
    - good at answering complex questions in simple language
    - an expert in all programming languages
    - able to infer the intent of the user's question

    The user will ask a question about their codebase, and you will answer it.

    When the user asks their question, you will answer it by searching the codebase for the answer.
    Answer the question using the code file(s) below:
    ----------------
        {context}"""
    # Define your user message template
    general_user_template = "Question:```{question}```"

    # Create message prompt templates from your message templates
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        general_system_template
    )
    user_message_prompt = HumanMessagePromptTemplate.from_template(
        general_user_template
    )

    # Create a chat prompt template from your message prompt templates
    qa_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, user_message_prompt]
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return conversation_chain


def handle_user_input(user_input):
    response = st.session_state.conversation({"question": user_input})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with repo")
    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        """
        Remember to add your `OPENAI_API_KEY` and `ACTIVELOOP_TOKEN` to your .env file.
        """
        gpt_model = st.selectbox("Select OpenAI GPT model", ("gpt-3.5-turbo", "gpt-4"))

        st.subheader("If you don't have an existing Activeloop dataset enter below")
        github_url = st.text_input(
            "Enter GitHub repo URL (for example: `https://github.com/username/my_repo`)"
        )
        repo_branch = st.text_input(
            "Enter GitHub repo branch (for example: `master`)", "master"
        )
        activeloop_url = st.text_input(
            "Enter the Activeloop dataset URL where you wish to save your dataset (for example: `hub://username/my_dataset`)"
        )

        if st.button("Create dataset and start chatting"):
            with st.spinner("Processing..."):
                with tempfile.TemporaryDirectory() as local_path:
                    # get code files
                    docs = load_github_repo(
                        github_url, local_path, repo_branch="master"
                    )
                    # get code chunks
                    chunks = split_documents(docs)
                    # create vector store
                    vectorstore = create_vectorstore(chunks, activeloop_url)
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore, gpt_model
                    )

        st.subheader("If you already have an existing Activeloop dataset enter below")

        activeloop_url = st.text_input(
            "Enter your existing Activeloop dataset URL here (for  example: `hub://username/my_dataset`)"
        )

        if st.button("Load dataset and start chatting"):
            with st.spinner("Processing..."):
                # load vector store
                vectorstore = load_vectorstore(activeloop_url)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, gpt_model
                )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Error handling?
    st.header("Chat with repo")
    user_input = st.text_area("Ask question to repo")
    if user_input:
        handle_user_input(user_input)


if __name__ == "__main__":
    main()
