import streamlit as st
# import tiktoken 
from loguru import logger 

from langchain.chains import ConversationalRetrievalChain 
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.document_loaders.csv_loader import CSVLoader

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings 

from langchain.memory import ConversationBufferMemory 
from langchain.vectorstores import Chroma

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback 
from langchain.memory import StreamlitChatMessageHistory
import os

def main():
    st.set_page_config(
        page_title="ChatBot Demo",
        page_icon="random",
    )
    st.title(":red[QA Chat]")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None


    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        process = st.button("Process")
        
    if process:
        if not openai_api_key:
            st.info("Please add your API key to continue.")
            st.stop()
        embedding = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding)
        st.session_state.conversation = get_conversation_chain(vectordb, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                        "content" :"안녕하세요! 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    if query := st.chat_input("질문을 입력해주세요."): 
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type = "stuff",
        retriever=vectorstore.as_retriever(search_type = 'mmr'),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

if __name__ == '__main__':
    main()