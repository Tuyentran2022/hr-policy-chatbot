from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

import streamlit as st


def build_chat_history(chat_history_list):
    # this function takes in the Chat History Message in a List of Tuples format
    # and turns it into a series of Human an AI Message objects

    chat_history = []
    for message in chat_history_list:
        chat_history.append(HumanMessage(content=message[0]))
        chat_history.append(AIMessage(content=message[1]))

    return chat_history

@st.cache_resource(show_spinner=False)
def load_rag_components():
    #Load the local FAISS database where the entire website is stored as Embedding vectors
    # load FAISS one time by this function
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    new_db= FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    llm = ChatOllama(model="llama3.2:latest",temperature=0)

    return new_db, llm


@st.cache_resource(show_spinner=False)
def load_rag_chain():
    # Create a ConversationalBufferMemory object with 'chat_history'
    # Create a ConersationalRetrievalChain object with the FAISS DB as Retriever

    new_db, llm = load_rag_components()
    
    condense_question_system_template = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood"
        "without the chat history. DO NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    retriever = new_db.as_retriever(search_kwargs={"k":3})

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_question_prompt
    )

    system_prompt = (
        "You are an AI assistant specialized in HR policies. "

        "Answer ONLY using the information provided in the retrieved context. "

        "If the retrieved context contains partial information that is relevant,"
        "answer using that information and clearly state that it is partial. "

        "Only respond with"
        "'I couldn't find this information in the HR policy documents.'"
        "when the retrieved context contains no relevant information at all."

        "Do not make up information or rely on your own knowledge. "

        "Keep your answer concise (maximum 3 short paragraphs). "

        "Use bullet points whenever appropriate. "

        "At the end of the answer, mention the source document if available.\n\n"

        "{context}"
    )

    qa_prompt= ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]   
    )
  
    qa_chain= create_stuff_documents_chain(llm, qa_prompt)

    convo_qa_chain= create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain


def query(question, chat_history):
    """
    This function:
    - Receives 2 parameters: 'question' and 'chat_history'
    - Invoke the Retriever object with the Query and Chat History
    - Return the response
    """
    rag_chain = load_rag_chain()

    result = rag_chain.invoke(
        {
            "input": question,
            "chat_history": build_chat_history(chat_history),
        }
    )

    return result


def show_ui():
    """
    - Implement the streamlit UI
    - Implement 2 session_state vatiables - "messages" - to contain the accumulating Questions and Answers the 
    'chat_history' - the accumulating question-answering pairs as a list of Tuples to be saved to the served to the Retriever
    - For each user query, the response is obtained by invoking the 'query' function and the chat histories are
    """

    st.title("Your Human Resources Chatbot")
    st.image("banner-AI.png")
    st.subheader("Please enter your HR query ")

    #Initilize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    #Display chat message from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter your HR Policy related Query: "):
        with st.chat_message("user"):
            st.markdown(prompt)
        # Invoke the function with Retriever with chat history and display responses in chat container
        with st.spinner("Working on your query...."):
            response = query(question=prompt, chat_history=st.session_state.chat_history)

        with st.chat_message("assistant"):
            st.markdown(response["answer"])

        # Append user message to chat history

        st.session_state.messages.append({"role":"user", "content":prompt})
        st.session_state.messages.append({"role":"assistant", "content": response["answer"]})
        st.session_state.chat_history.extend([(prompt, response["answer"])])

# program Entry...
if __name__ == "__main__":
    show_ui()









