import streamlit as st
import openai

# used to create the memory
from langchain.memory import ConversationBufferMemory

# used to load text
from langchain.document_loaders import WebBaseLoader

# used to create the retriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from llama_index import SimpleDirectoryReader


# used to create the retrieval tool
from langchain.agents import tool

# used to create the prompt template
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFDirectoryLoader

# used to create the agent executor
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor


# set the secure key
openai_api_key = st.secrets.openai_key

# add a heading for your app.
st.header("Chat with the LLM Agents blog 💬 📚")

# Initialize the memory
# This is needed for both the memory and the prompt
memory_key = "history"

if "memory" not in st.session_state.keys():
    st.session_state.memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

# Initialize the chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about LLM based agents!"}
    ]

# create the document database
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the LLM blog – hang tight!."):
        #loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        #data = loader.load()
        # Set the directory containing the PDF documents
        #pdf_folder = "./data"

        # Create a DirectoryLoader object
        #loader = DirectoryLoader(pdf_folder)

        #reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        loader = PyPDFDirectoryLoader("./data")
        data = loader.load()

        # Load all of the PDF documents in the folder
        #data = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_documents(texts, embeddings)
        return db

db = load_data()

# instantiate the database retriever
retriever = db.as_retriever()

# define the retriever tool
@tool
def tool(query):
    "Searches and returns documents regarding the llm powered autonomous agents blog"
    docs = retriever.get_relevant_documents(query)
    return docs

tools = [tool]

# define the prompt
system_message = SystemMessage(
        content=(
            "Provide answers only from given documents; do not provide answers from other sources."
            "Always provide the file name as source along with answer"
            "If you don't find the information in the documents, say you don't know the answer based on the documents"
            

            
        )
)
prompt_template = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )

# instantiate the large language model
llm = ChatOpenAI(temperature = 0, openai_api_key=openai_api_key,return_source_documents=True)

# instantiate agent
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# Prompt for user input and display message history
if prompt := st.chat_input("Your LLM based agent related question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Pass query to chat engine and display response
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor({"input": prompt})
            st.write(response["output"])
            message = {"role": "assistant", "content": response["output"]}
            st.session_state.messages.append(message) # Add response to message
