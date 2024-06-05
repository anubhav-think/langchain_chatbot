# import os
# from dotenv import load_dotenv,find_dotenv
# load_dotenv(find_dotenv())
# api_key = os.environ.get("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = api_key

import time
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model='gpt-4')


airbus_loader = PyPDFLoader("pdfs/Airbus-Annual-Report-2023.pdf")
airbus_pages = airbus_loader.load_and_split()

sing_air_loader = PyPDFLoader("pdfs/annualreport2223.pdf")
sing_air_pages = sing_air_loader.load_and_split()

embeddings = OpenAIEmbeddings()

airbus_vectorstore = FAISS.from_documents(airbus_pages, embedding=embeddings)
sing_air_vectorstore = FAISS.from_documents(sing_air_pages, embedding=embeddings)

airbus_retriever =RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=airbus_vectorstore.as_retriever())

sing_air_retriever =RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=sing_air_vectorstore.as_retriever())


tools = [
    Tool(
        name = "Airbus store",
        func=airbus_retriever.run,
        description="useful for when you need to answer questions about Airbus."
    ),
    Tool(
        name = "Singapore Airlines store",
        func=sing_air_retriever.run,
        description="useful for when you need to answer questions about Singapore airlines."
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(tools,
                               llm,
                               agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                               memory=memory,
                               verbose=True)

def make_output(input):
    output = agent_chain.run(input=input)
    return output

def modify_output(input):
    for text in input.split():
        yield text + " "
        time.sleep(0.05)