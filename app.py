from flask import Flask, render_template, jsonify, request,make_response
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import Dict
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatOpenAI(temperature=0.4, max_tokens=500) 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",      
    history_messages_key="history", 
    output_messages_key="answer",    
)

@app.route("/")
def index():
    resp = make_response(render_template("chat.html"))
    if "session_id" not in request.cookies:
        # Use something better in production (uuid, auth user id)
        resp.set_cookie("session_id", request.remote_addr or "anon")
    return resp

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    session_id = request.cookies.get("session_id", request.remote_addr or "anon")

    # Invoke the history-wrapped chain and include a configurable session_id
    response = conversational_rag.invoke(
        {"input": msg},
        config={
            "configurable": {"session_id": session_id}
        },
    )
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)