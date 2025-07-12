"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
import logging
import sys
import unicodedata
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from typing import List
from sudachipy import tokenizer, dictionary
from langchain_community.agent_toolkits import SlackToolkit
from langchain.agents import AgentType, initialize_agent
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from docx import Document
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import LLMChain
import datetime
import constants as ct

############################################################
# 設定関連
############################################################
load_dotenv()

############################################################
# 関数定義
############################################################

def build_error_message(message):
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])

def create_rag_chain(db_name):
    logger = logging.getLogger(ct.LOGGER_NAME)

    docs_all = []
    if db_name == ct.DB_ALL_PATH:
        folders = os.listdir(ct.RAG_TOP_FOLDER_PATH)
        for folder_path in folders:
            if folder_path.startswith("."):
                continue
            add_docs(f"{ct.RAG_TOP_FOLDER_PATH}/{folder_path}", docs_all)
    else:
        folder_path = ct.DB_NAMES[db_name]
        add_docs(folder_path, docs_all)

    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    embeddings = OpenAIEmbeddings()
    if os.path.isdir(db_name):
        db = Chroma(persist_directory=".db", embedding_function=embeddings)
    else:
        db = Chroma.from_documents(splitted_docs, embedding=embeddings, persist_directory=".db")
    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, retriever, question_generator_prompt
    )

    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, question_answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def add_docs(folder_path, docs_all):
    files = os.listdir(folder_path)
    for file in files:
        file_extension = os.path.splitext(file)[1]
        if file_extension in ct.SUPPORTED_EXTENSIONS:
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](f"{folder_path}/{file}")
        else:
            continue
        docs = loader.load()
        docs_all.extend(docs)

def run_company_doc_chain(param):
    ai_msg = st.session_state.company_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])
    return ai_msg["answer"]

def run_service_doc_chain(param):
    ai_msg = st.session_state.service_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])
    return ai_msg["answer"]

def run_customer_doc_chain(param):
    ai_msg = st.session_state.customer_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])
    return ai_msg["answer"]

def run_faq_doc_chain(param):
    ai_msg = st.session_state.company_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])
    return ai_msg["answer"]

def delete_old_conversation_log(result):
    response_tokens = len(st.session_state.enc.encode(result))
    st.session_state.total_tokens += response_tokens
    while st.session_state.total_tokens > ct.MAX_ALLOWED_TOKENS:
        removed_message = st.session_state.chat_history.pop(1)
        removed_tokens = len(st.session_state.enc.encode(removed_message.content))
        st.session_state.total_tokens -= removed_tokens

def execute_agent_or_chain(chat_message):
    logger = logging.getLogger(ct.LOGGER_NAME)
    if st.session_state.agent_mode == ct.AI_AGENT_MODE_ON:
        st_callback = StreamlitCallbackHandler(st.container())
        result = st.session_state.agent_executor.invoke({"input": chat_message}, {"callbacks": [st_callback]})
        response = result["output"]
    else:
        result = st.session_state.rag_chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.extend([HumanMessage(content=chat_message), AIMessage(content=result["answer"])])
        response = result["answer"]

    if response != ct.NO_DOC_MATCH_MESSAGE:
        st.session_state.answer_flg = True
    
    return response
def notice_slack(chat_message):
    toolkit = SlackToolkit()
    tools = toolkit.get_tools()
    agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    )

    loader = CSVLoader(ct.EMPLOYEE_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs = loader.load()
    loader = CSVLoader(ct.INQUIRY_HISTORY_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs_history = loader.load()

    # SlackIDと氏名をマッピングする辞書を作成
    slackid_to_name = {}
    for doc in docs:
        lines = doc.page_content.split("\n")
        slack_id = ""
        name = ""
        for line in lines:
            if line.startswith("SlackID: "):
                slack_id = line.replace("SlackID: ", "").strip()
            elif line.startswith("氏名: "):
                name = line.replace("氏名: ", "").strip()
        if slack_id and name:
            slackid_to_name[slack_id] = name

    for doc in docs:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    for doc in docs_history:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    docs_all = adjust_reference_data(docs, docs_history)
    docs_all_page_contents = [doc.page_content for doc in docs_all]

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs_all, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})
    bm25_retriever = BM25Retriever.from_texts(
        docs_all_page_contents,
        preprocess_func=preprocess_func,
        k=ct.TOP_K
    )
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=ct.RETRIEVER_WEIGHTS
    )

    employees = retriever.invoke(chat_message)
    context = get_context(employees)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_EMPLOYEE_SELECTION)
    ])
    output_parser = CommaSeparatedListOutputParser()
    format_instruction = output_parser.get_format_instructions()

    messages = prompt_template.format_prompt(context=context, query=chat_message, format_instruction=format_instruction).to_messages()
    employee_id_response = st.session_state.llm(messages)
    employee_ids = output_parser.parse(employee_id_response.content)

    target_employees = get_target_employees(employees, employee_ids)
    slack_ids = get_slack_ids(target_employees)

    # SlackIDではなく、氏名で表示するように変更
    mention_names = [slackid_to_name.get(slack_id, slack_id) for slack_id in slack_ids]
    mention_text = "と".join([f"「{name}」" for name in mention_names])

    context = get_context(target_employees)
    now_datetime = get_datetime()

    prompt = PromptTemplate(
        input_variables=["slack_id_text", "query", "context", "now_datetime"],
        template=ct.SYSTEM_PROMPT_NOTICE_SLACK,
    )
    prompt_message = prompt.format(slack_id_text=mention_text, query=chat_message, context=context, now_datetime=now_datetime)

    agent_executor.invoke({"input": prompt_message})

    return ct.CONTACT_THANKS_MESSAGE

def adjust_reference_data(docs, docs_history):
    docs_all = []
    for row in docs:
        row_lines = row.page_content.split("\n")
        row_dict = {item.split(": ")[0]: item.split(": ")[1] for item in row_lines if ": " in item}
        employee_id = row_dict.get("従業員ID", "")

        doc = ""

        same_employee_inquiries = []
        for row_history in docs_history:
            row_history_lines = row_history.page_content.split("\n")
            row_history_dict = {item.split(": ")[0]: item.split(": ")[1] for item in row_history_lines if ": " in item}
            if row_history_dict.get("従業員ID", "") == employee_id:
                same_employee_inquiries.append(row_history_dict)

        new_doc = Document()

        if same_employee_inquiries:
            doc += "【従業員情報】\n"
            row_data = "\n".join(row_lines)
            doc += row_data + "\n=================================\n"
            doc += "【この従業員の問い合わせ対応履歴】\n"
            for inquiry_dict in same_employee_inquiries:
                for key, value in inquiry_dict.items():
                    doc += f"{key}: {value}\n"
                doc += "---------------\n"
            new_doc.page_content = doc
        else:
            new_doc.page_content = row.page_content
        new_doc.metadata = {}

        docs_all.append(new_doc)
    
    return docs_all

def get_target_employees(employees, employee_ids):
    target_employees = []
    duplicate_check = []
    target_text = "従業員ID"
    for employee in employees:
        num = employee.page_content.find(target_text)
        employee_id = employee.page_content[num+len(target_text)+2:].split("\n")[0]
        if employee_id in employee_ids:
            if employee_id in duplicate_check:
                continue
            duplicate_check.append(employee_id)
            target_employees.append(employee)
    
    return target_employees

def get_slack_ids(target_employees):
    target_text = "SlackID"
    slack_ids = []
    for employee in target_employees:
        num = employee.page_content.find(target_text)
        slack_id = employee.page_content[num+len(target_text)+2:].split("\n")[0]
        slack_ids.append(slack_id)
    
    return slack_ids

def create_slack_id_text(slack_ids):
    slack_id_text = ""
    for i, id in enumerate(slack_ids):
        slack_id_text += f"「{id}」"
        if not i == len(slack_ids)-1:
            slack_id_text += "と"
    
    return slack_id_text

def get_context(docs):
    context = ""
    for i, doc in enumerate(docs, start=1):
        context += "===========================================================\n"
        context += f"{i}人目の従業員情報\n"
        context += "===========================================================\n"
        context += doc.page_content + "\n\n"

    return context

def get_datetime():
    dt_now = datetime.datetime.now()
    now_datetime = dt_now.strftime('%Y年%m月%d日 %H:%M:%S')

    return now_datetime

def preprocess_func(text):
    tokenizer_obj = dictionary.Dictionary(dict="full").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text ,mode)
    words = [token.surface() for token in tokens]
    words = list(set(words))

    return words

def adjust_string(s):
    if type(s) is not str:
        return s
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    return s
