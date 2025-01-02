from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import logging

app = Flask(__name__)

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CURL_CA_BUNDLE'] = ''

# 创建 conversations 表
conn = sqlite3.connect('chat.db')
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, timestamp REAL)')
conn.commit()
conn.close()

# 尝试加载本地模型
try:
    llm = ChatOllama(model="glm4:latest")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# 尝试加载嵌入模型
try:
    embedding_model = SentenceTransformer('/Users/rock/aigcs/all-MiniLM-L12-v2')
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: e")

# 把模型加载放在函数外，避免重复加载
class EmbeddingAdapter:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


def check_context_and_invoke(question):
    # 获取历史对话记录
    conn = sqlite3.connect('chat.db')
    cursor = conn.cursor()
    # 按时间戳降序排序，获取最新的一条记录
    cursor.execute('SELECT question, answer FROM conversations ORDER BY timestamp DESC LIMIT 1')
    conversations = cursor.fetchall()
    conn.close()

    if conversations:
        last_question, last_answer = conversations[0]  # 获取最新的一条记录
        print("最新一条记录：", last_question, last_answer)
    else:
        last_question, last_answer = "", ""

    # 如果问题是翻译请求，且上一条回答存在
    if "翻译" in question and last_answer:
        translation_prompt = f"将以下内容翻译成英文：{last_answer}"
        question = translation_prompt  # 将翻译请求作为新问题

    # 其他逻辑保持不变
    history = ""
    if len(conversations) > 1:
        for q, a in conversations[1:]:
            history += f"Q: {q}\nA: {a}\n"

    with open("gansu.txt", "r", encoding="utf-8") as f:
        knowledge_base = f.read()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    knowledge_texts = text_splitter.create_documents([knowledge_base])

    embedding_adapter = EmbeddingAdapter(embedding_model)
    vector_store = Chroma.from_documents(documents=knowledge_texts, embedding=embedding_adapter)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.001
        }
    )

    prompt_template = """
        请用中文回答问题。已知上一轮对话：Q: {last_question}\nA: {last_answer}，以及更早的历史对话：{history}，还有知识库信息：{knowledge_base} 直接依靠自身知识完整作答，不用强行关联上下文；若有匹配信息，则依据上下文精准作答。请回答：{question}。
        Answer:
        """
    prompt = PromptTemplate.from_template(prompt_template)

    contexts = retriever.invoke(question)
    print("contexts======》:", contexts)

    if not contexts:
        input_dict = {
            "last_question": last_question,
            "last_answer": last_answer,
            "history": history,
            "knowledge_base": knowledge_base,
            "question": question
        }
    else:
        if isinstance(contexts, dict) and "documents" in contexts:
            context_text = " ".join([doc.page_content for doc in contexts["documents"]])
            input_dict = {
                "last_question": last_question,
                "last_answer": last_answer,
                "history": history,
                "knowledge_base": knowledge_base,
                "context": context_text,
                "question": question
            }
        else:
            input_dict = {
                "last_question": last_question,
                "last_answer": last_answer,
                "history": history,
                "knowledge_base": knowledge_base,
                "context": contexts,
                "question": question
            }

    chain = (
        RunnablePassthrough.assign(last_question=lambda x: x.get("last_question", ""))
        .assign(last_answer=lambda x: x.get("last_answer", ""))
        .assign(history=lambda x: x.get("history", ""))
        .assign(knowledge_base=lambda x: x.get("knowledge_base", ""))
        .assign(context=lambda x: x.get("context", ""))
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke(input_dict)

    if isinstance(result, AIMessage):
        result = result.content
    return result


@app.route('/get_conversations', methods=['GET'])
def get_conversations():
    conn = sqlite3.connect('chat.db')
    cursor = conn.cursor()
    cursor.execute('SELECT question, answer FROM conversations ORDER BY timestamp ASC')
    conversations = cursor.fetchall()
    conn.close()
    result = []
    for q, a in conversations:
        entry = {"question": q, "answer": a}
        result.append(entry)
    return jsonify(result)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form.get('question')
        answer = check_context_and_invoke(question)
        print("Answer to be sent to frontend:", answer)
        try:
            # 保存对话记录到SQLite
            conn = sqlite3.connect('chat.db')
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, timestamp REAL)')
            import time
            current_time = time.time()
            cursor.execute('INSERT INTO conversations (question, answer, timestamp) VALUES (?,?,?)', (question, answer, current_time))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database insert error: {e}")

        response_data = {
            "question": question,
            "answer": answer
        }
        return jsonify(response_data)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, port=5001)