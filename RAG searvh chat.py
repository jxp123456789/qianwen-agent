from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# ✅ 新版正确导入（修复所有 chains 不存在问题）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 加载本地大模型（ollama）
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen:1.8b"
)

# 加载文档
loader = TextLoader(r"E:\big model use\wenben\input.txt", encoding='utf-8')
docs = loader.load()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

# 向量库（使用本地下载的模型）
embedding = HuggingFaceEmbeddings(
    model_name='E:\\big model use\\code\\learn\\models\\bge-large-zh-v1.5'
)
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever()

# 提示词
prompt = ChatPromptTemplate.from_template("""
根据以下已知信息回答用户问题。
已知信息：{context}

用户问题：{question}
""")

# ✅ 新版 LCEL 链式写法（完全替代 RetrievalQA）
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

# 运行
user_question = "五虎上将有哪些？"
result = rag_chain.invoke(user_question)
print("回答：", result)