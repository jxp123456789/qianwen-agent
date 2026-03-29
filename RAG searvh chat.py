from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 加载本地大模型
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b"
)

# 加载文档（自己把 sanguoyanyi.txt 放在同目录）
loader = TextLoader("sanguoyanyi.txt", encoding='utf-8')
docs = loader.load()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

# 向量库
embedding = HuggingFaceEmbeddings(model_name='models/AI-ModelScope/bge-large-zh-v1___5')
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever()

# 提示词
system_message = SystemMessagePromptTemplate.from_template(
    "根据以下已知信息回答用户问题。\n 已知信息{context}"
)
human_message = HumanMessagePromptTemplate.from_template("用户问题:{question}")
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# 检索链
qa = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": chat_prompt}
)

# 测试
user_question = "五虎上将有哪些？"
print(qa.invoke(user_question))