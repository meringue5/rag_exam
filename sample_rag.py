#!/usr/bin/env python
# coding: utf-8

# ## RAG 기본 파이프라인(1~8단계)
# 

# In[ ]:


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI


# 아래는 기본적인 RAG 구조 이해를 위한 뼈대코드(skeleton code) 입니다.
# 
# 각 단계별 모듈의 내용을 앞으로 상황에 맞게 변경하면서 문서에 적합한 구조를 찾아갈 수 있습니다.
# 
# (각 단계별로 다양한 옵션을 설정하거나 새로운 기법을 적용할 수 있습니다.)

# In[2]:


#실습용 AOAI 환경변수 읽기
import os

AOAI_ENDPOINT=os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY=os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O=os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_GPT4O_MINI=os.getenv("AOAI_DEPLOY_GPT4O_MINI")
AOAI_DEPLOY_EMBED_3_LARGE=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")
AOAI_DEPLOY_EMBED_3_SMALL=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL")
AOAI_DEPLOY_EMBED_ADA=os.getenv("AOAI_DEPLOY_EMBED_ADA")


# In[3]:


# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("./data/extract_text/AI_Paradigm_Shift_Driven_by_DeepSeek.pdf")
docs = loader.load()
print(f"문서의 페이지수: {len(docs)}")


# 페이지의 내용을 출력합니다.

# In[4]:


print(docs[10].page_content)


# `metadata` 를 확인합니다.

# In[5]:


docs[10].__dict__


# In[6]:


# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의수: {len(split_documents)}")


# In[7]:


# 단계 3: 임베딩(Embedding) 생성
embeddings = AzureOpenAIEmbeddings(
    model=AOAI_DEPLOY_EMBED_3_LARGE,
    openai_api_version="2024-02-01",
    api_key= AOAI_API_KEY,
    azure_endpoint=AOAI_ENDPOINT
    )


# In[8]:


# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)


# In[9]:


for doc in vectorstore.similarity_search("메타"):
    print(doc.page_content)


# In[10]:


# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()


# 검색기에 쿼리를 날려 검색된 chunk 결과를 확인합니다.

# In[11]:


# 검색기에 쿼리를 날려 검색된 chunk 결과를 확인합니다.
retriever.invoke("메타의 24분기 사업실적은?")


# In[12]:


# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Context: 
{context}

#Question:
{question}

#Answer:"""
)


# In[14]:


# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.

# ChatOpenAI 언어 모델을 초기화합니다. temperature는 0으로 설정합니다.
llm = AzureChatOpenAI(
    openai_api_version="2024-02-01",
    azure_deployment=AOAI_DEPLOY_GPT4O_MINI,
    temperature=0.0,
    api_key= AOAI_API_KEY,  
    azure_endpoint=AOAI_ENDPOINT
)


# In[15]:


# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# 생성된 체인에 쿼리(질문)을 입력하고 실행합니다.

# In[16]:


# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "메타의 4분기 영업 실적은?"
response = chain.invoke(question)
print(response)


# ## 전체 코드

# In[ ]:


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("./data/extract_text/AI_Paradigm_Shift_Driven_by_DeepSeek.pdf")
docs = loader.load()

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# 단계 3: 임베딩(Embedding) 생성
embeddings = AzureOpenAIEmbeddings(
    model=AOAI_DEPLOY_EMBED_3_LARGE,
    openai_api_version="2024-02-01",
    api_key= AOAI_API_KEY,
    azure_endpoint=AOAI_ENDPOINT
    )

# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Context: 
{context}

#Question:
{question}

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.

# ChatOpenAI 언어 모델을 초기화합니다. temperature는 0으로 설정합니다.
# ChatOpenAI 언어 모델을 초기화합니다. temperature는 0으로 설정합니다.
llm = AzureChatOpenAI(
    openai_api_version="2024-02-01",
    azure_deployment=AOAI_DEPLOY_GPT4O_MINI,
    temperature=0.0,
    api_key= AOAI_API_KEY,  
    azure_endpoint=AOAI_ENDPOINT
)


# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# In[ ]:


# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "메타의 4분기 영업 실적은?"
response = chain.invoke(question)
print(response)

