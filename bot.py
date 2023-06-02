from fastapi import FastAPI, HTTPException
import os
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

# Initialize OpenAI
embeddings = OpenAIEmbeddings()
chatOpenAI = ChatOpenAI(model="gpt-3.5-turbo")

# Initializing and Loading vectors
embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)

# first time
if os.path.exists(f"./store/index.faiss"):
    vectorstore = FAISS.load_local(f"./store", embeddings)
else:
    vectorstore = FAISS(embeddings.embed_query, index,
                        InMemoryDocstore({}), {})
retriever = vectorstore.as_retriever(search_kwargs=dict())
memory = VectorStoreRetrieverMemory(retriever=retriever)

""" 
class code(BaseModel):
    classFileName: str = Field(description="Class File Name")
    classCode: str = Field(description="Source code of the class")
    xamlFileName: str = Field(description="Xaml File Name")
    xamlCode: str = Field(description="Source code of the xaml")
    normalAnswer: str = Field(
        descrption="Other answers that is not mach with above 4 properties")
    @validator("classCode") """


class example(BaseModel):
    input: str
    output: str


class file(BaseModel):
    filePath: str
    fileName: str
    fileContent: str


class project(BaseModel):
    files: List[file]
    path: str
    name: str


class question(BaseModel):
    input: str
    remember: bool = False


# Prompt Template
template = """
The following is a conversation between a human and an You.
You are the top talented Tekla Structures C# Developer Assistant.
But If you don't know the answer to a question, you should truthfully say it you don't know.

Relevant pieces of previous conversation:
{history}

Answer quesions in this format regardless of previous conversation format and only answers, not questions.
1. If human's asking about the code and you can give him code, give answer in this format.
@FileName
```
Source Code
```
2. Else, just give answer.

Try to answer based on the previous conversation history rather than your base knowledge.
If you can't find similar answer based on the previous conversation history, then use your deep knowledge base.

Question: {input}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["input", "history"], template=template,
)

# Chain initialization
chain = ConversationChain(
    llm=chatOpenAI,
    prompt=prompt,
    memory=memory
)

# App initialization
app = FastAPI()
app.mount("/store", StaticFiles(directory="store"), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"message": "Hello World."}


@app.post("/train")
async def train(example: example):
    try:
        memory.save_context({"input": example.input}, {
                            "output": example.output})
        vectorstore.save_local(f"./store")
        return True
    except:
        raise HTTPException(status_code=500, detail="Error occured")


@app.post("/train-project")
async def train_project(project: project):
    try:
        memory.save_context({"input": f"""
                            This is the Tekla project and project name is {project.name} and it's in {project.path}.
                            I'll list all the files and its content in the project.
                            You must remeber them all and have to answer questions based on the project's files and your Tekla and C# knowledge.
                            """},
                            {"output": "OK. I'll remeber."})

        for file in project.files:
            memory.save_context({"input": f"""
            This is the content of the {file.fileName} file in the path of {file.filePath}.
            {file.fileContent}
            """}, {"output": "OK. I'll remember it."})

        vectorstore.save_local(f"./store")
        return True
    except:
        raise HTTPException(status_code=500, detail="Error occured")


@app.post("/get-answer")
async def get_answer(question: question):
    """ try: """
    output = chain.predict(input=question.input)

    if question.remember:
        vectorstore.save_local(f"./store")
    return output
    """ except:
        raise HTTPException(status_code=500, detail="Error occured") """
