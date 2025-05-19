# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:09:12 2025

@author: msoldner
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

load_dotenv()

#OPENAI_API_KEY = ""
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SPLIT_CHUNK_SIZE = 100
SPLIT_CHUNK_OVERLAP = 2

os.environ["LANGCHAIN_PROJECT"] = "ProofOfConcept"

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}
def load_file(): 
    # Specify the folder containing the PDF files
    pdf_folder = "resource"

    # List to store all pages from all PDFs
    all_pages = []

    # Iterate through all files in the folder
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            print(f"Loading: {file_path}")
            
            # Load the PDF
            loader = PyPDFLoader(file_path)
            
            # Append pages to the list
            for page in loader.lazy_load():
                all_pages.append(page)

    # Combine all pages into a single text string
    text = "".join(page.page_content for page in all_pages)

    print(f"Total characters loaded: {len(text)}")
    return all_pages



os.environ["NVIDIA_API_KEY"] = OPENAI_API_KEY



embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")
vector_store = InMemoryVectorStore(embeddings)


llm = ChatNVIDIA(model="meta/llama3-70b-instruct")



'''Load the PDF Document. Adjust the link to your file'''
docs = load_file()
#loader = PyPDFLoader("Taycan-Porsche-Connect-Gut-zu-wissen-Die-Anleitung.pdf")
#pages = []
#for page in loader.lazy_load():
#    pages.append(page)
#text = ""

#Extract document to string
#for document in loader.lazy_load():
    #print(document)
#    text += document.page_content
#docs = pages    
#print(f"Total characters: {len(pages[0].page_content)}")




#Textsplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=SPLIT_CHUNK_SIZE,  # chunk size (characters)
    chunk_overlap=SPLIT_CHUNK_OVERLAP,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")


#Store documents
document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])

#docs = retriever.invoke(query)

prompt = hub.pull("rlm/rag-prompt")



# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()



while True:
    print("Chatbot: Wie kann ich helfen?")
    question = input("Bitte Frage eingeben: ")
    
    response = graph.invoke({"question": question})
    answer = response["answer"]
    print(answer)








