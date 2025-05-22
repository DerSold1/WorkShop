# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:09:12 2025

@author: msoldner
"""
import threading
import asyncio
import tkinter as tk
import tkinter
import random
import time
from gtts import gTTS
import os
from playsound import playsound
import tempfile
#import speech_recognition as sr
import pyttsx3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import embeddings
from langchain_openai import OpenAIEmbeddings
from tkinter import ttk
from  langchain_core.embeddings import Embeddings
import langchain_openai.embeddings.base
#from langchain_openai.embeddings.base.OpenAIEmbeddings import embed_query
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain import hub
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from IPython.display import Image, display
import getpass
import speech_recognition as sr


def recognize_speech():
    # Initialisiere den Recognizer
    recognizer = sr.Recognizer()

    # Verwende das Mikrofon als Audioquelle
    with sr.Microphone() as source:
        print("Sprechen Sie jetzt...")
        audio = recognizer.listen(source, phrase_time_limit=10)  # Aufnahme der Spracheingabe

    try:
        # Verwende Google Web Speech API zur Spracherkennung
        text = recognizer.recognize_google(audio, language="de-DE")
        print("Sie sagten: " + text)
        return text
    except sr.UnknownValueError:
        print("Google Web Speech API konnte die Sprache nicht verstehen.")
    except sr.RequestError as e:
        print(f"Fehler bei der Anfrage an Google Web Speech API: {e}")


OPENAI_API_KEY = ""

#initialize speech output
engine = pyttsx3.init()

SPLIT_CHUNK_SIZE = 500
SPLIT_CHUNK_OVERLAP = 20

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

os.environ["NVIDIA_API_KEY"] = OPENAI_API_KEY
#os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")#"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"
#os.environ["LANGCHAIN_PROJECT"] = "ProofOfConcept"
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")
vector_store = InMemoryVectorStore(embeddings)
llm = ChatNVIDIA(model="microsoft/phi-3.5-mini-instruct", temperature=0.2)



#Document Loader
loader = PyPDFLoader("Taycan-Porsche-Connect-Gut-zu-wissen-Die-Anleitung.pdf")
pages = []
for page in loader.lazy_load():
    pages.append(page)
text = ""

#Extract document into string
for document in loader.lazy_load():
    #print(document)
    text += document.page_content
docs = pages    
print(f"Total characters: {len(pages[0].page_content)}")
#print(docs[0].page_content[:500])



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



"""
prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

assert len(example_messages) == 1
print(example_messages[0].content)
"""


root = tk.Tk()
root.title("Chatbot")
# Create the chatbot's text area
text_area = tk.Text(root, bg="white", width=50, height=20)
text_area.pack()
text_area.tag_config("user", foreground = "blue")
# Create the user's input field
input_field = tk.Entry(root, width=50)
input_field.bind("<Return>", (lambda event: send_message()))
input_field.pack()


def on_select(event):
    selected_value = combo.get()
    print(f"Ausgewählt: {selected_value}")
    llm = ChatNVIDIA(model=selected_value, temperature=0.2)

# Erstelle eine Combobox
combo = ttk.Combobox(root, values=["microsoft/phi-3.5-mini-instruct", "meta/llama-3.1-70b-instruct"])
combo.bind("<<ComboboxSelected>>", on_select)  # Event-Handler für Auswahl
combo.config(width=30)
combo.pack(pady=10)  # Füge die Combobox zum Fenster hinzu
combo.current(0)


# Create the send button
send_button = tk.Button(root, text="Send", command=lambda: send_message())
send_button.pack()

def chatbot_response(user_input):
  # Normalize the user's input
  user_input = user_input.lower()

  # Check for specific keywords in the user's input
  if "movie" in user_input:
    return "I recommend checking out the IMDb website for movie recommendations. They have a wide variety of genres and ratings to choose from."
  elif "weather" in user_input:
    return "You can check the weather by using a weather website or app. Some popular ones include Weather.com and The Weather Channel app."
  elif "news" in user_input:
    return "There are many websites and apps that offer the latest news updates, such as CNN, Fox News, and NBC News."
  elif "joke" in user_input:
    return "Why couldn't the bicycle stand up by itself? Because it was two-tired!"
  else:
    # If no keywords are detected, select a random response from the list
    return random.choice(responses)
def reply(name):
    showinfo(title="Reply", message = "Hello %s!" % name)


# Funktion, die aufgerufen wird, wenn eine Auswahl getroffen wird
def on_select(event):
    selected_value = combo.get()
    llm = ChatNVIDIA(model=selected_value, temperature=0.2)
    print(f"Ausgewählt: {selected_value}")

def updateGui(user_input, response):
  text_area.insert(tk.END, f"User: {user_input}\n", "user")
  text_area.insert(tk.END, f"Chatbot: {response}\n")

    
def ttS(txt):
    # Erstelle das gTTS-Objekt
    tts = gTTS(text=txt, lang='de', lang_check =True)
    
    # Speichere die Sprachausgabe in einer temporären MP3-Datei
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        tts.save(temp_file.name)  # Speichere die MP3-Datei
        temp_file_path = temp_file.name  # Speichere den Pfad der temporären Datei
    
    # Spiele die Audiodatei ab
    playsound(temp_file_path)
    
    # Optional: Lösche die temporäre Datei nach der Wiedergabe
    os.remove(temp_file_path)


def updateSpeech(response):
    def speak():
        ttS(response)
        #engine.say(response)
        #engine.runAndWait()
    # Starte die Sprachausgabe in einem separaten Thread
    speech_thread = threading.Thread(target=speak)
    speech_thread.start()
def send_message():
  # Get the user's input
  user_input = recognize_speech()

  # Clear the input field


  # Generate a response from the chatbot
  #response = chatbot_response(user_input)
  response = graph.invoke({"question": user_input})
  response = response["answer"]
  print(response)
  updateGui(user_input, response)
  #engine.say(response)
  #engine.runAndWait()
  updateSpeech(response)






root.mainloop()
while True:
    send_message()
#response = graph.invoke({"question": "Mit welchen Schritten richte ich Apple Music ein?"})
#print(response)
#engine.say(response)
#engine.runAndWait()
#answer = response["answer"]
#print(answer)








