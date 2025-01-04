import streamlit as st
import torch
from huggingface_hub import notebook_login
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load the model and tokenizer from Hugging Face
MODEL_NAME = "Yermalovich/results"  # Replace with your model name


@st.cache_resource
def load_model():
    pipe = pipeline("text-generation", model="Yermalovich/results")
    return pipe


pipe = load_model()


@st.cache_resource
def load_rag_mode():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder


embedder = load_rag_mode()


@st.cache_data
def load_index(index_file, csv_file):
    df = pd.read_csv(csv_file)  # load the dataset
    index = faiss.read_index(index_file)  # load the FAISS index
    return index, df


index, df = load_index("RAG/context.index", "RAG/med_data.csv")


def retrieve_context(query, index, df, embedder, top_k=1):
    query_vector = embedder.encode([query]).astype("float32")  # embed the query
    distances, indices = index.search(query_vector, top_k)  # search the FAISS index
    return [df.iloc[i]["context"] for i in indices[0]]  # retrieve contexts by index


def generate_answer(pipe, promt_rag_question):
    answer = pipe(promt_rag_question)
    return answer[0]['generated_text']


# Streamlit UI
st.title("Medical Question Answering App")
st.write("Enter a question and the model will generate an answer.")

# Input fields
question = st.text_input("Question", placeholder="What are the symptoms of hypertension?")

# Generate answer when the user clicks the button
if st.button("Generate Answer"):
    if question:
        contexts = retrieve_context(question, index, df, embedder)
        extracted_text = ' '.join(contexts)
        promt_rag_question = f"You are a medical assistant. Based on the question, provide an accurate answer:\nContext: {extracted_text}\nQuestion: {question}\nAnswer:"

        with st.spinner("Generating answer..."):
            answer = generate_answer(pipe, promt_rag_question)
        st.success("Answer:")
        st.write(answer)
