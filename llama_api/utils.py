from transformers import BitsAndBytesConfig, AutoConfig, AutoTokenizer, pipeline, AutoModelForCausalLM
from torch import cuda
import torch
from time import time
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

DEVICE = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
model_id = 'meta-llama/Llama-2-7b-chat-hf'

def time_record(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


@time_record
def load_model():
    # bnb_config = BitsAndBytesConfig(
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    model_config = AutoConfig.from_pretrained(
        model_id,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        device_map='auto',
    )
    print("Model loaded")
    return model


@time_record
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Tokenizer loaded")
    return tokenizer

@time_record
def get_query_pipeline():
    model = load_model()
    tokenizer = get_tokenizer()
    query_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Query pipeline loaded")
    return query_pipeline

@time_record
def test_query_pipeline(tokenizer, pipeline, prompt_to_test):
    time1 = time()
    sequences = pipeline(
        prompt_to_test,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200
    )
    time2 = time()
    print(f"Time to generate: {time2 - time1:.2f} seconds")
    for seq in sequences:
        print(f"Generated: {seq['generated_text']}")


@time_record
def get_llm():
    llm = HuggingFacePipeline(
        pipeline=get_query_pipeline()
    )
    llm(prompt="Please tell me the capital of Turkey")
    print("LLM loaded")
    return llm


@time_record
def pdf_loader():
    pdf_loader = PyPDFDirectoryLoader('llama_api/pdfs')
    documents = pdf_loader.load()
    print(f"Number of documents: {len(documents)}")
    return documents

@time_record
def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    print(f"Number of text chunks: {len(text_chunks)}")
    return text_chunks

@time_record
def get_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Embeddings loaded")
    return embeddings

@time_record
def get_verctordb(embeddings, text_chunks):
    vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory="chroma_db")
    print("Vector DB loaded")
    return vectordb

@time_record
def get_qa(llm, vectordb):
    retriever = vectordb.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    print("QA loaded")
    return qa

