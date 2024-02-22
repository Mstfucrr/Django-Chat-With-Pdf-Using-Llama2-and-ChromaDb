from django.apps import AppConfig
from llama_api.utils import get_llm, pdf_loader, get_text_chunks, get_embeddings, get_verctordb, get_qa


class LlamaApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'llama_api'

    # Initialize components here

    llm = get_llm()
    documents = pdf_loader()

    text_chunks = get_text_chunks(documents)
    embeddings = get_embeddings()
    vector_db = get_verctordb(embeddings=embeddings, text_chunks=text_chunks)
    qa = get_qa(llm=llm, vectordb=vector_db)
    