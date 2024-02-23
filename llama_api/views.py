import pickle
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from llama_api.apps import LlamaApiConfig
from langchain.chains import RetrievalQA

@require_GET
def index(request):
    return JsonResponse({"message": "Welcome to the Llama API"})

@csrf_exempt
@require_POST
def query(request : HttpRequest):
    qa = LlamaApiConfig.qa
    data = request.POST
    query = data['query']
    response = qa(query)

    result_text = response.get("result", "")
    formatted_result = {
        "result": result_text,
        "status": "success" if result_text else "error",
        "message": "Query processed successfully" if result_text else "Error processing the query",
    }

    return JsonResponse(formatted_result)

@require_GET
def get_docs(request):
    if len(LlamaApiConfig.text_chunks) == 0:
        return JsonResponse({"message": "No documents found"})
    js_list = [
        {
            "source": text_chunk.metadata['source'],
            "content": text_chunk.page_content
        }
        for text_chunk in LlamaApiConfig.text_chunks
    ]
    return JsonResponse({"message": "Success", "data": js_list})

