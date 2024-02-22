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
    qa: RetrievalQA = request.session.get('qa')
    print("QA", qa)
    if qa is None:
        qa = LlamaApiConfig.qa
        print("QA is generated")
    data = request.POST
    query = data['query']
    response = qa(query)
    return JsonResponse({"result : " : response["result"]})

@require_GET
def get_docs(request):
    return JsonResponse({
        "docs": [
            {
                "title": "Query",
                "description": "Query the model with a question",
                "method": "POST",
                "path": "/api/query",
                "body": {
                    "query": "string"
                }
            }
        ]
    })

