from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
from .services.forecast import forecast_sales
from .services.chatbot import chat_with_rag
from .embeddings.embed import index_embeddings
from .rag.retrieval import search_chroma

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    csv: bytes

class EmbedIndexRequest(BaseModel):
    csv: bytes
    text_column: str

class ChatRequest(BaseModel):
    query: str

@app.post("/forecast")
async def forecast(csv: UploadFile = File(...)):
    try:
        df = pd.read_csv(csv.file)
        print("Uploaded DF:", df.head())  # debug print
        result = forecast_sales(df)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

@app.post("/embed-index")
async def embed_index(csv: UploadFile = File(...), text_column: str = Form(...)):
    df = pd.read_csv(csv.file)
    n_indexed = index_embeddings(df, text_column)
    return {"indexed": n_indexed}

@app.post("/chat")
async def chat(query: str = Form(...)):
    answer = chat_with_rag(query)
    return {"answer": answer}

@app.post("/vector-search")
async def vector_search(query: str = Form(...), n_results: int = Form(5)):
    results = search_chroma(query, n_results)
    return {"results": results}