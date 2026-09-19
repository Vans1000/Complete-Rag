from TreeRag import TreeRAG
from fastapi import (
    FastAPI, File as FastAPIFile, UploadFile, HTTPException, 
    BackgroundTasks, Depends, Query as FastAPIQuery, Form
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
import uvicorn
from contextlib import asynccontextmanager
import tempfile
import os
import shutil
import httpx
import uuid as uuid_mod
import json
from Tokenizer import Tokenizer
from VectorDatabase import VectorDatabase
from File import File as DocumentProcessor
from WebSearch import WebSearch
from LLM import OpenAILLM, OllamaLLM, RAGChat
from HuggingFaceDataset import HuggingFaceDataset
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams
from fastapi import APIRouter

app_state = {
    "tokenizer": None,
    "vector_db": None,
    "web_search": None,
    "rag_chat": None,
    "tree_rag_enabled": None
}

upload_progress: Dict[str, dict] = {}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_web_search: bool = False
    stream: bool = False


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    use_web_search: bool = False
    clear_history: bool = False
    force_web: bool = False
    ingest_web: bool = False


class IngestURLRequest(BaseModel):
    url: str
    recursive: bool = False


class LLMConfig(BaseModel):
    provider: str 
    model: str 
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class CollectionConfig(BaseModel):
    name: str
    host: str = "localhost"
    port: int = 6333


class SwitchRequest(BaseModel):
    collection_name: str


class HFRequest(BaseModel):
    dataset_path: str
    subset: Optional[str] = None
    text_column: str = "text"
    id_column: str = "id"
    split: str = "train"
    batch_size: int = 1024
    workers: int = 4


class ScrollRequest(BaseModel):
    limit: int = 20
    offset: Optional[str] = None


class TokenizerConfig(BaseModel):
    dense_model: str = "clip-ViT-B-32"
    sparse_model: str = "naver/splade-v3"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    vision_rerank_model: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    caption_model: str = "None"


class WebSearchConfig(BaseModel):
    max_results: int = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup"""
    yield
    pass


app = FastAPI(
    title="RAG Engine API",
    description="Multimodal RAG with Web Search and LLM Integration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_chat():
    if app_state["rag_chat"] is None:
        raise HTTPException(status_code=503, detail="LLM not configured. Call /config/llm first.")
    return app_state["rag_chat"]


@app.get("/collections/active")
async def get_active_collection():
    """Return the currently active collection name"""
    if app_state["vector_db"] is None:
        raise HTTPException(status_code=503, detail="Vector DB not initialized")
    return {"collection": app_state["vector_db"].collection_name}

@app.get("/config/tree_rag")
async def get_tree_rag():
    return {"enabled": app_state.get("tree_rag_enabled", False)}

@app.post("/config/tree_rag")
async def set_tree_rag(data: dict):
    app_state["tree_rag_enabled"] = data.get("enabled", False)
    return {"status": "ok"}

@app.post("/config/llm")
async def configure_llm(config: LLMConfig):
    """Configure the LLM provider"""
    try:
        prov = config.provider.lower() if config.provider else "openai"
        
        if prov == "ollama":
            llm = OllamaLLM(
                model=config.model,
                base_url=config.base_url or "http://localhost:11434"
            )
        else:
            llm = OpenAILLM(
                api_key=config.api_key or "sk-no-key-required",
                base_url=config.base_url,
                model=config.model
            )
        
        app_state["rag_chat"] = RAGChat(
            llm=llm,
            vector_db=app_state["vector_db"],
            tokenizer=app_state["tokenizer"],
            web_search=app_state["web_search"]
        )
        
        return {"status": "success", "provider": config.provider, "model": config.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/llm/models")
async def get_available_models(provider: str = FastAPIQuery(...)):
    """Return suggested models for provider"""
    p = provider.lower()
    if p == "openai":
        return {
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "default_base": "https://api.openai.com/v1"
        }
    elif p == "ollama":
        return {
            "models": ["llama3.2", "mistral", "gemma2", "mixtral", "llama3.1"],
            "default_base": "http://localhost:11434"
        }
    else:
        return {"models": [], "default_base": ""}

@app.get("/config/llm")
async def get_llm_config():
    """Return the current LLM configuration."""
    rag_chat = app_state.get("rag_chat")
    if rag_chat is None:
        return {"provider": None, "model": None}
    llm = rag_chat.llm
    provider = "ollama" if hasattr(llm, "base_url") and "ollama" in llm.base_url else "openai"
    return {
        "provider": provider,
        "model": llm.model if hasattr(llm, "model") else None,
        "base_url": getattr(llm, "base_url", None)
    }
    
@app.get("/config/tokenizer")
async def get_tokenizer_config():
    if app_state["tokenizer"] is None:
        raise HTTPException(status_code=503, detail="Tokenizer not initialized")
    return {
        "dense_model": app_state["tokenizer"].dense_model_name,
        "sparse_model": app_state["tokenizer"].sparse_model_name,
        "cross_encoder_model": app_state["tokenizer"].cross_encoder_model_name,
        "vision_rerank_model": app_state["tokenizer"].vision_rerank_model_name,
        "caption_model": app_state["tokenizer"].caption_model_name
    }


@app.post("/config/tokenizer")
async def configure_tokenizer(config: TokenizerConfig):
    """Reconfigure tokenizer models (will reload models)"""
    try:
        app_state["tokenizer"] = Tokenizer(
            dense_model=config.dense_model,
            sparse_model=config.sparse_model,
            cross_encoder_model=config.cross_encoder_model,
            vision_rerank_model=config.vision_rerank_model,
            caption_model=config.caption_model
        )
        if app_state["rag_chat"]:
            app_state["rag_chat"].tokenizer = app_state["tokenizer"]
        if app_state["web_search"]:
            app_state["web_search"].tokenizer = app_state["tokenizer"]
        return {"status": "success", "message": "Tokenizer reconfigured"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/websearch")
async def get_websearch_config():
    if app_state["web_search"] is None:
        return {"max_results": 10}
    return {"max_results": app_state["web_search"].max_results}


@app.post("/config/websearch")
async def configure_websearch(config: WebSearchConfig):
    try:
        if app_state["web_search"] is None:
            app_state["web_search"] = WebSearch(
                tokenizer=app_state["tokenizer"],
                vector_db=app_state["vector_db"],
                max_results=config.max_results
            )
        else:
            app_state["web_search"].max_results = config.max_results
        if app_state["rag_chat"]:
            app_state["rag_chat"].web_search = app_state["web_search"]
        return {"status": "success", "max_results": config.max_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
@app.post("/api/query")
async def query(request: QueryRequest):
    """Search vector database without LLM generation"""
    try:
        results = app_state["vector_db"].Search(
            tokenizer=app_state["tokenizer"],
            query_text=request.query,
            top_k=request.top_k
        )
        
        formatted_results = []
        for res in results:
            payload = res.payload
            formatted_results.append({
                "id": res.id,
                "score": res.score,
                "type": payload.get('type', 'unknown'),
                "content": payload.get('text', payload.get('caption', '')),
                "source": payload.get('path', payload.get('source', 'unknown')),
                "page": payload.get('page', None)
            })
        
        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
@app.post("/api/chat")
@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    """Chat with RAG (non-streaming)"""
    rag_chat = get_rag_chat()
    
    if request.clear_history:
        rag_chat.clear_history()
    
    try:
        result = rag_chat.query(
            question=request.message,
            use_web_search=request.use_web_search,
            force_web=request.force_web,
            ingest_web=request.ingest_web,
            stream=False
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses"""
    rag_chat = get_rag_chat()
    
    if request.clear_history:
        rag_chat.clear_history()
    
    def generate():
        stream_gen = rag_chat.query(
            question=request.message,
            use_web_search=request.use_web_search,
            force_web=request.force_web,
            ingest_web=request.ingest_web,
            stream=True
        )
        for chunk in stream_gen:
            if chunk:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


def progress_callback(upload_id: str, status: str, progress: int, message: str):
    """Callback function to update process progress based on encoder state."""
    upload_progress[upload_id] = {
        "status": status,
        "progress": progress,
        "message": message
    }


@app.post("/ingest/file")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = FastAPIFile(...),
    collection: Optional[str] = Form(None),
    tree_rag_enabled: Optional[bool] = Form(False)
):
    upload_id = str(uuid_mod.uuid4())
    original_filename = file.filename or "unknown.bin"
    use_tree_rag = tree_rag_enabled or app_state.get("tree_rag_enabled", False)
    if use_tree_rag:
        rag_chat = app_state.get("rag_chat")
        if rag_chat is None or rag_chat.llm is None:
            raise HTTPException(
                status_code=400,
                detail="TreeRAG requires an LLM. Please configure your LLM first."
            )
    upload_progress[upload_id] = {
        "filename": original_filename,
        "status": "uploading",
        "progress": 0,
        "message": "Uploading file..."
    }
    print(app_state["tokenizer"])
    print(app_state["vector_db"])
    
    try:
        if collection and collection != app_state["vector_db"].collection_name:
            app_state["vector_db"] = VectorDatabase(collection_name=collection)
            if app_state["rag_chat"]:
                app_state["rag_chat"].vector_db = app_state["vector_db"]
        
        temp_dir = tempfile.mkdtemp()
        safe_name = os.path.basename(original_filename)
        file_path = os.path.join(temp_dir, safe_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        upload_progress[upload_id]["status"] = "processing"
        upload_progress[upload_id]["progress"] = 15
        upload_progress[upload_id]["message"] = "File saved, processing and indexing..."
        
        def process():
            try:
                tree_rag_engine = None
                if use_tree_rag:
                    tree_rag_engine = TreeRAG(
                        tokenizer=app_state["tokenizer"],
                        vector_db=app_state["vector_db"],
                        llm=app_state["rag_chat"].llm
                    )

                DocumentProcessor.ProcessDocuments(
                    file_path, 
                    app_state["tokenizer"], 
                    app_state["vector_db"],
                    tree_rag_engine=tree_rag_engine
                )
                
                upload_progress[upload_id]["status"] = "completed"
                upload_progress[upload_id]["progress"] = 100
                upload_progress[upload_id]["message"] = "Successfully uploaded and indexed in Vector DB"
            except Exception as e:
                upload_progress[upload_id]["status"] = "error"
                upload_progress[upload_id]["progress"] = 0
                upload_progress[upload_id]["message"] = str(e)
                print(f"[Ingest Error] {e}")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        background_tasks.add_task(process)
        
        return {
            "status": "processing", 
            "filename": original_filename,
            "upload_id": upload_id,
            "collection": collection or app_state["vector_db"].collection_name,
            "message": "Document ingestion started in background"
        }    
    except Exception as e:
        upload_progress[upload_id]["status"] = "error"
        upload_progress[upload_id]["message"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingest/progress/{upload_id}")
async def get_upload_progress(upload_id: str):
    """Get upload/processing progress"""
    if upload_id not in upload_progress:
        raise HTTPException(status_code=404, detail="Upload not found")
    return upload_progress[upload_id]


@app.post("/ingest/web")
async def ingest_web(request: IngestURLRequest, background_tasks: BackgroundTasks):
    """Ingest images, documents, and text files from web search or URL"""
    try:
        if app_state["web_search"] is None:
            raise HTTPException(status_code=503, detail="Web search not initialized")
        
        upload_id = str(uuid_mod.uuid4())
        upload_progress[upload_id] = {
            "url": request.url,
            "status": "downloading",
            "progress": 0,
            "message": "Downloading content from web..."
        }

        def process_web():
            try:
                temp_dir = tempfile.mkdtemp()
                url = request.url

                if any(url.lower().endswith(ext) for ext in ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.webp']):
                    local_filename = os.path.basename(url.split('?')[0])
                    file_path = os.path.join(temp_dir, local_filename)

                    with httpx.Client(follow_redirects=True) as client:
                        resp = client.get(url)
                        resp.raise_for_status()
                        with open(file_path, "wb") as f:
                            f.write(resp.content)

                    upload_progress[upload_id]["status"] = "encoding"
                    upload_progress[upload_id]["progress"] = 20
                    upload_progress[upload_id]["message"] = "Encoding and uploading content..."

                    DocumentProcessor.ProcessDocuments(
                        file_path,
                        app_state["tokenizer"],
                        app_state["vector_db"]
                    )
                else:
                    upload_progress[upload_id]["status"] = "encoding"
                    upload_progress[upload_id]["progress"] = 20
                    upload_progress[upload_id]["message"] = "Parsing and encoding webpage..."

                    def on_encoder_finished():
                        progress_callback(
                            upload_id, 
                            status="encoding_finished", 
                            progress=90, 
                            message="Encoding finished. Storing vectors..."
                        )

                    app_state["web_search"].search_and_ingest(
                        url,
                        recursive=request.recursive,
                        on_encoder_finished=on_encoder_finished
                    )

                upload_progress[upload_id]["status"] = "completed"
                upload_progress[upload_id]["progress"] = 100
                upload_progress[upload_id]["message"] = "Complete"

            except Exception as e:
                upload_progress[upload_id]["status"] = "error"
                upload_progress[upload_id]["message"] = str(e)
                print(f"[Web Ingest Error] {e}")
            finally:
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir, ignore_errors=True)

        background_tasks.add_task(process_web)
        
        return {
            "status": "processing",
            "query": request.url,
            "upload_id": upload_id,
            "message": "Web ingestion started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/huggingface")
async def ingest_huggingface(request: HFRequest, background_tasks: BackgroundTasks):
    """Import HuggingFace dataset"""
    try:
        ingestor = HuggingFaceDataset(app_state["tokenizer"], app_state["vector_db"])
        background_tasks.add_task(
            ingestor.ingest,
            dataset_path=request.dataset_path,
            name=request.subset,
            text_column=request.text_column,
            id_column=request.id_column,
            split=request.split,
            batch_size=request.batch_size,
            num_workers=request.workers
        )
        return {
            "status": "processing", 
            "dataset": request.dataset_path,
            "message": "Dataset import started in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
async def list_collections():
    """List all Qdrant collections"""
    try:
        collections = app_state["vector_db"].client.get_collections()
        return {"collections": [c.name for c in collections.collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections")
async def create_collection(config: CollectionConfig):
    """Create new collection"""
    try:
        client = QdrantClient(host=config.host, port=config.port)
        if not client.collection_exists(config.name):
            client.create_collection(
                collection_name=config.name,
                vectors_config={
                    "image_dense": VectorParams(size=512, distance=Distance.COSINE),
                    "text_dense": VectorParams(size=512, distance=Distance.COSINE),
                    "caption_dense": VectorParams(size=512, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "text_sparse": SparseVectorParams(index={}),
                    "caption_sparse": SparseVectorParams(index={})
                }
            )
        return {"status": "created", "name": config.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections/switch")
async def switch_collection(request: SwitchRequest):
    """Switch active collection"""
    try:
        app_state["vector_db"] = VectorDatabase(collection_name=request.collection_name)
        if app_state["rag_chat"]:
            app_state["rag_chat"].vector_db = app_state["vector_db"]
        return {"status": "switched", "collection": request.collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/stats")
async def collection_stats():
    """Get collection statistics"""
    try:
        stats = app_state["vector_db"].client.get_collection(
            app_state["vector_db"].collection_name
        )
        points_count = app_state["vector_db"].client.count(
            collection_name=app_state["vector_db"].collection_name
        )
        return {
            "collection_name": app_state["vector_db"].collection_name,
            "vectors_count": points_count.count,
            "config": {
                "vector_size": 512,
                "distance": "Cosine"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dashboard/points")
async def scroll_points(request: ScrollRequest):
    """Scroll through points for dashboard"""
    try:
        offset_uuid = None
        if request.offset:
            try:
                offset_uuid = request.offset
            except:
                pass
                
        results = app_state["vector_db"].client.scroll(
            collection_name=app_state["vector_db"].collection_name,
            limit=request.limit,
            offset=offset_uuid,
            with_payload=True,
            with_vectors=False
        )
        return {
            "points": [{"id": str(p.id), "payload": p.payload} for p in results[0]],
            "next_offset": str(results[1]) if results[1] else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_db": app_state["vector_db"] is not None,
        "tokenizer": app_state["tokenizer"] is not None,
        "llm": app_state["rag_chat"] is not None
    }


def start_api_server(tokenizer, vector_db, web_search, host="0.0.0.0", port=8000):
    """Initialize state and start server"""
    app_state["tokenizer"] = tokenizer
    app_state["vector_db"] = vector_db
    app_state["web_search"] = web_search
    
    uvicorn.run(app, host=host, port=port)