import argparse
from File import File
from Tokenizer import Tokenizer
from VectorDatabase import VectorDatabase
from PIL import Image
import os
import warnings
import sys
import multiprocessing
from Wikipedia import Wikipedia
from HuggingFaceDataset import HuggingFaceDataset
from transformers import logging
from TreeRag import TreeRAG

from WebSearch import WebSearch
from LLM import OpenAILLM, OllamaLLM, RAGChat
import api

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
if sys.platform == "darwin":
    multiprocessing.set_start_method("spawn", force=True)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DISABLE_TRANSFORMERS_AUTOCONVERSION"] = "1"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()



def search_system(tokenizer, vector_db, query=None, query_image_path=None, tree_level: int = None):
    query_img = Image.open(query_image_path) if query_image_path else None

    results = vector_db.Search(
        tokenizer=tokenizer,
        query_text=query,
        query_image=query_img,
        top_k=10,
        tree_level=tree_level
    )

    print(f"\nResults for query: '{query}'")
    print("-" * 50)

    for i, res in enumerate(results, 1):
        payload = res.payload

        if payload['type'] == 'image':
            content = payload.get('caption', payload.get('text', 'No caption'))
            content_type = f"IMAGE (Page {payload.get('page', 'N/A')})"
        else:
            content = payload.get('text', 'N/A')
            if payload.get('page', 'N/A') == 0:
                content_type = f"TEXT (Title {payload.get('path', 'N/A')})"
            else:
                content_type = f"TEXT (Page {payload.get('page', 'N/A')})"

        content = content

        if len(content) > 300:
            content = content[:300].rsplit(' ', 1)[0] + "..."

        print(f"{i}. [{content_type}] Score: {res.score:.3f}")

        if payload['type'] == 'image':
            print(f"   Caption: {content}")
        else:
            print(f"   Content: {content}")
        print()

    return results


def interactive_chat_mode(rag_chat: RAGChat, ingest_web_default: bool = False, self_correction: bool = False):
    """Interactive CLI chat mode"""
    print("\n🤖 RAG Chat Mode")
    print("Commands:")
    print("  /web      - Toggle web search (If local results are low/missing)")
    print("  /forceweb - Toggle forced web search (Always search web)")
    print("  /clear    - Clear chat history")
    print("  /exit     - Exit chat")
    print("-" * 50)
    
    use_web = False
    force_web = False
    
    while True:
        try:
            mode_prefix = "Local"
            if force_web:
                mode_prefix = "FORCE-WEB"
            elif use_web:
                mode_prefix = "Web+"
                
            user_input = input(f"\n[{mode_prefix}] You: ").strip()
            
            if not user_input:
                continue
                
            if user_input == "/exit":
                break
            elif user_input == "/clear":
                rag_chat.clear_history()
                print("Chat history cleared.")
                continue
            elif user_input == "/web":
                use_web = not use_web
                force_web = False  
                print(f"Web search: {'ON' if use_web else 'OFF'}")
                continue
            elif user_input == "/forceweb":
                force_web = not force_web
                use_web = False   
                print(f"Forced web search: {'ON' if force_web else 'OFF'}")
                continue
            
            print("Assistant: ", end="", flush=True)
            
            response_gen = rag_chat.query(
                question=user_input,
                use_web_search=use_web or force_web,
                force_web=force_web,
                ingest_web=ingest_web_default,
                self_correction=self_correction,
                stream=True
            )
            
            full_response = ""
            for chunk in response_gen:
                print(chunk, end="", flush=True)
                full_response += chunk
            print()
            
        except KeyboardInterrupt:
            print("\nUse /exit to quit.")
        except Exception as e:
            print(f"\nError: {e}")

def web_search_mode(web_search: WebSearch, query: str, ingest: bool = False):
    """CLI mode for web search"""
    print(f"\n🔍 Searching web for: {query}")
    
    if ingest:
        print("Ingesting results into vector database...")
        results = web_search.search_and_ingest(query)
        print(f"Ingested content from {len(results)} sources")
    else:
        results = web_search.search(query)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['href']}")
            print(f"   {result['body'][:200]}...")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process documents and store in vector database.")
    parser.add_argument("--documents", nargs="+", default=None, help="Paths to the documents to be processed.")
    parser.add_argument("--document_dir", type=str, default=None, help="Directory containing documents to be processed.")
    parser.add_argument("--collection_name", type=str, help="Name of the vector database collection.", required=True)
    parser.add_argument("--dense_model", type=str, default="clip-ViT-B-32", help="Name of the dense model to use.")
    parser.add_argument("--sparse_model", type=str, default="naver/splade-v3", help="Name of the sparse model to use.")
    parser.add_argument("--cross_encoder_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2", help="Name of the cross-encoder model to use.")
    parser.add_argument("--vision_rerank_model", type=str, default="nvidia/llama-nemotron-rerank-vl-1b-v2", help="Name of the vision rerank model to use.")
    parser.add_argument("--caption_model", type=str, default="vikhyatk/moondream2", help="Name of the caption model to use (or use None for speed).")
    parser.add_argument("--query", type=str, default=None, help="Query for retrieval.")
    
    parser.add_argument("--web_search", type=str, default=None, help="Perform web search with query")
    parser.add_argument("--ingest_web", action="store_true", help="Ingest web search results into vector DB")
    parser.add_argument("--web_results", type=int, default=10, help="Number of web results to fetch")
    
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "ollama"], help="LLM provider")
    parser.add_argument("--llm_model", type=str, default="google/gemma-4-31b", help="LLM model name")
    parser.add_argument("--llm_base_url", type=str, default=None, help="Base URL for LLM API (e.g., http://localhost:11434 for Ollama)[Include /v1 for OpenAI]")
    parser.add_argument("--ask", type=str, default=None, help="Single question to ask (non-interactive)")
    parser.add_argument("--use_web", action="store_true", help="Use web search in RAG queries")
    parser.add_argument("--force_web", action="store_true", default=False, help="Force web search for all queries, ignoring local results")

    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--api_host", type=str, default="0.0.0.0", help="API server host")
    parser.add_argument("--api_port", type=int, default=8000, help="API server port")
    
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--wikipedia", action='store_true', help="Ingest Wikipedia dataset.")
    dataset_group.add_argument("--huggingface_dataset", type=str, help="HuggingFace dataset path.")

    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for dataset ingestion.")
    parser.add_argument("--workers", type=int, default=4, help="Number of embedding worker threads.")
    parser.add_argument("--hf_subset", type=str, default=None, help="Subset of the HF dataset (e.g., '20231101.en').")
    parser.add_argument("--hf_split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'test').")
    parser.add_argument("--hf_text_col", type=str, default="text", help="Column name containing the document text.")
    parser.add_argument("--hf_id_col", type=str, default="id", help="Column name containing the unique identifier.")
    parser.add_argument("--hf_extra_cols", nargs="+", default=[], help="Extra metadata columns to include.")
    parser.add_argument("--self_correction", action="store_true", help="Enable self-correction in RAG queries (LLM evaluates and rewrites queries)")
    parser.add_argument("--tree_rag", action="store_true", help="Enable TreeRAG (hierarchical tree indexing)")
    args = parser.parse_args()
    if args.force_web:
        args.use_web = True
    if not args.self_correction:
        args.self_correction = False
    dataset_enabled = args.wikipedia or args.huggingface_dataset

    if not dataset_enabled:
        if args.batch_size != 1024 or args.workers != 4:
            parser.error("--batch_size and --workers can only be used with --wikipedia or --huggingface_dataset.")
    if not args.huggingface_dataset:
        if args.hf_subset != None or args.hf_split != "train" or args.hf_text_col != "text" or args.hf_id_col != "id":
            parser.error("--hf_subset, --hf_split, --hf_text_col, and --hf_id_col can only be used with --huggingface_dataset.")
    
    print(">>> loading Tokenizer...", flush=True)

    tokenizer = Tokenizer(
        dense_model=args.dense_model,
        sparse_model=args.sparse_model,
        cross_encoder_model=args.cross_encoder_model,
        vision_rerank_model=args.vision_rerank_model,
        caption_model=args.caption_model
    )
    
    print(">>> Tokenizer ready", flush=True)

    vector_db = VectorDatabase(collection_name=args.collection_name)
    
    
    print(">>> Vector DB ready", flush=True)

    web_search = None
    if args.web_search or args.use_web or args.ingest_web or args.api:
        web_search = WebSearch(tokenizer, vector_db, max_results=args.web_results)
    
    rag_chat = None
    if args.chat or args.ask or args.api:
        if args.llm_provider == "openai":
            llm = OpenAILLM(model=args.llm_model, base_url=args.llm_base_url)
        else:
            llm = OllamaLLM(model=args.llm_model, base_url=args.llm_base_url or "http://localhost:11434")
        
        rag_chat = RAGChat(llm=llm, vector_db=vector_db, tokenizer=tokenizer, web_search=web_search)

    if args.web_search:
        web_search_mode(web_search, args.web_search, args.ingest_web)
        if not args.chat and not args.ask: 
            sys.exit(0)

    if args.api:
        print(f"🚀 Starting API server on http://{args.api_host}:{args.api_port}")
        print("Configure LLM via POST /config/llm")
        api.start_api_server(tokenizer, vector_db, web_search, args.api_host, args.api_port)
        sys.exit(0)
    
    llm = None
    if args.chat or args.ask or args.api or args.tree_rag:
        if args.llm_provider == "openai":
            llm = OpenAILLM(model=args.llm_model, base_url=args.llm_base_url)
        else:
            llm = OllamaLLM(model=args.llm_model, base_url=args.llm_base_url or "http://localhost:11434")
    
    
    if args.tree_rag:
        if llm is None:
            raise ValueError("TreeRAG requires an LLM. Please configure --llm_provider and --llm_model.")
        tree_rag_engine = TreeRAG(tokenizer=tokenizer, vector_db=vector_db, llm=llm)
    else:
        tree_rag_engine = None
    if args.documents is not None:
        document_paths = args.documents
        for file_path in document_paths:
            File.ProcessDocuments(file_path, tokenizer, vector_db, tree_rag_engine=tree_rag_engine)

    if args.document_dir is not None:
        for x in args.document_dir:
            File.ProcessDocuments(x, tokenizer, vector_db, tree_rag_engine=tree_rag_engine)
            
    if args.wikipedia:
        wikipedia = Wikipedia(tokenizer, vector_db)
        wikipedia.ingest_all(batch_size=args.batch_size)

    if args.huggingface_dataset:
        dataset_path = args.huggingface_dataset
        subset = args.hf_subset
        text_col = args.hf_text_col
        id_col = args.hf_id_col
        extra_cols = args.hf_extra_cols
        ingestor = HuggingFaceDataset(tokenizer, vector_db)
        ingestor.ingest(
            dataset_path=dataset_path,
            name=subset,           
            text_column=text_col,           
            id_column=id_col,            
            split=args.hf_split,                
            batch_size=args.batch_size,
            num_workers=args.workers,
            extra_metadata_columns=extra_cols 
        )

   
    if args.ask:
        print(f"\nQ: {args.ask}")
        print("A: ", end="", flush=True)
        
        result = rag_chat.query(args.ask, use_web_search=args.use_web, stream=True, self_correction=args.self_correction)
        
        if isinstance(result, dict):
            print(result.get('answer', ''))
            if result.get('sources'):
                print("\nSources:")
                for src in result['sources']:
                    print(f"  - {src['source']} ({src.get('type', 'unknown')})")
        elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
            full_text = ""
            for chunk in result:
                print(chunk, end="", flush=True)
                full_text += chunk
            print()
        else:
            print(result)
                
    elif args.chat:
        interactive_chat_mode(rag_chat, ingest_web_default=args.ingest_web, self_correction=args.self_correction)
        
    elif args.query is not None:
        search_system(query=args.query, tokenizer=tokenizer, vector_db=vector_db)
