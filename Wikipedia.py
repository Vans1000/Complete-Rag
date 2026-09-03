import uuid
import multiprocessing as mp
import queue
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter

def _worker_fetch_and_split(worker_id, num_workers, output_queue):
    """Worker process to fetch, shard, and chunk Wikipedia articles independently."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=500, separators=["\n\n", "\n", " ", ""]
    )
    
    # Streaming + sharding natively implements the "article_num % num_workers == worker_id" logic
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    shard = ds.shard(num_shards=num_workers, index=worker_id)
    
    try:
        for article in shard:
            chunks = splitter.split_text(article['text'])
            metadatas = []
            for i, chunk_text in enumerate(chunks):
                metadatas.append({
                    "text": chunk_text, 
                    "title": article['title'], 
                    "chunk_idx": i+1,
                    "type": "text", 
                    "page": 0, 
                    "path": f"wikipedia/{article['title']}"
                })
            # Send processed article to main process (blocks if queue is full to prevent OOM)
            output_queue.put((chunks, metadatas))
    except Exception as e:
        print(f"Worker {worker_id} encountered an error: {e}")
    finally:
        # Signal to the main process that this worker is completely done
        output_queue.put(None)

class Wikipedia:
    def __init__(self, tokenizer, vector_db):
        self.tokenizer = tokenizer
        self.vector_db = vector_db

    def ingest_all(self, batch_size=1024, num_workers=4):
        print(f"🚀 Running Multi-Process Distributed Ingestion (Batch Size: {batch_size}, Workers: {num_workers})")
        
        # Huge queue capacity to buffer chunks and aggressively utilize extra RAM
        output_queue = mp.Queue(maxsize=20000) 
        
        workers = []
        for i in range(num_workers):
            p = mp.Process(target=_worker_fetch_and_split, args=(i, num_workers, output_queue))
            p.start()
            workers.append(p)
            
        active_workers = num_workers
        batch_chunks = []
        batch_metadatas = []
        
        article_count = 0
        total_chunks_ingested = 0
        
        while active_workers > 0:
            try:
                # Timeout unblocks the loop periodically (safe checks)
                item = output_queue.get(timeout=1)
            except queue.Empty:
                continue
                
            if item is None:
                active_workers -= 1
                continue
                
            chunks, metadatas = item
            batch_chunks.extend(chunks)
            batch_metadatas.extend(metadatas)
            article_count += 1
            
            # Slice strictly into maximum batch chunks to maximize GPU efficiency
            while len(batch_chunks) >= batch_size:
                process_chunks = batch_chunks[:batch_size]
                process_metadatas = batch_metadatas[:batch_size]
                
                batch_chunks = batch_chunks[batch_size:]
                batch_metadatas = batch_metadatas[batch_size:]
                
                self._process_and_upload_batch(process_chunks, process_metadatas)
                total_chunks_ingested += len(process_chunks)
                print(f"✅ Progress: ~{article_count} articles | Total Chunks Uploaded: {total_chunks_ingested}", end='\r')
                
        # Flush any remaining chunks trailing at the end
        if len(batch_chunks) > 0:
            self._process_and_upload_batch(batch_chunks, batch_metadatas)
            total_chunks_ingested += len(batch_chunks)
            
        for p in workers:
            p.join()
            
        print(f"\n\n✨ INGESTION COMPLETE ✨")
        print(f"Final Count: {article_count} articles")
        print(f"Total Points in Qdrant: {total_chunks_ingested}")

    def _process_and_upload_batch(self, texts, metadatas):
        _, dense_vectors = self.tokenizer.BatchDenseVectorRetrieval(text_lists=texts)
        sparse_vectors = self.tokenizer.BatchSparseTokens(texts)

        for i in range(len(texts)):
            string_id = f"wiki_{metadatas[i]['title']}_ch{metadatas[i]['chunk_idx']}"
            doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))
            self.vector_db.upload_text(doc_id=doc_uuid, dense_vector=dense_vectors[i], 
                                     sparse_vector=sparse_vectors[i], metadata=metadatas[i])