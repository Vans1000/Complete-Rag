import uuid
import multiprocessing as mp
import queue
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter

def _worker_fetch_and_split(worker_id, num_workers, output_queue, dataset_config):
    """
    Worker process to fetch, shard, and chunk ANY Hugging Face dataset.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=dataset_config.get("chunk_size", 3000), 
        chunk_overlap=dataset_config.get("chunk_overlap", 500), 
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Load dataset dynamically based on config
    ds = load_dataset(
        dataset_config["path"], 
        dataset_config.get("name"), 
        split=dataset_config.get("split", "train"), 
        streaming=True
    )
    shard = ds.shard(num_shards=num_workers, index=worker_id)
    
    text_column = dataset_config.get("text_column", "text")
    id_column = dataset_config.get("id_column", "id")

    try:
        for entry in shard:
            # Extract text and handle potential missing keys
            content = entry.get(text_column, "")
            if not content:
                continue

            chunks = splitter.split_text(content)
            metadatas = []
            
            # Identify unique identifier for the source (e.g., Title or ID)
            source_id = str(entry.get(id_column, uuid.uuid4().hex))

            for i, chunk_text in enumerate(chunks):
                # Build metadata dynamically by including all original columns or specific ones
                meta = {
                    "text": chunk_text, 
                    "source_id": source_id,
                    "chunk_idx": i + 1,
                    "type": "text",
                    "path": f"hf://{dataset_config['path']}/{source_id}"
                }
                # Optional: Merge other useful columns into metadata
                for col in dataset_config.get("extra_metadata_columns", []):
                    if col in entry:
                        meta[col] = entry[col]

                metadatas.append(meta)
            
            output_queue.put((chunks, metadatas))
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
    finally:
        output_queue.put(None)

class HuggingFaceDataset:
    def __init__(self, tokenizer, vector_db):
        self.tokenizer = tokenizer
        self.vector_db = vector_db

    def ingest(self, dataset_path, text_column="text", id_column="id", 
               name=None, split="train", batch_size=1024, num_workers=4, 
               extra_metadata_columns=None):
        
        dataset_config = {
            "path": dataset_path,
            "name": name,
            "split": split,
            "text_column": text_column,
            "id_column": id_column,
            "extra_metadata_columns": extra_metadata_columns or [],
            "chunk_size": 3000,
            "chunk_overlap": 500
        }

        print(f"🚀 Ingesting {dataset_path} (Text: {text_column}, Workers: {num_workers})")
        
        output_queue = mp.Queue(maxsize=10000) 
        workers = []
        for i in range(num_workers):
            p = mp.Process(target=_worker_fetch_and_split, args=(i, num_workers, output_queue, dataset_config))
            p.start()
            workers.append(p)
            
        active_workers = num_workers
        batch_chunks = []
        batch_metadatas = []
        total_ingested = 0
        
        while active_workers > 0:
            try:
                item = output_queue.get(timeout=1)
            except queue.Empty:
                continue
                
            if item is None:
                active_workers -= 1
                continue
                
            chunks, metadatas = item
            batch_chunks.extend(chunks)
            batch_metadatas.extend(metadatas)
            
            while len(batch_chunks) >= batch_size:
                self._process_and_upload_batch(batch_chunks[:batch_size], batch_metadatas[:batch_size])
                total_chunks = len(batch_chunks[:batch_size])
                total_ingested += total_chunks
                batch_chunks = batch_chunks[batch_size:]
                batch_metadatas = batch_metadatas[batch_size:]
                print(f"✅ Total Chunks Uploaded: {total_ingested}", end='\r')
                
        if batch_chunks:
            self._process_and_upload_batch(batch_chunks, batch_metadatas)
            total_ingested += len(batch_chunks)
            
        for p in workers:
            p.join()
        print(f"\n✨ Ingestion Complete. Total points: {total_ingested}")

    def _process_and_upload_batch(self, texts, metadatas):
        _, dense_vectors = self.tokenizer.BatchDenseVectorRetrieval(text_lists=texts)
        sparse_vectors = self.tokenizer.BatchSparseTokens(texts)

        for i in range(len(texts)):
            # Generate a consistent UUID based on the source and chunk index
            unique_key = f"{metadatas[i]['source_id']}_ch{metadatas[i]['chunk_idx']}"
            doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_key))
            self.vector_db.upload_text(
                doc_id=doc_uuid, 
                dense_vector=dense_vectors[i], 
                sparse_vector=sparse_vectors[i], 
                metadata=metadatas[i]
            )