from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector
from pathlib import Path
import math


class VectorDatabase:
    def __init__(self, host="localhost", port=6333, collection_name="documents", use_grpc=False, grpc_port=6334):
        self.client = QdrantClient(host=host, port=port, grpc_port=grpc_port, prefer_grpc=use_grpc)
        self.collection_name = collection_name
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
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

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def upload_text(self, doc_id, dense_vector, sparse_vector, metadata):
        indices = sparse_vector.nonzero().flatten().cpu().tolist()
        values = sparse_vector[indices].cpu().tolist()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=doc_id,
                    vector={
                        "text_dense": dense_vector.cpu().tolist(),
                        "text_sparse": SparseVector(indices=indices, values=values)
                    },
                    payload=metadata
                )
            ]
        )

    def upload_image(self, doc_id, dense_vector, caption_dense_vector, sparse_caption_vector, metadata):
        if sparse_caption_vector is not None:
            indices = sparse_caption_vector.nonzero().flatten().cpu().tolist()
            values = sparse_caption_vector[indices].cpu().tolist()
        else:
            indices = []
            values = []

        cap_dense = caption_dense_vector.cpu().tolist() if caption_dense_vector is not None else [0.0] * 512

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=doc_id,
                    vector={
                        "image_dense": dense_vector.cpu().tolist(),
                        "caption_dense": cap_dense,
                        "caption_sparse": SparseVector(indices=indices, values=values)
                    },
                    payload=metadata
                )
            ]
        )

    def Search(self, tokenizer, query_text=None, query_image=None, top_k=20, tree_level: int = None):
        results = []
        query_filter = None
        if tree_level is not None:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="level", match=models.MatchValue(value=tree_level))]
            )
        if query_text:
            query_dense = tokenizer.DenseVectorRetrieval(textList=[query_text])[1]
            query_sparse = tokenizer.SparseTokens(query_text)
            sparse_indices = query_sparse.nonzero().flatten().cpu().tolist()
            sparse_values = query_sparse[sparse_indices].cpu().tolist()
            sparse_vec = models.SparseVector(indices=sparse_indices, values=sparse_values)

            text_query_results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=query_dense.tolist(), using="text_dense", limit=top_k, filter=query_filter),
                    models.Prefetch(query=sparse_vec, using="text_sparse", limit=top_k, filter=query_filter),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            ).points
            results.extend(text_query_results)

            image_query_results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=query_dense.tolist(), using="image_dense", limit=top_k, filter=query_filter),
                    models.Prefetch(query=query_dense.tolist(), using="caption_dense", limit=top_k, filter=query_filter),
                    models.Prefetch(query=sparse_vec, using="caption_sparse", limit=top_k, filter=query_filter),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            ).points
            results.extend(image_query_results)

        if query_image:
            img_dense = tokenizer.DenseVectorRetrieval(pil_image=query_image)[0]

            caption = tokenizer.ImageCaptioning(query_image)
            cap_dense = tokenizer.DenseVectorRetrieval(textList=[caption])[1]
            cap_sparse = tokenizer.SparseTokens(caption)
            cap_sparse_indices = cap_sparse.nonzero().flatten().cpu().tolist()
            cap_sparse_values = cap_sparse[cap_sparse_indices].cpu().tolist()
            cap_sparse_vec = models.SparseVector(indices=cap_sparse_indices, values=cap_sparse_values)

            image_query_results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=img_dense.cpu().tolist(), using="image_dense", limit=top_k, filter=query_filter),
                    models.Prefetch(query=cap_dense.cpu().tolist(), using="caption_dense", limit=top_k, filter=query_filter),
                    models.Prefetch(query=cap_sparse_vec, using="caption_sparse", limit=top_k, filter=query_filter),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            ).points
            results.extend(image_query_results)

        seen = set()
        unique_results = []
        for result in results:
            if result.id not in seen:
                unique_results.append(result)
                seen.add(result.id)
        unique_results.sort(key=lambda x: x.score, reverse=True)

        if query_text:
            image_results = [res for res in unique_results if res.payload.get('type') == 'image']
            pil_images = []
            valid_indices = []
            
            # Load images from cache (instead of re-parsing with PyMuPDF)
            for i, res in enumerate(image_results):
                cache_path = res.payload.get('image_cache_path')
                
                if cache_path and Path(cache_path).exists():
                    try:
                        pil_image = Image.open(cache_path).convert("RGB")
                        pil_images.append(pil_image)
                        valid_indices.append(i)
                    except Exception as e:
                        print(f"Warning: Could not load image from cache: {e}")
                        continue
                else:
                    print(f"Warning: Image cache not found: {cache_path}")

            if pil_images:
                captions = [image_results[idx].payload.get('caption', '') for idx in valid_indices]
                vision_scores = tokenizer.VisionRerank(query_text, pil_images, captions=captions)
                for i, score in enumerate(vision_scores):
                    idx = valid_indices[i]
                    image_results[idx].score += VectorDatabase.sigmoid(score) * 2.0

                # Clean up loaded images
                for img in pil_images:
                    img.close()

            candidates_text = [res.payload.get('caption', res.payload.get('text', '')) for res in unique_results]
            if candidates_text:
                cross_encoder_scores = tokenizer.CrossEncoderRanking(query_text, candidates_text)
                for i, score in enumerate(cross_encoder_scores):
                    unique_results[i].score += VectorDatabase.sigmoid(score) * 2.0

        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:top_k]