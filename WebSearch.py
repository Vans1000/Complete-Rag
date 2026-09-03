import re
import uuid
import asyncio
import aiohttp
import trafilatura
from ddgs import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urlparse
from typing import List, Dict, Optional
import hashlib

class WebSearch:
    def __init__(self, tokenizer, vector_db=None, max_results=10):
        self.tokenizer = tokenizer
        self.vector_db = vector_db
        self.max_results = max_results
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def search(self, query: str, num_results: int = None) -> List[Dict]:
        """Perform web search using DuckDuckGo"""
        num_results = num_results or self.max_results
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        
        return [{
            'title': r['title'],
            'href': r['href'],
            'body': r['body']
        } for r in results]
    
    async def fetch_content(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch and extract clean text from URL with fallback to meta tags"""
        try:
            connector = aiohttp.TCPConnector(limit=0)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, timeout=timeout, ssl=False, read_bufsize=2**16) as response:
                    html = await response.text()
                    content = trafilatura.extract(html, include_comments=False, 
                                                include_tables=True, 
                                                include_links=False,
                                                deduplicate=True)

                    if not content or len(content.strip()) < 100:
                        meta_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                        if meta_match:
                            content = meta_match.group(1)
                        else:
                            meta_match = re.search(r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                            if meta_match:
                                content = meta_match.group(1)

                    return content or ""
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    async def search_and_fetch(self, query: str, ingest: bool = False) -> List[Dict]:
        """Search and fetch full content from results"""
        search_results = self.search(query)
        
        tasks = [self.fetch_content(r['href']) for r in search_results]
        contents = await asyncio.gather(*tasks)
        
        results = []
        for sr, content in zip(search_results, contents):
            if content:
                results.append({
                    'title': sr['title'],
                    'url': sr['href'],
                    'snippet': sr['body'],
                    'content': content
                })
                
                if ingest and self.vector_db:
                    self._ingest_web_content(content, sr['title'], sr['href'])
        
        return results
    
    def _ingest_web_content(self, content: str, title: str, url: str):
        """Chunk and ingest web content into vector DB"""
        chunks = self.splitter.split_text(content)
        
        for i, chunk in enumerate(chunks):
            _, dense_vector = self.tokenizer.DenseVectorRetrieval(textList=[chunk])
            sparse_vector = self.tokenizer.SparseTokens(chunk)
            
            url_hash = hashlib.md5(f"{url}_{i}".encode()).hexdigest()
            doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, url_hash))
            
            metadata = {
                'text': chunk,
                'title': title,
                'source': url,
                'chunk_idx': i + 1,
                'type': 'web',
                'path': url
            }
            
            self.vector_db.upload_text(
                doc_id=doc_uuid,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                metadata=metadata
            )
    
    def search_and_ingest(self, query: str) -> List[Dict]:
        """Synchronous wrapper for search and fetch with ingestion"""
        return asyncio.run(self.search_and_fetch(query, ingest=True))
    
    def get_context_for_llm(self, query: str, top_k: int = 3) -> str:
        """Search web and format results as context string for LLM"""
        results = asyncio.run(self.search_and_fetch(query, ingest=False))
        
        context_parts = []
        for i, result in enumerate(results[:top_k], 1):
            context_parts.append(
                f"[Source {i}: {result['title']}]\n"
                f"SOURCE LINK: {result['url']}\n"
                f"TITLE: {result['title']}\n"
                f"CONTENT: {result['content'][:2000]}..."
            )
        
        return "\n\n".join(context_parts)