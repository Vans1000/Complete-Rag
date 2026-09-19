import os
import json
import openai
from typing import List, Dict, Optional, Generator, Union
from abc import ABC, abstractmethod
import requests


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Generator]:
        pass
    
    @abstractmethod
    def generate_with_context(self, query: str, context: str, stream: bool = False) -> Union[str, Generator]:
        pass


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = model
        self.temperature = temperature
        
        self.system_prompt = """You are a helpful AI assistant with access to retrieved documents and web search results. 
Use the provided context to answer the user's question. If the context doesn't contain the answer, say so clearly.

### LATEX & MATHEMATICAL FORMULAS:
- When returning mathematical equations or formulas present in the context, ALWAYS preserve valid LaTeX formatting.
- Use `$$ ... $$` for block/display equations and `$ ... $` for inline formulas.

### UNIT CONVERSION RULES:
- If the user is asking about a location in the United States, ALWAYS provide weather and measurements in Imperial units (Fahrenheit, mph, inches).
- If the source data is in Metric (Celsius, km/h), you MUST convert it to Imperial before responding.
- For non-US locations, use the units standard for that region unless the user specifies otherwise.

### CITATION RULES:
- Always cite your sources by placing the 'SOURCE LINK' and 'CONTENT' in parentheses at the end of the sentence, like (Source: https://example.com - Content: ...).
"""
    
    def chat(self, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Generator]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', 4096)
        )
        
        if stream:
            return (chunk.choices[0].delta.content or "" for chunk in response 
                    if chunk.choices[0].delta.content)
        else:
            return response.choices[0].message.content
    
    def generate_with_context(self, query: str, context: str, stream: bool = False, 
                             chat_history: Optional[List[Dict]] = None, force_web: bool = False) -> Union[str, Generator]:
        system_content = self.system_prompt
        if force_web:
            system_content += "\nEXTREMELY IMPORTANT: Prioritize the 'CRITICAL LIVE WEB DATA' provided. If local documents conflict with the web data, trust the web data."
        
        messages = [{"role": "system", "content": system_content}]
        if chat_history:
            messages.extend(chat_history)
        
        user_prompt = f"""Context:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above. Cite sources when applicable."""
        
        messages.append({"role": "user", "content": user_prompt})
        
        return self.chat(messages, stream=stream)


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "mlx-community/Qwen3.8-27B-4bit", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.system_prompt = """You are a helpful AI assistant. Use the provided context to answer questions accurately.
When returning mathematical formulas, format them in valid LaTeX notation ($$for block formulas,$ for inline formulas). Cite sources when possible."""

    def chat(self, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Generator]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": kwargs.get('temperature', 0.7)
            }
        }
        
        response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=stream)
        response.raise_for_status()
        
        if stream:
            def generate():
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'message' in data and 'content' in data['message']:
                                yield data['message']['content']
                        except json.JSONDecodeError:
                            continue
            return generate()
        else:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'message' in data:
                            full_response += data['message'].get('content', '')
                    except json.JSONDecodeError:
                        continue
            return full_response
    
    def generate_with_context(self, query: str, context: str, stream: bool = False,
                             chat_history: Optional[List[Dict]] = None) -> Union[str, Generator]:
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if chat_history:
            messages.extend([{"role": m["role"], "content": m["content"]} for m in chat_history])
        
        user_prompt = f"""Context:
{context}

Question: {query}"""
        
        messages.append({"role": "user", "content": user_prompt})
        return self.chat(messages, stream=stream)


class RAGChat:
    def __init__(self, llm: BaseLLM, vector_db, tokenizer, web_search: Optional = None):
        self.llm = llm
        self.vector_db = vector_db
        self.tokenizer = tokenizer
        self.web_search = web_search
        self.chat_history = []
   
    def clear_history(self):
        self.chat_history = []
        
    
    def _retrieve_and_generate(self, question: str, query_text: str, use_web_search: bool, 
                              force_web: bool, top_k: int, stream: bool, ingest_web: bool):
        """
        Returns: (result, sources, context)
        If stream=True, result is a generator.
        """
        context_parts = []
        sources = []
        
        if force_web and self.web_search:
            if ingest_web:
                web_results = self.web_search.search_and_ingest(query_text)
                web_context_parts = []
                for i, res in enumerate(web_results[:10], 1):
                    content = res.get('text', res.get('content', ''))[:2000]
                    web_context_parts.append(f"[Source {i}: {res['title']}]\nCONTENT: {content}")
                web_context = "\n\n".join(web_context_parts)
                if web_context:
                    context_parts.append(f"### CRITICAL LIVE WEB DATA:\n{web_context}")
                    sources.append({"source": "Live Web Search", "type": "web"})
            else:
                web_context = self.web_search.get_context_for_llm(query_text)
                if web_context:
                    context_parts.append(f"### CRITICAL LIVE WEB DATA:\n{web_context}")
                    sources.append({"source": "Live Web Search", "type": "web"})
        
        results = self.vector_db.Search(
            tokenizer=self.tokenizer,
            query_text=query_text,
            top_k=top_k
        )
        
        for i, res in enumerate(results, 1):
            if force_web and res.score < 0.7:
                continue
            payload = res.payload
            content = payload.get('text', payload.get('caption', ''))
            file_path = payload.get('path', payload.get('source', 'Unknown'))
            page_num = payload.get('page', 'N/A')
            source_identifier = f"{file_path}: Page {page_num}"
            context_parts.append(f"{i}. SOURCE LINK: {source_identifier}\nCONTENT: {content}")
            sources.append({
                "source": source_identifier,
                "score": res.score,
                "type": payload.get('type', 'text')
            })
        
        if use_web_search and not force_web and self.web_search:
            if ingest_web:
                web_results = self.web_search.search_and_ingest(query_text)
                web_context_parts = []
                for i, res in enumerate(web_results[:10], 1):
                    content = res.get('text', res.get('content', ''))[:2000]
                    web_context_parts.append(f"[Source {i}: {res['title']}]\nCONTENT: {content}")
                web_context = "\n\n".join(web_context_parts)
                if web_context:
                    context_parts.append(f"[Web Search Results]\n{web_context}")
                    sources.append({"source": "web_search", "type": "web"})
            else:
                web_context = self.web_search.get_context_for_llm(query_text)
                if web_context:
                    context_parts.append(f"[Web Search Results]\n{web_context}")
                    sources.append({"source": "web_search", "type": "web"})
        
        context = "\n\n".join(context_parts)
        
        if not context_parts:
            empty_result = {"answer": "No relevant context found.", "sources": []}
            if stream:
                def empty_gen():
                    yield "No relevant context found."
                return empty_gen(), [], ""
            else:
                return empty_result, [], ""
        
        # Generation
        if stream:
            response_gen = self.llm.generate_with_context(
                question, context, stream=True, chat_history=self.chat_history[-5:]
            )
            return response_gen, sources, context
        else:
            response = self.llm.generate_with_context(
                question, context, stream=False, chat_history=self.chat_history[-5:]
            )
            return {"answer": response, "sources": sources}, sources, context


    def _rewrite_query(self, original_query: str, feedback: str) -> str:
        """Rewrite query to improve results"""
        prompt = f"The search query '{original_query}' failed because: {feedback}. Rewrite it to be more specific and likely to retrieve relevant information. Output ONLY the improved query."
        try:
            result = self.llm.chat([{"role": "user", "content": prompt}], stream=False)
            return result.strip().strip('"\'')
        except:
            return original_query

    def _evaluate_answer(self, question: str, context: str, answer: str) -> tuple[bool, str]:
        answer_str = str(answer).lower()
        
        failure_indicators = [
            "does not contain",
            "does not have", 
            "no information",
            "i am sorry",
            "i apologize",
            "unable to find",
            "no relevant context",
            "not found in",
            "cannot answer",
            "don't have access"
        ]
        
        if any(indicator in answer_str for indicator in failure_indicators):
            return False, "insufficient_context"
        
        if len(context.strip()) < 100:
            return False, "no_context_retrieved"
            
        return True, ""

    def query(self, question: str, use_web_search: bool = False, force_web: bool = False, 
              top_k: int = 5, stream: bool = False, ingest_web: bool = False,
              max_iterations: int = 2, self_correction: bool = False) -> Union[str, Generator, dict]:
   
        if not self_correction:
            response, sources, context = self._retrieve_and_generate(
                question, question, use_web_search, force_web, top_k, stream, ingest_web
            )
            
            if stream:
                # Yield chunks directly without buffering
                full_response = ""
                for chunk in response:
                    full_response += chunk
                    yield chunk
                self.chat_history.append({"role": "user", "content": question})
                self.chat_history.append({"role": "assistant", "content": full_response})
            else:
                self.chat_history.append({"role": "user", "content": question})
                self.chat_history.append({"role": "assistant", "content": response.get('answer', str(response))})
                return response
        
        current_query = question
        current_force_web = force_web
        current_use_web = use_web_search
        
        for attempt in range(max_iterations):
            result, sources, context = self._retrieve_and_generate(
                question, current_query, current_use_web, current_force_web, 
                top_k, False, ingest_web
            )
            
            answer_text = result.get('answer', '') if isinstance(result, dict) else str(result)
            
            if attempt < max_iterations - 1:
                is_good, feedback = self._evaluate_answer(question, context, answer_text)
                if not is_good:
                    print(f"[Self-Correction] Attempt {attempt+1} failed: {feedback}")
                    
                    if self.web_search and not current_force_web:
                        current_force_web = True
                        current_use_web = True
                        print(f"[Self-Correction] Escalating to forced web search...")
                    else:
                        current_query = self._rewrite_query(current_query, feedback)
                        print(f"[Self-Correction] Rewriting query: {current_query}")
                    continue
            
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer_text})
            
            if stream:
                def gen():
                    yield answer_text
                return gen()
            else:
                return result
        
        return result