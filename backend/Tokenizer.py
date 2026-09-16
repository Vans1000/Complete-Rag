import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, LlavaForConditionalGeneration, logging
import os
import warnings
from transformers import logging
import numpy as np

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DISABLE_TRANSFORMERS_AUTOCONVERSION"] = "1" 
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

class Tokenizer:
    def __init__(self, dense_model="clip-ViT-B-32", sparse_model="naver/splade-v3", cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2", vision_rerank_model="nvidia/llama-nemotron-rerank-vl-1b-v2", caption_model="None"):
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        self.cross_encoder_model_name = cross_encoder_model
        self.vision_rerank_model_name = vision_rerank_model
        self.caption_model_name = caption_model
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        self.dense_model = SentenceTransformer(dense_model, model_kwargs={"use_fast": True}).to(self.device)
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_model, use_fast=True)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_model).to(self.device)
        self.cross_encoder_model = CrossEncoder(cross_encoder_model)
        processor_kwargs = {
            "trust_remote_code": True,
            "max_input_tiles": 6,
            "use_thumbnail": True,
            "rerank_max_length": 10240, 
            "use_fast": False
        }
        self.vision_processor = AutoProcessor.from_pretrained(
            vision_rerank_model, 
            **processor_kwargs
        )
        self.vision_model = AutoModelForSequenceClassification.from_pretrained(
            vision_rerank_model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",  
            device_map="auto",
            use_safetensors=True
        ).to(self.device).eval()
        
        if caption_model != "None":
            if 'moondream2' in str(caption_model):
                self.caption_model_type = 'moondream2'
                try:
                    self.caption_model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", trust_remote_code=True, dtype=torch.float16,use_safetensors=True).to(self.device).eval()
                except Exception as e:
                    print(f"Error loading caption model: {e}")
                    self.caption_model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", trust_remote_code=True, safetensors=False).to(self.device).eval()
                self.caption_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)
            if 'salesforce' in str(caption_model):
                self.caption_model_type = 'salesforce'
                self.caption_processor = BlipProcessor.from_pretrained(caption_model, use_fast=True)
                try:
                    self.caption_model = BlipForConditionalGeneration.from_pretrained(
                    caption_model, use_fast=True
                    ).to(self.device).eval()
                except Exception as e:
                    print(f"Error loading caption model: {e}")
                    self.caption_model = BlipForConditionalGeneration.from_pretrained(
                    caption_model, use_fast=True, safetensors=False
                    ).to(self.device).eval()
            if 'florence' in str(caption_model):
                self.caption_model_type = 'florence'
                self.caption_processor = AutoProcessor.from_pretrained(caption_model, trust_remote_code=True, use_fast=True)
                try:
                    self.caption_model = AutoModelForCausalLM.from_pretrained(
                        caption_model, 
                        trust_remote_code=True, 
                        dtype=torch.float16,
                        use_safetensors=True
                    ).to(self.device).eval()
                except Exception as e:
                    print(f"Error loading caption model: {e}")
                    self.caption_model = AutoModelForCausalLM.from_pretrained(
                        caption_model, 
                        trust_remote_code=True, 
                        dtype=torch.float16,
                        safetensors=False
                    ).to(self.device).eval()

            if 'llava' in str(caption_model):
                self.caption_model_type = 'llava'
                try:
                    self.caption_model = LlavaForConditionalGeneration.from_pretrained(
                    caption_model, 
                    torch_dtype=torch.float16, 
                    use_safetensors=True
                    ).to(self.device).eval()
                except Exception as e:
                    print(f"Error loading caption model: {e}")
                    self.caption_model = LlavaForConditionalGeneration.from_pretrained(
                    caption_model, 
                    torch_dtype=torch.float16, 
                    safetensors=False
                    ).to(self.device).eval()
                self.caption_processor = AutoProcessor.from_pretrained(caption_model)
        else:
            self.caption_model_type = None
        self.i = 0 
    def DenseVectorRetrieval(self, pil_image=None, textList=None):
        img_emb = None
        text_emb = None
        if pil_image is not None:
            with torch.no_grad():
                img_emb = self.dense_model.encode(pil_image, batch_size=512, convert_to_tensor=True, device=self.device)
            if img_emb.dim() > 1:
                img_emb = img_emb.squeeze()
        if textList is not None:
            with torch.no_grad():
                text_emb = self.dense_model.encode(textList, batch_size=512, convert_to_tensor=True, device=self.device)
            if text_emb.dim() > 1:
                text_emb = text_emb.squeeze()
        return img_emb, text_emb
    def SparseTokens(self, text):
        tokens = self.sparse_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.sparse_model(**tokens)
        vec = torch.max(torch.log(1 + torch.relu(outputs.logits))*tokens['attention_mask'].unsqueeze(-1), dim=1)[0]
        return vec.squeeze()
    def ImageCaptioning(self, pil_image):
        if self.caption_model_type is not None:
            if self.caption_model_type == 'moondream2':
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                caption = self.caption_model.caption(pil_image, length="short")
                caption = str(caption).strip('Caption: {\'caption\': \'').rstrip('\'}')
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                return caption
            if self.caption_model_type == 'llava':
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                prompt = "USER: <image>\nDescribe this image in great detail. ASSISTANT:"
                
                self.i += 1
                print(f"'{self.i}' Generating LLaVA caption...")

                inputs = self.caption_processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
                
                inputs["pixel_values"] = inputs["pixel_values"].to(self.caption_model.dtype)

                with torch.no_grad():
                    generated_ids = self.caption_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        num_beams=1,
                    )

                generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if "ASSISTANT:" in generated_text:
                    caption = generated_text.split("ASSISTANT:")[-1].strip()
                else:
                    caption = generated_text.strip()
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                return caption
            if self.caption_model_type == 'salesforce':
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.caption_model.generate(**inputs, max_new_tokens=50)
                
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                return caption
            if self.caption_model_type == 'florence':
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                prompt = "<MORE_DETAILED_CAPTION>"
                self.i += 1
                print(f"'{self.i}'Generating caption for image with prompt: '{prompt}'")
                inputs = self.caption_processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(self.caption_model.dtype)
                with torch.no_grad():

                    generated_ids = self.caption_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512,
                        do_sample=False,
                        num_beams=1,
                        use_cache=False, 
                    )

                generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                
                parsed_answer = self.caption_processor.post_process_generation(
                    generated_text, 
                    task=prompt, 
                    image_size=(pil_image.width, pil_image.height)
                )
                del inputs, generated_ids, generated_text
            
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                return parsed_answer[prompt]

        
    def CrossEncoderRanking(self, query, candidates):
        processed_candidates = []
        for item in candidates:
            if isinstance(item, Image.Image) or (isinstance(item, dict) and 'pil_image' in item):
                caption = self.ImageCaptioning(item)
                processed_candidates.append(caption)
            else:
                processed_candidates.append(item)
       
        pairs = [[query, txt] for txt in processed_candidates]
        scores = self.cross_encoder_model.predict(pairs)

        return scores.tolist()
    
    def VisionRerank(self, query_text, pil_images, captions=None, top_k=5):
    
        if not pil_images:
            return []

        examples = []
        for i, img in enumerate(pil_images):
            caption = captions[i] if captions and i < len(captions) else ""
            examples.append({
                "question": query_text,
                "doc_text": caption,  
                "doc_image": img
            })

        batch_dict = self.vision_processor.process_queries_documents_crossencoder(examples)
        
        batch_dict = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_dict.items()
        }

        with torch.no_grad():
            outputs = self.vision_model(**batch_dict, return_dict=True)
            
        logits = outputs.logits.squeeze(-1)
        
        return logits.cpu().tolist()
    
    
    
    
    def BatchSparseTokens(self, texts, batch_size=64):  
        vocab_size = self.sparse_model.config.vocab_size
        n_samples = len(texts)
        
        result = np.zeros((n_samples, vocab_size), dtype=np.float32)
        
        chunk_size = 64  
        
        for i in range(0, n_samples, chunk_size):
            end_i = min(i + chunk_size, n_samples)
            batch = texts[i:end_i]
            
            tokens = self.sparse_tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            with torch.no_grad():
                outputs = self.sparse_model(**tokens)
            
            activated = torch.log1p(torch.relu(outputs.logits)) * tokens['attention_mask'].unsqueeze(-1)
            vecs = torch.max(activated, dim=1)[0]
            
            result[i:end_i] = vecs.cpu().numpy()
            
            del outputs, activated, vecs, tokens
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        return torch.from_numpy(result)
    

    def BatchDenseVectorRetrieval(self, pil_images=None, text_lists=None, batch_size=64):
        """
        Memory-efficient batch dense embedding for large batches of images or texts.
        Processes in chunks to avoid OOM on GPU/MPS devices.
        """
        img_embeddings = []
        text_embeddings = []
        
        if pil_images is not None and len(pil_images) > 0:
            n_samples = len(pil_images)
            
            for i in range(0, n_samples, batch_size):
                end_i = min(i + batch_size, n_samples)
                batch = pil_images[i:end_i]
                
                with torch.no_grad():
                    batch_emb = self.dense_model.encode(
                        batch, 
                        batch_size=len(batch), 
                        convert_to_tensor=True, 
                        device=self.device,
                        show_progress_bar=False
                    )
                
                img_embeddings.append(batch_emb.cpu())
                
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()
            
            img_embeddings = torch.cat(img_embeddings, dim=0) if img_embeddings else None
        
        if text_lists is not None and len(text_lists) > 0:
            n_samples = len(text_lists)
            
            for i in range(0, n_samples, batch_size):
                end_i = min(i + batch_size, n_samples)
                batch = text_lists[i:end_i]
                
                with torch.no_grad():
                    batch_emb = self.dense_model.encode(
                        batch, 
                        batch_size=len(batch), 
                        convert_to_tensor=True, 
                        device=self.device,
                        show_progress_bar=False
                    )
                
                text_embeddings.append(batch_emb.cpu())
                
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()
            
            text_embeddings = torch.cat(text_embeddings, dim=0) if text_embeddings else None
        
        return img_embeddings, text_embeddings