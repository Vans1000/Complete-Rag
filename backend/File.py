from pathlib import Path
from PIL import Image
import io
import uuid
import hashlib
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling_core.types.doc import PictureItem, TextItem, FormulaItem, TableItem
from docling.datamodel import settings
from Tokenizer import Tokenizer
from VectorDatabase import VectorDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
import TreeRag
from typing import Optional
class File:
    def __init__(self, file_path, tokenizer=None, vector_db=None):
        self.filePath = file_path
        self.tokenizer = tokenizer
        markdown, text, images = File.ProcessDocuments(file_path, tokenizer, vector_db)
        self.markdown = markdown
        self.text = text
        self.images = images

    @staticmethod
    def get_image_cache_path(file_path, img_idx):
        """Generate a cache path for extracted images"""
        cache_dir = Path(".image_cache")
        cache_dir.mkdir(exist_ok=True)
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return cache_dir / f"{file_hash}_{img_idx}.png"

    @staticmethod
    def pil_image_to_bytes(pil_image, format='PNG'):
        """Convert PIL Image to bytes"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return buffer.getvalue()

    @staticmethod
    def extract_pil_image_from_item(item):
        """Extract PIL image from a Docling PictureItem across different API versions"""
        # Try different attribute paths depending on docling version
        if hasattr(item, 'image') and item.image is not None:
            img = item.image
            if isinstance(img, Image.Image):
                return img
            if hasattr(img, 'pil_image') and img.pil_image is not None:
                return img.pil_image
            if hasattr(img, 'to_pil') and callable(img.to_pil):
                return img.to_pil()
            if callable(img):
                try:
                    return img()
                except Exception:
                    pass

        if hasattr(item, 'pil_image') and item.pil_image is not None:
            return item.pil_image

        if hasattr(item, 'get_image') and callable(item.get_image):
            try:
                return item.get_image()
            except Exception:
                pass

        return None

    @staticmethod
    def create_converter():
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 1.5
        

        try:

            pipeline_options.ocr_options = EasyOcrOptions(use_gpu=False)
        except ImportError:
            pass

        #settings.perf.doc_batch_size = 1  
        #settings.perf.doc_batch_concurrency = 1
        
        try:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            return converter
        except TypeError:
            return DocumentConverter(pipeline_options=pipeline_options)

    @staticmethod
    def Parsing(file_path, tokenizer=None):
        if tokenizer is None:
            tokenizer = Tokenizer()

        converter = File.create_converter()
        result = converter.convert(str(file_path))
        doc = result.document

        text_data = []
        image_data = []

        full_text = doc.export_to_markdown()

        # Prevent splitting inside equations by prioritizing newline/paragraph breaks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n$$\n\n", "\n\n", "\n", " ", ""]
        )

        page_texts = {}

        for item, level in doc.iterate_items():
            text_content = None
            
            if isinstance(item, TextItem):
                text_content = item.text
            
            elif isinstance(item, FormulaItem):
                # Check all common Docling LaTeX attributes
                latex_val = None
                if hasattr(item, 'latex') and item.latex:
                    latex_val = item.latex
                elif hasattr(item, 'text') and item.text:
                    latex_val = item.text
                elif hasattr(item, 'formula') and item.formula:
                    latex_val = item.formula

                if latex_val:
                    latex_str = str(latex_val).strip()
                    # Ensure properly formatted display LaTeX
                    if not (latex_str.startswith("$$") and latex_str.endswith("$$")):
                        latex_str = latex_str.strip("$")
                        text_content = f"$$ {latex_str} $$"
                    else:
                        text_content = latex_str
            
            elif isinstance(item, TableItem):
                if hasattr(item, 'export_to_markdown'):
                    text_content = item.export_to_markdown()
                else:
                    text_content = str(item.text) if hasattr(item, 'text') else None

            if text_content:
                page_no = 0 
                if hasattr(item, 'prov') and item.prov:
                    try:
                        p = item.prov[0]
                        if hasattr(p, 'page_no') and p.page_no is not None:
                            page_no = p.page_no
                    except (IndexError, AttributeError):
                        pass
                
                if page_no not in page_texts:
                    page_texts[page_no] = []
                
                page_texts[page_no].append(text_content)

        if not page_texts:
            page_texts[1] = [full_text]

        for page_num in sorted(page_texts.keys()):
            page_text = "\n\n".join(page_texts[page_num]).strip()
            if not page_text:
                continue

            chunks = text_splitter.split_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                _, text_dense = tokenizer.DenseVectorRetrieval(textList=[chunk])
                text_sparse = tokenizer.SparseTokens(chunk)

                display_page = page_num
                
                text_data.append({
                    'text': chunk,
                    'page': display_page,
                    'chunk_idx': chunk_idx + 1,
                    'type': 'text',
                    'dense_vector': text_dense,
                    'sparse_vector': text_sparse,
                    'path': str(file_path)
                }) 

        img_idx = 0
        for item, level in doc.iterate_items():
            if not isinstance(item, PictureItem):
                continue

            pil_image = File.extract_pil_image_from_item(item)

            if pil_image is None:
                img_idx += 1
                continue

            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            image_bytes = File.pil_image_to_bytes(pil_image)

            # Save to cache
            cache_path = File.get_image_cache_path(file_path, img_idx)
            pil_image.save(str(cache_path), 'PNG')

            # Get page number
            page_no = 1
            if hasattr(item, 'prov') and item.prov:
                try:
                    p = item.prov[0]
                    if hasattr(p, 'page_no') and p.page_no is not None:
                        page_no = p.page_no
                except (IndexError, AttributeError):
                    pass

            caption = tokenizer.ImageCaptioning(pil_image)
            if caption is None:
                caption = ""
                caption_dense_vector = None
                caption_sparse_vector = None
            else:
                _, caption_dense_vector = tokenizer.DenseVectorRetrieval(textList=[caption])
                caption_sparse_vector = tokenizer.SparseTokens(caption)

            image_dense_vector, _ = tokenizer.DenseVectorRetrieval(pil_image=pil_image)

            image_data.append({
                'image_bytes': image_bytes,
                'image_cache_path': str(cache_path),
                'caption': caption,
                'dense_vector': image_dense_vector,
                'dense_caption_vector': caption_dense_vector,
                'sparse_caption_vector': caption_sparse_vector,
                'page': page_no,
                'img_idx': img_idx + 1,
                'type': 'image',
                'path': str(file_path)
            })

            img_idx += 1
            del pil_image

        return full_text, text_data, image_data


    @staticmethod
    def ProcessDocuments(path, tokenizer, vector_db, tree_rag_engine: Optional['TreeRAG'] = None):
        if Path(path).is_file():
            full_text, text_data, image_data = File.Parsing(path, tokenizer)
            
            if tree_rag_engine:
            # TreeRAG handles embedding and hierarchical ingestion for text
                tree_rag_engine.build_tree_and_ingest(leaf_chunks=text_data)
            else:
            # Standard flat chunking upload
                for idx, data in enumerate(text_data):
                    dense_vec = data.pop('dense_vector')
                    sparse_vec = data.pop('sparse_vector')
                    string_id = f"{Path(path).name}_text_{idx}"
                    doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))
                    vector_db.upload_text(
                        doc_id=doc_uuid,
                        dense_vector=dense_vec,
                        sparse_vector=sparse_vec,
                        metadata=data
                    )

            for idx, data in enumerate(image_data):
                dense_vector = data.pop('dense_vector')
                caption_dense_vector = data.pop('dense_caption_vector')
                sparse_caption_vector = data.pop('sparse_caption_vector')
                string_id = f"{Path(path).name}_img_{idx}"
                doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))
                data.pop('image_bytes', None)
                vector_db.upload_image(
                    doc_id=doc_uuid,
                    dense_vector=dense_vector,
                    caption_dense_vector=caption_dense_vector,
                    sparse_caption_vector=sparse_caption_vector,
                    metadata=data
                )
            return full_text, text_data, image_data
        else:
            for x in Path(path).iterdir():
                File.ProcessDocuments(x, tokenizer, vector_db, tree_rag_engine=tree_rag_engine)
