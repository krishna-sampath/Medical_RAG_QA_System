#!/usr/bin/env python3
"""
RAG Pipeline for Medical Document Analysis
==========================================

This script implements a complete RAG (Retrieval-Augmented Generation) pipeline
for analyzing medical documents and answering queries about medical classifications.

Components:
1. Document Ingestion and Chunking
2. Embedding and Vector Store
3. Query Interface
4. LLM Integration (Fast Mode)
5. Output Generation
6. Gradio Web Interface (Bonus)
7. Caching and Relevance Reranking (Bonus)

Author: AI Assistant
Date: 2024
"""

import os
import re
import logging
import hashlib
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# PDF processing
import pypdf

# Gradio for web interface
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of query-answer pairs."""
    
    def __init__(self, cache_file="cache/query_cache.pkl"):
        self.cache_file = cache_file
        self.cache = self._load_query_cache()
    
    def _load_query_cache(self):
        """Load cached queries from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    logger.info(f"Loaded {len(cache)} cached queries")
                    return cache
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_query_cache(self):
        """Save cache to file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _generate_cache_key(self, query, use_real_llm=False, use_mmr=False):
        """Generate cache key based on query and settings."""
        key_data = f"{query}_{use_real_llm}_{use_mmr}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_answer(self, query, use_real_llm=False, use_mmr=False):
        """Get cached answer if available."""
        key = self._generate_cache_key(query, use_real_llm, use_mmr)
        return self.cache.get(key)
    
    def cache_answer(self, query, answer, use_real_llm=False, use_mmr=False):
        """Cache a query-answer pair."""
        key = self._generate_cache_key(query, use_real_llm, use_mmr)
        self.cache[key] = answer
        self._save_query_cache()
    
    def clear_cache(self):
        """Clear all cached queries."""
        self.cache = {}
        self._save_query_cache()
    
    def get_cache_stats(self):
        """Get cache statistics."""
        cache_size = len(self.cache)
        cache_size_mb = os.path.getsize(self.cache_file) / (1024 * 1024) if os.path.exists(self.cache_file) else 0
        return {
            "total_cached_queries": cache_size,
            "cache_size_mb": cache_size_mb
        }


class MedicalRAGPipeline:
    """
    Medical RAG Pipeline for processing medical documents and answering classification questions.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 vector_store_type: str = "faiss"):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: HuggingFace model name for embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            vector_store_type: Type of vector store (faiss or chroma)
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_type = vector_store_type
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.cache_manager = CacheManager()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline components."""
        try:
            logger.info("Initializing RAG pipeline components...")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            logger.info("Text splitter initialized")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Embeddings initialized with model: {self.embedding_model}")
            
            # Initialize basic LLM (will be replaced by real LLM when needed)
            self.llm = None
            logger.info("LLM initialized (Real LLM mode)")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _initialize_real_llm(self):
        """Initialize real LLM for accurate responses."""
        try:
            logger.info("Initializing real LLM...")
            
            # Use a smaller, more suitable model for medical text generation
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_name = "microsoft/DialoGPT-medium"  # Better for conversational responses
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Create pipeline with optimized settings
            from transformers import pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,  # Increased for better responses
                temperature=0.3,  # Lower temperature for more focused responses
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2  # Prevent repetitive text
            )
            
            # Create LangChain wrapper
            from langchain.llms import HuggingFacePipeline
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            logger.info(f"Real LLM initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing real LLM: {e}")
            raise
    
    def setup_qa_chain(self):
        """Set up the question-answering chain with real LLM."""
        try:
            if self.vector_store is None:
                logger.warning("Vector store not available, skipping QA chain setup")
                return
            
            # Create a clean prompt template for medical classifications
            prompt_template = """You are a medical classification expert. Based on the provided medical document context, answer the question accurately and concisely.

Context: {context}

Question: {question}

Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": prompt}
            )
            
            logger.info("QA chain set up successfully")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text_content = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        else:
                            logger.warning(f"Page {page_num + 1} appears to be empty or image-based")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                
                logger.info(f"Extracted {len(text_content)} characters from PDF")
                return text_content
                
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            raise
    
    def chunk_document(self, text: str) -> List[str]:
        """
        Split document into manageable chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        try:
            logger.info("Chunking document...")
            
            # Clean the text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Log chunk statistics
            chunk_lengths = [len(chunk) for chunk in chunks]
            logger.info(f"Chunk statistics - Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}, Avg: {np.mean(chunk_lengths):.1f}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise
    
    def create_vector_store(self, chunks: List[str], store_path: str = "vector_store") -> None:
        """
        Create and populate vector store with document chunks.
        
        Args:
            chunks: List of text chunks
            store_path: Path to save vector store
        """
        try:
            logger.info("Creating vector store...")
            
            if self.vector_store_type == "faiss":
                # Create FAISS vector store
                self.vector_store = FAISS.from_texts(
                    chunks, 
                    self.embeddings
                )
                
                # Save the vector store
                self.vector_store.save_local(store_path)
                logger.info(f"FAISS vector store saved to {store_path}")
                
            elif self.vector_store_type == "chroma":
                # Create Chroma vector store
                self.vector_store = Chroma.from_texts(
                    chunks,
                    self.embeddings,
                    persist_directory=store_path
                )
                logger.info(f"Chroma vector store saved to {store_path}")
            
            logger.info(f"Vector store created with {len(chunks)} documents")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vector_store(self, store_path: str) -> None:
        """
        Load existing vector store.
        
        Args:
            store_path: Path to vector store
        """
        try:
            logger.info(f"Loading vector store from {store_path}")
            
            if self.vector_store_type == "faiss":
                self.vector_store = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
            elif self.vector_store_type == "chroma":
                self.vector_store = Chroma(
                    persist_directory=store_path,
                    embedding_function=self.embeddings
                )
            
            logger.info("Vector store loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def rerank_with_mmr(self, query: str, k: int = 5) -> List[Document]:
        """Rerank documents using Maximal Marginal Relevance."""
        try:
            # Get initial similarity search results
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            # Extract documents and scores
            docs = [doc for doc, score in docs_and_scores]
            scores = [score for doc, score in docs_and_scores]
            
            # Apply MMR reranking
            selected_docs = []
            selected_indices = []
            
            # Select the first document (highest similarity)
            selected_indices.append(0)
            selected_docs.append(docs[0])
            
            # For remaining documents, use MMR
            for i in range(1, min(k, len(docs))):
                max_mmr_score = -1
                best_idx = -1
                
                for j in range(len(docs)):
                    if j in selected_indices:
                        continue
                    
                    # Calculate MMR score: λ * similarity - (1-λ) * redundancy
                    similarity = scores[j]
                    
                    # Calculate redundancy (max similarity to already selected docs)
                    redundancy = max([
                        self._calculate_similarity(docs[j].page_content, docs[selected_idx].page_content)
                        for selected_idx in selected_indices
                    ])
                    
                    # MMR score (λ = 0.7 for balance between relevance and diversity)
                    lambda_param = 0.7
                    mmr_score = lambda_param * similarity - (1 - lambda_param) * redundancy
                    
                    if mmr_score > max_mmr_score:
                        max_mmr_score = mmr_score
                        best_idx = j
                
                if best_idx != -1:
                    selected_indices.append(best_idx)
                    selected_docs.append(docs[best_idx])
            
            logger.info(f"MMR reranking completed: {len(selected_docs)} results")
            return selected_docs
            
        except Exception as e:
            logger.error(f"Error in MMR reranking: {e}")
            # Fallback to regular similarity search
            return self.vector_store.similarity_search(query, k=k)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        try:
            # Simple cosine similarity using embeddings
            embedding1 = self.embeddings.embed_query(text1)
            embedding2 = self.embeddings.embed_query(text2)
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def query(self, question: str, use_cache: bool = True, use_mmr: bool = False, use_real_llm: bool = True, bypass_cache: bool = False) -> str:
        """Query the RAG pipeline."""
        logger.info(f"Processing query: {question}")
        
        # Check cache first (unless bypassing)
        if use_cache and not bypass_cache:
            cached_answer = self.cache_manager.get_cached_answer(question, use_real_llm, use_mmr)
            if cached_answer:
                logger.info("Returning cached answer")
                return cached_answer
        
        # Load vector store if not already loaded
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            try:
                self.load_vector_store("medical_vector_store")
            except Exception as e:
                logger.error(f"Could not load vector store: {e}")
                return "Error: No document has been processed yet. Please upload a PDF file first."
        
        # Perform similarity search
        try:
            if use_mmr:
                docs = self.rerank_with_mmr(question, k=5)
            else:
                docs = self.vector_store.similarity_search(question, k=5)
            
            # Combine context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate answer using Real LLM
            if self.llm:
                try:
                    # Use real LLM with direct invocation
                    logger.info("Using real LLM for RAG response")
                    
                    # Construct a focused prompt for medical classification
                    prompt = f"""You are a medical classification expert. Based on the provided medical document context, answer the question accurately and concisely.

Context: {context}

Question: {question}

Answer:"""
                    
                    # Use the LLM directly
                    result = self.llm.invoke(prompt)
                    
                    # Extract the answer
                    answer = str(result).strip()
                    
                    # Clean up the answer - remove everything before "Answer:"
                    if "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()
                    
                    # If answer is still empty or just whitespace, try a different approach
                    if not answer or answer.strip() == "" or len(answer.strip()) < 20:
                        # Try with a more direct prompt
                        direct_prompt = f"Medical question: {question}\n\nMedical context: {context}\n\nProvide the ICD-10 classification:"
                        result = self.llm.invoke(direct_prompt)
                        answer = str(result).strip()
                        
                        # Clean up any prompt artifacts
                        if "Medical question:" in answer:
                            answer = answer.split("Medical question:")[-1].strip()
                        if "Medical context:" in answer:
                            answer = answer.split("Medical context:")[-1].strip()
                        if "Provide the ICD-10 classification:" in answer:
                            answer = answer.split("Provide the ICD-10 classification:")[-1].strip()
                    
                    # Debug: Log the actual response
                    logger.info(f"Real LLM raw result: {result}")
                    logger.info(f"Real LLM answer: '{answer}'")
                    
                    # Check if answer is empty or too short
                    if not answer or len(answer.strip()) < 10:
                        logger.warning(f"Real LLM response too short or empty: '{answer}'")
                        raise Exception("Real LLM generated empty or invalid response")
                    
                    logger.info("Real LLM response generated successfully")
                    if use_cache and not bypass_cache:
                        self.cache_manager.cache_answer(question, answer, use_real_llm, use_mmr)
                    return answer
                except Exception as e:
                    logger.warning(f"Real LLM failed: {e}")
                    return f"Error: Real LLM failed to generate response. Please try again or check your input."
            else:
                # LLM not initialized
                return "Error: Real LLM not initialized. Please ensure Real LLM is enabled."
                
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return f"Error processing query: {str(e)}"
    
    def process_document(self, pdf_path: str, store_path: str = "vector_store", setup_qa: bool = True) -> None:
        """
        Complete document processing pipeline.
        
        Args:
            pdf_path: Path to PDF document
            store_path: Path to save vector store
            setup_qa: Whether to set up QA chain for real RAG
        """
        try:
            logger.info("Starting document processing pipeline...")
            
            # 1. Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # 2. Chunk the document
            chunks = self.chunk_document(text)
            
            # 3. Create vector store
            self.create_vector_store(chunks, store_path)
            
            # 4. Set up QA chain for real RAG (optional)
            if setup_qa:
                try:
                    self.setup_qa_chain()
                    logger.info("QA chain set up for real RAG responses")
                except Exception as e:
                    logger.warning(f"Could not set up QA chain: {e}")
                    logger.info("Will use direct LLM responses only")
            
            logger.info("Document processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            raise
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_manager.get_cache_stats()


def create_gradio_interface():
    """Create Gradio web interface."""
    
    # Initialize pipeline
    rag_pipeline = MedicalRAGPipeline()
    
    def process_upload_and_query(file, question, use_cache, use_mmr, use_real_llm, bypass_cache):
        """Process uploaded file and answer question."""
        try:
            if file is None:
                return "Please upload a PDF file first.", ""
            
            # Initialize real LLM if requested
            if use_real_llm:
                try:
                    yield "🔄 Initializing Real LLM (this may take 1-2 minutes)...", ""
                    rag_pipeline._initialize_real_llm()
                    yield "✅ Real LLM initialized successfully!", ""
                except Exception as e:
                    return f"❌ Error initializing real LLM: {str(e)}", ""
            
            # Process the uploaded file
            temp_path = file.name
            yield "📄 Processing PDF document...", ""
            rag_pipeline.process_document(temp_path, "temp_vector_store", setup_qa=use_real_llm)
            yield "✅ Document processed successfully!", ""
            
            # Answer the question
            yield "🔍 Generating answer...", ""
            answer = rag_pipeline.query(question, use_cache=use_cache, use_mmr=use_mmr, use_real_llm=use_real_llm, bypass_cache=bypass_cache)
            
            # Debug: Log the final answer
            logger.info(f"Final answer for Gradio: '{answer}'")
            
            # Ensure we have a valid answer
            if not answer or answer.strip() == "":
                answer = "No answer generated. Please try again."
            
            final_result = f"✅ Document processed successfully!\n\n📄 File: {os.path.basename(temp_path)}\n\n❓ Question: {question}\n\n💡 Answer: {answer}"
            
            # Return the final result (this should update the Gradio interface)
            yield final_result, ""
            
        except Exception as e:
            logger.error(f"Error in process_upload_and_query: {e}")
            yield f"❌ Error: {str(e)}", ""
    
    def query_only(question, use_cache, use_mmr, use_real_llm, bypass_cache):
        """Query without uploading a new file."""
        try:
            if not question.strip():
                return "Please enter a question."
            
            # Initialize real LLM if requested
            if use_real_llm:
                try:
                    yield "🔄 Initializing Real LLM (this may take 1-2 minutes)..."
                    rag_pipeline._initialize_real_llm()
                    yield "✅ Real LLM initialized successfully!"
                except Exception as e:
                    return f"❌ Error initializing real LLM: {str(e)}"
            
            # Answer the question
            yield "🔍 Generating answer..."
            answer = rag_pipeline.query(question, use_cache=use_cache, use_mmr=use_mmr, use_real_llm=use_real_llm, bypass_cache=bypass_cache)
            
            # Debug: Log the final answer
            logger.info(f"Final answer for Gradio (query only): '{answer}'")
            
            # Ensure we have a valid answer
            if not answer or answer.strip() == "":
                answer = "No answer generated. Please try again."
            
            final_result = f"❓ Question: {question}\n\n💡 Answer: {answer}"
            
            # Return the final result (single string, not tuple)
            yield final_result
            
        except Exception as e:
            logger.error(f"Error in query_only: {e}")
            yield f"❌ Error: {str(e)}"
    
    def get_cache_stats():
        """Get cache statistics."""
        try:
            stats = rag_pipeline.get_cache_stats()
            return f"📊 Cache Statistics:\n\n" \
                   f"• Total cached queries: {stats['total_cached_queries']}\n" \
                   f"• Cache size: {stats['cache_size_mb']:.2f} MB"
        except Exception as e:
            return f"❌ Error getting cache stats: {str(e)}"
    
    def clear_cache():
        """Clear the cache."""
        try:
            rag_pipeline.cache_manager.clear_cache()
            return "✅ Cache cleared successfully!"
        except Exception as e:
            return f"❌ Error clearing cache: {str(e)}"
    
    # Create Gradio interface
    with gr.Blocks(title="Medical RAG Pipeline", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🏥 Medical RAG Pipeline")
        gr.Markdown("Upload a medical PDF document and ask questions about medical classifications.")
        
        with gr.Tab("📄 Upload & Query"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is the ICD-10 classification for recurrent depressive disorder in remission?",
                        lines=3
                    )
                    with gr.Row():
                        cache_checkbox = gr.Checkbox(label="Use Caching", value=True)
                        mmr_checkbox = gr.Checkbox(label="Use MMR Reranking", value=True)
                        real_llm_checkbox = gr.Checkbox(label="Use Real LLM (slower but more accurate)", value=True)
                        bypass_cache_checkbox = gr.Checkbox(label="Bypass Cache", value=False)
                    submit_btn = gr.Button("🚀 Process & Query", variant="primary")
                
                with gr.Column():
                    output = gr.Textbox(label="Results", lines=10)
                    error_output = gr.Textbox(label="Status", lines=3)
            
            submit_btn.click(
                process_upload_and_query,
                inputs=[file_input, question_input, cache_checkbox, mmr_checkbox, real_llm_checkbox, bypass_cache_checkbox],
                outputs=[output, error_output],
                show_progress=True
            )
        
        with gr.Tab("❓ Query Only"):
            gr.Markdown("**Note**: This tab allows you to ask questions about previously processed documents. Make sure you've uploaded and processed a document in the 'Upload & Query' tab first, or run the pipeline from command line.")
            
            with gr.Row():
                with gr.Column():
                    query_only_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about the processed document...",
                        lines=3
                    )
                    with gr.Row():
                        cache_checkbox2 = gr.Checkbox(label="Use Caching", value=True)
                        mmr_checkbox2 = gr.Checkbox(label="Use MMR Reranking", value=True)
                        real_llm_checkbox2 = gr.Checkbox(label="Use Real LLM (slower but more accurate)", value=True)
                        bypass_cache_checkbox2 = gr.Checkbox(label="Bypass Cache", value=False)
                    query_btn = gr.Button("🔍 Query", variant="primary")
                
                with gr.Column():
                    query_output = gr.Textbox(label="Answer", lines=8)
            
            query_btn.click(
                query_only,
                inputs=[query_only_input, cache_checkbox2, mmr_checkbox2, real_llm_checkbox2, bypass_cache_checkbox2],
                outputs=query_output,
                show_progress=True
            )
        
        with gr.Tab("⚙️ Cache Management"):
            with gr.Row():
                stats_btn = gr.Button("📊 Get Cache Stats", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear Cache", variant="stop")
            
            cache_output = gr.Textbox(label="Cache Information", lines=6)
            
            stats_btn.click(get_cache_stats, outputs=cache_output)
            clear_btn.click(clear_cache, outputs=cache_output)
        
        gr.Markdown("---")
        gr.Markdown("### Features")
        gr.Markdown("""
        - **Document Upload**: Upload PDF medical documents
        - **Intelligent Querying**: Ask questions about medical classifications
        - **Caching**: Fast responses for repeated queries
        - **MMR Reranking**: Better context selection using Maximal Marginal Relevance
        - **Real LLM**: Uses actual language model for accurate responses
        - **Real-time Processing**: Immediate document processing and querying
        """)
        
        gr.Markdown("### Tips")
        gr.Markdown("""
        - **For accurate responses**: Real LLM provides context-aware medical classifications
        - **Large documents**: Processing may take 30-60 seconds
        - **LLM initialization**: Real LLM takes 1-2 minutes to initialize on first use
        - **Connection issues**: If you see 'connection lost', wait for the process to complete
        """)
    
    return interface


def main():
    """Main function to demonstrate the RAG pipeline."""
    
    print("=" * 60)
    print("MEDICAL RAG PIPELINE")
    print("=" * 60)
    
    # Ask user for mode preference
    print("\nSelect Mode:")
    print("1. Real LLM Mode (DialoGPT-medium - ~2 minutes total)")
    print("2. Interactive Mode (Real LLM with options)")
    print("3. Gradio Web Interface")
    
    while True:
        try:
            choice = input("\nEnter your choice (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    # Determine initial mode
    if choice == '1':
        use_real_llm = True
        print("\nStarting in Real LLM Mode (this will take ~1 minute to initialize)...")
    elif choice == '2':
        use_real_llm = True
        print("\nStarting in Interactive Mode (Real LLM)...")
    else:  # choice == '3'
        print("\nStarting Gradio web interface...")
        create_gradio_interface().launch(server_name="0.0.0.0", server_port=7860)
        return
    
    try:
        # Create pipeline instance
        rag_pipeline = MedicalRAGPipeline()
        
        # Initialize real LLM
        if use_real_llm:
            print("🔄 Initializing Real LLM (this may take 1-2 minutes)...")
            rag_pipeline._initialize_real_llm()
            print("✅ Real LLM initialized successfully!")
        
        # Process document
        pdf_path = "9241544228_eng.pdf"
        if os.path.exists(pdf_path):
            print(f"📄 Processing document: {pdf_path}")
            rag_pipeline.process_document(pdf_path, "medical_vector_store", setup_qa=use_real_llm)
            print("✅ Document processed successfully!")
        else:
            print(f"❌ Document not found: {pdf_path}")
            return
        
        # Interactive querying
        if choice == '2':
            print("\n" + "=" * 40)
            print("INTERACTIVE MODE")
            print("=" * 40)
            print("Commands:")
            print("- Type your question to get an answer")
            print("- Type 'quit' to exit")
            print("- Type 'cache' to see cache statistics")
            print("- Type 'clear' to clear cache")
            print("=" * 40)
            
            while True:
                try:
                    user_input = input("\n❓ Your question: ").strip()
                    
                    if user_input.lower() == 'quit':
                        print("👋 Goodbye!")
                        break
                    elif user_input.lower() == 'cache':
                        stats = rag_pipeline.get_cache_stats()
                        print(f"📊 Cache Statistics:")
                        print(f"   • Total cached queries: {stats['total_cached_queries']}")
                        print(f"   • Cache size: {stats['cache_size_mb']:.2f} MB")
                    elif user_input.lower() == 'clear':
                        rag_pipeline.cache_manager.clear_cache()
                        print("🗑️ Cache cleared!")
                    elif user_input:
                        print("🔍 Generating answer...")
                        answer = rag_pipeline.query(user_input, use_real_llm=use_real_llm)
                        print(f"💡 Answer: {answer}")
                    else:
                        print("Please enter a question or command.")
                        
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        else:
            # Single query mode
            question = "Give me the correct coded classification for the following diagnosis: 'Recurrent depressive disorder, currently in remission'"
            print(f"\n❓ Question: {question}")
            print("🔍 Generating answer...")
            answer = rag_pipeline.query(question, use_real_llm=use_real_llm)
            print(f"💡 Answer: {answer}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return


if __name__ == "__main__":
    main() 