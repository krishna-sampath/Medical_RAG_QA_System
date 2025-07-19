# Medical RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) pipeline specifically designed for medical document analysis and ICD-10 classification queries. This system processes medical PDF documents and provides accurate, context-aware responses to medical classification questions.

## 🚀 Features

### Core Functionality
- **Document Processing**: Upload and process medical PDF documents
- **Intelligent Querying**: Ask questions about medical classifications
- **Real LLM Integration**: Uses Microsoft DialoGPT-medium for accurate responses
- **Vector Search**: FAISS and ChromaDB vector stores for efficient document retrieval
- **MMR Reranking**: Maximal Marginal Relevance for better context selection
- **Caching System**: Persistent cache for fast repeated queries

### Advanced Features
- **Gradio Web Interface**: User-friendly web interface with real-time processing
- **Multiple Vector Stores**: Support for both FAISS and ChromaDB
- **Document Chunking**: Intelligent text splitting for optimal processing
- **Error Handling**: Robust error handling and fallback mechanisms
- **Performance Optimization**: Optimized for both CPU and GPU environments

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended for large documents)
- **Storage**: 2GB free space for models and cache
- **GPU**: Optional but recommended for faster Real LLM processing

### Dependencies
All required packages are listed in `requirements.txt`. Key dependencies include:
- LangChain for RAG pipeline
- Transformers for Real LLM (DialoGPT-medium)
- FAISS/ChromaDB for vector storage
- Gradio for web interface
- PyPDF for document processing

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Wundrsight_Assignment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python rag_pipeline.py --help
```

## 🚀 Usage

### Command Line Interface

#### 1. Real LLM Mode (Recommended)
```bash
python rag_pipeline.py
```
Then select option 1 for Real LLM mode.

#### 2. Interactive Mode
```bash
python rag_pipeline.py
```
Then select option 2 for interactive mode with Real LLM.

#### 3. Gradio Web Interface
```bash
python rag_pipeline.py
```
Then select option 3 for the web interface.

### Gradio Web Interface

1. **Start the Interface**
   ```bash
   python rag_pipeline.py
   # Select option 3
   ```

2. **Access the Interface**
   - Open your browser to `http://localhost:7860`
   - Upload a medical PDF document
   - Ask questions about medical classifications

3. **Available Tabs**
   - **Upload & Query**: Process new documents and ask questions
   - **Query Only**: Ask questions about previously processed documents
   - **Cache Management**: View and manage cached responses

## ⚡ Performance Characteristics

### Real LLM Performance
- **Initialization Time**: 1-2 minutes on first use
- **Response Generation**: 10-30 seconds per query
- **Memory Usage**: ~2GB RAM during processing
- **CPU Usage**: High during LLM inference

### Processing Times
- **Document Processing**: 30-60 seconds for large PDFs
- **Vector Store Creation**: 20-40 seconds for 1000+ chunks
- **Query Response**: 10-30 seconds with Real LLM
- **Cache Hits**: <1 second for repeated queries

### Current Limitations
- **4-Hour Session Limit**: Due to memory constraints and model loading
- **Single Document Processing**: One document at a time
- **CPU-Only Default**: GPU acceleration requires manual configuration
- **Model Size**: DialoGPT-medium requires significant memory

## 🔧 Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0  # For GPU acceleration
export TRANSFORMERS_CACHE=/path/to/cache  # For model caching
```

### Model Configuration
The system uses Microsoft DialoGPT-medium by default. You can modify the model in `rag_pipeline.py`:
```python
model_name = "microsoft/DialoGPT-medium"  # Change this line
```

### Vector Store Configuration
```python
# In MedicalRAGPipeline.__init__()
vector_store_type = "faiss"  # or "chroma"
chunk_size = 500
chunk_overlap = 50
```

## 📊 Cache Management

### Cache Features
- **Persistent Storage**: Cache survives program restarts
- **Query-Based Keys**: Unique keys for different query types
- **Size Management**: Automatic cache size monitoring
- **Manual Control**: Clear cache through interface

### Cache Statistics
- View cache statistics in the Gradio interface
- Monitor cache size and hit rates
- Clear cache when needed

## 🐛 Troubleshooting

### Common Issues

#### 1. Real LLM Initialization Fails
**Problem**: "Error initializing real LLM"
**Solution**: 
- Ensure sufficient RAM (4GB+ available)
- Check internet connection for model download
- Try restarting the application

#### 2. Slow Response Times
**Problem**: Queries taking too long
**Solution**:
- Use cache for repeated queries
- Reduce document chunk size
- Consider GPU acceleration

#### 3. Memory Issues
**Problem**: "Out of memory" errors
**Solution**:
- Close other applications
- Reduce chunk size
- Use smaller model variant

#### 4. Gradio Interface Issues
**Problem**: Interface not loading
**Solution**:
- Check if port 7860 is available
- Ensure Gradio is properly installed
- Try different port: `interface.launch(server_port=7861)`

### Performance Optimization

#### For Faster Processing
1. **Use GPU**: Install CUDA version of PyTorch
2. **Increase RAM**: Ensure 8GB+ available
3. **SSD Storage**: Use SSD for faster I/O
4. **Cache Management**: Clear cache regularly

#### For Better Accuracy
1. **Larger Chunks**: Increase chunk_size for more context
2. **MMR Reranking**: Enable for better document selection
3. **Real LLM**: Always use Real LLM for accurate responses

## 🔮 Future Improvements

### Planned Enhancements
1. **Multi-Document Support**: Process multiple documents simultaneously
2. **GPU Acceleration**: Automatic GPU detection and utilization
3. **Model Optimization**: Quantized models for faster inference
4. **Batch Processing**: Process multiple queries at once
5. **Advanced Caching**: Intelligent cache management
6. **API Integration**: REST API for external applications
7. **Real-time Updates**: Live document processing
8. **Custom Models**: Fine-tuned medical classification models

### Performance Improvements
1. **Response Time**: Target <5 seconds for Real LLM responses
2. **Memory Usage**: Reduce to <1GB RAM usage
3. **Initialization**: Target <30 seconds for LLM initialization
4. **Session Duration**: Extend beyond 4-hour limitation
5. **Concurrent Users**: Support multiple simultaneous users

### Technical Enhancements
1. **Model Switching**: Support for multiple LLM models
2. **Document OCR**: Better handling of scanned documents
3. **Multi-language Support**: Support for non-English documents
4. **Advanced Reranking**: More sophisticated document selection
5. **Streaming Responses**: Real-time response generation

## 📝 API Reference

### Main Classes

#### MedicalRAGPipeline
```python
pipeline = MedicalRAGPipeline(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=50,
    vector_store_type="faiss"
)
```

#### CacheManager
```python
cache = CacheManager(cache_file="cache/query_cache.pkl")
```

### Key Methods

#### Document Processing
```python
pipeline.process_document(pdf_path, store_path, setup_qa=True)
```

#### Query Processing
```python
answer = pipeline.query(
    question="What is the ICD-10 classification for...",
    use_cache=True,
    use_mmr=True,
    use_real_llm=True,
    bypass_cache=False
)
```

#### Cache Management
```python
stats = pipeline.get_cache_stats()
pipeline.cache_manager.clear_cache()
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make your changes
5. Test thoroughly
6. Submit a pull request

### Testing
```bash
# Run basic tests
python -m pytest tests/

# Run performance tests
python tests/test_performance.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤖 AI Usage Disclosure (ChatGPT/Claude)

### Development Approach
This Medical RAG Pipeline was developed using a collaborative approach between human expertise and AI assistance:

**Human Contribution:**
- **Core Concept & Architecture**: The fundamental RAG pipeline design, medical classification focus, and overall system architecture were conceived and developed independently
- **Implementation Logic**: All core algorithms, business logic, and medical domain knowledge integration were developed from scratch
- **Model Selection**: Decisions regarding Real LLM implementation, vector store choices, and performance optimization strategies were made independently
- **Problem Solving**: All debugging, error resolution, and system optimization were handled through independent analysis and testing

**AI Assistance:**
- **Documentation**: AI tools assisted with README organization, code comments, and technical writing
- **Code Structure**: AI helped with code organization, class structure, and file organization
- **Dependency Management**: AI assisted with requirements.txt creation and version compatibility
- **Error Handling**: AI provided suggestions for robust error handling patterns

### Professional Standards
This project maintains professional development standards:
- All core functionality was independently developed and tested
- AI assistance was used primarily for documentation and code organization
- The final implementation represents original work with appropriate AI collaboration
- All medical classification logic and RAG pipeline core functionality are original implementations

## 🙏 Acknowledgments

- **LangChain**: For the RAG pipeline framework
- **Hugging Face**: For the transformer models
- **Gradio**: For the web interface
- **FAISS**: For efficient vector search
- **Microsoft**: For the DialoGPT model

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This system is designed for educational and research purposes. For clinical use, please ensure compliance with relevant medical regulations and standards. 