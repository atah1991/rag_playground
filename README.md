# Multimodal RAG System for E-commerce

A Retrieval Augmented Generation (RAG) system that enables multimodal search across product images and descriptions using CLIP embeddings.

## Features

- Unified embedding space using CLIP for both images and text
- MongoDB integration for storing and retrieving embeddings
- Text-based queries that match against both visual and textual content
- Configurable weighting between image and text similarity scores
- Modular design with separate utility functions and demo notebook

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

Organize your product data in subfolders with the following structure:
```
data/
  product1/
    description.json
    image.jpg
  product2/
    description.json
    image.png
  ...
```

### JSON Format
Each product's `description.json` should follow this structure:
```json
{
    "name": "Product Name",
    "description": "Detailed product description",
    "details": {
        "color": "Product color",
        "material": "Product material",
        "style": "Product style",
        "care": "Care instructions"
    }
}
```

### Image Requirements
- Supported formats: JPG, JPEG, PNG
- One image per product folder
- Clear, well-lit product images

## Technical Details

### Embedding Generation
- Uses CLIP (Contrastive Language-Image Pre-training) model for both image and text embeddings
- Text embeddings are generated in two spaces:
  1. Text encoder space (for text-to-text comparison)
  2. Image-aligned space (for text-to-image comparison)
- All embeddings are L2-normalized for cosine similarity computation

### Similarity Search
- Text queries are embedded into both text and image spaces
- Similarity scores are computed using cosine similarity
- Final ranking combines image and text similarities with configurable weights
- Default weight distribution: 50% image similarity, 50% text similarity

## Usage

1. Import the RAG system:
```python
from multimodal_rag import MultimodalRAG
```

2. Initialize the system:
```python
rag = MultimodalRAG(
    mongodb_uri="mongodb://localhost:27017",
    db_name="ecommerce",
    collection_name="products"
)
```

3. Load and process data:
```python
items = rag.load_data("path/to/data/directory")
rag.store_embeddings(items)
```

4. Perform searches:
```python
# Default weights (50-50)
results = rag.search("blue cotton shirt")

# Custom weights (e.g., 70% image, 30% text)
results = rag.search("blue cotton shirt", weight_img=0.7)
```

## Components

- `multimodal_rag.py`: Core implementation of the RAG system
- `multimodal_rag_demo.ipynb`: Jupyter notebook demonstrating system usage
- `requirements.txt`: Project dependencies

## Dependencies

- torch
- transformers (CLIP)
- pymongo
- Pillow (PIL)
- numpy
- jupyter

## Notes

- The system uses CLIP's unified latent space to ensure meaningful comparisons between images and text
- Text queries are automatically projected into both text and image spaces for comprehensive matching
- Similarity scores are normalized and weighted to provide balanced results
- Memory usage is optimized using `torch.no_grad()` during inference
- MongoDB is used for efficient storage and retrieval of embeddings

## Performance Considerations

- First-time initialization may take longer due to CLIP model download
- Using GPU (if available) can significantly speed up embedding generation
- Consider batch processing for large datasets
- Embeddings are cached in MongoDB to avoid regeneration 