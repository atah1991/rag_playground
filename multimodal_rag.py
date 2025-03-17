import os
import json
from typing import List, Tuple, Dict
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pymongo import MongoClient
import numpy as np

class MultimodalRAG:
    def __init__(self, mongodb_uri: str, db_name: str, collection_name: str):
        """Initialize the MultimodalRAG system."""
        # Initialize CLIP for both image and text embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize MongoDB connection
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """Generate embedding for an image using CLIP."""
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.detach().numpy()[0]

    def embed_text(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for text using CLIP.
        Returns both image-aligned and text-aligned embeddings.
        """
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            # Get text features in text encoder space
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Project text into image space using CLIP's projection
            image_aligned_text = self.clip_model.text_projection(text_features)
            image_aligned_text = image_aligned_text / image_aligned_text.norm(dim=-1, keepdim=True)
            
        return (
            image_aligned_text.detach().numpy()[0],  # For comparing with images
            text_features.detach().numpy()[0]        # For comparing with text
        )

    def load_data(self, data_dir: str) -> List[Dict]:
        """Load data from subfolders containing JSON and image files."""
        items = []
        for subfolder in os.listdir(data_dir):
            subfolder_path = os.path.join(data_dir, subfolder)
            if os.path.isdir(subfolder_path):
                # Find JSON and image files
                json_file = None
                image_file = None
                for file in os.listdir(subfolder_path):
                    if file.endswith('.json'):
                        json_file = os.path.join(subfolder_path, file)
                    elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_file = os.path.join(subfolder_path, file)
                
                if json_file and image_file:
                    with open(json_file, 'r') as f:
                        description = json.load(f)
                    items.append({
                        'subfolder': subfolder,
                        'description': description,
                        'image_path': image_file
                    })
        return items

    def store_embeddings(self, items: List[Dict]) -> None:
        """Store embeddings and metadata in MongoDB."""
        for item in items:
            # Generate image embedding
            image_embedding = self.embed_image(item['image_path'])
            
            # Generate text embeddings for various text fields
            description_text = f"{item['description']['name']}. {item['description']['description']}"
            details = item['description']['details']
            details_text = f"Color: {details['color']}. Material: {details['material']}. Style: {details['style']}. Care: {details['care']}."
            full_text = f"{description_text} {details_text}"
            
            # Get both image-aligned and text-space embeddings
            text_img_aligned, text_embedding = self.embed_text(full_text)
            
            document = {
                'subfolder': item['subfolder'],
                'description': item['description'],
                'image_path': item['image_path'],
                'image_embedding': image_embedding.tolist(),
                'text_embedding': text_embedding.tolist(),
                'text_img_aligned': text_img_aligned.tolist()
            }
            self.collection.insert_one(document)

    def compute_similarity(self, query_img_aligned: np.ndarray, query_text: np.ndarray, weight_img: float = 0.5, top_k: int = 3) -> List[Dict]:
        """
        Compute weighted similarity between query embeddings and stored embeddings.
        weight_img: weight for image similarity (1 - weight_img will be used for text)
        """
        results = []
        for doc in self.collection.find():
            img_emb = np.array(doc['image_embedding'])
            txt_emb = np.array(doc['text_embedding'])
            txt_img_aligned = np.array(doc['text_img_aligned'])
            
            # Compute cosine similarities in appropriate spaces
            img_sim = np.dot(query_img_aligned, img_emb)
            txt_sim = np.dot(query_text, txt_emb)
            
            # Compute weighted similarity
            weighted_sim = weight_img * img_sim + (1 - weight_img) * txt_sim
            
            results.append({
                'subfolder': doc['subfolder'],
                'description': doc['description'],
                'image_path': doc['image_path'],
                'similarity': weighted_sim,
                'image_similarity': img_sim,
                'text_similarity': txt_sim
            })
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def search(self, query_text: str, weight_img: float = 0.5, top_k: int = 3) -> List[Dict]:
        """
        Perform search using text query.
        The query will be compared against both image and text embeddings with weights.
        weight_img: weight for image similarity (1 - weight_img will be used for text)
        """
        # Generate embeddings for the query text in both spaces
        query_img_aligned, query_text_emb = self.embed_text(query_text)
        
        return self.compute_similarity(
            query_img_aligned,
            query_text_emb,
            weight_img=weight_img,
            top_k=top_k
        ) 