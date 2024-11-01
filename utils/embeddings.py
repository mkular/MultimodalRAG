import torch
import numpy as np
from PIL import Image
from transformers import ClapModel, ClapProcessor
import librosa
from sentence_transformers import SentenceTransformer
import docker.errors
from qdrant_client import models, QdrantClient
import docker
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from loguru import logger

class MultimodalEmbedder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.clip_model = 'clip-ViT-B-32'
        self.embedder = SentenceTransformer(self.clip_model)
        self.model_name = "laion/larger_clap_general"
        self.clap_model = ClapModel.from_pretrained(self.model_name)
        self.clap_processor = ClapProcessor.from_pretrained(self.model_name)

    def get_image_embedding(self, image_path):
        """Get CLIP embedding for an image."""
        image = Image.open(image_path).convert('RGB')
        image_embedding = self.embedder.encode([image], convert_to_tensor=True, show_progress_bar=True)
        return image_embedding

    def get_video_embedding(self, video_frames):
        """Get CLIP embedding for a video by averaging frame embeddings."""
        frame_embeddings = []
        frames_list = [Image.fromarray(frame).convert('RGB') for frame in video_frames]
        frame_embeddings = self.embedder.encode(frames_list, convert_to_tensor=True)
        return frame_embeddings

    def get_audio_embedding(self, audio_path):
        """Get CLAP embedding for audio."""
        with torch.no_grad():
            audio_data, sr = librosa.load(audio_path, sr=48000)
            #audio_data = audio_data.reshape(1, -1)
            inputs = self.clap_processor(audios=audio_data, sampling_rate=sr, return_tensors="pt")
            #one_embed= model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
            audio_embedding = self.clap_model.get_audio_features(**inputs)[0]
        return audio_embedding
    
    def get_query_embedding(self, query):
        text_embedding = self.embedder.encode([query], convert_to_tensor=True, show_progress_bar=True)
        ##for audio_collection
        text_inputs = self.clap_processor(text=query, return_tensors="pt")
        audio_embedding = self.clap_model.get_text_features(**text_inputs)[0]
        return text_embedding, audio_embedding



class QdrantHelper():
    """
    Helper methods to interacting with qdrant db.
    """
    def __init__(self):
        self.client = docker.from_env()
        self.qclient = QdrantClient(host="localhost", port=6333)
        self._init_collections()
        logger.success("Successfully connected to Qdrant")
    def _init_collections(self):
        """Initialize collections for different media types."""
        collections_config = {
            "image_collection": 512,  # CLIP image embedding size
            "video_collection": 512,  # CLIP video embedding size
            "audio_collection": 512   # CLAP audio embedding size
        }
        
        for collection_name, vector_size in collections_config.items():
            try:
                collection_present = False
                collection_list = self.qclient.get_collections().collections
                for collection in collection_list:
                    if collection.name == collection_name:
                        logger.info(f"Collection {collection_name} already exists")
                        collection_present = True
                if not collection_present:
                    logger.info(f"Creating collection {collection_name}")
                    self.qclient.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )
            except Exception as e:
                logger.info(f"Exception found {e}")

    def upsert_points(self, collection_name, id, emb_vector, payload):
        self.qclient.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=id, vector=emb_vector, payload=payload
                )
            ],
        )

    def search_collection(self, collection_name, embedded_query, top_k):
        search_result = self.qclient.search(
            collection_name=collection_name,
            query_vector=embedded_query,
            limit=top_k  # Return top 3 results
        )
        return search_result

