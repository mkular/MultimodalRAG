import cv2
import numpy as np
from typing import List
from utils.embeddings import QdrantHelper
from utils.embeddings import MultimodalEmbedder
from loguru import logger
from utils.auth import GooglePhotosAuth
from utils.media_downloader import MediaDownloader
from utils.embeddings import MultimodalEmbedder
from utils.audio_extractor import AudioExtractor

class VideoProcessor:
    @staticmethod
    def extract_frames(video_path: str, num_frames: int = 10) -> List[np.ndarray]:
        """Extract evenly spaced frames from a video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
        cap.release()
        return frames

class MediaFetcher():

    def __init__(self, get_media):
        #self.qdrant_manager = QdrantManager(settings.QDRANT_HOST, settings.QDRANT_PORT)
        self.media_embedder = MultimodalEmbedder()        
        self.qclient = QdrantHelper()        
        if get_media == True:
            self._prepare_embeddings()
        #self.qclient = QdrantClient(host="localhost", port=6333)

    def _prepare_embeddings(self):
        logger.info("get_media is True so we need to fetch it via Google Photos")
        try:
            auth = GooglePhotosAuth()
            credentials = auth.authenticate()
            downloader = MediaDownloader(credentials)
            media_items = downloader.list_media_items()
            image_id, video_id, audio_id = 1,1,1
            for item in media_items:
                media_path = downloader.download_media(item, 'downloads')
                if item['mimeType'].startswith('image'):
                    collection_name = "image_collection"
                    # Process image
                    logger.info(f"embedding image file {media_path}")
                    image_embedding = self.media_embedder.get_image_embedding(media_path)
                    image_embedding = image_embedding.flatten()
                    payload={"file_path":media_path,}
                    self.qclient.upsert_points(collection_name, image_id, image_embedding, payload)
                    image_id += 1                
                else:
                    # Extract audio
                    audio_path = AudioExtractor.extract_audio(media_path, 'audio')                
                    # Extract video frames
                    frames = VideoProcessor.extract_frames(media_path)                
                    # Get embeddings
                    logger.info(f"embedding video file {media_path}")
                    collection_name = "video_collection"
                    video_embeddings = self.media_embedder.get_video_embedding(frames)
                    payload={"file_path":media_path,}
                    for video_embedding in video_embeddings:
                        self.qclient.upsert_points(collection_name, video_id, video_embedding, payload)
                        video_id += 1
                    logger.info(f"embedding video file {audio_path}")
                    audio_embedding = self.media_embedder.get_audio_embedding(audio_path)
                    collection_name = "audio_collection"
                    payload={"file_path":audio_path,}
                    self.qclient.upsert_points(collection_name, audio_id, audio_embedding, payload)
                    audio_id += 1
        except Exception as e:
            logger.error(f"Encountered unexpected exception while downloading and embedding media {e}")
