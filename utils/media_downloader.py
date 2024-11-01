from googleapiclient.discovery import build
import requests
import os

class MediaDownloader:
    def __init__(self, credentials):
        self.API_VERSION = 'v1'
        self.DISCOVERY_URL = f'https://photoslibrary.googleapis.com/$discovery/rest?version={self.API_VERSION}'
        self.service = build('photoslibrary', 'v1', credentials=credentials, discoveryServiceUrl=self.DISCOVERY_URL, static_discovery=False)
        self.album_title = "Multimodal-Rag"

    def list_media_items(self, page_size=100):
        """List media items from Google Photos Album"""
        album_id = ""
        body = {
            'albumId': album_id,
            'pageSize': 100  # Maximum allowed page size
        }
        #results = self.service.mediaItems().list(pageSize=page_size).execute()
        results = self.service.mediaItems().search(body=body).execute()
        return results.get('mediaItems', [])

    def download_media(self, media_item, output_dir):
        """Download a media item from Google Photos."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{media_item['id']}.{media_item['mimeType'].split('/')[-1]}"
        filepath = os.path.join(output_dir, filename)
        
        if media_item['mimeType'].startswith('video'):
            download_url = f"{media_item['baseUrl']}=dv"
        else:
            download_url = f"{media_item['baseUrl']}=d"

        response = requests.get(download_url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        return filepath
