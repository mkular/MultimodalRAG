from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle

SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
API_NAME = 'photoslibrary'
API_VERSION = 'v1'

class GooglePhotosAuth:
    def __init__(self, credentials_path='credentials.json', token_path='token.pickle'):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.credentials = None

    def authenticate(self):
        """Authenticate with Google Photos API."""
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                self.credentials = pickle.load(token)

        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                self.credentials = flow.run_local_server(port=0)

            with open(self.token_path, 'wb') as token:
                pickle.dump(self.credentials, token)

        return self.credentials