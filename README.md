# MultimodalRAG
Multimodal Rag on Google Photos (Image, Video and Audio)

## Project Description
This program performs multimodal rag over the images, videos and extracted audio (from videos) obtained from Google Photos Album. 
Qdrant is used for storing embeddings and their correponding payload/metadate.
Streamlit is used to obtain user query and display all results.

![Multimodal RAG workflow](https://github.com/user-attachments/assets/a0ecaadf-eaab-47ad-aad4-98a3bb39b344)

## Basic workflow
1. Given an id of google photos album containing images and video, once initiated, it will download all contents of a given album.
2. Images are embedded (using CLIP model) in image_collection of qdrant vector database.
3. Frames of videos are embedded (using CLIP model) in video_collection of qdrant vector database.
4. Audio extracted from each video is embedded (using CLAP model) in audio_collection of qdrant vector database.
5. Now on streamlit, the user can enter their query.
6. This query is then embedded both using CLIP and CLAP models and matching candidates are fetched via ANN in Qdrant across all 3 collections.
7. The images are fed to a multimodal LLM (LLaVA) for its description in nautral language.
8. Results are then shown to the user in streamlit ui. These results are divided into 3 sections:
   8.1. Multimodal RAG results: output of LLM.
   8.2. Images
   8.3. Videos
   8.4. Audio

## Instructions to run[^1]
1. Create a conda env in order to run and install all dependencies.
   <pre>conda create multimodal_rag python=3.12
   conda activate multimodal_rag
2. Install qdrant container (ensure you have docker installed first) and ensure that the container is up and running.
   please refer: https://qdrant.tech/documentation/guides/installation/ for installation instructions
3. Install all the libraries mentioned in requirements.txt.
   <pre>pip install -e .
4. Add right album id in utils/media_downloader.py.
5. Ensure that you have active api key to download google photos album. Info can be found at console.cloud.google.com. This youtube video was helpful: https://www.youtube.com/watch?v=vbHp_FiJXqA
7. Thats it! now run it via:
   <pre> python -m streamlit run main_app.py</pre>
   
  [^1]: __**NOTE**__ : If running for the first time, it will take time to fetch the data, db data to be perared. Subsequent runs should go through quicker as the Qdrant DB would already have embeddings present.
