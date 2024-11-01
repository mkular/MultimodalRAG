
from loguru import logger
from utils.utils import MediaFetcher
import streamlit as st
from ollama_llm import llm
from utils.embeddings import MultimodalEmbedder
import os

def get_candidates(app_obj, search_query):
    """This function downloads the media and creates embeddings for all modalities if get_media == True.
    Otherwise it will embed the query and compare it semantically across all modalities
    and present result.
    """
    try:        
        emb_obj = MultimodalEmbedder()
        text_emb, audio_emb = emb_obj.get_query_embedding(query=search_query)
        text_emb = text_emb.squeeze(0)
        multimodal_collections = [{"name": "image_collection"}, 
                                {"name": "audio_collection"}, 
                                {"name": "video_collection"}]

        for i, collection in enumerate(multimodal_collections):
            if collection["name"] == "audio_collection":
                query_emb = audio_emb
                threshold = 0.1
            else:
                query_emb = text_emb
                threshold = 0.20
            hits = app_obj.qclient.search_collection(collection_name=collection["name"], embedded_query=query_emb, top_k=3)
            multimodal_collections[i]["hits"] = [hit for hit in hits if hit.score > threshold]
            
        logger.info(f"printing collection {multimodal_collections}")
    except Exception as e:
        logger.error("Encountered exception in get_candidates: {e}")

    return multimodal_collections


def run_streamlit(app_obj) -> None:    
    st.set_page_config(layout="wide")
    logger.info("Running multimodal rag")
    st.title(":blue[MultiModal Rag on Google Photos]", )
    search_query = st.text_input("Enter your query below:")    
    if st.button("Search"):
        if search_query:
            st.header(":blue[Multimodal Rag Results]", divider="blue")
            clean_results = get_candidates(app_obj, search_query)
            llm_obj = llm(clean_results, search_query)
            llm_input = llm_obj.format_results()
            st.subheader(llm_obj.generate_result(llm_input))            
            for result in clean_results:
                append_path = "/Users/mandeepkular/Projects/google-photos-multimodal/"
                if result["name"] == "image_collection":
                    logger.info(f"image result is {result}")
                    images = [append_path + hit.payload['file_path'] for hit in result["hits"]]
                    logger.info(f"images are {images}")
                elif result["name"] == "video_collection":
                    logger.info(f"video result is {result}")
                    videos = [append_path + hit.payload['file_path'] for hit in result["hits"]]
                else:
                    logger.info(f"audio result is {result}")
                    audios = [append_path + hit.payload['file_path'] for hit in result["hits"]]
            st.subheader("Images")
            st.image(images, width=500, caption=[0,1,2])
            st.subheader("Videos")
            col1, col2, col3, _, _ = st.columns(5)
            with col1:
                st.video(videos[0])
            st.subheader("Audio")
            col1, col2, col3, _, _ = st.columns(5)
            with col1:
                st.audio(audios[0])

if __name__ == "__main__":
    if len(os.listdir("downloads")) > 0:
        logger.info("setting get_media to False")
        get_media = False
    else:
        logger.info("setting get_media to True")
        get_media = True
    app_obj = MediaFetcher(get_media)
    if app_obj:
        run_streamlit(app_obj)
