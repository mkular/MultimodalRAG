from moviepy.editor import VideoFileClip
import os

class AudioExtractor:
    @staticmethod
    def extract_audio(video_path, output_dir):
        """Extract audio from video file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        video = VideoFileClip(video_path)
        audio_path = os.path.join(output_dir, 
                                os.path.splitext(os.path.basename(video_path))[0] + '.mp3')
        video.audio.write_audiofile(audio_path)
        video.close()
        
        return audio_path