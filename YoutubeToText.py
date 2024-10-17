import os
import yt_dlp
from pydub import AudioSegment
import whisper
import re
import tempfile
from datetime import timedelta
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from tqdm import tqdm

def download_audio(youtube_url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, 'temp_audio.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        video_title = info_dict.get('title', 'output')
        video_id = info_dict.get('id')

    # Sanitize the video title to remove problematic characters
    sanitized_title = sanitize_unicode(video_title)

    audio_path = os.path.join(output_path, "temp_audio.mp3")
    # Log the processed video
    log_processed_video(video_id, sanitized_title, youtube_url)
    
    return sanitized_title, audio_path

def log_processed_video(video_id, video_title, youtube_url):
    log_file = "processed_videos.log"
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("Processed Videos:\n")
    
    with open(log_file, "r", encoding="utf-8") as f:
        logged_videos = f.read()
    
    if video_id not in logged_videos:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{video_id}: {video_title} - {youtube_url}\n")

def sanitize_filename(filename):
    sanitized = re.sub(r'[\/*?"<>|:\[\]]', "_", filename)
    if not sanitized:
        sanitized = "output"
    return sanitized

def sanitize_unicode(text):
    # Remove or replace any characters that can't be encoded
    sanitized = ''.join(char if char.isprintable() else '_' for char in text)
    return sanitized if sanitized else "output"

def extract_speech_segments(input_audio_path, temp_dir):
    torch.set_num_threads(1)
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    # Convert MP3 to WAV format using pydub
    wav_path = os.path.join(temp_dir, "temp_audio.wav")
    audio = AudioSegment.from_mp3(input_audio_path)
    audio.export(wav_path, format="wav")

    wav = read_audio(wav_path)
    speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)

    # Clean up temporary WAV file
    os.remove(wav_path)

    # Ensure segments are at most 30 seconds long
    max_segment_length = 30  # seconds
    refined_speech_timestamps = []
    for segment in speech_timestamps:
        start = segment['start']
        end = segment['end']
        while end - start > max_segment_length:
            refined_speech_timestamps.append({'start': start, 'end': start + max_segment_length})
            start += max_segment_length
        refined_speech_timestamps.append({'start': start, 'end': end})

    return refined_speech_timestamps

def split_audio_default(input_audio_path, chunk_length_ms, temp_dir):
    audio = AudioSegment.from_mp3(input_audio_path)
    chunk_paths = []
    for idx, chunk in enumerate(audio[::chunk_length_ms]):
        chunk_path = os.path.join(temp_dir, f"chunk_{idx}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)
    return chunk_paths

def split_audio_by_speech(input_audio_path, speech_timestamps, temp_dir):
    audio = AudioSegment.from_mp3(input_audio_path)
    chunk_paths = []
    
    for idx, segment in enumerate(speech_timestamps):
        start_ms = segment['start'] * 1000
        end_ms = segment['end'] * 1000
        
        chunk = audio[start_ms:end_ms]
        chunk_path = os.path.join(temp_dir, f"chunk_{idx}_{start_ms//1000}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)

    return chunk_paths

def transcribe_audio_whisper(audio_path, model):
    try:
        result = model.transcribe(audio_path)
        return result.get("text", "No transcription found."), result.get("segments", [])
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return "", []

def format_timestamp(seconds):
    timestamp = str(timedelta(seconds=seconds))
    return timestamp.split('.')[0]  # Remove microseconds

def main():
    print("Select Whisper model to use:")
    print("1. tiny.en          (39 M parameters, ~1 GB VRAM, ~10x speed, English only)")
    print("2. base.en          (74 M parameters, ~1 GB VRAM, ~7x speed, English only)")
    print("3. medium.en        (769 M parameters, ~5 GB VRAM, ~2x speed, English only)")
    print("4. large            (1550 M parameters, ~10 GB VRAM, 1x speed, multilingual)")
    print("5. large-v2         (?)")
    print("6. large-v3         (?)")
    print("7. large-v3-turbo   (?)")
    print("8. turbo            (809 M parameters, ~6 GB VRAM, ~8x speed, multilingual)")

    model_selection = input("Enter the number corresponding to the model you want to use (1-8) [default: 8]: ") or '8'
    model_names = {
        "1": "tiny.en",
        "2": "base.en",
        "3": "medium.en",
        "4": "large",
        "5": "large-v2",
        "6": "large-v3",
        "7": "large-v3-turbo",
        "8": "turbo"
    }

    model_name = model_names.get(model_selection, "tiny")  # Default to "tiny" if invalid selection

    # Step 1: Choose chunking method
    print("Select chunking method to use:")
    print("1. Use Silero VAD for speech-based chunking")
    print("2. Use default 30-second intervals for chunking with phrase segmentation")
    print("3. Use default 15-second intervals for chunking")
    chunking_selection = input("Enter the number corresponding to the chunking method you want to use (1-3) [default: 3]: ") or '3'

    youtube_url = input("Enter YouTube video URL: ")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 2: Download audio from YouTube
        print("Downloading audio from YouTube...")
        video_title, output_mp3_path = download_audio(youtube_url, temp_dir)
        sanitized_title = sanitize_filename(video_title)
        print("Audio downloaded.")

        # Step 3: Split audio based on chosen method
        if chunking_selection == "1":
            chunking_method = "SileroVAD"
            print("Extracting speech segments from audio using Silero VAD...")
            speech_timestamps = extract_speech_segments(output_mp3_path, temp_dir)
            if not speech_timestamps:
                print("No speech detected in the audio.")
                return
            print("Splitting audio into speech segments...")
            chunk_paths = split_audio_by_speech(output_mp3_path, speech_timestamps, temp_dir)
        elif chunking_selection == "2":
            chunking_method = "Default30s"
            print("Splitting audio into 30-second intervals with phrase segmentation...")
            chunk_paths = split_audio_default(output_mp3_path, 30000, temp_dir)
        else:
            chunking_method = "Default15s"
            print("Splitting audio into 15-second intervals...")
            chunk_paths = split_audio_default(output_mp3_path, 15000, temp_dir)
        
        print(f"Audio split into {len(chunk_paths)} speech chunks.")

        # Step 4: Load Whisper model
        print(f"Loading Whisper model ({model_name})...")
        model = whisper.load_model(model_name)

        # Step 5: Transcribe each chunk using Whisper and write to output file with timestamps
        output_text_file = f"{sanitized_title}_{model_name}_{chunking_method}.txt"
        try:
            with open(output_text_file, "w", encoding="utf-8") as f:
                current_time_ms = 0
                for idx, chunk_path in enumerate(tqdm(chunk_paths, desc="Transcribing chunks", unit="chunk")):
                    transcript, segments = transcribe_audio_whisper(chunk_path, model)
                    if segments:
                        for segment in segments:
                            start_time = current_time_ms + segment["start"] * 1000
                            try:
                                f.write(f"[{format_timestamp(start_time / 1000)}] {segment['text']}\n")
                            except UnicodeEncodeError as e:
                                print(f"Error writing segment: {e}")
                                f.write(f"[{format_timestamp(start_time / 1000)}] [UNREADABLE SEGMENT]\n")
                    else:
                        try:
                            f.write(transcript + "\n")
                        except UnicodeEncodeError as e:
                            print(f"Error writing transcript: {e}")
                            f.write("[UNREADABLE TRANSCRIPT]\n")
                    current_time_ms += len(AudioSegment.from_mp3(chunk_path))
            print(f"Transcript saved to {output_text_file}")
        except FileNotFoundError as e:
            print(f"Error writing to file {output_text_file}: {e}")
            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".txt", encoding="utf-8") as temp_file:
                current_time_ms = 0
                for idx, chunk_path in enumerate(tqdm(chunk_paths, desc="Transcribing chunks", unit="chunk")):
                    transcript, segments = transcribe_audio_whisper(chunk_path, model)
                    if segments:
                        for segment in segments:
                            start_time = current_time_ms + segment["start"] * 1000
                            try:
                                temp_file.write(f"[{format_timestamp(start_time / 1000)}] {segment['text']}\n")
                            except UnicodeEncodeError as e:
                                print(f"Error writing segment: {e}")
                                temp_file.write(f"[{format_timestamp(start_time / 1000)}] [UNREADABLE SEGMENT]\n")
                    else:
                        try:
                            temp_file.write(transcript + "\n")
                        except UnicodeEncodeError as e:
                            print(f"Error writing transcript: {e}")
                            temp_file.write("[UNREADABLE TRANSCRIPT]\n")
                    current_time_ms += len(AudioSegment.from_mp3(chunk_path))
                print(f"Transcript saved to temporary file {temp_file.name}")

if __name__ == "__main__":
    main()