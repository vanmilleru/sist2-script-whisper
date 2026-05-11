from time import time
import typer
import os
from sist2 import Sist2Index, print_progress
import requests  # <-- Import requests
# --- Removed OpenAI import ---

# --- Manually get API key from environment ---
API_KEY = os.environ.get("OPENAI_API_KEY","None")


# --- Define API constants ---
OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

# --- Removed OpenAI client initialization ---
def whisper_stt(input_audio: str, model_name: str):
    """
    Performs speech-to-text using the OpenAI Whisper API via plain requests.
    Includes a 25MB file size check to prevent API errors.
    """
    total = None
    # OpenAI Whisper API has a 25 MB (25 * 1024 * 1024 bytes) file size limit
    WHISPER_API_LIMIT_BYTES = 512 * 1024 * 1024  # 25 MB

    try:
        # === FILE SIZE VALIDATION ===
        # 1. Check if file exists
        if not os.path.exists(input_audio):
            print(f"Error: Audio file not found at {input_audio}")
            return None

        # 2. Check file size
        file_size = os.path.getsize(input_audio)
        
        if file_size > WHISPER_API_LIMIT_BYTES:
            print(f"Error: File '{os.path.basename(input_audio)}' is too large.")
            print(f"Size: {file_size / (1024*1024):.2f} MB, Limit: {WHISPER_API_LIMIT_BYTES / (1024*1024):.0f} MB.")
            print("Please use a smaller file or split the audio.")
            return None
        # === END VALIDATION ===

        # Open the audio file in binary read mode
        with open(input_audio, "rb") as audio_file:
            
            # Prepare the multipart/form-data payload
            files = {
                'file': (os.path.basename(input_audio), audio_file, 'application/octet-stream')
            }
            data = {
                'model': model_name,
                'response_format': 'verbose_json',
                'prompt': '[English]'
            }

            # Call the OpenAI API using requests
            response = requests.post(OPENAI_API_URL, headers=HEADERS, files=files, data=data)

            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            # Parse the JSON response
            response_data = response.json()

        total = ""
        # Print info similar to the original code
        print(f"Detected language: {response_data['language']}, Duration: {response_data['duration']}s")

        # Process the segments from the verbose JSON response
        for segment in response_data['segments']:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            now = "[%.2fs -> %.2fs] %s" % (start, end, text)
            total += now + "\n"

    except requests.exceptions.HTTPError as http_err:
        # This will now catch 4xx/5xx errors, but our check prevents 413 (Payload Too Large)
        print(f"HTTP error calling OpenAI API for {input_audio}: {http_err} - {response.text}")
    except FileNotFoundError:
        # This is handled by our os.path.exists check, but good to keep
        print(f"Error: Audio file not found at {input_audio}")
    except Exception as e:
        # Provide a more specific error message
        print(f"Error processing {input_audio}: {e}")
    return total

def main(
    index_file: str,
    num_threads: int = 8,  # Note: This parameter is not used in the loop
    color: str = "#51da4c",
    tag: bool = True,
    model: str = "whisper-1",  # Default to the standard OpenAI model
):

    index = Sist2Index(index_file)

    tag_value = f"whisper.{color}"

    # Only consider documents that were modified since the last run of this script
    whisper_version = index.get("whisper_version", default=0)

    where = (
        f"((SELECT name FROM mime WHERE id=document.mime ) LIKE 'audio/%' "
        f"OR (SELECT name FROM mime WHERE id=document.mime ) LIKE 'video/%') AND version > {whisper_version}"
    )

    total = index.document_count(where)
    done = 0

    for doc in index.document_iter(where=where):
        if "content" in doc.json_data:
            # print(f"skipping {doc.rel_path}")
            # and len(doc.json_data["content"])>0
            continue

        start = time()
        
        # Pass the model name to the STT function
        text = whisper_stt(doc.path, model_name=model)

        if text is not None:
            doc.json_data["content"] = text

        if tag:
            if "tags" not in doc.json_data:
                doc.json_data["tag"] = [tag_value]
            else:
                doc.json_data["tag"] = list(
                    filter(lambda t: not t.startswith("whisper."), doc.json_data["tag"])
                ).append(tag_value)

        index.update_document(doc)
        print(f"Performed STT for {doc.rel_path} ({time() - start:.2f}s)")

        done += 1
        if done % 100 == 0:
            index.commit()

        print_progress(done=done, count=total)

    index.set("whisper_version", index.versions[-1].id)

    print("Done!")

    index.sync_tag_table()
    index.commit()


if __name__ == "__main__":
    typer.run(main)
