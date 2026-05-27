import os
from google import genai
from PIL import Image

def extract_blurry_text(image_path):
    print(f"Loading image: {image_path}")
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}.")
        return

    print("Initializing Gemini Client...")
    client = genai.Client()
    
    prompt = """
    This is a highly compressed and blurry scan of a newspaper. 
    Please transcribe the text of the articles to the best of your ability. 
    Use the visible words, headlines, and standard English grammar/context to fill in the words that are blurry or pixelated. 
    If a section is completely unrecoverable, put [unreadable] in its place.
    """
    
    print("Sending to Gemini Vision API for transcription...")
    # Using the current standard free-tier model
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt, img]
    )
    
    print("\n--- TRANSCRIBED TEXT ---\n")
    print(response.text)
    print("\n------------------------\n")

if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set in this terminal session.")
        print("Run this in your terminal first: export GEMINI_API_KEY='your_key_here'")
        exit(1)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_IMAGE = os.path.join(BASE_DIR, "temp_crop.jpg")
    
    extract_blurry_text(INPUT_IMAGE)