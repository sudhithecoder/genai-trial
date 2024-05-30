import os
import yaml
from dotenv import load_dotenv
import streamlit as st
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import requests

# Load environment variables
load_dotenv()

# Initialize the translator
translator = Translator()

# Load configuration from YAML file
def load_configuration(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_configuration()

# Supported languages dictionary loaded from configuration
LANGUAGES = config.get("languages", {})

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to send a question to the RAGAPI endpoint and get the answer
def get_answer_from_api(question,rag_api_url):
    response = requests.post(rag_api_url, json={"question": question})
    if response.status_code == 200:
        return response.json().get("answer")
    else:
        return "Error: Unable to get a response from the chatbot."

def translate_api_request(text, src_lang, dest_lang, api_url):
    payload = {'text': text, 'src_lang': src_lang, 'dest_lang': dest_lang}
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        return response.json().get('translated_text')
    else:
        st.error("Error: Translation API request failed.")
        return None

def recognize_speech(lang_code):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        
    try:
        query = recognizer.recognize_google(audio, language=lang_code)
        return query
    except sr.UnknownValueError:
        return "Sorry, I did not understand that."
    except sr.RequestError:
        return "Sorry, the service is unavailable."

def text_to_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    audio_file_path = "response.mp3"
    tts.save(audio_file_path)
    return audio_file_path

def generate_chat_message(content, user):
    return f"""
    <div style="display: flex; justify-content: flex-start; align-items: flex-start; margin-bottom: 20px;">
        <div style="background-color: #f8f8f8; border-radius: 5px; padding: 10px; max-width: 70%; margin-right: 10px;">
            <strong>{user}:</strong> {content}
        </div>
    </div>
    """

def display_chat_history():
    for message in st.session_state.messages:
        if message["role"] == "user":
           st.markdown(generate_chat_message(message['content'], 'You'), unsafe_allow_html=True)     
        else:
           st.markdown(generate_chat_message(message['content'], 'HSBC Assistant'), unsafe_allow_html=True)

def main():
    # Define the API endpoints
    rag_api_url = config.get("rag_api_url", "http://localhost:8000/ask")
    translation_api_url = config.get("translation_api_url", 'http://127.0.0.1:5000/translate')
    hsbc_img_url = config.get("hsbc_image_url")

    # Streamlit app
    # Set page configuration for a Google-like UI
    st.set_page_config(
        page_title="HSBC Helper Chatbot",
        initial_sidebar_state="collapsed",
        page_icon=hsbc_img_url,
    )

    # Display the logo
    st.image(hsbc_img_url, width=100)

    # Language selection
    selected_language = st.selectbox("Choose your language:", list(LANGUAGES.keys()))
    lang_code = LANGUAGES[selected_language]

    # Display chat history
    display_chat_history()

    # Text input
    text_input = st.chat_input(f"Type your message here ({selected_language}):")

    # Voice input
    if st.button("Talk"):
        query = recognize_speech(lang_code)
        if query:
            st.write(f"Recognized: {query}")
            text_input = query

    if text_input:
        translated_text = translate_api_request(text_input, lang_code, "en", translation_api_url)

        if translated_text:
            # Get the answer from the RAGAPI endpoint
            answer = get_answer_from_api(translated_text,rag_api_url)
            response = translate_api_request(answer, "en", lang_code, translation_api_url)
            st.write(f"Chatbot ({selected_language}): {response}")

            audio_file = text_to_speech(response, lang_code)
            audio_bytes = open(audio_file, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")

            st.session_state.messages.append({"role": "user", "content": text_input})
            st.session_state.messages.append({"role": "chatbot", "content": response})
        else:
            st.error("Translation failed. Please check your input and try again.")

if __name__ == "__main__":
    main()
