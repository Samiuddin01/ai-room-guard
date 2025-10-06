import speech_recognition as sr
import pyttsx3

# Initialize TTS engine
tts_engine = pyttsx3.init()

def speak(text: str):
    """ Speaks the given text using the system's TTS engine. """
    print(f"AGENT: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen_for_command(activation_phrase: str) -> bool:
    """ Listens for an activation phrase and returns True if heard. """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"\nListening for '{activation_phrase}'...")
        r.pause_threshold = 1.0
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=4)
            text = r.recognize_google(audio).lower()
            print(f"USER: {text}")
            return activation_phrase.lower() in text
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            print("...") # Silence or unintelligible audio
            return False
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return False
