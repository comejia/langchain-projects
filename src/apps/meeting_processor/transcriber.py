import openai  # Para transcripción Whisper
from src.config.settings import settings


def transcribe_media(file_path: str) -> str:
    """Transcribe un archivo de audio o video utilizando el modelo Whisper de OpenAI.
     Args:
        file_path (str): Ruta al archivo de audio o video.
    Returns:
        str: Transcripción del contenido del archivo.
    """

    print("🎙️ Transcribiendo con OpenAI Whisper API directa...")
    client = openai.OpenAI(api_key=settings.api_key)

    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="es",
            prompt="Esta es una reunión de trabajo en español con multiples participantes.",
            response_format="text",
        )

    print(f"✓ Transcripción completada: {len(transcript)} caracteres")
    return transcript
