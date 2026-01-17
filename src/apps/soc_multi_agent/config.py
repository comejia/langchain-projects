import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API keys principales
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")

    # Gmail configuration
    GMAIL_CREDENTIALS_FILE = os.getenv("GMAIL_CREDENTIALS_FILE")
    GMAIL_TOKEN_FILE = os.getenv("GMAIL_TOKEN_FILE")

    # SOC email configuration
    SOC_EMAIL_RECIPIENT = os.getenv("SOC_EMAIL_RECIPIENT")
    SOC_EMAIL_SENDER = os.getenv("SOC_EMAIL_SENDER")

    # APIs opcionales para Threat inteligence
    ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY")
    URLVOID_API_KEY = os.getenv("URLVOID_API_KEY")

    # Configuración del SOC
    WEBHOOK_PORT = os.getenv("WEBHOOK_PORT", 8000)
    DASHBOARD_PORT = os.getenv("DASHBOARD_PORT", 8501)

    @classmethod
    def validate_required_config(cls):
        required_keys = [
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            ("TAVILY_API_KEY", cls.TAVILY_API_KEY),
            ("VIRUSTOTAL_API_KEY", cls.VIRUSTOTAL_API_KEY),
        ]

        missing_keys = [key for key, value in required_keys if not value]

        if missing_keys:
            raise ValueError(f"Missing required configuration: {missing_keys}")


config = Config()


if __name__ == "__main__":
    config.validate_required_config()
