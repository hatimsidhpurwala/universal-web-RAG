"""
Constants and static assets used throughout the application.
"""

# Application metadata
APP_NAME = "Smart Web Scraper with Hybrid RAG"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "An intelligent AI assistant that scrapes websites, processes "
    "multiple input types, builds a knowledge base, and answers "
    "questions with automatic web research capabilities."
)

# URL regex pattern for auto-detection
URL_PATTERN = r"https?://[^\s<>\"'\)\]]+(?:\.[^\s<>\"'\)\]]+)+"

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_DOC_EXTENSIONS = {".docx", ".doc"}
SUPPORTED_SLIDE_EXTENSIONS = {".pptx", ".ppt"}
SUPPORTED_SHEET_EXTENSIONS = {".xlsx", ".xls", ".csv"}

ALL_SUPPORTED_EXTENSIONS = (
    SUPPORTED_IMAGE_EXTENSIONS
    | SUPPORTED_PDF_EXTENSIONS
    | SUPPORTED_DOC_EXTENSIONS
    | SUPPORTED_SLIDE_EXTENSIONS
    | SUPPORTED_SHEET_EXTENSIONS
)

# Streamlit UI constants
MAX_UPLOAD_SIZE_MB = 50
CHAT_AVATAR_USER = "👤"
CHAT_AVATAR_AI = "🤖"
