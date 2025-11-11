"""Chat module for IceNet"""

from icenet.chat.chatbot import SimpleChatbot, run_chat_loop
from icenet.chat.retrieval import RetrievalChatbot
from icenet.chat.ollama_manager import OllamaManager

__all__ = ["SimpleChatbot", "run_chat_loop", "RetrievalChatbot", "OllamaManager"]
