"""
Simple chatbot interface for IceNet
"""

import torch
from typing import Optional, List, Dict
import logging
from pathlib import Path
from icenet.chat.retrieval import RetrievalChatbot

logger = logging.getLogger(__name__)


class SimpleChatbot:
    """Simple chatbot that uses a trained model"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        data_dir: str = "~/icenet/training",
    ):
        """
        Initialize chatbot

        Args:
            model_path: Path to trained model checkpoint (optional)
            max_length: Maximum response length
            temperature: Sampling temperature (higher = more creative)
            data_dir: Directory with training data for retrieval
        """
        self.model_path = model_path
        self.max_length = max_length
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.conversation_history: List[Dict[str, str]] = []

        # Use retrieval chatbot by default (works immediately!)
        # Ensure path is expanded
        expanded_dir = str(Path(data_dir).expanduser())
        self.retrieval_bot = RetrievalChatbot(data_dir=expanded_dir)

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a trained model"""
        try:
            # For now, this is a placeholder
            # In a real implementation, you'd load the actual model
            logger.info(f"Loading model from {model_path}")
            # self.model = torch.load(model_path)
            # self.tokenizer = load_tokenizer()
            logger.warning("Model loading not fully implemented yet")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def chat(self, user_input: str) -> str:
        """
        Generate a response to user input

        Args:
            user_input: User's message

        Returns:
            Chatbot response
        """
        # Use retrieval bot (works immediately with your data!)
        response = self.retrieval_bot.chat(user_input)

        # Also add to our history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def _generate_response(self, user_input: str) -> str:
        """
        Generate response (placeholder for now)

        This would use the actual model once trained
        """
        if self.model is None:
            return self._fallback_response(user_input)

        # TODO: Implement actual model inference
        # This is where you'd:
        # 1. Tokenize the input
        # 2. Run through the model
        # 3. Decode the output

        return "Model inference not yet implemented"

    def _fallback_response(self, user_input: str) -> str:
        """Simple rule-based fallback responses"""
        user_lower = user_input.lower()

        # Simple pattern matching responses
        if "hello" in user_lower or "hi" in user_lower:
            return "Hello! I'm IceNet AI. I don't have a trained model yet, but I'm ready to learn from your files! Use 'icenet train-local' to train me on your documents."

        elif "help" in user_lower:
            return """I can help you with:
- Training on your local files: Use 'icenet train-local /path/to/your/files'
- Once trained, I can answer questions about your data
- Type 'exit' or 'quit' to end our conversation"""

        elif "train" in user_lower:
            return "To train me on your files, exit this chat and run: icenet train-local /path/to/your/directory"

        elif any(word in user_lower for word in ["bye", "exit", "quit"]):
            return "Goodbye! Train me on your files to make me smarter!"

        else:
            return """I don't have a trained model yet, so I can't answer that question.

To train me on your files, run:
  icenet train-local /path/to/your/documents

This will teach me about your data so I can answer questions!"""

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.retrieval_bot.clear_history()

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history

    def save_conversation(self, output_path: str):
        """Save conversation to file"""
        import json

        with open(output_path, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

        logger.info(f"Conversation saved to {output_path}")


def run_chat_loop(model_path: Optional[str] = None, data_dir: str = "~/icenet/training"):
    """
    Run interactive chat loop

    Args:
        model_path: Optional path to trained model
        data_dir: Directory with training data
    """
    # ANSI color codes for terminal
    USER_COLOR = "\033[96m"      # Cyan for user
    BOT_COLOR = "\033[92m"       # Green for bot
    RESET_COLOR = "\033[0m"      # Reset to default
    BOLD = "\033[1m"             # Bold text

    print("\n" + "=" * 60)
    print("IceNet AI Chatbot - Chat About Your Files!")
    print("=" * 60)

    chatbot = SimpleChatbot(model_path=model_path, data_dir=data_dir)

    # Check if we have training data
    stats = chatbot.retrieval_bot.get_stats()

    if stats['has_data']:
        print(f"✓ Ready to chat! I have data from {stats['metadata'].get('total_files', '?')} files")
        print(f"  ({stats['chunks_loaded']} chunks loaded)")

        # Check if file list is available in metadata
        if not stats['metadata'].get('files'):
            print(f"\n💡 Tip: Re-train to enable file list viewing:")
            print(f"  icenet train-local /path/to/your/files --yes")
    else:
        print("⚠ No training data found!")
        print("  Train me first with: icenet train-local /path/to/files")
        print("  For example: icenet train-local ~/Documents")

    print("\nCommands:")
    print("  'exit' or 'quit' - End the conversation")
    print("  'clear' - Clear conversation history")
    print("  'save' - Manually save conversation")
    print("  'list' - List saved conversations")
    print("  'load <id>' - Load a previous conversation")
    print("\n💡 Conversations are automatically saved after each message")
    print("=" * 60 + "\n")

    while True:
        try:
            # Get user input with color
            user_input = input(f"{BOLD}{USER_COLOR}You:{RESET_COLOR} ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"\n{BOLD}{BOT_COLOR}Chatbot:{RESET_COLOR} Goodbye! 👋\n")
                break

            elif user_input.lower() == 'clear':
                chatbot.retrieval_bot.clear_history()
                chatbot.clear_history()
                print("\n[Conversation history cleared]\n")
                continue

            elif user_input.lower() == 'save':
                filepath = chatbot.retrieval_bot.save_conversation()
                print(f"\n[Conversation saved to {filepath}]\n")
                continue

            elif user_input.lower() == 'list':
                conversations = chatbot.retrieval_bot.list_conversations()
                if not conversations:
                    print("\n[No saved conversations found]\n")
                else:
                    print("\n📚 Saved Conversations:")
                    print("-" * 60)
                    for conv in conversations[:10]:  # Show last 10
                        timestamp = conv['timestamp'][:19].replace('T', ' ')
                        print(f"  ID: {conv['id']}")
                        print(f"  Time: {timestamp}")
                        print(f"  Messages: {conv['messages']}")
                        print("-" * 60)
                    print(f"\nTo load: type 'load <id>'\n")
                continue

            elif user_input.lower().startswith('load '):
                conversation_id = user_input[5:].strip()
                if chatbot.retrieval_bot.load_conversation(conversation_id):
                    print(f"\n[Loaded conversation: {conversation_id}]")
                    print(f"[{len(chatbot.retrieval_bot.conversation_history)} messages restored]\n")
                else:
                    print(f"\n[Failed to load conversation: {conversation_id}]\n")
                continue

            # Get response with streaming
            print(f"\n{BOLD}{BOT_COLOR}Chatbot:{RESET_COLOR} ", end='', flush=True)

            # Stream response in real-time
            response_stream = chatbot.retrieval_bot.chat(user_input, stream=True)

            # Handle both streaming and non-streaming responses
            if hasattr(response_stream, '__iter__') and not isinstance(response_stream, str):
                # Streaming response - display token by token
                import sys
                for token in response_stream:
                    print(token, end='', flush=True)
                    sys.stdout.flush()
                print("\n")  # End with newline

                # Also update SimpleChatbot history
                full_response = chatbot.retrieval_bot.conversation_history[-1]['content']
                chatbot.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                chatbot.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
            else:
                # Non-streaming fallback
                print(f"{response_stream}\n")
                chatbot.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                chatbot.conversation_history.append({
                    "role": "assistant",
                    "content": response_stream
                })

        except KeyboardInterrupt:
            print("\n\nChatbot: Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            logger.error(f"Chat error: {e}", exc_info=True)
