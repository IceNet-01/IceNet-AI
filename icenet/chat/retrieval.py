"""
Simple retrieval-based chatbot that works immediately
Uses Ollama for intelligent responses with your data!
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RetrievalChatbot:
    """Smart chatbot that searches your data and uses AI for responses"""

    def __init__(self, data_dir: str = "~/icenet/training", use_ollama: bool = True):
        """
        Initialize retrieval chatbot

        Args:
            data_dir: Directory with training data
            use_ollama: Whether to use Ollama for AI responses (default: True)
        """
        self.data_dir = Path(data_dir).expanduser()
        self.chunks: List[str] = []
        self.metadata: Dict = {}
        self.conversation_history: List[Dict] = []
        self.use_ollama = use_ollama
        self.ollama_manager = None

        # Try to load data automatically
        self.load_data()

        # Setup Ollama if requested
        if self.use_ollama:
            self._setup_ollama()

    def _setup_ollama(self):
        """Setup Ollama integration"""
        try:
            from icenet.chat.ollama_manager import OllamaManager
            self.ollama_manager = OllamaManager()

            # Check if Ollama is ready
            if not self.ollama_manager.is_running():
                logger.info("Ollama not running, will use fallback mode")
                self.use_ollama = False
            elif not self.ollama_manager.has_model(self.ollama_manager.default_model):
                logger.info("Ollama model not found, will use fallback mode")
                self.use_ollama = False
        except Exception as e:
            logger.warning(f"Ollama setup failed: {e}, using fallback mode")
            self.use_ollama = False

    def load_data(self) -> bool:
        """Load training data from directory"""
        data_file = self.data_dir / "training_data.txt"
        metadata_file = self.data_dir / "metadata.json"

        if not data_file.exists():
            logger.warning(f"No training data found at {data_file}")
            return False

        try:
            # Load chunks
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.chunks = [
                    chunk.strip()
                    for chunk in content.split('\n---\n')
                    if chunk.strip()
                ]

            # Load metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)

            logger.info(f"Loaded {len(self.chunks)} chunks from {self.metadata.get('total_files', '?')} files")
            return True

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return False

    def search(self, query: str, top_k: int = 3, min_score: int = 3) -> List[str]:
        """
        Search for relevant chunks

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum relevance score (default: 3)

        Returns:
            List of relevant text chunks
        """
        if not self.chunks:
            return []

        # Simple keyword-based search (case-insensitive)
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Score each chunk
        scored_chunks = []
        for chunk in self.chunks:
            chunk_lower = chunk.lower()

            # Count matching words
            chunk_words = set(chunk_lower.split())
            matches = len(query_words & chunk_words)

            # Bonus for exact phrase match
            if query_lower in chunk_lower:
                matches += 10

            # Only include if score meets minimum threshold
            if matches >= min_score:
                scored_chunks.append((matches, chunk))

        # Sort by score and return top_k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def chat(self, user_input: str, stream: bool = False):
        """
        Generate response to user input

        Args:
            user_input: User's message
            stream: If True, returns generator for streaming output

        Returns:
            Response string (or generator if stream=True)
        """
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Check if data is loaded
        if not self.chunks:
            response = """I don't have any training data yet!

To train me on your files, run:
  icenet train-local /path/to/your/folder

For example:
  icenet train-local ~/Documents
  icenet train-local ~/Desktop/MyNotes

Then I'll be able to answer questions about your files!"""

            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            return response

        # Check if user is asking about the training data itself
        meta_keywords = ['files', 'training', 'trained', 'data', 'loaded', 'what do you know',
                         'what information', 'what have you learned', 'your data', 'my files']
        is_meta_question = any(keyword in user_input.lower() for keyword in meta_keywords)

        # Search for relevant information
        results = self.search(user_input, top_k=3)

        # Use Ollama for intelligent responses if available
        if self.use_ollama and self.ollama_manager:
            if not results:
                # Check if asking about training data
                if is_meta_question:
                    # Provide info about training data
                    file_list = self.metadata.get('files', [])
                    file_info = "\n".join([f"- {f}" for f in file_list[:20]])  # Show up to 20 files

                    response = self.ollama_manager.chat(
                        prompt=user_input,
                        context=f"Training data info:\n- Total files: {self.metadata.get('total_files', 0)}\n- Total chunks: {len(self.chunks)}\n\nFiles:\n{file_info}",
                        system_prompt=f"You are IceNet AI. The user is asking about their training data. I've provided information about the {self.metadata.get('total_files', 0)} files you were trained on. Use this information to answer their question about what files/data you have access to. Be specific and helpful.",
                        stream=stream
                    )
                else:
                    # No relevant data - answer as a general AI assistant
                    response = self.ollama_manager.chat(
                        prompt=user_input,
                        system_prompt=f"You are IceNet AI, a helpful AI assistant. The user has trained you on {self.metadata.get('total_files', 0)} files from their computer, but this question doesn't seem related to those files. Answer the question normally using your general knowledge. Only mention the training files if the user specifically asks about them or their data.",
                        stream=stream
                    )
            else:
                # Check if this is a meta-question even though we have results
                if is_meta_question and any(word in user_input.lower() for word in ['what files', 'which files', 'my files', 'your files', 'files do', 'list files']):
                    # User wants to know about the files themselves, not search them
                    file_list = self.metadata.get('files', [])
                    file_info = "\n".join([f"- {f}" for f in file_list[:20]])

                    response = self.ollama_manager.chat(
                        prompt=user_input,
                        context=f"Training data info:\n- Total files: {self.metadata.get('total_files', 0)}\n- Total chunks: {len(self.chunks)}\n\nFiles:\n{file_info}",
                        system_prompt=f"You are IceNet AI. The user is asking about their training data. I've provided a list of the {self.metadata.get('total_files', 0)} files you were trained on. Answer their question by telling them about these files. Be specific and helpful.",
                        stream=stream
                    )
                else:
                    # Build context from search results
                    context = "\n\n---\n\n".join(results)

                    # Get AI response with context
                    response = self.ollama_manager.chat(
                        prompt=user_input,
                        context=context,
                        system_prompt=f"You are IceNet AI, a helpful AI assistant. The user has trained you on {self.metadata.get('total_files', 'some')} files. I've provided some context from those files below. IMPORTANT: Only use this context if it's actually relevant to answering the question. If the question is general knowledge (like 'what year did we land on the moon?' or 'when does it snow?'), answer normally and ignore the file context. If the context IS relevant (like the user asks about their code, documents, or files), then use it to provide a helpful answer. Be conversational and natural.",
                        stream=stream
                    )
        else:
            # Fallback: basic responses without AI
            if not results:
                response = f"""I couldn't find anything about "{user_input}" in your files.

I have data from {self.metadata.get('total_files', 'some')} files.

Try asking about:
- Topics mentioned in your files
- Keywords from your documents
- General questions about your data

Or train me on more files with:
  icenet train-local /path/to/more/files

💡 Tip: Install Ollama for much better AI responses!
  Run: icenet setup-ollama"""
            else:
                # Build response from search results
                response = f"Based on your files, here's what I found:\n\n"

                for i, result in enumerate(results, 1):
                    # Truncate long results
                    display_text = result[:500] + "..." if len(result) > 500 else result
                    response += f"📄 Result {i}:\n{display_text}\n\n"

                response += f"\n💡 This is from {self.metadata.get('total_files', 'your')} files I was trained on."
                response += f"\n\n💡 Tip: Install Ollama for intelligent AI responses instead of raw data dumps!"
                response += f"\n  Run: icenet setup-ollama"

        # Handle streaming vs non-streaming
        if stream:
            # Return generator wrapper that collects response for history
            return self._stream_and_collect(response)
        else:
            # Add to history immediately for non-streaming
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            return response

    def _stream_and_collect(self, response_generator):
        """
        Wrapper that yields from generator and collects full response

        Args:
            response_generator: Generator yielding response tokens

        Yields:
            Response tokens
        """
        collected = []
        for token in response_generator:
            collected.append(token)
            yield token

        # Add complete response to history
        full_response = ''.join(collected)
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_stats(self) -> Dict:
        """Get statistics about loaded data"""
        return {
            "chunks_loaded": len(self.chunks),
            "metadata": self.metadata,
            "has_data": len(self.chunks) > 0,
        }
