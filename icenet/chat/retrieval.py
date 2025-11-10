"""
Simple retrieval-based chatbot that works immediately
No training needed - just searches your data!
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RetrievalChatbot:
    """Simple chatbot that searches through your training data"""

    def __init__(self, data_dir: str = "~/icenet/training"):
        """
        Initialize retrieval chatbot

        Args:
            data_dir: Directory with training data
        """
        self.data_dir = Path(data_dir).expanduser()
        self.chunks: List[str] = []
        self.metadata: Dict = {}
        self.conversation_history: List[Dict] = []

        # Try to load data automatically
        self.load_data()

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

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search for relevant chunks

        Args:
            query: Search query
            top_k: Number of results to return

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

            if matches > 0:
                scored_chunks.append((matches, chunk))

        # Sort by score and return top_k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def chat(self, user_input: str) -> str:
        """
        Generate response to user input

        Args:
            user_input: User's message

        Returns:
            Response string
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

        # Search for relevant information
        results = self.search(user_input, top_k=3)

        if not results:
            response = f"""I couldn't find anything about "{user_input}" in your files.

I have data from {self.metadata.get('total_files', 'some')} files.

Try asking about:
- Topics mentioned in your files
- Keywords from your documents
- General questions about your data

Or train me on more files with:
  icenet train-local /path/to/more/files"""
        else:
            # Build response from search results
            response = f"Based on your files, here's what I found:\n\n"

            for i, result in enumerate(results, 1):
                # Truncate long results
                display_text = result[:500] + "..." if len(result) > 500 else result
                response += f"📄 Result {i}:\n{display_text}\n\n"

            response += f"\n💡 This is from {self.metadata.get('total_files', 'your')} files I was trained on."

        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

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
