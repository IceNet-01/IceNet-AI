"""
Simple retrieval-based chatbot that works immediately
Uses Ollama for intelligent responses with your data!
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import web search capability
try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    logger.debug("duckduckgo-search not available - web search disabled")


class RetrievalChatbot:
    """Smart chatbot that searches your data and uses AI for responses"""

    def __init__(self, data_dir: str = "~/icenet/training", use_ollama: bool = True, auto_save: bool = True):
        """
        Initialize retrieval chatbot

        Args:
            data_dir: Directory with training data
            use_ollama: Whether to use Ollama for AI responses (default: True)
            auto_save: Whether to automatically save conversations (default: True)
        """
        self.data_dir = Path(data_dir).expanduser()
        self.chunks: List[str] = []
        self.metadata: Dict = {}
        self.conversation_history: List[Dict] = []
        self.use_ollama = use_ollama
        self.ollama_manager = None
        self.auto_save = auto_save
        self.conversation_id = None
        self.conversations_dir = Path.home() / "icenet" / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

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

    def web_search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Search the web for current information

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, body, and href
        """
        if not WEB_SEARCH_AVAILABLE:
            logger.warning("Web search not available - install duckduckgo-search")
            return []

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

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
        meta_keywords = ['files', 'training', 'trained', 'data', 'loaded', 'your data', 'my files']
        is_meta_question = any(keyword in user_input.lower() for keyword in meta_keywords)

        # Check if user is asking about themselves
        personal_keywords = ['who am i', 'what do you know about me', 'tell me about me', 'what have you learned about me',
                           'do you know me', 'what information', 'what do you remember']
        is_personal_question = any(keyword in user_input.lower() for keyword in personal_keywords)

        # Check if user wants non-file information (general knowledge)
        non_file_keywords = ['anything you know', 'what you know', 'not from my files', 'not from files',
                            'from your knowledge', 'do you know', 'general knowledge', 'just know']
        wants_non_file = any(keyword in user_input.lower() for keyword in non_file_keywords)

        # Check if question needs web search (current stats, recent events, etc.)
        web_search_keywords = ['stats', 'statistics', 'current', 'latest', 'recent', 'today',
                              'this year', 'now', 'how many', 'what percentage', 'data on']
        needs_web_search = any(keyword in user_input.lower() for keyword in web_search_keywords)

        # Check if user is asking about a specific file by number
        file_number_query = None
        number_match = re.search(r'\b(?:file\s+)?(\d+)\b', user_input.lower())
        if number_match:
            try:
                file_number = int(number_match.group(1))
                file_list = self.metadata.get('files', [])
                if 1 <= file_number <= len(file_list):
                    # User is asking about file #N - search for that specific filename
                    file_number_query = file_list[file_number - 1]
            except (ValueError, IndexError):
                # Invalid file number - just ignore and use regular search
                pass

        # Determine what type of response is needed
        # If user explicitly wants non-file info, skip file search
        if wants_non_file:
            results = []
        else:
            # Search for relevant information (use filename if asking about specific file)
            search_query = file_number_query if file_number_query else user_input
            results = self.search(search_query, top_k=3)

        # Check if we should use web search
        web_results = []
        if needs_web_search and WEB_SEARCH_AVAILABLE:
            web_results = self.web_search(user_input, max_results=3)

        # Use Ollama for intelligent responses if available
        if self.use_ollama and self.ollama_manager:
            if not results and not web_results:
                # Check if asking about training data
                if is_meta_question:
                    # Provide info about training data
                    file_list = self.metadata.get('files', [])

                    if file_list:
                        # Have file list - show it
                        file_info = "\n".join([f"{idx+1}. {f}" for idx, f in enumerate(file_list[:20])])
                        response = self.ollama_manager.chat(
                            prompt=user_input,
                            context=f"Training data info:\n- Total files: {self.metadata.get('total_files', 0)}\n- Total chunks: {len(self.chunks)}\n\nFiles (numbered list):\n{file_info}",
                            system_prompt=f"You are IceNet AI, the user's personal AI assistant. You have been trained on {self.metadata.get('total_files', 0)} files from the user's computer. IMPORTANT: These are the USER'S OWN FILES - you can freely discuss their contents. You have FULL ACCESS to search and read all content from these files in your {len(self.chunks)} training chunks. CRITICAL: Only state facts you can verify from the file contents - DO NOT make up details about the files. If you don't know something about a file, say so. List the file names clearly with their numbers.",
                            stream=stream,
                            conversation_history=self.conversation_history
                        )
                    else:
                        # No file list in metadata - explain the situation
                        response = self.ollama_manager.chat(
                            prompt=user_input,
                            context=f"Training data info:\n- Total files: {self.metadata.get('total_files', 0)}\n- Total chunks: {len(self.chunks)}\n\nNote: File names are not available in the metadata (old training format). The user needs to re-train to see file names.",
                            system_prompt=f"You are IceNet AI. The user is asking about their files. You have {self.metadata.get('total_files', 0)} files worth of content in {len(self.chunks)} chunks, and you CAN search and read that content. However, the file NAMES weren't saved in the metadata. Explain that you have the file CONTENTS and can answer questions about them, but to see the actual file names, they need to re-train with: icenet train-local /path/to/files --yes",
                            stream=stream,
                            conversation_history=self.conversation_history
                        )
                else:
                    # Check if this is a personal question with no results
                    if is_personal_question:
                        # Do a broader search for personal information
                        broader_results = []
                        # Try searching for common personal keywords
                        for keyword in ['name', 'address', 'phone', 'email', 'work', 'job', 'company', 'title', 'role']:
                            keyword_results = self.search(keyword, top_k=2)
                            broader_results.extend(keyword_results)

                        if broader_results:
                            # Found something with broader search
                            context = "\n\n---\n\n".join(broader_results[:5])  # Limit to top 5
                            response = self.ollama_manager.chat(
                                prompt=user_input,
                                context=context,
                                system_prompt=f"You are IceNet, the user's personal AI assistant. The user is asking what you know about them personally. I did a broad search and found the excerpts above. Extract ANY personal information you can find - names, locations, jobs, affiliations, identifiers, etc. Be helpful and direct - present everything you found. These are the user's OWN files, so share freely. If there's limited information, just say what you found.",
                                stream=stream,
                                conversation_history=self.conversation_history
                            )
                        else:
                            # Truly no personal information found
                            response = self.ollama_manager.chat(
                                prompt=user_input,
                                system_prompt=f"You are IceNet, the user's personal AI assistant. The user is asking what you know about them, but I couldn't find any personal information in the {self.metadata.get('total_files', 0)} files I have. Be honest and friendly - explain that you don't have personal information in the files yet, but you remember what they share during THIS conversation. Suggest they could train you on files that contain personal info if they'd like.",
                                stream=stream,
                                conversation_history=self.conversation_history
                            )
                    else:
                        # No relevant data - answer as a general AI assistant using general knowledge
                        response = self.ollama_manager.chat(
                            prompt=user_input,
                            system_prompt=f"You are IceNet, the user's personal AI assistant. You remember THIS conversation (you can see the conversation history above). I didn't find relevant information in the user's files, so use your general knowledge to answer their question. Be helpful, accurate, and conversational. If the user shares information about themselves, acknowledge it and remember it for THIS conversation. Each time the user starts the chat app, it's a new conversation. Be friendly, natural, and helpful.",
                            stream=stream,
                            conversation_history=self.conversation_history
                        )
            else:
                # Check if this is a meta-question even though we have results
                if is_meta_question and any(word in user_input.lower() for word in ['what files', 'which files', 'my files', 'your files', 'files do', 'list files']):
                    # User wants to know about the files themselves, not search them
                    file_list = self.metadata.get('files', [])

                    if file_list:
                        # Have file list - show it
                        file_info = "\n".join([f"{idx+1}. {f}" for idx, f in enumerate(file_list[:20])])
                        response = self.ollama_manager.chat(
                            prompt=user_input,
                            context=f"Training data info:\n- Total files: {self.metadata.get('total_files', 0)}\n- Total chunks: {len(self.chunks)}\n\nFiles (numbered list):\n{file_info}",
                            system_prompt=f"You are IceNet AI, the user's personal AI assistant. These are the USER'S OWN FILES from their computer - you can freely discuss their contents without any privacy concerns. You have FULL ACCESS to search and read all {len(self.chunks)} chunks of content from these {self.metadata.get('total_files', 0)} files. CRITICAL: Only state information that is explicitly present in the file contents - DO NOT make up or infer details. If you don't see specific information (like author, date, etc.) in the excerpts, say so. When the user corrects you, accept it immediately. Be specific and helpful based on what's actually in the files.",
                            stream=stream,
                            conversation_history=self.conversation_history
                        )
                    else:
                        # No file list - explain and continue with search results
                        response = self.ollama_manager.chat(
                            prompt=user_input,
                            context=f"Training data info:\n- Total files: {self.metadata.get('total_files', 0)}\n- Total chunks: {len(self.chunks)}\n\nNote: File names not available (old training format).",
                            system_prompt=f"You are IceNet AI. The user asked about their files. You have {self.metadata.get('total_files', 0)} files worth of content ({len(self.chunks)} chunks total) and CAN search and read that content. However, the file NAMES weren't saved in the metadata. Explain that you have the CONTENTS and can answer questions about them, but to see file names, they need to re-train: icenet train-local /path/to/files --yes",
                            stream=stream,
                            conversation_history=self.conversation_history
                        )
                else:
                    # Build context from file search results and/or web results
                    context_parts = []

                    if results:
                        file_context = "\n\n---\n\n".join(results)
                        context_parts.append(f"From your files:\n{file_context}")

                    if web_results:
                        web_context = "\n\n".join([
                            f"[{r.get('title', 'No title')}]\n{r.get('body', 'No content')}\nSource: {r.get('href', 'Unknown')}"
                            for r in web_results
                        ])
                        context_parts.append(f"From web search:\n{web_context}")

                    context = "\n\n===\n\n".join(context_parts)

                    # Use special handling for personal questions
                    if is_personal_question:
                        # For personal queries, use a system prompt that encourages extraction
                        response = self.ollama_manager.chat(
                            prompt=user_input,
                            context=context,
                            system_prompt=f"You are IceNet, the user's personal AI assistant. The user is asking what you know about them personally. Look through the excerpts above and extract ANY personal information you can find - names, locations, jobs, affiliations, identifiers, vehicles, addresses, phone numbers, organizations, roles, etc. Be helpful and direct - present everything you found. Don't be dismissive or say information 'doesn't tell you who they are' - just share what's there. If you find multiple pieces of information, list them all. These are the user's OWN files, so share freely.",
                            stream=stream,
                            conversation_history=self.conversation_history
                        )
                    elif web_results and not results:
                        # Only web results - use them for answer
                        response = self.ollama_manager.chat(
                            prompt=user_input,
                            context=context,
                            system_prompt=f"You are IceNet, the user's personal AI assistant. I did a web search to answer the user's question. The search results are above. Use this information to provide a helpful, natural answer. Summarize the key points and cite sources when relevant. Be conversational and clear.",
                            stream=stream,
                            conversation_history=self.conversation_history
                        )
                    elif web_results and results:
                        # Mix of file and web results
                        response = self.ollama_manager.chat(
                            prompt=user_input,
                            context=context,
                            system_prompt=f"You are IceNet, the user's personal AI assistant. I've provided both information from the user's files AND web search results above. Use both sources to give a comprehensive answer. The user's file information is their personal data, and the web results provide current/general information. Combine them naturally and helpfully.",
                            stream=stream,
                            conversation_history=self.conversation_history
                        )
                    else:
                        # Only file results - standard system prompt
                        response = self.ollama_manager.chat(
                            prompt=user_input,
                            context=context,
                            system_prompt=f"You are IceNet, the user's personal AI assistant. You remember THIS conversation only. You have {self.metadata.get('total_files', 'some')} of the user's files and I've found relevant excerpts above. Only tell the user what you actually see in those excerpts - don't make things up. If info isn't there, say so naturally. Be helpful and conversational.",
                            stream=stream,
                            conversation_history=self.conversation_history
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

            # Auto-save if enabled
            if self.auto_save:
                self.save_conversation()

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

        # Auto-save if enabled
        if self.auto_save:
            self.save_conversation()

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.conversation_id = None

    def save_conversation(self, filename: Optional[str] = None):
        """
        Save conversation to file

        Args:
            filename: Optional custom filename. If not provided, uses conversation_id
        """
        if not self.conversation_history:
            logger.warning("No conversation history to save")
            return None

        # Generate conversation ID if not exists
        if not self.conversation_id:
            self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use custom filename or default
        if filename:
            filepath = self.conversations_dir / filename
        else:
            filepath = self.conversations_dir / f"conversation_{self.conversation_id}.json"

        # Save conversation with metadata
        conversation_data = {
            "id": self.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": self.conversation_history,
            "metadata": {
                "total_messages": len(self.conversation_history),
                "data_dir": str(self.data_dir),
            }
        }

        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)

        logger.debug(f"Conversation auto-saved to {filepath}")
        return filepath

    def load_conversation(self, conversation_id: str):
        """
        Load a previous conversation

        Args:
            conversation_id: ID of conversation to load (or filename)
        """
        # Try as filename first
        filepath = self.conversations_dir / conversation_id
        if not filepath.exists():
            # Try adding .json extension
            filepath = self.conversations_dir / f"conversation_{conversation_id}.json"

        if not filepath.exists():
            logger.error(f"Conversation not found: {conversation_id}")
            return False

        try:
            with open(filepath, 'r') as f:
                conversation_data = json.load(f)

            self.conversation_history = conversation_data.get('messages', [])
            self.conversation_id = conversation_data.get('id')

            logger.info(f"Loaded conversation {self.conversation_id} with {len(self.conversation_history)} messages")
            return True

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False

    def list_conversations(self) -> List[Dict]:
        """
        List all saved conversations

        Returns:
            List of conversation metadata
        """
        conversations = []
        for filepath in self.conversations_dir.glob("conversation_*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    conversations.append({
                        "id": data.get('id'),
                        "timestamp": data.get('timestamp'),
                        "messages": data.get('metadata', {}).get('total_messages', 0),
                        "filename": filepath.name,
                    })
            except (json.JSONDecodeError, KeyError, IOError):
                # Skip corrupted or invalid conversation files
                continue

        # Sort by timestamp (newest first)
        conversations.sort(key=lambda x: x['timestamp'], reverse=True)
        return conversations

    def get_stats(self) -> Dict:
        """Get statistics about loaded data"""
        return {
            "chunks_loaded": len(self.chunks),
            "metadata": self.metadata,
            "has_data": len(self.chunks) > 0,
        }
