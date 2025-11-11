"""
Ollama integration for IceNet - Automatic setup and management
"""

import subprocess
import requests
import time
import logging
import platform
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class OllamaManager:
    """Manages Ollama installation, models, and chat"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama manager

        Args:
            base_url: Ollama API base URL
        """
        self.base_url = base_url
        self.default_model = "llama3.2:latest"  # Good balance of speed/quality

    def is_installed(self) -> bool:
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(
                ["which", "ollama"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False

    def is_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def install_ollama(self) -> bool:
        """
        Auto-install Ollama (macOS only for now)

        Returns:
            True if successful, False otherwise
        """
        system = platform.system()

        if system == "Darwin":  # macOS
            print("\n📥 Installing Ollama via Homebrew...")
            print("   This may take a few minutes...\n")

            try:
                # Check if Homebrew is installed
                brew_check = subprocess.run(
                    ["which", "brew"],
                    capture_output=True
                )

                if brew_check.returncode != 0:
                    print("❌ Homebrew not found. Please install Homebrew first:")
                    print("   Visit: https://brew.sh")
                    return False

                # Install Ollama
                subprocess.check_call(["brew", "install", "ollama"])
                print("✅ Ollama installed successfully!")
                return True

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install Ollama: {e}")
                return False

        else:
            print(f"\n⚠️  Auto-install not supported on {system}")
            print("Please install Ollama manually:")
            print("   Visit: https://ollama.com/download")
            return False

    def start_server(self) -> bool:
        """
        Start Ollama server in background

        Returns:
            True if started successfully
        """
        try:
            print("🚀 Starting Ollama server...")

            # Start in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Wait for server to start
            for i in range(10):
                time.sleep(1)
                if self.is_running():
                    print("✅ Ollama server started!")
                    return True

            print("⚠️  Server started but not responding yet. Please wait...")
            return False

        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False

    def list_models(self) -> List[str]:
        """Get list of installed models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []

    def has_model(self, model_name: str) -> bool:
        """Check if a model is installed"""
        models = self.list_models()
        return model_name in models

    def pull_model(self, model_name: str) -> bool:
        """
        Download a model

        Args:
            model_name: Name of model to download

        Returns:
            True if successful
        """
        print(f"\n📥 Downloading {model_name}...")
        print("   This may take a few minutes (one-time setup)...\n")

        try:
            # Use ollama pull command
            subprocess.check_call(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            print(f"✅ {model_name} downloaded successfully!")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    def setup(self, model_name: Optional[str] = None) -> bool:
        """
        Complete setup: install, start, download model

        Args:
            model_name: Model to use (default: llama3.2)

        Returns:
            True if setup successful
        """
        model = model_name or self.default_model

        print("\n" + "=" * 60)
        print("🤖 IceNet AI - Ollama Setup")
        print("=" * 60)

        # Step 1: Check/Install Ollama
        if not self.is_installed():
            print("\n📦 Ollama not found. Installing...")
            if not self.install_ollama():
                return False
        else:
            print("✅ Ollama is installed")

        # Step 2: Start server if not running
        if not self.is_running():
            if not self.start_server():
                print("\n💡 Please start Ollama manually:")
                print("   Open a new terminal and run: ollama serve")
                return False
        else:
            print("✅ Ollama server is running")

        # Step 3: Download model if needed
        if not self.has_model(model):
            print(f"\n📥 Model '{model}' not found. Downloading...")
            if not self.pull_model(model):
                return False
        else:
            print(f"✅ Model '{model}' is ready")

        print("\n" + "=" * 60)
        print("🎉 Setup complete! Ready to chat with AI.")
        print("=" * 60 + "\n")

        return True

    def chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Send a chat request to Ollama

        Args:
            prompt: User's question/prompt
            model: Model to use (default: llama3.2)
            context: Additional context to include
            system_prompt: System prompt for the model
            stream: If True, yields tokens as they arrive (generator)

        Returns:
            Model's response (or generator if stream=True)
        """
        model_name = model or self.default_model

        # Build the full prompt
        if context:
            full_prompt = f"""Based on the following information:

{context}

Question: {prompt}

Please provide a helpful, accurate response."""
        else:
            full_prompt = prompt

        try:
            # Use Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": full_prompt,
                    "stream": stream,
                    "system": system_prompt or "You are a helpful AI assistant.",
                },
                stream=stream  # Enable streaming for requests library
            )

            if response.status_code == 200:
                if stream:
                    # Return generator for streaming
                    return self._stream_response(response)
                else:
                    return response.json()['response']
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "Sorry, I encountered an error. Please try again."

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return "Sorry, I couldn't connect to the AI model."

    def _stream_response(self, response):
        """
        Generator that yields tokens from streaming response

        Args:
            response: Streaming response object

        Yields:
            Text chunks as they arrive
        """
        import json

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue

    def create_fine_tuned_model(
        self,
        base_model: str,
        training_data: str,
        model_name: str,
        system_prompt: str = None,
    ) -> bool:
        """
        Create a fine-tuned model using Ollama

        Args:
            base_model: Base model to fine-tune from
            training_data: Path to training data
            model_name: Name for the new model
            system_prompt: Custom system prompt

        Returns:
            True if successful
        """
        print(f"\n🎓 Creating fine-tuned model: {model_name}")
        print(f"   Base model: {base_model}")

        try:
            # Create Modelfile
            modelfile_content = f"""FROM {base_model}

# Custom system prompt
SYSTEM {system_prompt or "You are a helpful assistant trained on the user's data."}

# Training data
"""
            # Add training examples from data
            modelfile_path = Path.home() / "icenet" / f"Modelfile-{model_name}"
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)

            # Create model
            subprocess.check_call(
                ["ollama", "create", model_name, "-f", str(modelfile_path)]
            )

            print(f"✅ Fine-tuned model '{model_name}' created!")
            return True

        except Exception as e:
            logger.error(f"Fine-tuning error: {e}")
            return False
