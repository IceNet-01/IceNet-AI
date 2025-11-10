"""
Main terminal UI application inspired by nomadnet
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    Label,
    Input,
    RichLog,
    DataTable,
    TabbedContent,
    TabPane,
)
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import RenderableType
import logging
from datetime import datetime

from icenet.core.engine import IceNetEngine
from icenet.core.device import DeviceManager
from icenet.data.local_loader import LocalFileLoader
from icenet.chat.chatbot import SimpleChatbot


logger = logging.getLogger(__name__)


class StatusPanel(Static):
    """Status panel showing system info"""

    def __init__(self, device_manager: DeviceManager):
        super().__init__()
        self.device_manager = device_manager

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        info = self.device_manager.get_info_dict()

        status_text = Text()
        status_text.append("IceNet AI\n", style="bold cyan")
        status_text.append(f"Device: {info['device']}\n", style="green")
        status_text.append(f"Memory: {info['unified_memory_gb']} GB\n")
        status_text.append(f"CPU Cores: {info['cpu_cores']}\n")
        status_text.append(f"GPU Cores: {info['gpu_cores']}\n")
        status_text.append(f"MPS: {'✓' if info['has_mps'] else '✗'}\n")
        status_text.append(f"Neural Engine: {'✓' if info['has_neural_engine'] else '✗'}\n")

        yield Static(status_text)


class LogPanel(Static):
    """Log panel for displaying training logs"""

    def __init__(self):
        super().__init__()
        self.log_content = RichLog()

    def compose(self) -> ComposeResult:
        yield self.log_content

    def add_log(self, message: str, level: str = "info"):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if level == "error":
            style = "red"
        elif level == "warning":
            style = "yellow"
        elif level == "success":
            style = "green"
        else:
            style = "white"

        text = Text(f"[{timestamp}] {message}", style=style)
        self.log_content.write(text)


class IceNetApp(App):
    """
    Main IceNet terminal application

    Inspired by nomadnet's clean terminal interface
    """

    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        background: $primary;
        color: $text;
        height: 3;
    }

    Footer {
        background: $primary-darken-2;
        color: $text;
    }

    .status-panel {
        width: 30;
        height: 100%;
        background: $panel;
        border: solid $primary;
        padding: 1;
    }

    .main-panel {
        width: 1fr;
        height: 100%;
        background: $background;
        padding: 1;
    }

    .log-panel {
        height: 15;
        background: $panel;
        border: solid $accent;
        padding: 1;
    }

    Button {
        width: 100%;
        margin: 1;
    }

    Input {
        width: 100%;
        margin: 1;
    }

    .section-title {
        color: $accent;
        text-style: bold;
        margin: 1;
    }

    DataTable {
        height: 100%;
    }

    TabbedContent {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("e", "tab_easy", "Easy Train"),
        Binding("c", "tab_chat", "Chat"),
        Binding("t", "tab_train", "Train"),
        Binding("m", "tab_model", "Model"),
        Binding("s", "tab_settings", "Settings"),
        Binding("h", "tab_help", "Help"),
    ]

    TITLE = "IceNet AI - Apple M4 Pro Optimized"

    def __init__(self):
        super().__init__()
        self.device_manager = DeviceManager()
        self.engine = None
        self.chatbot = SimpleChatbot()
        self.file_loader = None
        self.scanned_files = []

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()

        with Horizontal():
            # Left sidebar - Status
            with Vertical(classes="status-panel"):
                yield Static("System Status", classes="section-title")
                yield StatusPanel(self.device_manager)

            # Main content area
            with Vertical(classes="main-panel"):
                with TabbedContent():
                    # Home tab
                    with TabPane("Home", id="tab-home"):
                        yield Static("Welcome to IceNet AI", classes="section-title")
                        yield Static(
                            "A powerful AI system optimized for Apple M4 Pro\n\n"
                            "🚀 NEW - No Technical Skills Needed!\n"
                            "  • [e] Easy Train - Train on your files in seconds\n"
                            "  • [c] Chat - Talk with your AI\n\n"
                            "Features:\n"
                            "  • Metal Performance Shaders (MPS) acceleration\n"
                            "  • Train on ANY folder (documents, code, notes)\n"
                            "  • Multiple model architectures\n"
                            "  • Real-time monitoring\n\n"
                            "Quick Start:\n"
                            "  [e] Easy Train   [c] Chat   [t] Advanced   [h] Help   [q] Quit"
                        )

                    # Easy Train tab (NEW!)
                    with TabPane("Easy Train", id="tab-easy"):
                        yield Static("🚀 Train on Your Files - No Tech Skills Needed!", classes="section-title")
                        yield Static("Step 1: Choose a folder")
                        yield Input(
                            placeholder="e.g., /Users/you/Documents or ~/Desktop/MyNotes",
                            id="easy-train-path"
                        )
                        yield Static("\nStep 2: Click to start!")
                        yield Button("📂 Scan Files", id="scan-files", variant="primary")
                        yield Button("⚡ Start Training", id="easy-train-start", variant="success")
                        yield Static("\nWhat will happen:")
                        yield Static(
                            "  1. IceNet scans your folder for text files\n"
                            "  2. Shows you what it found\n"
                            "  3. Prepares training data automatically\n"
                            "  4. Ready to chat about your documents!\n\n"
                            "Supported files: .txt, .md, .py, .js, .html, and 25+ more!"
                        )

                    # Chat tab (NEW!)
                    with TabPane("Chat", id="tab-chat"):
                        yield Static("💬 Chat with IceNet AI", classes="section-title")
                        yield Static(
                            "Start a conversation! Type your message below.\n"
                            "Note: Train on your files first for better responses!"
                        )
                        yield Input(placeholder="Type your message here...", id="chat-input")
                        yield Button("Send", id="send-chat", variant="success")
                        yield Static("\nConversation:", classes="section-title")
                        yield RichLog(id="chat-log")
                        yield Static("\nCommands: Type 'clear' to reset, 'exit' to quit chat")

                    # Train tab
                    with TabPane("Train", id="tab-train"):
                        yield Static("Training", classes="section-title")
                        yield Input(placeholder="Config path (e.g., configs/example.yaml)", id="config-input")
                        yield Button("Start Training", id="start-train", variant="success")
                        yield Button("Stop Training", id="stop-train", variant="error")
                        yield Static("Training Progress:", classes="section-title")
                        yield DataTable(id="metrics-table")

                    # Model tab
                    with TabPane("Model", id="tab-model"):
                        yield Static("Model Management", classes="section-title")
                        yield Input(placeholder="Model checkpoint path", id="model-input")
                        yield Button("Load Model", id="load-model", variant="primary")
                        yield Button("Save Model", id="save-model", variant="primary")
                        yield Static("Model Info:", classes="section-title")
                        yield Static(id="model-info")

                    # Settings tab
                    with TabPane("Settings", id="tab-settings"):
                        yield Static("Settings", classes="section-title")
                        yield Static("Device: Auto (MPS preferred)")
                        yield Static("Mixed Precision: FP16")
                        yield Static("Batch Size: Auto")

                    # Help tab
                    with TabPane("Help", id="tab-help"):
                        yield Static("Help & Documentation", classes="section-title")
                        yield Static(
                            "Keyboard Shortcuts:\n"
                            "  q - Quit application\n"
                            "  e - Easy Train (NEW! No tech skills needed)\n"
                            "  c - Chat with IceNet AI (NEW!)\n"
                            "  t - Advanced Train tab\n"
                            "  m - Model tab\n"
                            "  s - Settings tab\n"
                            "  h - Help tab\n\n"
                            "🚀 Easy Training (No Config Needed!):\n"
                            "  1. Press 'e' to open Easy Train tab\n"
                            "  2. Enter path to your folder (e.g., ~/Documents)\n"
                            "  3. Click 'Scan Files' to see what's found\n"
                            "  4. Click 'Start Training' to prepare data\n"
                            "  5. Press 'c' to chat about your files!\n\n"
                            "💬 Chatting:\n"
                            "  1. Press 'c' to open Chat tab\n"
                            "  2. Type your message and click 'Send'\n"
                            "  3. Chat naturally with IceNet!\n\n"
                            "Advanced Training:\n"
                            "  1. Create a YAML config file\n"
                            "  2. Go to Train tab (press 't')\n"
                            "  3. Enter config path\n"
                            "  4. Click 'Start Training'\n\n"
                            "For more information, visit:\n"
                            "  https://github.com/IceNet-01/IceNet-AI"
                        )

                # Log panel at bottom
                with Container(classes="log-panel"):
                    yield Static("Logs", classes="section-title")
                    yield LogPanel()

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted"""
        # Setup metrics table
        table = self.query_one("#metrics-table", DataTable)
        table.add_columns("Epoch", "Train Loss", "Val Loss", "Time")

        # Initial log
        log_panel = self.query_one(LogPanel)
        log_panel.add_log("IceNet initialized successfully", "success")
        log_panel.add_log(f"Device: {self.device_manager.device}", "info")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        log_panel = self.query_one(LogPanel)

        if button_id == "start-train":
            config_input = self.query_one("#config-input", Input)
            config_path = config_input.value

            if not config_path:
                log_panel.add_log("Please enter a config path", "error")
                return

            log_panel.add_log(f"Starting training with config: {config_path}", "info")
            # TODO: Implement training start

        elif button_id == "stop-train":
            log_panel.add_log("Stopping training...", "warning")
            # TODO: Implement training stop

        elif button_id == "load-model":
            model_input = self.query_one("#model-input", Input)
            model_path = model_input.value

            if not model_path:
                log_panel.add_log("Please enter a model path", "error")
                return

            log_panel.add_log(f"Loading model from: {model_path}", "info")
            # TODO: Implement model loading

        elif button_id == "save-model":
            model_input = self.query_one("#model-input", Input)
            model_path = model_input.value

            if not model_path:
                log_panel.add_log("Please enter a model path", "error")
                return

            log_panel.add_log(f"Saving model to: {model_path}", "info")
            # TODO: Implement model saving

        elif button_id == "scan-files":
            # Scan files for easy training
            path_input = self.query_one("#easy-train-path", Input)
            scan_path = path_input.value.strip()

            if not scan_path:
                log_panel.add_log("Please enter a folder path", "error")
                return

            # Expand ~ to home directory
            from pathlib import Path
            scan_path = str(Path(scan_path).expanduser())

            log_panel.add_log(f"📂 Scanning files in: {scan_path}", "info")

            try:
                self.file_loader = LocalFileLoader(scan_path, recursive=True)
                self.scanned_files = self.file_loader.scan_files()
                stats = self.file_loader.get_statistics(self.scanned_files)

                log_panel.add_log(f"✓ Found {stats['total_files']} files ({stats['total_size_mb']:.2f} MB)", "success")
                log_panel.add_log(f"  Ready to train! Click 'Start Training' button.", "info")
            except Exception as e:
                log_panel.add_log(f"❌ Error scanning files: {e}", "error")

        elif button_id == "easy-train-start":
            # Start easy training
            if not self.scanned_files:
                log_panel.add_log("❌ Please scan files first!", "error")
                return

            log_panel.add_log("⚡ Preparing training data...", "info")

            try:
                import json
                chunks = self.file_loader.load_as_chunks(chunk_size=1000, overlap=100)
                stats = self.file_loader.get_statistics(self.scanned_files)

                # Save training data
                output_dir = Path.home() / "icenet" / "training"
                output_dir.mkdir(parents=True, exist_ok=True)

                data_file = output_dir / "training_data.txt"
                with open(data_file, 'w', encoding='utf-8') as f:
                    for chunk in chunks:
                        f.write(chunk + "\n\n---\n\n")

                # Save metadata
                metadata_file = output_dir / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump({
                        'total_files': stats['total_files'],
                        'total_chunks': len(chunks),
                        'chunk_size': 1000,
                        'statistics': stats,
                    }, f, indent=2)

                # Reload the chatbot with new data
                from icenet.chat.retrieval import RetrievalChatbot
                self.chatbot.retrieval_bot = RetrievalChatbot(str(output_dir))

                log_panel.add_log(f"✅ READY! Processed {stats['total_files']} files ({len(chunks)} chunks)", "success")
                log_panel.add_log("💬 Go to Chat tab and start asking questions!", "success")
                log_panel.add_log("💾 Data saved for future sessions", "info")
            except Exception as e:
                log_panel.add_log(f"❌ Error preparing data: {e}", "error")

        elif button_id == "send-chat":
            # Send chat message
            chat_input = self.query_one("#chat-input", Input)
            user_message = chat_input.value.strip()

            if not user_message:
                return

            # Get chat log
            chat_log = self.query_one("#chat-log", RichLog)

            # Show user message
            user_text = Text(f"You: {user_message}", style="cyan")
            chat_log.write(user_text)

            # Get response from chatbot
            response = self.chatbot.chat(user_message)

            # Show bot response
            bot_text = Text(f"IceNet: {response}", style="green")
            chat_log.write(bot_text)
            chat_log.write("")  # Empty line for spacing

            # Clear input
            chat_input.value = ""

    def action_tab_train(self) -> None:
        """Switch to train tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-train"

    def action_tab_model(self) -> None:
        """Switch to model tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-model"

    def action_tab_settings(self) -> None:
        """Switch to settings tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-settings"

    def action_tab_help(self) -> None:
        """Switch to help tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-help"

    def action_tab_easy(self) -> None:
        """Switch to easy train tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-easy"

    def action_tab_chat(self) -> None:
        """Switch to chat tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-chat"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input fields"""
        if event.input.id == "chat-input":
            # Simulate clicking the send button
            send_button = self.query_one("#send-chat", Button)
            self.on_button_pressed(Button.Pressed(send_button))


def run_ui():
    """Run the IceNet UI"""
    app = IceNetApp()
    app.run()
