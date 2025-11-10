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
                            "Features:\n"
                            "  • Metal Performance Shaders (MPS) acceleration\n"
                            "  • Easy training with YAML configs\n"
                            "  • Multiple model architectures\n"
                            "  • Real-time monitoring\n\n"
                            "Use tabs or keyboard shortcuts to navigate:\n"
                            "  [t] Train   [m] Model   [s] Settings   [h] Help   [q] Quit"
                        )

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
                            "  t - Switch to Train tab\n"
                            "  m - Switch to Model tab\n"
                            "  s - Switch to Settings tab\n"
                            "  h - Switch to Help tab\n\n"
                            "Training:\n"
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


def run_ui():
    """Run the IceNet UI"""
    app = IceNetApp()
    app.run()
