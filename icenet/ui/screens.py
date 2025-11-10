"""Screen definitions for IceNet UI"""

from textual.screen import Screen
from textual.containers import Container
from textual.widgets import Static


class MainScreen(Screen):
    """Main screen"""

    def compose(self):
        yield Container(
            Static("IceNet AI - Main Screen"),
        )


class TrainScreen(Screen):
    """Training screen"""

    def compose(self):
        yield Container(
            Static("Training Screen"),
        )


class ModelScreen(Screen):
    """Model management screen"""

    def compose(self):
        yield Container(
            Static("Model Management Screen"),
        )


class SettingsScreen(Screen):
    """Settings screen"""

    def compose(self):
        yield Container(
            Static("Settings Screen"),
        )
