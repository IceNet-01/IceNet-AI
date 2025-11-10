# Contributing to IceNet AI

Thank you for your interest in contributing to IceNet AI!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/IceNet-AI.git
   cd IceNet-AI
   ```

3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all public functions/classes
- Run `black` for formatting: `black .`
- Run `flake8` for linting: `flake8 .`

## Testing

Run tests with pytest:
```bash
pytest tests/
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add your feature"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

## Reporting Issues

When reporting issues, please include:
- Your macOS version
- Apple Silicon model (M1, M2, M3, M4)
- IceNet version
- Steps to reproduce
- Error messages/logs

## Feature Requests

We welcome feature requests! Please open an issue and describe:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for contributing to IceNet AI!
