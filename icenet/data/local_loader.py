"""
Local file data loader for training on computer files
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class LocalFileLoader:
    """Load and process local files for training"""

    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.h',
        '.html', '.css', '.json', '.yaml', '.yml', '.sh', '.bash',
        '.rs', '.go', '.rb', '.php', '.swift', '.kt', '.ts', '.jsx', '.tsx',
        '.r', '.sql', '.xml', '.csv', '.log', '.conf', '.cfg',
        '.pdf', '.docx', '.doc', '.rtf'  # Document formats
    }

    def __init__(
        self,
        root_path: Union[str, Path],
        recursive: bool = True,
        exclude_dirs: Optional[List[str]] = None,
        max_file_size_mb: float = 10.0,
    ):
        """
        Initialize local file loader

        Args:
            root_path: Root directory to scan
            recursive: Whether to scan subdirectories
            exclude_dirs: Directories to exclude (e.g., ['.git', 'node_modules'])
            max_file_size_mb: Maximum file size to process in MB
        """
        self.root_path = Path(root_path)
        self.recursive = recursive
        self.exclude_dirs = exclude_dirs or [
            '.git', 'node_modules', '__pycache__', 'venv', '.venv',
            'build', 'dist', '.idea', '.vscode', 'target'
        ]
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes

    def scan_files(self) -> List[Path]:
        """
        Scan directory for supported files

        Returns:
            List of file paths
        """
        files = []

        if not self.root_path.exists():
            logger.error(f"Path does not exist: {self.root_path}")
            return files

        if self.root_path.is_file():
            if self._is_supported(self.root_path):
                files.append(self.root_path)
            return files

        # Scan directory
        pattern = '**/*' if self.recursive else '*'

        for file_path in self.root_path.glob(pattern):
            if not file_path.is_file():
                continue

            # Check if in excluded directory
            if self._is_excluded(file_path):
                continue

            # Check extension and size
            if self._is_supported(file_path) and self._check_size(file_path):
                files.append(file_path)

        logger.info(f"Found {len(files)} files to process")
        return files

    def _extract_text_from_pdf(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF file"""
        try:
            import pypdf
            text = []
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)
        except ImportError:
            logger.warning(f"pypdf not installed. Install with: pip install pypdf")
            return None
        except Exception as e:
            logger.debug(f"Failed to extract text from PDF {file_path}: {e}")
            return None

    def _extract_text_from_docx(self, file_path: Path) -> Optional[str]:
        """Extract text from DOCX file"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)
            return "\n".join(text)
        except ImportError:
            logger.warning(f"python-docx not installed. Install with: pip install python-docx")
            return None
        except Exception as e:
            logger.debug(f"Failed to extract text from DOCX {file_path}: {e}")
            return None

    def load_texts(self, files: Optional[List[Path]] = None) -> List[str]:
        """
        Load text content from files

        Args:
            files: List of files to load (if None, scans automatically)

        Returns:
            List of text contents
        """
        if files is None:
            files = self.scan_files()

        texts = []
        successful = 0
        failed = 0

        for file_path in files:
            try:
                content = None

                # Handle different file types
                if file_path.suffix.lower() == '.pdf':
                    content = self._extract_text_from_pdf(file_path)
                elif file_path.suffix.lower() in ['.docx', '.doc']:
                    content = self._extract_text_from_docx(file_path)
                else:
                    # Text-based files
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                if content and content.strip():  # Only add non-empty files
                    # Add file path as header for context
                    text = f"# File: {file_path.relative_to(self.root_path)}\n\n{content}"
                    texts.append(text)
                    successful += 1
                elif content is None:
                    failed += 1
            except Exception as e:
                logger.debug(f"Failed to read {file_path}: {e}")
                failed += 1

        logger.info(f"Successfully loaded {successful} files, failed {failed}")
        return texts

    def load_as_chunks(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        files: Optional[List[Path]] = None
    ) -> List[str]:
        """
        Load files and split into chunks for training

        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            files: List of files to load

        Returns:
            List of text chunks
        """
        texts = self.load_texts(files)
        chunks = []

        for text in texts:
            # Split into chunks with overlap
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]

                if chunk.strip():
                    chunks.append(chunk)

                start += chunk_size - overlap

        logger.info(f"Created {len(chunks)} chunks from {len(texts)} files")
        return chunks

    def get_statistics(self, files: Optional[List[Path]] = None) -> Dict:
        """
        Get statistics about the dataset

        Args:
            files: List of files to analyze

        Returns:
            Dictionary with statistics
        """
        if files is None:
            files = self.scan_files()

        total_size = sum(f.stat().st_size for f in files)

        # Group by extension
        by_extension = {}
        for file_path in files:
            ext = file_path.suffix
            by_extension[ext] = by_extension.get(ext, 0) + 1

        return {
            'total_files': len(files),
            'total_size_mb': total_size / (1024 * 1024),
            'by_extension': by_extension,
            'root_path': str(self.root_path),
        }

    def _is_supported(self, file_path: Path) -> bool:
        """Check if file extension is supported"""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _is_excluded(self, file_path: Path) -> bool:
        """Check if file is in excluded directory"""
        parts = file_path.parts
        return any(excluded in parts for excluded in self.exclude_dirs)

    def _check_size(self, file_path: Path) -> bool:
        """Check if file size is within limit"""
        try:
            return file_path.stat().st_size <= self.max_file_size
        except:
            return False
