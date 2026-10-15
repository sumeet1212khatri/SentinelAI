"""
pdf_chunker.py — Section- and table-aware PDF parser for the enterprise manual.
Uses PyMuPDF (fitz) for high-fidelity extraction with layout awareness.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of text from the PDF with metadata."""
    chunk_id: int
    text: str
    page_number: int
    section_title: str = ""
    chunk_type: str = "text"  # "text", "table", "heading", "list"
    start_char: int = 0
    end_char: int = 0
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class ParsedDocument:
    """Complete parsed document."""
    chunks: List[DocumentChunk]
    total_pages: int
    total_chunks: int
    filename: str
    sections: List[str] = field(default_factory=list)


class PDFChunker:
    """
    Section- and table-aware PDF parser.
    Features:
    - Heading/section detection
    - Table extraction
    - Smart chunking with overlap
    - Metadata preservation
    """

    # Patterns for section detection
    _HEADING_PATTERNS = [
        re.compile(r'^(\d+\.?\d*\.?\d*)\s+(.+)$'),                    # "1.2.3 Title"
        re.compile(r'^(Chapter|Section|Part|Appendix)\s+\w+', re.IGNORECASE),
        re.compile(r'^[A-Z][A-Z\s]{5,}$'),                            # ALL CAPS headings
        re.compile(r'^(Introduction|Overview|Summary|Conclusion|References|Glossary|FAQ)', re.IGNORECASE),
    ]

    # Table indicators
    _TABLE_INDICATORS = [
        re.compile(r'\|.*\|.*\|'),           # Pipe-delimited tables
        re.compile(r'\t.*\t'),                # Tab-delimited
        re.compile(r'^\s*[-+]+\s*$'),         # Separator lines
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file into structured chunks.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedDocument with all chunks and metadata.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path}")
        doc = fitz.open(str(pdf_path))

        all_blocks = []
        sections_found = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text blocks with position info
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]

            page_text_parts = []
            for block in blocks:
                if block.get("type", 0) == 0:  # Text block
                    lines_text = []
                    for line in block.get("lines", []):
                        spans_text = ""
                        max_font_size = 0
                        is_bold = False
                        for span in line.get("spans", []):
                            spans_text += span.get("text", "")
                            font_size = span.get("size", 12)
                            if font_size > max_font_size:
                                max_font_size = font_size
                            if "bold" in span.get("font", "").lower():
                                is_bold = True

                        line_text = spans_text.strip()
                        if line_text:
                            lines_text.append({
                                "text": line_text,
                                "font_size": max_font_size,
                                "is_bold": is_bold,
                            })

                    for line_info in lines_text:
                        page_text_parts.append({
                            "text": line_info["text"],
                            "page": page_num + 1,
                            "font_size": line_info["font_size"],
                            "is_bold": line_info["is_bold"],
                        })

            # Also extract any tables
            try:
                tables = page.find_tables()
                if tables and tables.tables:
                    for table in tables.tables:
                        table_text = self._format_table(table)
                        if table_text.strip():
                            page_text_parts.append({
                                "text": table_text,
                                "page": page_num + 1,
                                "font_size": 12,
                                "is_bold": False,
                                "is_table": True,
                            })
            except Exception:
                pass  # Table extraction may not be available in all versions

            all_blocks.extend(page_text_parts)

        total_pages = len(doc)
        doc.close()

        # Build sections and chunk
        chunks = self._build_chunks(all_blocks, sections_found)

        result = ParsedDocument(
            chunks=chunks,
            total_pages=total_pages,
            total_chunks=len(chunks),
            filename=pdf_path.name,
            sections=sections_found,
        )

        logger.info(f"Parsed {result.total_pages} pages into {result.total_chunks} chunks")
        return result

    def _is_heading(self, text: str, font_size: float = 12, is_bold: bool = False) -> bool:
        """Detect if a line is a section heading."""
        if font_size > 14 and is_bold:
            return True
        if font_size > 16:
            return True
        for pattern in self._HEADING_PATTERNS:
            if pattern.match(text.strip()):
                return True
        return False

    def _format_table(self, table) -> str:
        """Format a table into readable text."""
        try:
            rows = table.extract()
            if not rows:
                return ""

            lines = []
            for row in rows:
                cells = [str(cell) if cell is not None else "" for cell in row]
                lines.append(" | ".join(cells))

            return "\n".join(lines)
        except Exception:
            return ""

    def _build_chunks(
        self,
        blocks: List[dict],
        sections_found: List[str],
    ) -> List[DocumentChunk]:
        """Build overlapping chunks from extracted blocks."""
        chunks = []
        current_section = "Document Start"
        current_text = ""
        current_page = 1
        chunk_id = 0

        for block in blocks:
            text = block["text"]
            page = block.get("page", 1)
            font_size = block.get("font_size", 12)
            is_bold = block.get("is_bold", False)
            is_table = block.get("is_table", False)

            # Check if this is a heading
            if self._is_heading(text, font_size, is_bold) and not is_table:
                # Flush current buffer as a chunk
                if len(current_text.strip()) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        text=current_text.strip(),
                        page_number=current_page,
                        section_title=current_section,
                        chunk_type="text",
                    ))
                    chunk_id += 1

                current_section = text.strip()
                sections_found.append(current_section)
                # Start new chunk with overlap from previous
                overlap_text = current_text[-self.chunk_overlap:] if len(current_text) > self.chunk_overlap else ""
                current_text = overlap_text + "\n" + text + "\n"
                current_page = page
                continue

            # Handle tables as separate chunks
            if is_table:
                if len(current_text.strip()) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        text=current_text.strip(),
                        page_number=current_page,
                        section_title=current_section,
                        chunk_type="text",
                    ))
                    chunk_id += 1
                    current_text = ""

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=text.strip(),
                    page_number=page,
                    section_title=current_section,
                    chunk_type="table",
                ))
                chunk_id += 1
                continue

            # Accumulate text
            current_text += " " + text
            current_page = page

            # Check if we exceeded chunk size
            if len(current_text.split()) >= self.chunk_size:
                words = current_text.split()
                chunk_words = words[:self.chunk_size]
                overlap_words = words[max(0, self.chunk_size - self.chunk_overlap):]

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=" ".join(chunk_words).strip(),
                    page_number=current_page,
                    section_title=current_section,
                    chunk_type="text",
                ))
                chunk_id += 1

                current_text = " ".join(overlap_words)

        # Flush remaining text
        if len(current_text.strip()) >= self.min_chunk_size:
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=current_text.strip(),
                page_number=current_page,
                section_title=current_section,
                chunk_type="text",
            ))

        return chunks
