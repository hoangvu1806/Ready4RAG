"""
PDF-to-Markdown converter.

Renders each PDF page to a PNG image and sends it to a vision LLM for
structured Markdown extraction. Pages are processed concurrently.
"""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
from PIL import Image
from tqdm import tqdm

from .config import PDF2MDConfig
from .llm_providers import create_provider


_EXTRACTION_PROMPT_BASE = """\
You are an expert OCR system that converts PDF page images to high-quality Markdown.

Extract ALL text content from this page image with MAXIMUM ACCURACY.

REQUIREMENTS:

1. TEXT ACCURACY (highest priority)
   - Preserve every character exactly, including diacritics and special symbols.
   - Do not skip, omit, or hallucinate any content.
   - Preserve all numbers, dates, measurements, and units exactly as shown.

2. DOCUMENT STRUCTURE
   - Use Markdown headings to represent the logical hierarchy: #, ##, ###, ####.
   - Maintain the original reading order (top-to-bottom, left-to-right).
   - Use horizontal rules (---) only to separate clearly distinct major sections.

3. TABLES
   - Convert all tables to Markdown pipe syntax.
   - Preserve all cell content; do not omit any rows or columns.

4. LISTS
   - Ordered lists: 1., 2., 3.
   - Unordered lists: -
   - Preserve nesting with proper indentation.
"""

_IMAGE_SECTION = """\
5. IMAGES AND FIGURES
   - For each image or figure on the page, insert: {{IMAGE_PLACEHOLDER}}
   - If a caption is visible, use: ![caption]({{IMAGE_PLACEHOLDER}})
   - Maintain the position of the placeholder relative to surrounding text.
"""

_FORMULA_SECTION = """\
6. MATHEMATICAL FORMULAS
   - Inline math: $formula$
   - Display/block math: $$formula$$
   - Use standard LaTeX: \\frac{}{}, \\sum, \\int, \\alpha, \\beta, etc.
   - Ensure all LaTeX is syntactically valid.
"""

_EXTRACTION_PROMPT_FOOTER = """\
7. SPECIAL FORMATTING
   - Bold: **text**, italic: *text*, code/monospace: `code`
   - Blockquotes: >
   - Footnotes and references must be preserved exactly.

8. OUTPUT FORMAT
   - Output ONLY the Markdown content; no preamble, no explanations.
   - Start directly with the content.
   - Preserve the original language of the document.

Extract the content now:
"""


class PDFToMarkdownConverter:
    """Converts PDF documents to Markdown using a vision LLM per page."""

    def __init__(self, config: Optional[PDF2MDConfig] = None):
        self._config = config or PDF2MDConfig()
        self._config.validate()
        self._provider = create_provider(self._config)
        self._images_dir: Optional[Path] = None

    def _build_prompt(self, has_embedded_images: bool) -> str:
        prompt = _EXTRACTION_PROMPT_BASE
        if has_embedded_images:
            prompt += _IMAGE_SECTION
        if self._config.preserve_formulas:
            prompt += _FORMULA_SECTION
        prompt += _EXTRACTION_PROMPT_FOOTER
        return prompt

    def _render_page(self, page: fitz.Page) -> bytes:
        """Render a PDF page to a PNG byte string."""
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if self._config.image_size:
            img.thumbnail(
                (self._config.image_size, self._config.image_size),
                Image.Resampling.LANCZOS,
            )

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _extract_page_images(
        self, page: fitz.Page, page_index: int
    ) -> List[Dict]:
        """Extract embedded images from a page and save them to disk."""
        results = []
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                img_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                filename = f"page_{page_index + 1}_img_{img_index + 1}_{img_hash}.{ext}"
                dest = self._images_dir / filename
                dest.write_bytes(image_bytes)

                rects = page.get_image_rects(xref)
                results.append(
                    {
                        "relative_path": f"images/{filename}",
                        "bbox": rects[0] if rects else None,
                    }
                )
            except Exception as exc:
                print(
                    f"Warning: could not extract image {img_index} "
                    f"from page {page_index + 1}: {exc}"
                )
        return results

    def _process_page(
        self, page_index: int, page: fitz.Page
    ) -> Tuple[int, str]:
        """Process one page: extract images (optional), OCR via LLM."""
        embedded_images: List[Dict] = []
        if self._config.extract_images and self._images_dir:
            embedded_images = self._extract_page_images(page, page_index)

        image_data = self._render_page(page)
        prompt = self._build_prompt(has_embedded_images=bool(embedded_images))
        markdown = self._provider.extract_text(image_data, prompt)

        # Substitute image placeholders with real relative paths.
        for img_info in embedded_images:
            if "{{IMAGE_PLACEHOLDER}}" in markdown:
                markdown = markdown.replace(
                    "{{IMAGE_PLACEHOLDER}}", img_info["relative_path"], 1
                )
            else:
                rel = img_info["relative_path"]
                markdown += f"\n\n![Extracted image]({rel})\n"

        return page_index, markdown

    def convert_file(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Convert a PDF file to a Markdown file.

        Args:
            pdf_path: Path to the source PDF.
            output_path: Destination Markdown path. Auto-generated when None.

        Returns:
            Absolute path to the generated Markdown file.
        """
        src = Path(pdf_path)
        if not src.exists():
            raise FileNotFoundError(f"PDF file not found: {src}")

        if output_path is None:
            out_dir = Path(self._config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"{src.stem}.md"
        else:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)

        if self._config.extract_images:
            self._images_dir = dest.parent / "images"
            self._images_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(src)
        total = len(doc)
        print(f"Converting {src.name} ({total} pages) ...")

        page_contents: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self._config.max_workers) as pool:
            futures = {
                pool.submit(self._process_page, i, doc[i]): i
                for i in range(total)
            }
            with tqdm(total=total, desc="Processing pages") as bar:
                for future in as_completed(futures):
                    idx, content = future.result()
                    page_contents[idx] = content
                    bar.update(1)

        doc.close()

        sections = []
        for i in range(total):
            if i > 0:
                sections.append("\n---\n")
            sections.append(page_contents[i])

        dest.write_text("\n".join(sections), encoding="utf-8")
        print(f"Saved: {dest}")
        return str(dest)

    def convert_bytes(self, pdf_bytes: bytes) -> str:
        """
        Convert PDF bytes to a Markdown string (no file I/O).

        Args:
            pdf_bytes: Raw PDF content.

        Returns:
            Markdown string with pages separated by '---'.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total = len(doc)

        page_contents: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self._config.max_workers) as pool:
            futures = {
                pool.submit(self._process_page, i, doc[i]): i
                for i in range(total)
            }
            for future in as_completed(futures):
                idx, content = future.result()
                page_contents[idx] = content

        doc.close()

        sections = []
        for i in range(total):
            if i > 0:
                sections.append("\n---\n")
            sections.append(page_contents[i])

        return "\n".join(sections)
