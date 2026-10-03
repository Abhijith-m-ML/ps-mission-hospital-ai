import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging_config import logger
from app.ingestion.document_parser import CleanedDocument


@dataclass
class ChunkMetadata:
    """Metadata attached to an individual chunk."""
    url: str
    title: str
    section: str
    source: str = "website"
    chunk_index: int = 0
    total_chunks: int = 0
    char_count: int = 0
    word_count: int = 0
    content_type: str = "webpage"
    doctor_name: Optional[str] = None
    qualification: Optional[str] = None
    department: Optional[str] = None
    schedule_text: Optional[str] = None
    image_url: Optional[str] = None
    source_url: Optional[str] = None
    facility_name: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        # Omit None values to ensure Pinecone compatibility
        return {k: v for k, v in raw.items() if v is not None}



@dataclass
class Chunk:
    """A semantic chunk derived from a cleaned document ready for vector embedding."""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
        }


class DocumentChunker:
    """Structure-aware chunker that divides cleaned documents into semantic,
    heading-preserving chunks with configurable size and overlap.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

        # Safety checks for overlap bounds
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(
                "chunk_overlap (%d) >= chunk_size (%d); reducing overlap to %d",
                self.chunk_overlap,
                self.chunk_size,
                max(0, self.chunk_size // 4),
            )
            self.chunk_overlap = max(0, self.chunk_size // 4)

    def chunk_document(self, document: CleanedDocument) -> List[Chunk]:
        """Splits a CleanedDocument into structure-aware, deterministic chunks."""
        text = document.content.strip()
        if not text:
            logger.info("Document content is empty for URL: %s. Returning 0 chunks.", document.url)
            return []

        doc_id = self._generate_document_id(document.url)
        known_headings = document.metadata.get("headings", [])
        
        # 1. Parse document into logical sections based on headings
        sections = self._extract_sections(text, document.title, known_headings)
        logger.info(
            "Divided document [%s] into %d structural sections (chunk_size=%d, overlap=%d)",
            document.title,
            len(sections),
            self.chunk_size,
            self.chunk_overlap,
        )

        # 2. Chunk each section respecting boundaries
        raw_chunks: List[Tuple[str, str]] = []  # List of (section_name, chunk_content)
        for section_name, section_text, is_heading in sections:
            section_chunks = self._chunk_section(section_name, section_text, is_heading)
            for sc in section_chunks:
                raw_chunks.append((section_name, sc))

        # Handle edge case: if raw_chunks is empty for some reason, fallback to full text
        if not raw_chunks and text:
            raw_chunks.append((document.title or "General", text))

        total_count = len(raw_chunks)
        final_chunks: List[Chunk] = []

        # 3. Create deterministic Chunk objects for text sections
        doc_dept = document.metadata.get("department", "")
        for idx, (sec_name, chunk_content) in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(doc_id, idx, chunk_content)
            
            # Determine content type for standard chunk
            sec_lower = sec_name.lower()
            is_facility = any(k in sec_lower for k in ["facilities", "services", "specialities", "diagnostic"])
            chunk_content_type = "facility" if is_facility else "webpage"

            chunk_meta = ChunkMetadata(
                url=document.url,
                title=document.title or "Untitled Document",
                section=sec_name,
                source=document.metadata.get("source", "website"),
                chunk_index=idx,
                total_chunks=total_count,
                char_count=len(chunk_content),
                word_count=len(chunk_content.split()),
                content_type=chunk_content_type,
                department=doc_dept if doc_dept else None,
                facility_name=sec_name if is_facility else None,
                source_url=document.url,
            )
            final_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    content=chunk_content,
                    metadata=chunk_meta.to_dict(),
                )
            )

        # 4. Generate structured doctor chunks when doctor entities were discovered
        doctors_list = document.metadata.get("doctors", []) or []
        for d_idx, doc_info in enumerate(doctors_list):
            d_name = doc_info.get("doctor_name", "").strip()
            if not d_name:
                continue

            d_dept = doc_info.get("department", "").strip() or doc_dept or "General"
            d_qual = doc_info.get("qualification", "").strip()
            d_sched = doc_info.get("schedule_text", "").strip()
            d_img = doc_info.get("image_url", "").strip()
            d_url = doc_info.get("source_url", "").strip() or document.url

            doc_content = (
                f"Doctor Profile: {d_name}\n"
                f"Department: {d_dept}\n"
                f"Qualification: {d_qual if d_qual else 'Specialist Consultant'}\n"
                f"OP Consultation Timings: {d_sched if d_sched else 'Hospital outpatient schedule'}\n"
                f"Profile Photo: {d_img if d_img else 'N/A'}\n"
                f"Hospital: P.S. Mission Hospital"
            )

            current_idx = len(final_chunks)
            chunk_id = self._generate_chunk_id(doc_id, current_idx, doc_content)

            chunk_meta = ChunkMetadata(
                url=d_url,
                title=document.title or "P.S. Mission Hospital",
                section=f"Our Consultants - {d_name}",
                source=document.metadata.get("source", "website"),
                chunk_index=current_idx,
                total_chunks=0,  # updated below
                char_count=len(doc_content),
                word_count=len(doc_content.split()),
                content_type="doctor",
                doctor_name=d_name,
                qualification=d_qual,
                department=d_dept,
                schedule_text=d_sched,
                image_url=d_img,
                source_url=d_url,
            )

            final_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    content=doc_content,
                    metadata=chunk_meta.to_dict(),
                )
            )

        # Update total_chunks across all chunks
        total_final = len(final_chunks)
        for c in final_chunks:
            c.metadata["total_chunks"] = total_final

        logger.info(
            "Successfully created %d chunks (including %d doctor chunks) for document: %s (doc_id=%s)",
            total_final,
            len(doctors_list),
            document.url,
            doc_id,
        )
        return final_chunks


    def _extract_sections(
        self,
        text: str,
        document_title: str,
        known_headings: List[str],
    ) -> List[Tuple[str, str, bool]]:
        """Separates the text into sections grouped by their preceding heading."""
        # Standardize known headings for matching
        normalized_headings = {h.strip().lower(): h.strip() for h in known_headings if h.strip()}

        paragraphs = text.split("\n\n")
        sections: List[Tuple[str, List[str], bool]] = []
        current_section = document_title or "General"
        is_heading_section = current_section.lower() in normalized_headings
        current_paragraphs: List[str] = []

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue

            # Check if this paragraph is a known heading
            p_lower = p_clean.lower()
            if p_lower in normalized_headings:
                if current_paragraphs:
                    sections.append((current_section, current_paragraphs, is_heading_section))
                    current_paragraphs = []
                current_section = normalized_headings[p_lower]
                is_heading_section = True
            else:
                current_paragraphs.append(p_clean)

        if current_paragraphs:
            sections.append((current_section, current_paragraphs, is_heading_section))

        # Join paragraphs within each section
        return [(name, "\n\n".join(paras), is_h) for name, paras, is_h in sections if paras]

    def _chunk_section(
        self,
        section_name: str,
        section_text: str,
        is_heading_section: bool = True,
    ) -> List[str]:
        """Chunks a single section into pieces respecting sentences, chunk_overlap,
        and preserving the section heading as the header of every sub-chunk.
        """
        section_text = section_text.strip()
        if not section_text:
            return [section_name] if (section_name and is_heading_section) else []

        # If the entire section fits within chunk_size, format and return intact
        formatted_direct = self._format_chunk(section_name, section_text, is_heading_section)
        if len(formatted_direct) <= self.chunk_size:
            return [formatted_direct]

        # Break section into sentences/units
        sentences = self._split_into_sentences(section_text)
        if not sentences:
            return [formatted_direct]

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_len = len(sentence) + (1 if current_sentences else 0)

            # If a single sentence exceeds chunk_size, split it into hard window chunks
            if sentence_len > self.chunk_size and not current_sentences:
                hard_chunks = self._hard_split(sentence, self.chunk_size, self.chunk_overlap)
                for hc in hard_chunks:
                    chunks.append(self._format_chunk(section_name, hc, is_heading_section))
                i += 1
                continue

            # If adding this sentence exceeds chunk_size and we already have content
            if current_len + sentence_len > self.chunk_size and current_sentences:
                chunk_str = " ".join(current_sentences).strip()
                chunks.append(self._format_chunk(section_name, chunk_str, is_heading_section))

                # Attempt sentence-level rewind first
                overlap_len = 0
                rewind_count = 0
                while rewind_count < len(current_sentences):
                    prev_sent = current_sentences[-(rewind_count + 1)]
                    if overlap_len + len(prev_sent) > self.chunk_overlap and rewind_count > 0:
                        break
                    overlap_len += len(prev_sent)
                    rewind_count += 1

                # If we have multiple sentences and can cleanly rewind
                if 0 < rewind_count < len(current_sentences):
                    current_sentences = current_sentences[-rewind_count:]
                    current_len = sum(len(s) for s in current_sentences) + (len(current_sentences) - 1)
                else:
                    # Fallback to word-boundary overlap prefix to ensure context continuity
                    overlap_prefix = self._get_overlap_prefix(chunk_str, self.chunk_overlap)
                    if overlap_prefix and self.chunk_overlap > 0:
                        current_sentences = [overlap_prefix]
                        current_len = len(overlap_prefix)
                    else:
                        current_sentences = []
                        current_len = 0

                # Advance to next sentence
                current_sentences.append(sentence)
                current_len += sentence_len + (1 if len(current_sentences) > 1 else 0)
                i += 1
                continue

            # Otherwise, append sentence
            current_sentences.append(sentence)
            current_len += sentence_len
            i += 1

        # Flush any remaining sentences
        if current_sentences:
            chunk_str = " ".join(current_sentences).strip()
            chunks.append(self._format_chunk(section_name, chunk_str, is_heading_section))

        return chunks

    @staticmethod
    def _format_chunk(section_name: str, body: str, is_heading_section: bool = True) -> str:
        """Attaches section heading to the top of chunk body separated by newline."""
        body = body.strip()
        if not is_heading_section or not section_name:
            return body

        # Remove redundant leading duplicate of section name if present in body
        if body.startswith(f"{section_name}\n"):
            return body
        if body.startswith(section_name):
            body = body[len(section_name):].strip()

        return f"{section_name}\n{body}" if body else section_name

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Splits text into sentences, preserving periods, numbers, and list markers."""
        # First split on line breaks so lists, tables, and headers stay distinct
        lines = text.split("\n")
        units: List[str] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # If line is a list item or table row, treat it as its own unit
            if line_str.startswith("- ") or line_str.startswith("* ") or " | " in line_str:
                units.append(line_str)
                continue

            # Split regular text on sentence terminators while protecting abbreviations
            # Regex splits on periods/exclamations/questions followed by space and uppercase/number
            sentence_splits = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line_str)
            for s in sentence_splits:
                s_strip = s.strip()
                if s_strip:
                    units.append(s_strip)

        return units

    @staticmethod
    def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fallback for oversized individual sentences or unbroken strings."""
        step = max(1, chunk_size - overlap)
        chunks = []
        for start in range(0, len(text), step):
            end = min(len(text), start + chunk_size)
            sub = text[start:end].strip()
            if sub:
                chunks.append(sub)
            if end >= len(text):
                break
        return chunks

    @staticmethod
    def _get_overlap_prefix(text: str, overlap_size: int) -> str:
        """Extracts trailing text of at most overlap_size, snapped to word boundaries."""
        if not text or overlap_size <= 0:
            return ""
        if len(text) <= overlap_size:
            return text

        tail = text[-overlap_size:].strip()
        space_idx = tail.find(" ")
        if space_idx != -1 and space_idx < len(tail) - 1:
            tail = tail[space_idx + 1:].strip()
        return tail

    @staticmethod
    def _generate_document_id(url: str) -> str:
        """Generates a stable 16-character hexadecimal hash for a document URL."""
        normalized = url.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _generate_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
        """Generates a stable, deterministic chunk ID from document_id, index, and content snippet."""
        seed = f"{document_id}:{chunk_index}:{content[:64]}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
