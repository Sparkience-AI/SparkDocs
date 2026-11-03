"""Structure-aware, bounded chunk construction."""

from sparkdocs.models import Chunk, IngestedDocument


def _split_text(text: str, target_characters: int, overlap_characters: int) -> list[str]:
    if len(text) <= target_characters:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target_characters, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + target_characters // 2:
                end = boundary
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        start = max(end - overlap_characters, start + 1)
    return parts


def build_chunks(
    document: IngestedDocument,
    *,
    target_characters: int = 1_200,
    overlap_characters: int = 180,
) -> tuple[Chunk, ...]:
    """Split within source sections so every chunk retains a precise locator."""

    if target_characters < 200:
        raise ValueError("Chunk target must be at least 200 characters.")
    if overlap_characters < 0 or overlap_characters >= target_characters // 2:
        raise ValueError("Chunk overlap must be non-negative and less than half the target.")
    chunks: list[Chunk] = []
    for section_index, section in enumerate(document.sections):
        parts = _split_text(section.text, target_characters, overlap_characters)
        for part_index, text in enumerate(parts):
            locator = section.locator
            if len(parts) > 1:
                locator = f"{locator} · chunk {part_index + 1}/{len(parts)}"
            chunks.append(
                Chunk(
                    chunk_id=f"s{section_index + 1}-c{part_index + 1}",
                    locator=locator,
                    text=text,
                )
            )
    return tuple(chunks)
