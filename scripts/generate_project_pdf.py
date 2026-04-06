from __future__ import annotations

from pathlib import Path
import textwrap


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT = 48
TOP = 800
BOTTOM = 42
FONT_SIZE = 11
LEADING = 14
WRAP_WIDTH = 92


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def normalize_line(line: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2192": "->",
        "\u2026": "...",
        "\u2713": "[OK]",
    }
    for src, dst in replacements.items():
        line = line.replace(src, dst)
    return line.encode("ascii", "replace").decode("ascii")


def wrap_markdown(text: str) -> list[str]:
    lines: list[str] = []
    in_code = False

    for raw_line in text.splitlines():
        line = normalize_line(raw_line.rstrip())

        if line.strip().startswith("```"):
            in_code = not in_code
            lines.append("")
            continue

        if in_code:
            lines.append(line)
            continue

        if not line.strip():
            lines.append("")
            continue

        stripped = line.lstrip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip().upper()
            lines.append(title)
            lines.append("")
            continue

        indent = ""
        content = line
        if stripped.startswith("- "):
            indent = "  "
            content = "* " + stripped[2:]
        elif stripped.startswith(tuple(f"{i}. " for i in range(1, 10))):
            indent = "   "
            content = stripped

        wrapped = textwrap.wrap(
            content,
            width=WRAP_WIDTH,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])

    return lines


def paginate(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    y = TOP

    for line in lines:
        if y - LEADING < BOTTOM:
            pages.append(current)
            current = []
            y = TOP
        current.append(line)
        y -= LEADING

    if current:
        pages.append(current)

    return pages


def make_content_stream(lines: list[str]) -> bytes:
    parts = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LEFT} {TOP} Td"]
    first = True
    for line in lines:
        safe = escape_pdf_text(line)
        if not first:
            parts.append(f"0 -{LEADING} Td")
        parts.append(f"({safe}) Tj")
        first = False
    parts.append("ET")
    stream = "\n".join(parts).encode("latin-1", "replace")
    return stream


def build_pdf(pages: list[list[str]], output_path: Path) -> None:
    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    content_ids: list[int] = []

    placeholder_pages_id = len(objects) + 1

    for page_lines in pages:
        stream = make_content_stream(page_lines)
        content_id = add_object(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
        )
        content_ids.append(content_id)
        page_obj = (
            f"<< /Type /Page /Parent {placeholder_pages_id} 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        page_id = add_object(page_obj)
        page_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add_object(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    )

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF"
        ).encode("ascii")
    )

    output_path.write_bytes(pdf)


def main() -> None:
    src = Path("docs/project_system_guide.md")
    dst = Path("docs/project_system_guide.pdf")
    text = src.read_text(encoding="utf-8")
    lines = wrap_markdown(text)
    pages = paginate(lines)
    build_pdf(pages, dst)
    print(f"PDF created at {dst}")


if __name__ == "__main__":
    main()
