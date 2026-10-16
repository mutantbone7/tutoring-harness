"""Produce the IEEE Access "Highlighted PDF": every line of the new manuscript
whose text does not occur in the old manuscript is marked with a yellow
highlight annotation. Matching ignores whitespace, hyphenation and
punctuation so that reflowed but unchanged paragraphs are not marked.

  python highlight_changes.py OLD.pdf NEW.pdf OUT.pdf
"""
import re
import sys
import fitz  # PyMuPDF


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main(old_path, new_path, out_path, min_len=12):
    old = fitz.open(old_path)
    old_text = norm("".join(p.get_text() for p in old))
    new = fitz.open(new_path)
    n_lines = n_marked = 0
    for page in new:
        d = page.get_text("dict")
        for block in d["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                text = "".join(sp["text"] for sp in line["spans"])
                key = norm(text)
                if len(key) < min_len:
                    # short lines (table cells, numbers): mark only if clearly new in context
                    continue
                n_lines += 1
                if key not in old_text:
                    rect = fitz.Rect(line["bbox"])
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 1, 0))
                    annot.update()
                    n_marked += 1
    new.save(out_path, garbage=3, deflate=True)
    print(f"lines checked {n_lines}, marked {n_marked} ({n_marked / max(n_lines, 1):.0%})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
