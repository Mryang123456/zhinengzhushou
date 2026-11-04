"""确定性文本切分，保留重叠上下文。"""

def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap。")
    return [text[start:start + chunk_size] for start in range(0, len(text), chunk_size - overlap)]
