from typing import List, Tuple

from PIL import Image, ImageDraw
from doctr.models import ocr_predictor

ocr = ocr_predictor(
    pretrained=True,
    det_arch="db_mobilenet_v3_large",
    reco_arch="crnn_mobilenet_v3_large",
)


def extract_text_and_boxes(image) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    result = ocr([image])
    text_lines: List[str] = []
    boxes: List[Tuple[int, int, int, int]] = []
    for page in result.pages:
        page_height, page_width = page.dimensions
        for block in page.blocks:
            for line in block.lines:
                line_words: List[str] = []
                for word in line.words:
                    line_words.append(word.value)
                    (x0, y0), (x1, y1) = word.geometry
                    left = int(x0 * page_width)
                    top = int(y0 * page_height)
                    right = int(x1 * page_width)
                    bottom = int(y1 * page_height)
                    boxes.append((left, top, right, bottom))
                if line_words:
                    text_lines.append(" ".join(line_words))
            if block.lines:
                text_lines.append("")
    return "\n".join(text_lines).strip(), boxes


def highlight_text_boxes(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 2,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for left, top, right, bottom in boxes:
        draw.rectangle([left, top, right, bottom], outline=color, width=width)
    return annotated


def extract_text_from_image(image: Image.Image) -> str:
    text, _ = extract_text_and_boxes(image)
    return text
