from typing import Literal
from PIL import Image

def make_grid(images: list[Image.Image], mode: Literal['h', 'v']) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    heights = [img.height for img in images]
    widths = [img.width for img in images]

    if mode == 'h':
        assert min(heights) == max(heights)
        h = heights[0]
        
        grid = Image.new("RGB", size=(sum(widths), h))
        sum_widths = 0
        for img in images:
            grid.paste(img, box=(sum_widths, 0))
            sum_widths += img.width
    if mode == 'v':
        assert min(widths) == max(widths), (min(widths), max(widths))
        w = widths[0]
        
        grid = Image.new("RGB", size=(w, sum(heights)))
        sum_heights = 0
        for img in images:
            grid.paste(img, box=(0, sum_heights))
            sum_heights += img.height

    return grid
