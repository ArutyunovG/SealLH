from seallh.helpers.transform.pad_to_aspect_ratio import _PadToAspectRatio, AlbumentationsTransform


class _PadToSquare(_PadToAspectRatio):
    """Pad the shorter side so the image becomes square (aspect_ratio=1)."""

    def __init__(self, position: str = "center", border_mode: int = 0,
                 fill: float = 0, p: float = 1.0):
        super().__init__(aspect_ratio=1.0, position=position,
                         border_mode=border_mode, fill=fill, p=p)


PadToSquare = lambda **cfg: AlbumentationsTransform(_PadToSquare, **cfg)
