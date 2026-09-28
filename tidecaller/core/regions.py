"""Screen regions stored relative to the Roblox client area."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Rect:
    """Absolute pixel rectangle."""
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_ltrb(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass
class Region:
    """Fractional rectangle (0..1) relative to a client rect."""
    x: float
    y: float
    w: float
    h: float

    def to_rect(self, client: Rect) -> Rect:
        return Rect(
            left=client.left + round(self.x * client.width),
            top=client.top + round(self.y * client.height),
            width=max(1, round(self.w * client.width)),
            height=max(1, round(self.h * client.height)),
        )

    @classmethod
    def from_rect(cls, rect: Rect, client: Rect) -> "Region":
        return cls(
            x=(rect.left - client.left) / client.width,
            y=(rect.top - client.top) / client.height,
            w=rect.width / client.width,
            h=rect.height / client.height,
        )
