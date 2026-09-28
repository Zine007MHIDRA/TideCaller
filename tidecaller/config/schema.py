"""Typed configuration. Every GUI tab edits one section of `Config`."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel as _PydanticModel, ConfigDict, Field

from tidecaller.core.regions import Region


class BaseModel(_PydanticModel):
    # GUI widgets assign raw values (e.g. "color"); validate so they become enums
    model_config = ConfigDict(validate_assignment=True)


class CaptureBackend(str, Enum):
    DXCAM = "dxcam"  # desktop duplication, fast; for strong PCs
    MSS = "mss"      # GDI BitBlt, plays nice with OBS/Discord screen share


class CastStyle(str, Enum):
    NORMAL = "normal"
    PERFECT = "perfect"


class ReleaseStyle(str, Enum):
    IDIOT_PROOF = "idiot_proof"  # release when the meter turns green, timeout fallback
    THRESHOLD = "threshold"      # release only on green (long safety cap)
    TIMED = "timed"              # fixed hold duration
    PREDICTIVE = "predictive"    # learned time-to-full minus input latency, green as fallback


class ShakeStyle(str, Enum):
    NONE = "none"  # rod/bait skips the shake: just wait for the reel
    NAVIGATION = "navigation"
    CIRCLE = "circle"
    PIXEL = "pixel"


class TrackStyle(str, Enum):
    LINE = "line"
    COLOR = "color"
    YOLO = "yolo"


class TotemTrigger(str, Enum):
    CYCLES = "cycles"
    TIMER = "timer"


class Hotkeys(BaseModel):
    area_editor: str = "f1"
    start: str = "f2"
    stop: str = "f3"


class Regions(BaseModel):
    """Stored as fractions of the Roblox client rect, so they survive resolution changes."""
    # defaults measured on a real 1920x1040 Fisch window
    fish_bar: Region = Region(0.29, 0.839, 0.42, 0.025)   # inner rows of the reel box, between its borders
    reel_progress: Region = Region(0.30, 0.902, 0.40, 0.009)
    shake: Region = Region(0.2, 0.15, 0.6, 0.6)
    # the meter sits right of the character; searched for, so this box can be generous
    cast_meter: Region = Region(0.45, 0.25, 0.25, 0.5)
    hotbar: Region = Region(0.3, 0.92, 0.4, 0.07)
    # "Current Bait ... Power" block that only shows while the rod is held
    rod_info: Region = Region(0.35, 0.855, 0.30, 0.065)


class MainSettings(BaseModel):
    capture: CaptureBackend = CaptureBackend.DXCAM
    hotkeys: Hotkeys = Hotkeys()
    auto_select_rod: bool = True
    rod_slot: int = Field(1, ge=1, le=9)
    pause_when_unfocused: bool = True
    always_on_top: bool = True
    show_debug_overlay: bool = False


class CastSettings(BaseModel):
    style: CastStyle = CastStyle.NORMAL
    release: ReleaseStyle = ReleaseStyle.IDIOT_PROOF
    fill_threshold: float = Field(0.95, ge=0.1, le=1.0)
    timed_hold_ms: int = Field(1000, ge=50, le=5000)
    timeout_ms: int = Field(2500, ge=200, le=10000)
    input_latency_ms: float = Field(35.0, ge=0, le=300)  # seed for the predictive estimator
    post_cast_delay_ms: int = 400


class ShakeSettings(BaseModel):
    style: ShakeStyle = ShakeStyle.CIRCLE
    navigation_key: str = "\\"
    click_interval_ms: int = Field(60, ge=10, le=1000)
    timeout_s: float = Field(12.0, ge=1, le=60)
    # rods whose lure stat is at/above this skip the shake entirely (e.g. Masterline); 0 disables
    skip_shake_lure_pct: float = Field(95.0, ge=0, le=5000)
    # always watch only for the reel this long first, so instant bites never get stray clicks
    instant_grace_s: float = Field(1.0, ge=0, le=10)
    min_button_area: int = 400
    max_button_area: int = 40000
    circularity: float = Field(0.7, ge=0.3, le=1.0)


class RodProfile(BaseModel):
    """Per-rod reel tuning. Bar speed/size differs a lot between rods."""
    name: str = "Default"
    kp: float = 0.012
    kd: float = 0.0025
    deadzone_px: int = 6
    # HSV bounds for color tracking
    fish_hsv_lo: tuple[int, int, int] = (0, 0, 20)
    fish_hsv_hi: tuple[int, int, int] = (180, 80, 50)
    bar_hsv_lo: tuple[int, int, int] = (0, 0, 200)
    bar_hsv_hi: tuple[int, int, int] = (180, 40, 255)
    # grayscale thresholds for line tracking
    line_bar_min: int = 200
    line_fish_max: int = 50
    # shape rules (fractions of the fish-bar box width); "Calibrate reel" fills these in for bar skins
    bar_gap_frac: float = 0.04       # bridge holes in the bar (fish icon, skin decorations)
    bar_min_frac: float = 0.0        # a white run narrower than this is not the catch bar
    fish_max_width_frac: float = 0.12
    fish_col_share: float = 0.3      # share of a column's pixels that must match the fish color
    bar_col_share: float = 0.4       # same for the bar; calibration raises it so diagonal objects (rod swing) fail
    calibrated: bool = False


class FishSettings(BaseModel):
    rod: str = "Default"
    track: TrackStyle = TrackStyle.LINE
    scan_fps: int = Field(120, ge=15, le=240)
    bag_spam: bool = True
    bag_key: str = "`"
    reel_timeout_s: float = 45.0
    yolo_model: str = "models/fishbar.onnx"


class TotemSettings(BaseModel):
    enabled: bool = False
    slot: int = Field(3, ge=1, le=9)
    trigger: TotemTrigger = TotemTrigger.CYCLES
    every_cycles: int = 60
    every_minutes: float = 15.0
    auto_potion: bool = False
    potion_slot: int = Field(4, ge=1, le=9)
    potion_every_minutes: float = 10.0


class DiscordSettings(BaseModel):
    enabled: bool = False
    webhook_url: str = ""
    ping_user_id: str = ""
    on_catch: bool = True
    on_error: bool = True
    summary_every_minutes: float = 30.0
    attach_screenshot: bool = True


CONFIG_VERSION = 3  # v2: regions measured on the real UI; v3: reel box moved onto the bar's inner rows


class Config(BaseModel):
    version: int = CONFIG_VERSION
    main: MainSettings = MainSettings()
    cast: CastSettings = CastSettings()
    shake: ShakeSettings = ShakeSettings()
    fish: FishSettings = FishSettings()
    totem: TotemSettings = TotemSettings()
    discord: DiscordSettings = DiscordSettings()
    regions: Regions = Regions()
    rods: dict[str, RodProfile] = Field(default_factory=lambda: {"Default": RodProfile()})

    def rod(self) -> RodProfile:
        """Custom tuning wins, then the stat-derived preset for that rod, then the default profile."""
        if self.fish.rod in self.rods:
            return self.rods[self.fish.rod]
        from tidecaller.config.rod_presets import preset

        return preset(self.fish.rod) or next(iter(self.rods.values()))

    @classmethod
    def load(cls, path: Path) -> "Config":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version", 1) < 3:
                data.pop("regions", None)  # older boxes were off; use the measured defaults
            data["version"] = CONFIG_VERSION
            cfg = cls.model_validate(data)
            cfg.save(path)
            return cfg
        cfg = cls()
        cfg.save(path)
        return cfg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2), encoding="utf-8")
