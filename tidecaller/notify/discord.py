"""Discord webhook notifications: test ping, catches, stuck alerts, periodic summaries."""
from __future__ import annotations

import io
import threading
import time
from typing import Callable

import requests

from tidecaller.config.schema import DiscordSettings

COLORS = {"catch": 0x2ECC71, "lost": 0xE67E22, "stuck": 0xE74C3C, "summary": 0x3498DB, "test": 0x9B59B6}


class DiscordNotifier:
    def __init__(self, cfg: DiscordSettings, screenshot: Callable[[], bytes | None] | None = None) -> None:
        self.cfg = cfg
        self.screenshot = screenshot
        self._last_summary = time.perf_counter()

    def _post(self, embed: dict, content: str = "", with_shot: bool = False) -> requests.Response | None:
        if not self.cfg.webhook_url:
            return None
        payload = {"username": "Tidecaller", "content": content, "embeds": [embed]}
        files = None
        if with_shot and self.cfg.attach_screenshot and self.screenshot:
            png = self.screenshot()
            if png:
                embed["image"] = {"url": "attachment://screen.png"}
                files = {"file": ("screen.png", io.BytesIO(png), "image/png")}
        if files:
            import json

            return requests.post(self.cfg.webhook_url, data={"payload_json": json.dumps(payload)}, files=files,
                                 timeout=10)
        return requests.post(self.cfg.webhook_url, json=payload, timeout=10)

    def _ping(self) -> str:
        return f"<@{self.cfg.ping_user_id}>" if self.cfg.ping_user_id else ""

    def test(self) -> bool:
        r = self._post({"title": "Webhook connected", "description": "Tidecaller will report here.",
                        "color": COLORS["test"]}, content=self._ping())
        return r is not None and r.ok

    def send(self, kind: str, stats=None, **kw) -> None:
        """Fire-and-forget; never blocks the macro loop."""
        if not self.cfg.enabled:
            return
        threading.Thread(target=self._send, args=(kind, stats), kwargs=kw, daemon=True).start()

    def _send(self, kind: str, stats, **kw) -> None:
        fields = []
        if stats is not None:
            now = time.perf_counter()
            fields = [
                {"name": "Catches", "value": str(stats.catches), "inline": True},
                {"name": "Lost", "value": str(stats.lost), "inline": True},
                {"name": "Per hour", "value": f"{stats.per_hour(now):.0f}", "inline": True},
            ]
        if kind == "catch":
            ok = kw.get("ok", True)
            self._post({"title": "Fish caught" if ok else "Fish escaped", "color": COLORS["catch" if ok else "lost"],
                        "fields": fields})
            if time.perf_counter() - self._last_summary >= self.cfg.summary_every_minutes * 60:
                self._last_summary = time.perf_counter()
                self._post({"title": "Session summary", "color": COLORS["summary"], "fields": fields},
                           with_shot=True)
        elif kind == "stuck":
            self._post({"title": "Macro got stuck - recovering", "description": f"Phase: `{kw.get('phase')}`",
                        "color": COLORS["stuck"], "fields": fields}, content=self._ping(), with_shot=True)
