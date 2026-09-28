"""Builds Qt forms straight from pydantic models, so every config field gets a widget for free."""
from __future__ import annotations

from enum import Enum
from typing import Callable, get_origin

from pydantic import BaseModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _bounds(model: type[BaseModel], name: str) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in model.model_fields[name].metadata:
        lo = getattr(m, "ge", lo)
        hi = getattr(m, "le", hi)
    return lo, hi


def build_form(obj: BaseModel, on_change: Callable[[], None], skip: tuple[str, ...] = ()) -> QWidget:
    """Nested models and tuples are skipped (edited elsewhere); everything else is live-bound."""
    w = QWidget()
    form = QFormLayout(w)
    cls = type(obj)
    for name, info in cls.model_fields.items():
        if name in skip:
            continue
        ann = info.annotation
        val = getattr(obj, name)

        def setter(v, _n=name):
            setattr(obj, _n, v)
            on_change()

        if isinstance(val, BaseModel) or get_origin(ann) in (tuple, dict) or isinstance(val, (tuple, dict)):
            continue
        lo, hi = _bounds(cls, name)
        if isinstance(val, bool):
            wid = QCheckBox()
            wid.setChecked(val)
            wid.toggled.connect(setter)
        elif isinstance(val, Enum):
            wid = QComboBox()
            members = list(type(val))
            wid.addItems([m.value for m in members])
            wid.setCurrentText(val.value)
            wid.currentTextChanged.connect(setter)
        elif isinstance(val, int):
            wid = QSpinBox()
            wid.setRange(int(lo if lo is not None else -1_000_000), int(hi if hi is not None else 1_000_000))
            wid.setValue(val)
            wid.valueChanged.connect(setter)
        elif isinstance(val, float):
            wid = QDoubleSpinBox()
            wid.setDecimals(4 if abs(val) < 0.1 and val != 0 else 2)
            wid.setSingleStep(0.001 if abs(val) < 0.1 and val != 0 else 0.05)
            wid.setRange(lo if lo is not None else -1e6, hi if hi is not None else 1e6)
            wid.setValue(val)
            wid.valueChanged.connect(setter)
        else:
            wid = QLineEdit(str(val))
            if "webhook" in name:
                wid.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
            wid.textChanged.connect(setter)
        wid.setObjectName(name)
        form.addRow(_label(name), wid)
    return w

