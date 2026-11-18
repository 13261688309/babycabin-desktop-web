"""Small, validated local preset store. Loading or saving never emits commands."""
import copy
import json
import os
from pathlib import Path
import tempfile


SLOTS = ('1', '2', '3')


DEFAULT_LIMITS = ((60, 120), (75, 105))


def validate_preset(value, limits=DEFAULT_LIMITS):
    if not isinstance(value, dict) or set(value) != {'name', 'pan', 'tilt', 'panSign', 'tiltSign'}:
        raise ValueError('Invalid preset fields')
    name = value['name']
    if (not isinstance(name, str) or not name.strip() or len(name) > 16
            or any(ord(char) < 32 or ord(char) == 127 for char in name)):
        raise ValueError('预置名称需为1至16个可见字符')
    for field, (low, high) in zip(('pan', 'tilt'), limits):
        if type(value[field]) is not int or not low <= value[field] <= high:
            raise ValueError('预置角度超出限位')
    if any(type(value[key]) is not int or value[key] not in (-1, 1) for key in ('panSign', 'tiltSign')):
        raise ValueError('Invalid axis direction')
    return {**value, 'name': name.strip()}


class PresetStore:
    def __init__(self, path=None, *, limits=DEFAULT_LIMITS):
        self.limits = limits
        self.path = Path(path) if path is not None else None
        self.slots = {}
        self.revision = 0
        self.error = None
        if self.path is not None and self.path.exists():
            try:
                if self.path.stat().st_size > 8192:
                    raise ValueError('Oversized preferences')
                from .telemetry import unique_object
                data = json.loads(self.path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
                if (not isinstance(data, dict) or set(data) != {'version', 'slots'}
                        or type(data['version']) is not int or data['version'] != 1
                        or not isinstance(data['slots'], dict) or set(data['slots']) - set(SLOTS)):
                    raise ValueError('Invalid preferences')
                self.slots = {key: validate_preset(value, self.limits) for key, value in data['slots'].items()}
            except (OSError, ValueError, UnicodeError, RecursionError):
                self.error = '预置文件无法读取，未覆盖原文件'

    def snapshot(self):
        return dict(slots=copy.deepcopy(self.slots), revision=self.revision,
                    persistent=self.path is not None, error=self.error)

    def get(self, slot):
        if type(slot) is not str or slot not in SLOTS or slot not in self.slots:
            raise ValueError('该预置位尚未保存')
        return copy.deepcopy(self.slots[slot])

    def change(self, slot, value=None):
        if type(slot) is not str or slot not in SLOTS:
            raise ValueError('Invalid preset slot')
        if self.error:
            raise ValueError(self.error)
        updated = copy.deepcopy(self.slots)
        if value is None:
            updated.pop(slot, None)
        else:
            updated[slot] = validate_preset(value, self.limits)
        if self.path is not None:
            temporary = None
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                                 prefix='.gimbal-preset-', delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(dict(version=1, slots=updated), stream, ensure_ascii=True, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            except OSError:
                raise ValueError('预置保存失败，未改变原设置') from None
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        self.slots = updated
        self.revision += 1
