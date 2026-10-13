import tkinter as tk
import pytest

from selective_hearing.ui import App


def test_ui_opens_without_audio_and_preserves_mute(monkeypatch):
    monkeypatch.setattr("selective_hearing.ui.devices", lambda **kwargs: [])
    root = tk.Tk()
    root.withdraw()
    app = App(root)
    try:
        root.update()
        assert app.session is None
        assert app.player is None
        app.choose("mute")
        app.update_people([{"id": "speaker_1", "ready": False}], ["speaker_1"])
        assert app.control.get()[0] == "mute"
        assert str(app.speaker_buttons["speaker_1"]["state"]) == "disabled"
        app.update_people([{"id": "speaker_1", "ready": True}])
        app.speaker_buttons["speaker_1"].invoke()
        assert app.control.get()[0] == "speaker_1"
        app.reset_people()
        assert not app.speaker_buttons
        assert app.control.get()[0] == "all"
    finally:
        app.close()
