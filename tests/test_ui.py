import tkinter as tk
import pytest

from selective_hearing.ui import App


def device(index, name, direction, default=False):
    return dict(id=index, name=name, api='Windows WASAPI', input=direction == 'input',
                output=direction == 'output', default_input=default and direction == 'input',
                default_output=default and direction == 'output')


def test_system_defaults_manual_choice_and_unplug_refresh(monkeypatch):
    listed = [device(0, 'AirPods Mic', 'input'), device(1, 'Laptop Mic', 'input', True),
              device(2, 'Speakers', 'output'), device(3, 'Headphones', 'output', True)]
    monkeypatch.setattr('selective_hearing.ui.devices', lambda **kw: listed)
    root = tk.Tk()
    root.withdraw()
    app = App(root)
    try:
        assert app.selected_device(app.input_box, app.inputs) == 1  # OS default beats AirPods
        assert app.selected_device(app.output_box, app.outputs) == 3
        assert '系统默认' in app.input_box.get()
        assert app.session is None and app.player is None
        app.input_box.current(0)
        app.remember_device('input')
        listed[:] = [device(5, 'Laptop Mic', 'input', True), device(6, 'AirPods Mic', 'input'),
                     device(7, 'Speakers', 'output', True), device(8, 'Headphones', 'output')]
        app.refresh_devices()
        assert app.selected_device(app.input_box, app.inputs) == 6  # IDs can change
        assert app.selected_device(app.output_box, app.outputs) == 7  # follows changed OS default
        listed[:] = [device(5, 'Laptop Mic', 'input', True)]
        app.refresh_devices()
        assert app.selected_device(app.input_box, app.inputs) == 5
        assert app.output_box.get() == '' and app.output_box.current() == -1
        listed[:] = [device(9, 'Other Mic', 'input')]
        app.refresh_devices()
        assert app.input_box.current() == -1  # never choose arbitrary device if default unavailable
        listed.clear()
        app.refresh_devices()
        assert not app.inputs and not app.outputs
        assert app.input_box.get() == app.output_box.get() == ''
    finally:
        app.close()


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
        assert str(app.speaker_buttons["speaker_1"]["state"]) == "normal"
        assert '确认中' in app.speaker_labels['speaker_1']['text']
        app.speaker_buttons['speaker_1'].invoke()
        assert app.control.get()[0] == 'speaker_1'
        app.speaker_mutes['speaker_1'].invoke()
        assert app.control.get()[0] == 'exclude:speaker_1'
        app.update_people([{"id": "speaker_1", "ready": True}])
        assert len(app.speaker_cards) == 1
        assert '确认中' not in app.speaker_labels['speaker_1']['text']
        app.speaker_buttons["speaker_1"].invoke()
        assert app.control.get()[0] == "speaker_1"
        app.reset_people()
        assert not app.speaker_buttons
        assert not app.speaker_cards and not app.speaker_mutes
        assert app.control.get()[0] == "all"
    finally:
        app.close()
