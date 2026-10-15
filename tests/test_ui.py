import tkinter as tk
import pytest
from threading import Event
from selective_hearing.file_ui import FileApp

from selective_hearing.ui import App


@pytest.fixture(scope='module')
def tk_session():
    # One Tcl interpreter, as in the real application; isolated windows per test.
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def root(tk_session):
    window = tk.Toplevel(tk_session)
    window.withdraw()
    yield window
    if window.winfo_exists():
        window.destroy()


class FakeLiveSession:
    def __init__(self, *args):
        self.cancel = Event()
        self.worker = None
        self.started = False
    def start(self):
        self.started = True
    def stop(self):
        self.cancel.set()
    def stats(self):
        return dict(level=0, input_drops=0, output_drops=0, queue_seconds=0,
                    software_age=0, empty_callbacks=0)


def drain(app):
    app.root.after_cancel(app.poll_after)
    app.poll()


def startup_fixture(monkeypatch, available=True):
    listed = [device(1, 'Mic', 'input', True), device(2, 'Headphones', 'output', True)] if available else []
    monkeypatch.setattr('selective_hearing.ui.devices', lambda **kw: listed)
    monkeypatch.setattr('selective_hearing.ui.LiveSession', FakeLiveSession)
    calls = []
    def load(app):
        calls.append('load')
        app.backend_busy = True
    monkeypatch.setattr(App, 'load_models', load)
    return calls


def test_default_launch_loads_then_starts_and_has_no_file_controls(monkeypatch, root):
    calls = startup_fixture(monkeypatch)
    app = App(root)
    try:
        root.update()
        assert calls == ['load'] and app.session is None
        assert not hasattr(app, 'file_button')
        app.events.put({'type': 'loaded', 'backend': object()})
        drain(app)
        assert app.session.started and not app.auto_start_pending
        app.stop()
        assert app.session is None
    finally: app.close()


@pytest.mark.parametrize('stage', ['before_load', 'during_load', 'failure', 'no_devices'])
def test_stop_or_startup_failure_never_starts_capture(monkeypatch, root, stage):
    calls = startup_fixture(monkeypatch, available=stage != 'no_devices')
    app = App(root)
    try:
        if stage == 'before_load':
            app.stop()
        root.update()
        if stage == 'before_load':
            assert not calls
        elif stage == 'during_load':
            app.stop()
            app.events.put({'type': 'loaded', 'backend': object()})
        elif stage == 'failure':
            app.events.put({'type': 'load_error', 'message': 'missing model'})
        else:
            app.events.put({'type': 'loaded', 'backend': object()})
        drain(app)
        assert app.session is None and not app.auto_start_pending
        if stage == 'no_devices':
            assert '未能开始收音' in app.status['text']
    finally: app.close()


def test_file_app_loads_without_capture_and_exports_custom_names(monkeypatch, root):
    calls = startup_fixture(monkeypatch)
    app = FileApp(root)
    try:
        root.update()
        assert calls == ['load']
        app.events.put({'type': 'loaded', 'backend': object()})
        drain(app)
        app.start_live()
        assert app.session is None and not app.auto_start_pending
        assert not app.input_box.winfo_manager() and not app.start_button.winfo_manager()
        assert app.file_button.winfo_manager()
        app.update_people([{'id': 'speaker_1', 'ready': False}])
        app.speaker_names['speaker_1'].set('  小王  ')
        app.commit_name('speaker_1')
        app.speaker_mutes['speaker_1'].invoke()
        app.update_people([{'id': 'speaker_1', 'ready': True}], ['speaker_1'])
        assert app.speaker_names['speaker_1'].get() == '小王'
        assert not app.control.track_enabled('speaker_1')
        app.file_report = {'speakers': [{'id': 'speaker_1', 'ready': True}]}
        assert app.named_report()['speakers'][0]['name'] == '小王'
        assert 'name' not in app.file_report['speakers'][0]
        app.speaker_names['speaker_1'].set(' ')
        app.commit_name('speaker_1')
        assert app.speaker_names['speaker_1'].get() == '说话人 1'
        app.reset_people()
        assert not app.speaker_names
    finally: app.close()


def device(index, name, direction, default=False):
    return dict(id=index, name=name, api='Windows WASAPI', input=direction == 'input',
                output=direction == 'output', default_input=default and direction == 'input',
                default_output=default and direction == 'output')


def test_system_defaults_manual_choice_and_unplug_refresh(monkeypatch, root):
    listed = [device(0, 'AirPods Mic', 'input'), device(1, 'Laptop Mic', 'input', True),
              device(2, 'Speakers', 'output'), device(3, 'Headphones', 'output', True)]
    monkeypatch.setattr('selective_hearing.ui.devices', lambda **kw: listed)
    app = App(root, auto_start=False)
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


def test_ui_opens_without_audio_and_preserves_mute(monkeypatch, root):
    monkeypatch.setattr("selective_hearing.ui.devices", lambda **kwargs: [])
    app = App(root, auto_start=False)
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
        assert app.control.track_enabled('speaker_1')
        app.speaker_mutes['speaker_1'].invoke()
        assert not app.control.track_enabled('speaker_1')
        app.update_people([{"id": "speaker_1", "ready": True}])
        assert len(app.speaker_cards) == 1
        assert '确认中' not in app.speaker_labels['speaker_1']['text']
        app.speaker_buttons["speaker_1"].invoke()
        assert app.control.track_enabled('speaker_1')
        app.reset_people()
        assert not app.speaker_buttons
        assert not app.speaker_cards and not app.speaker_mutes
        assert app.control.get()[0] == "all"
    finally:
        app.close()


def test_multiple_people_have_independent_controls_and_scrollable_cards(monkeypatch, root):
    monkeypatch.setattr('selective_hearing.ui.devices', lambda **kwargs: [])
    app = App(root, auto_start=False)
    try:
        app.update_people([{'id': f'speaker_{i}', 'ready': False} for i in range(1, 9)])
        app.speaker_mutes['speaker_2'].invoke()
        app.speaker_mutes['speaker_3'].invoke()
        app.speaker_buttons['speaker_1'].invoke()
        assert app.speaker_buttons['speaker_1']['text'] == '听他'
        assert app.control.track_enabled('speaker_1')
        assert not app.control.track_enabled('speaker_2')
        assert not app.control.track_enabled('speaker_3')
        assert app.control.track_enabled('speaker_4')
        assert len(app.speaker_cards) == 8
        root.update_idletasks()
        assert app.track_grid.winfo_reqheight() > int(app.track_canvas['height'])
        app.choose('mute')
        app.speaker_buttons['speaker_2'].invoke()
        app.update_people([{'id': 'speaker_9', 'ready': False}])
        assert app.control.track_enabled('speaker_2')
        assert not app.control.track_enabled('speaker_9')
        assert app.speaker_buttons['speaker_2']['style'] == 'Selected.TButton'
        app.reset_people()
        assert not app.track_area.winfo_manager()
    finally:
        app.close()
