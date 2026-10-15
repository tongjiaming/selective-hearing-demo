import tkinter as tk
import pytest
from threading import Event
from selective_hearing.file_ui import FileApp

from selective_hearing.ui import App


@pytest.fixture(autouse=True)
def isolated_profiles(monkeypatch, tmp_path):
    monkeypatch.setattr('selective_hearing.ui.default_profile_path',
                        lambda file_mode=False: tmp_path / ('file.sqlite3' if file_mode else 'live.sqlite3'))


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
    def __init__(self, *args, **kwargs):
        self.cancel = Event()
        self.worker = None
        self.started = False
        self.registry = kwargs.get('registry')
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


def test_listen_button_restores_audio_and_label_after_last_exclusion(monkeypatch, root):
    import numpy as np
    from selective_hearing.audio import LiveSession
    from selective_hearing.core import AudioPacket
    from types import SimpleNamespace
    monkeypatch.setattr('selective_hearing.ui.devices', lambda **kwargs: [])
    app = App(root, auto_start=False)
    try:
        app.update_people([{'id': 'speaker_1', 'ready': True}])
        session = LiveSession(0, 1, app.control, lambda _: None)
        session.playback.put(AudioPacket(np.full(960, .2, np.float32)))
        output = np.empty((320, 1), np.float32)
        status = SimpleNamespace(output_underflow=False)
        session._output(output, 320, None, status)
        np.testing.assert_allclose(output, .08)
        app.speaker_mutes['speaker_1'].invoke()
        session._output(output, 320, None, status)
        np.testing.assert_array_equal(output, 0)
        app.speaker_buttons['speaker_1'].invoke()
        # A subsequent model update must not put playback back in filtering mode.
        app.update_people([{'id': 'speaker_1', 'ready': True}], [])
        session._output(output, 320, None, status)
        np.testing.assert_allclose(output, .08)
        assert app.mode_label['text'] == '当前：全部原声'
        assert app.speaker_buttons['speaker_1']['style'] == 'Selected.TButton'
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


def test_names_save_on_close_restore_cards_and_delete(monkeypatch, tk_session):
    import numpy as np
    from selective_hearing.core import SpeakerRegistry
    monkeypatch.setattr('selective_hearing.ui.devices', lambda **kw: [])
    window = tk.Toplevel(tk_session); window.withdraw()
    app = App(window, auto_start=False)
    registry = SpeakerRegistry(store=app.profile_store)
    vector = np.eye(192, dtype=np.float32)[0]
    registry.assign([vector], [1.2])
    app.update_people(registry.snapshot())
    app.speaker_names['speaker_1'].set('小王')  # close without Return/FocusOut
    app.close()
    window = tk.Toplevel(tk_session); window.withdraw()
    restored = App(window, auto_start=False)
    try:
        assert restored.speaker_names['speaker_1'].get() == '小王'
        assert '已保存' in restored.speaker_labels['speaker_1']['text']
        restored.reset_people()
        assert restored.speaker_names['speaker_1'].get() == '小王'
        monkeypatch.setattr('selective_hearing.ui.messagebox.askyesno', lambda *args, **kwargs: True)
        restored.delete_person('speaker_1')
        assert not restored.speaker_cards and not restored.profile_store.summaries()
    finally:
        restored.close()


def test_auto_start_uses_saved_profiles_and_keeps_choice_during_loading(monkeypatch, root):
    import numpy as np
    from selective_hearing.ui import default_profile_path
    from selective_hearing.profiles import ProfileStore
    store = ProfileStore(default_profile_path())
    store.create(np.eye(192, dtype=np.float32)[0], 2, 2.4)
    startup_fixture(monkeypatch)
    app = App(root)
    try:
        root.update()
        app.set_person('speaker_1', False)
        app.events.put({'type': 'loaded', 'backend': object()})
        drain(app)
        assert app.session.registry.speakers[0].id == 'speaker_1'
        assert not app.control.track_enabled('speaker_1')
    finally:
        app.close()


def seed_person(app):
    import numpy as np
    speaker = app.profile_store.create(np.eye(192, dtype=np.float32)[0], 2, 2.4)
    app.restore_people()
    return speaker.id


def test_delete_while_capturing_prompts_and_cancel_keeps_capture(monkeypatch, root):
    startup_fixture(monkeypatch)
    app = App(root, auto_start=False)
    try:
        identity = seed_person(app)
        session = FakeLiveSession()
        app.session = session
        prompts = []
        def decline(*args, **kwargs):
            prompts.append((args, kwargs))
            return False
        monkeypatch.setattr('selective_hearing.ui.messagebox.askyesno', decline)
        app.delete_person(identity)
        assert prompts and prompts[0][1]['parent'] is root
        assert app.session is session and not session.cancel.is_set()
        assert app.pending_delete is None and app.profile_store.summaries()
    finally:
        app.close()


def test_delete_waits_for_inflight_worker_and_discards_stale_cards(monkeypatch, root):
    from types import SimpleNamespace
    from selective_hearing.profiles import ProfileStore
    startup_fixture(monkeypatch)
    app = App(root, auto_start=False)
    try:
        identity = seed_person(app)
        old_generation = app.generation
        stale_people = app.profile_store.summaries()
        session = FakeLiveSession()
        running = [True]
        session.worker = SimpleNamespace(is_alive=lambda: running[0])
        app.session = session
        monkeypatch.setattr('selective_hearing.ui.messagebox.askyesno', lambda *a, **kw: True)
        app.delete_person(identity)
        assert session.cancel.is_set() and app.pending_delete
        assert app.profile_store.summaries()  # no deletion during a possible write
        app.start_live()
        assert app.session is session and app.pending_delete
        running[0] = False
        app.events.put({'type': 'progress', 'speakers': stale_people, 'generation': old_generation})
        drain(app)
        assert app.session is None and app.pending_delete is None
        assert not app.speaker_cards and '已删除' in app.profile_note['text']
        assert not ProfileStore(app.profile_store.path).summaries()
        app.events.put({'type': 'progress', 'speakers': stale_people, 'generation': old_generation})
        drain(app)
        assert not app.speaker_cards  # queued old results cannot restore the card
    finally:
        app.pending_delete = None
        app.close()


def test_delete_cancels_file_processing_and_invalidates_playback(monkeypatch, root):
    startup_fixture(monkeypatch)
    app = FileApp(root)
    try:
        identity = seed_person(app)
        old_generation = app.generation
        app.file_busy = True
        app.file_packet = app.file_report = object()
        player = FakeLiveSession()
        app.player = player
        monkeypatch.setattr('selective_hearing.ui.messagebox.askyesno', lambda *a, **kw: True)
        app.delete_person(identity)
        assert app.pending_delete and app.file_cancel.is_set() and player.cancel.is_set()
        assert app.profile_store.summaries()
        # Completion of the cancelled task releases the deletion, but cannot
        # reinstall its old packet or show an error for intentional cancellation.
        app.events.put({'type': 'file_error', 'message': '已取消', 'generation': old_generation})
        drain(app)
        assert not app.file_busy and app.pending_delete is None
        assert not app.profile_store.summaries() and not app.speaker_cards
        assert app.file_packet is None and app.file_report is None
        assert str(app.play_button['state']) == str(app.export_button['state']) == 'disabled'
    finally:
        app.pending_delete = None
        app.close()


def test_delete_failure_preserves_card_and_allows_retry(monkeypatch, root):
    startup_fixture(monkeypatch)
    app = App(root, auto_start=False)
    try:
        identity = seed_person(app)
        monkeypatch.setattr('selective_hearing.ui.messagebox.askyesno', lambda *a, **kw: True)
        original_delete = app.profile_store.delete
        def fail(_):
            raise OSError('locked')
        monkeypatch.setattr(app.profile_store, 'delete', fail)
        app.delete_person(identity)
        assert app.pending_delete is None and identity in app.speaker_cards
        assert app.profile_store.summaries() and 'locked' in app.profile_note['text']
        monkeypatch.setattr(app.profile_store, 'delete', original_delete)
        app.delete_person(identity)
        assert not app.profile_store.summaries() and not app.speaker_cards
    finally:
        app.close()


def test_invalid_profile_blocks_automatic_capture_without_erasing_file(monkeypatch, root):
    from selective_hearing.ui import default_profile_path
    path = default_profile_path()
    path.write_bytes(b'broken profile')
    calls = startup_fixture(monkeypatch)
    app = App(root)
    try:
        root.update()
        assert not calls and app.session is None and app.profile_error
        assert path.read_bytes() == b'broken profile'
        assert '原文件未覆盖' in app.profile_note['text']
    finally:
        app.close()
