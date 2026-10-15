import tkinter as tk
import pytest
from threading import Event
from selective_hearing.file_ui import FileApp

from selective_hearing.ui import App


@pytest.fixture(autouse=True)
def isolated_profiles(monkeypatch, tmp_path):
    monkeypatch.setattr(App, 'watch_devices', lambda self: None)
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
        self.device_ids = args[:2]
        self.registry = kwargs.get('registry')
        self.window_seconds = kwargs.get('window_seconds')
    def start(self):
        self.started = True
    def stop(self):
        self.cancel.set()
    def stats(self):
        return dict(level=0, input_drops=0, output_drops=0, queue_seconds=0,
                    software_age=0, empty_callbacks=0)
    def active_speakers(self):
        return set()


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
        assert app.session.window_seconds == 3
        assert str(app.window_box['state']) == 'disabled'
        app.stop()
        assert app.session is None
        assert str(app.window_box['state']) == 'normal'
    finally: app.close()


@pytest.mark.parametrize('file_mode', [False, True])
def test_merged_draft_card_preserves_name_mute_and_speaking_indicator(monkeypatch, root, file_mode):
    import numpy as np
    from selective_hearing.core import SpeakerRegistry
    startup_fixture(monkeypatch)
    app = FileApp(root) if file_mode else App(root, auto_start=False)
    try:
        vector = np.zeros(192, np.float32)
        vector[0] = 1
        draft = app.profile_store.create(vector, 1, .25)
        target = app.profile_store.create(vector, 3, 3)
        app.profile_store.add_samples([(vector, .25, draft.id)])
        for _ in range(3):
            app.profile_store.add_samples([(vector, 1, target.id)])
        app.update_people(app.profile_store.summaries())
        app.speaker_names[draft.id].set('小王')
        app.commit_name(draft.id)
        app.speaker_mutes[draft.id].invoke()
        registry = SpeakerRegistry(store=app.profile_store)
        registry.assign([], [])
        # Saving stale but unedited cards must not overwrite the transferred name.
        assert app.save_names()
        assert app.profile_store.summaries()[0]['name'] == '小王'
        app.apply_profile_aliases(registry.aliases, registry.snapshot())
        app.update_people(registry.snapshot())
        assert list(app.speaker_cards) == [target.id]
        assert app.speaker_names[target.id].get() == '小王'
        assert not app.control.track_enabled(target.id)
        app.update_speaking([draft.id])
        assert app.active_speakers == {target.id}
        app.speaker_buttons[target.id].invoke()
        assert app.control.track_enabled(draft.id)
    finally:
        app.close()


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
        session.playback.put(AudioPacket(np.full(960, .2, np.float32),
                                        {'speaker_1': np.full(960, .1, np.float32)}))
        output = np.empty((320, 1), np.float32)
        status = SimpleNamespace(output_underflow=False)
        session._output(output, 320, None, status)
        np.testing.assert_allclose(output, .08)
        app.speaker_mutes['speaker_1'].invoke()
        session._output(output, 320, None, status)
        np.testing.assert_allclose(output, .04)  # unclassified sound is still on
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


@pytest.mark.parametrize('file_mode', [False, True])
def test_unclassified_buttons_are_independent_of_people_and_follow_global_controls(monkeypatch, root, file_mode):
    from selective_hearing.core import UNCLASSIFIED_ID
    startup_fixture(monkeypatch)
    app = FileApp(root) if file_mode else App(root, auto_start=False)
    try:
        app.update_people([{'id': 'speaker_1', 'ready': True}, {'id': 'speaker_2', 'ready': True}])
        app.speaker_mutes['speaker_1'].invoke()
        app.speaker_mutes['speaker_2'].invoke()
        assert app.control.track_enabled(UNCLASSIFIED_ID)
        assert '听 0 人' in app.mode_label['text'] and '未分类：听' in app.mode_label['text']
        app.unclassified_mute.invoke()
        assert not app.control.track_enabled(UNCLASSIFIED_ID)
        app.unclassified_listen.invoke()
        assert not app.control.track_enabled('speaker_1') and not app.control.track_enabled('speaker_2')
        assert app.unclassified_listen['style'] == 'Selected.TButton'
        app.choose('mute')
        assert app.unclassified_mute['style'] == 'Selected.TButton'
        app.unclassified_listen.invoke()
        assert not app.control.track_enabled('speaker_3')
        assert UNCLASSIFIED_ID not in app.speaker_names
        assert not app.profile_store.summaries()
        app.choose('all')
        assert app.control.track_enabled(UNCLASSIFIED_ID) and app.control.track_enabled('speaker_1')
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


def test_saved_person_border_follows_audio_even_if_muted_and_clears_on_stop(monkeypatch, root):
    import numpy as np
    import time
    from selective_hearing.audio import LiveSession
    from selective_hearing.core import AudioPacket
    startup_fixture(monkeypatch)
    app = App(root, auto_start=False)
    try:
        identity = seed_person(app)
        assert app.speaker_activity[identity]['text'] == '未在说话'
        app.session = LiveSession(0, 1, app.control, app.events.put)
        app.session.playback.put(AudioPacket(np.full(320, .1, np.float32),
                                            {identity: np.full(320, .1, np.float32)}))
        app.set_person(identity, False)
        app.session.playback.read(320, app.control.get()[0], 0, time.perf_counter())
        drain(app)
        assert app.speaker_cards[identity]['highlightbackground'] == '#16a34a'
        assert app.speaker_activity[identity]['text'] == '● 正在说话'
        assert not app.control.track_enabled(identity)
        app.stop()
        assert app.speaker_cards[identity]['highlightbackground'] == '#cbd5e1'
        assert app.speaker_activity[identity]['text'] == '未在说话'
    finally:
        app.close()


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


def test_custom_window_reaches_live_session_and_survives_reopen(monkeypatch, tk_session):
    startup_fixture(monkeypatch)
    window = tk.Toplevel(tk_session); window.withdraw()
    app = App(window, auto_start=False)
    app.backend = object()
    app.window_choice.set('2.5')
    app.start_live()
    assert app.session.window_seconds == 2.5
    app.close()
    window = tk.Toplevel(tk_session); window.withdraw()
    app = App(window, auto_start=False)
    try:
        assert app.window_choice.get() == '2.5'
        app.window_choice.set('nan')
        app.backend = object()
        app.start_live(automatic=True)
        assert app.session is None and '1 到 10' in app.status['text']
    finally:
        app.close()


def test_custom_window_reaches_file_processor(monkeypatch, root):
    import numpy as np
    from selective_hearing.core import AudioPacket
    startup_fixture(monkeypatch)
    app = FileApp(root)
    calls = []
    class InlineThread:
        def __init__(self, target, **kwargs): self.target = target
        def start(self): self.target()
    monkeypatch.setattr('selective_hearing.file_ui.Thread', InlineThread)
    monkeypatch.setattr('selective_hearing.file_ui.filedialog.askopenfilename', lambda **kw: 'sample.wav')
    def process(*args, **kwargs):
        calls.append(kwargs['window_seconds'])
        return AudioPacket(np.zeros(320, np.float32)), {'speakers': []}
    monkeypatch.setattr('selective_hearing.file_ui.process_file', process)
    try:
        app.backend = object()
        app.window_choice.set('6')
        app.import_file()
        assert calls == [6]
        assert str(app.window_box['state']) == 'disabled'
        drain(app)
        assert str(app.window_box['state']) == 'normal'
    finally:
        app.close()


def deliver_devices(app, items, **kwargs):
    app.events.put(dict(type='devices', items=items, epoch=app.device_epoch, **kwargs))
    drain(app)


def test_hotplug_lists_new_devices_without_restarting_or_using_helper_ids(monkeypatch, root):
    startup_fixture(monkeypatch)
    listed = [device(1, 'Mic', 'input', True), device(2, 'Headphones', 'output', True)]
    probes = []
    def local_devices(**kwargs):
        probes.append(kwargs)
        return list(listed)
    monkeypatch.setattr('selective_hearing.ui.devices', local_devices)
    app = App(root, auto_start=False)
    try:
        app.backend = object()
        identity = seed_person(app)
        app.start_live()
        app.set_person(identity, False)
        session = app.session
        before = len(probes)
        helper = [device(80, 'Mic', 'input'), device(81, 'Headphones', 'output'),
                  device(82, 'USB Mic', 'input', True), device(83, 'USB Headphones', 'output', True)]
        deliver_devices(app, helper)
        assert app.session is session and not session.cancel.is_set()
        assert len(probes) == before  # Never reinitialize active PortAudio.
        assert 'USB Mic' in app.input_box['values'][1]
        assert app.input_box.get() == 'Mic' and app.output_box.get() == 'Headphones'
        # Local IDs can differ from the probe process; switching must resolve names again.
        listed[:] = [device(11, 'Mic', 'input'), device(12, 'Headphones', 'output'),
                     device(13, 'USB Mic', 'input', True), device(14, 'USB Headphones', 'output', True)]
        app.input_box.current(1)
        app.remember_device('input')
        assert session.cancel.is_set() and app.session is not session
        assert app.session.device_ids == (13, 12)
        assert not app.control.track_enabled(identity)
    finally:
        app.close()


@pytest.mark.parametrize('cancel', [False, True])
def test_device_switch_waits_for_worker_and_stop_cancels_resume(monkeypatch, root, cancel):
    from types import SimpleNamespace
    startup_fixture(monkeypatch)
    app = App(root, auto_start=False)
    try:
        app.backend = object()
        app.start_live()
        session = app.session
        running = [True]
        session.worker = SimpleNamespace(is_alive=lambda: running[0])
        app.remember_device('input')
        assert session.cancel.is_set() and app.session is session
        assert app.pending_device_start == 'live'
        if cancel:
            app.stop()
        running[0] = False
        drain(app)
        assert app.pending_device_start is None
        if cancel:
            assert app.session is None
        else:
            assert app.session is not session and app.session.started
    finally:
        running[0] = False
        app.close()


def test_unplug_stops_and_does_not_fall_back_to_speakers(monkeypatch, root):
    startup_fixture(monkeypatch)
    app = App(root, auto_start=False)
    try:
        app.backend = object()
        app.start_live()
        session = app.session
        deliver_devices(app, [device(8, 'Mic', 'input', True), device(9, 'Speakers', 'output', True)])
        assert session.cancel.is_set() and app.session is None
        assert app.output_box.current() == -1
        assert app.pending_device_start is None and '已断开' in app.status['text']
    finally:
        app.close()


def test_refresh_during_capture_isolated_and_scan_failures_keep_audio(monkeypatch, root):
    startup_fixture(monkeypatch)
    app = App(root, auto_start=False)
    try:
        app.backend = object()
        app.start_live()
        session = app.session
        scanned = []
        monkeypatch.setattr(app, 'scan_devices', lambda: scanned.append(True))
        app.refresh_devices()
        assert scanned and app.session is session
        app.events.put(dict(type='device_error', message='timeout', epoch=app.device_epoch))
        drain(app)
        assert app.session is session and app.input_box.get()
        # A delayed old scan must not clear freshly resolved devices.
        app.events.put(dict(type='devices', items=[], epoch=app.device_epoch - 1))
        drain(app)
        assert app.session is session and app.inputs
    finally:
        app.close()


def test_file_device_switch_resolves_local_ids_without_starting_microphone(monkeypatch, root):
    startup_fixture(monkeypatch)
    listed = [device(2, 'Headphones', 'output', True)]
    monkeypatch.setattr('selective_hearing.ui.devices', lambda **kwargs: list(listed))
    opened = []
    class Player(FakeLiveSession):
        def __init__(self, packet, device, control):
            super().__init__()
            opened.append(device)
    monkeypatch.setattr('selective_hearing.file_ui.FilePlayer', Player)
    app = FileApp(root)
    try:
        app.file_packet = object()
        app.play_file()
        player = app.player
        deliver_devices(app, [device(50, 'Headphones', 'output'), device(51, 'USB', 'output', True)])
        assert app.player is player
        listed[:] = [device(22, 'Headphones', 'output'), device(23, 'USB', 'output', True)]
        app.output_box.current(1)
        app.remember_device('output')
        assert opened == [2, 23] and player.cancel.is_set()
        assert app.player is not player and app.session is None
    finally:
        app.close()
