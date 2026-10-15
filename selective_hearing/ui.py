"""Live Tk interface and shared controls for the separate file-test application."""
from queue import Queue, Empty
from threading import Thread, Event
import tkinter as tk
from tkinter import ttk, messagebox

from .core import OutputControl, SpeakerRegistry
from .profiles import ProfileStore, default_profile_path, validate_window_seconds
from .audio import devices, LiveSession, LIVE_WINDOW_SECONDS
from .models import ModelBackend
from .device_probe import fresh_devices, catalog_signature


class App:
    def __init__(self, root, *, auto_start=True, file_mode=False):
        self.root = root
        self.file_mode = file_mode
        self.auto_start_pending = auto_start and not file_mode
        self.startup_after = self.poll_after = None
        root.title('Selective Hearing · 音频文件测试' if file_mode else "Selective Hearing · 选择听谁说话")
        root.geometry('1000x980+80+30' if file_mode else '1000x880+80+30')
        root.minsize(860, 900 if file_mode else 760)
        root.configure(bg="#eef2f6")
        self.events = Queue()
        self.control = OutputControl()
        self.backend = None
        self.backend_busy = False
        self.session = None
        self.player = None
        self.file_packet = self.file_report = None
        self.file_cancel = Event()
        self.file_busy = False
        self.speaker_buttons = {}
        self.speaker_cards = {}
        self.speaker_labels = {}
        self.speaker_mutes = {}
        self.speaker_names = {}
        self.speaker_activity = {}
        self.active_speakers = set()
        self.saved_ids = set()
        self.pending_delete = None
        self.profile_store = None
        self.profile_error = None
        try:
            self.profile_store = ProfileStore(default_profile_path(file_mode))
        except Exception as exc:
            self.profile_error = str(exc)
            self.auto_start_pending = False
        self.source_devices = []
        self.manual_devices = {}
        self.device_after = None
        self.device_scan_busy = False
        self.device_epoch = 0
        self.pending_device_start = None
        self.alive = True
        self.generation = 0
        self._build()
        if self.profile_error:
            self.profile_note.config(text=f'人物资料加载失败：{self.profile_error}；原文件未覆盖。')
        else:
            self.restore_people()
        self.refresh_devices()
        self.device_after = self.root.after(4000, self.watch_devices)
        self.poll_after = self.root.after(100, self.poll)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        if auto_start or file_mode:
            self.startup_after = self.root.after(0, self.begin_startup)

    def begin_startup(self):
        self.startup_after = None
        if self.alive and (self.auto_start_pending or self.file_mode):
            self.load_models()

    def _build(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f6")
        style.configure("TLabel", background="#eef2f6", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=8)
        style.configure('Selected.TButton', background='#c7e7d4', foreground='#17482a')
        style.configure("TLabelframe", background="#eef2f6")
        style.configure("TLabelframe.Label", background="#eef2f6", font=("Microsoft YaHei UI", 10, "bold"))
        # Added settings/status rows must remain reachable on smaller displays.
        page = tk.Canvas(self.root, bg='#eef2f6', highlightthickness=0)
        page_scroll = ttk.Scrollbar(self.root, orient='vertical', command=page.yview)
        page_scroll.pack(side='right', fill='y')
        page.pack(side='left', fill='both', expand=True)
        page.configure(yscrollcommand=page_scroll.set)
        main = ttk.Frame(page, padding=16)
        main_window = page.create_window((0, 0), window=main, anchor='nw')
        main.bind('<Configure>', lambda _: page.configure(scrollregion=page.bbox('all')))
        page.bind('<Configure>', lambda event: page.itemconfigure(main_window, width=event.width))
        ttk.Label(main, text='音频文件测试' if self.file_mode else "选择听谁说话", font=("Microsoft YaHei UI", 24, "bold")).pack(anchor="w")
        subtitle = ('导入录音测试分离效果 · 不打开麦克风' if self.file_mode
                    else '启动后自动加载模型并收音 · 点“停止”可随时暂停')
        ttk.Label(main, text=subtitle, foreground="#536378").pack(anchor="w", pady=(4, 14))

        source = ttk.LabelFrame(main, text="1  音频设备", padding=12)
        source.pack(fill="x", pady=5)
        source.columnconfigure(1, weight=1)
        self.input_box = ttk.Combobox(source, state="readonly", width=55)
        if not self.file_mode:
            ttk.Label(source, text="输入麦克风").grid(row=0, column=0, padx=(0, 12))
            self.input_box.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(source, text="播放设备").grid(row=1, column=0, padx=(0, 12))
        self.output_box = ttk.Combobox(source, state="readonly", width=55)
        self.output_box.grid(row=1, column=1, sticky="ew", pady=4)
        self.input_box.bind('<<ComboboxSelected>>', lambda _: self.remember_device('input'))
        self.output_box.bind('<<ComboboxSelected>>', lambda _: self.remember_device('output'))
        self.refresh_button = ttk.Button(source, text="刷新设备", command=self.refresh_devices)
        self.refresh_button.grid(row=0, column=2, rowspan=2, padx=(10, 0))
        self.device_note = ttk.Label(source, text="", foreground="#67758a")
        self.device_note.grid(row=2, column=0, columnspan=3, sticky="w", pady=4)

        controls = ttk.LabelFrame(main, text="2  模型与运行", padding=12)
        controls.pack(fill="x", pady=5)
        row = ttk.Frame(controls)
        row.pack(fill="x")
        self.load_button = ttk.Button(row, text="加载人声模型", command=self.load_models)
        self.load_button.pack(side="left", padx=(0, 8))
        self.ai = tk.BooleanVar(value=True)
        self.ai_check = ttk.Checkbutton(row, text="自动识别人声并分离", variable=self.ai)
        self.start_button = ttk.Button(row, text="开始收音", command=self.start_live)
        if not self.file_mode:
            self.ai_check.pack(side="left", padx=6)
            self.start_button.pack(side="left", padx=8)
        ttk.Button(row, text="停止", command=self.stop).pack(side="left")
        window_row = ttk.Frame(controls)
        window_row.pack(anchor='w', pady=(8, 0))
        ttk.Label(window_row, text='分析窗口（秒）').pack(side='left', padx=(0, 8))
        default_window = 4 if self.file_mode else LIVE_WINDOW_SECONDS
        seconds = self.profile_store.window_seconds(default_window) if self.profile_store else default_window
        self.window_choice = tk.StringVar(value=f'{seconds:g}')
        self.window_box = ttk.Spinbox(window_row, textvariable=self.window_choice,
                                      from_=1, to=10, increment=.5, width=8,
                                      command=self.save_window)
        self.window_box.pack(side='left')
        self.window_box.bind('<Return>', lambda _: self.save_window())
        self.window_box.bind('<FocusOut>', lambda _: self.save_window())
        ttk.Label(window_row, text='1–10 秒；加长通常更稳，但延迟更高。停止后修改。',
                  foreground='#67758a').pack(side='left', padx=8)
        self.model_status = ttk.Label(controls, text="等待加载模型…", foreground="#67758a")
        self.model_status.pack(anchor="w", pady=(8, 0))
        timing_note = ('按所设窗口分块；处理完成后可循环试听、导出音轨。' if self.file_mode
                       else '默认 3 秒；首次识别到人物即显示，后续新人需跨片段确认。')
        ttk.Label(controls, text=timing_note, foreground="#9b6500").pack(anchor="w", pady=(4, 0))

        self.recognition_label = ttk.Label(controls, text='识别进度：等待音频', wraplength=900, foreground='#536378')
        self.recognition_label.pack(anchor='w', pady=(4, 0))

        people = ttk.LabelFrame(main, text="3  选择收听目标", padding=12)
        people.pack(fill="x", pady=5)
        self.people_row = ttk.Frame(people)
        self.people_row.pack(fill="x")
        ttk.Button(self.people_row, text="全部听（原声）", command=lambda: self.choose("all")).pack(side="left", padx=(0, 8))
        ttk.Button(self.people_row, text="全部静音", command=lambda: self.choose("mute")).pack(side="left", padx=(0, 8))
        self.track_area = ttk.Frame(people)
        self.track_canvas = tk.Canvas(self.track_area, height=156, highlightthickness=0, bg='#eef2f6')
        scrollbar = ttk.Scrollbar(self.track_area, orient='vertical', command=self.track_canvas.yview)
        scrollbar.pack(side='right', fill='y')
        self.track_canvas.pack(side='left', fill='x', expand=True)
        self.track_canvas.configure(yscrollcommand=scrollbar.set)
        self.track_grid = ttk.Frame(self.track_canvas)
        grid_window = self.track_canvas.create_window((0, 0), window=self.track_grid, anchor='nw')
        self.track_grid.columnconfigure((0, 1), weight=1)
        self.track_grid.bind('<Configure>', lambda _: self.track_canvas.configure(scrollregion=self.track_canvas.bbox('all')))
        self.track_canvas.bind('<Configure>', lambda e: self.track_canvas.itemconfigure(grid_window, width=e.width))
        self.people_note = ttk.Label(people, text="可陆续识别多个人；当前模型同时最多分离两路人声，背景音暂不能独立分离。", foreground="#67758a", wraplength=910)
        self.people_note.pack(anchor="w", pady=8)
        self.profile_note = ttk.Label(people, text='声纹和名字自动保存在本机；样本指已归类的逐段声纹；追加学习指模板更新，待定样本可随后改归类。删除资料时会自动停止音频。',
                                      foreground='#536378', wraplength=900)
        self.profile_note.pack(anchor='w', pady=(0, 4))
        ttk.Label(people, text='绿框表示识别到该人正在说话；不受静音开关影响，随处理后的音频显示，有延迟。',
                  foreground='#67758a', wraplength=900).pack(anchor='w', pady=(0, 4))
        self.mode_label = ttk.Label(people, text="当前：全部原声", font=("Microsoft YaHei UI", 11, "bold"))
        self.mode_label.pack(anchor="w")
        gain_row = ttk.Frame(people)
        gain_row.pack(fill="x", pady=(8, 0))
        ttk.Label(gain_row, text="播放音量").pack(side="left")
        self.gain = ttk.Scale(gain_row, from_=0, to=1, value=0.4,
                              command=lambda v: self.control.set(gain=float(v)))
        self.gain.pack(side="left", fill="x", expand=True, padx=12)
        ttk.Label(gain_row, text="软件静音不阻断外界漏音", foreground="#67758a").pack(side="right")

        if self.file_mode:
            self.build_file_controls(main)

        self.status = ttk.Label(main, text="就绪 · 所有处理在本机进行，录音不会自动保存或上传。", wraplength=930)
        self.status.pack(anchor="w", pady=(12, 4))
        self.stats_label = ttk.Label(main, text="", foreground="#67758a", wraplength=930)
        self.stats_label.pack(anchor="w")

    def device_choices(self):
        choices = {}
        for direction, box, items in [('input', self.input_box, getattr(self, 'inputs', [])),
                                      ('output', self.output_box, getattr(self, 'outputs', []))]:
            index = box.current()
            if 0 <= index < len(items):
                choices[direction] = (items[index]['api'], items[index]['name'])
        return choices

    def apply_device_list(self, listed, preserve=False):
        choices = self.device_choices() if preserve else self.manual_devices.copy()
        self.source_devices = listed
        self.inputs = [d for d in listed if d['input']]
        self.outputs = [d for d in listed if d['output']]
        for direction, box, items in [('input', self.input_box, self.inputs),
                                      ('output', self.output_box, self.outputs)]:
            box['values'] = [d['name'] + ('（系统默认）' if d[f'default_{direction}'] else '')
                             for d in items]
            key = choices.get(direction)
            matches = [i for i, d in enumerate(items) if (d['api'], d['name']) == key]
            selected = matches[0] if len(matches) == 1 else None
            if not preserve and selected is None:
                selected = next((i for i, d in enumerate(items) if d[f'default_{direction}']), None)
                self.manual_devices.pop(direction, None)
            box.set('')
            if selected is not None:
                box.current(selected)
        missing = [label for box, label in [(self.input_box, '输入'), (self.output_box, '输出')]
                   if box.current() < 0 and (not self.file_mode or label == '输出')]
        self.device_note.config(text=('、'.join(missing) + '设备未选中：请选择可用设备。' if missing
            else '每 4 秒自动检测设备；收音或播放时选择其他设备会切换，期间会短暂停顿。'))
        return not missing

    def refresh_devices(self):
        if self.session is not None or self.player is not None:
            self.scan_devices()
            return
        try:
            self.device_epoch += 1
            self.apply_device_list(devices(refresh=True))
        except Exception as exc:
            self.apply_device_list([])
            self.device_note.config(text=f'设备枚举失败：{exc}')

    def watch_devices(self):
        self.device_after = None
        if self.alive:
            self.scan_devices()
            self.device_after = self.root.after(4000, self.watch_devices)

    def scan_devices(self):
        if self.device_scan_busy or not self.alive:
            return
        self.device_scan_busy = True
        epoch = self.device_epoch
        def work():
            try:
                self.events.put({'type': 'devices', 'items': fresh_devices(), 'epoch': epoch})
            except Exception as exc:
                self.events.put({'type': 'device_error', 'message': str(exc), 'epoch': epoch})
        Thread(target=work, daemon=True).start()

    def accept_device_scan(self, event):
        self.device_scan_busy = False
        if event['epoch'] != self.device_epoch:
            return  # Ignore a snapshot started before local PortAudio was refreshed.
        if event['type'] == 'device_error':
            self.device_note.config(text='自动检测暂时失败，可点刷新设备重试。')
            return  # A failed probe is not evidence of unplugging.
        listed = event['items']
        if catalog_signature(listed) == catalog_signature(self.source_devices):
            return
        running = self.session is not None or self.player is not None
        valid = self.apply_device_list(listed, preserve=running or self.pending_device_start is not None)
        if running and not valid:
            self.stop()
            self.status.config(text='正在使用的设备已断开或无法唯一匹配，音频已停止。请选择设备后重新开始。')

    def prepare_audio_devices(self):
        # All streams must be closed first. Never pass IDs from the helper
        # process into this process; resolve selected names against a fresh list.
        if self.session is not None or self.player is not None:
            raise ValueError('上一轮音频正在结束，请稍后开始')
        self.device_epoch += 1
        listed = devices(refresh=True)
        if not self.apply_device_list(listed, preserve=True):
            raise ValueError('所选设备已不可用，请重新选择设备')

    def remember_device(self, direction):
        box, items = ((self.input_box, self.inputs) if direction == 'input'
                      else (self.output_box, self.outputs))
        index = box.current()
        if 0 <= index < len(items):
            device = items[index]
            self.manual_devices[direction] = (device['api'], device['name'])
            resume = self.pending_device_start
            if self.session is not None and not self.session.cancel.is_set():
                resume = 'live'
            elif self.player is not None:
                resume = 'file'
            if resume:
                self.stop()
                self.pending_device_start = resume
                self.status.config(text='正在切换音频设备…')
                self.finish_device_switch()

    def finish_device_switch(self):
        if self.pending_device_start is None or self.session is not None or self.pending_delete is not None:
            return
        resume, self.pending_device_start = self.pending_device_start, None
        if resume == 'live':
            self.start_live(automatic=True)
        else:
            self.play_file()

    def selected_device(self, box, items):
        index = box.current()
        if index < 0 or index >= len(items):
            raise ValueError("请先明确选择音频设备")
        return items[index]["id"]

    def load_models(self):
        if self.backend_busy or self.backend is not None:
            return
        self.backend_busy = True
        self.load_button.config(state="disabled")
        self.model_status.config(text="正在加载本地模型…")
        self.status.config(text='正在加载模型，完成后自动开始收音…' if self.auto_start_pending else '正在加载模型…')
        def work():
            try:
                backend = ModelBackend()
                self.events.put({"type": "loaded", "backend": backend})
            except Exception as exc:
                self.events.put({"type": "load_error", "message": str(exc)})
        Thread(target=work, daemon=True).start()

    def reset_people(self, reset_selection=True, save_names=True):
        if save_names and not self.save_names():
            raise ValueError('名字保存失败，请检查人物资料提示后重试。')
        for card in self.speaker_cards.values():
            card.destroy()
        self.speaker_buttons.clear()
        self.speaker_cards.clear()
        self.speaker_labels.clear()
        self.speaker_mutes.clear()
        self.speaker_names.clear()
        self.speaker_activity.clear()
        self.active_speakers.clear()
        self.saved_ids.clear()
        self.track_area.pack_forget()
        # Listening choices reset, but identities and names come from local storage.
        if reset_selection:
            self.choose("all")
        self.people_note.config(text="等待人声。首位人物首次识别即可选择，后续新人需跨片段确认。")
        self.restore_people()

    def restore_people(self):
        if self.profile_store is not None:
            self.update_people(self.profile_store.summaries())

    def save_names(self):
        success = True
        for identity in list(self.speaker_names):
            success = self.commit_name(identity) and success
        return success

    def delete_person(self, identity):
        if self.pending_delete is not None:
            self.profile_note.config(text='正在等待音频处理结束并删除资料，请稍候。')
            return
        if identity not in self.saved_ids:
            return
        name = self.speaker_names[identity].get()
        if not messagebox.askyesno('删除人物资料',
                f'删除“{name}”的名字和已学习声纹？\n将自动停止收音、播放和文件处理。删除后保持停止；再次开始收音时可能重新识别此人。',
                parent=self.root):
            return
        try:
            self.stop()
            self.pending_delete = (identity, name)
            self.profile_note.config(text=f'正在停止音频处理，随后删除“{name}”…')
            self.finish_delete()
        except Exception as exc:
            self.pending_delete = None
            self.profile_note.config(text=f'删除未完成：{exc}')

    def finish_delete(self):
        if self.pending_delete is None or self.file_busy:
            return
        if self.session is not None:
            if self.session.worker and self.session.worker.is_alive():
                return
            self.session = None
        identity, name = self.pending_delete
        try:
            if not self.save_names():
                raise ValueError('名字保存失败，人物资料尚未删除。')
            self.profile_store.delete(identity)
            self.saved_ids.discard(identity)
            self.file_packet = self.file_report = None
            if self.file_mode:
                self.play_button.config(state='disabled')
                self.export_button.config(state='disabled')
            self.reset_people(save_names=False)
            self.profile_note.config(text=f'已删除“{name}”的本机资料。')
            self.status.config(text='人物资料已删除，音频保持停止。可点击开始收音继续。' if not self.file_mode
                               else '人物资料已删除，播放已停止。可重新导入音频测试。')
        except Exception as exc:
            self.profile_note.config(text=f'删除未完成：{exc}')
        finally:
            self.pending_delete = None

    def choose(self, mode):
        self.control.set(mode=mode)
        self.refresh_track_controls()

    def set_person(self, identity, enabled):
        self.control.set_track(identity, enabled)
        self.refresh_track_controls()

    def refresh_track_controls(self):
        mode, _ = self.control.get()
        for identity in self.speaker_buttons:
            enabled = self.control.track_enabled(identity)
            self.speaker_buttons[identity].configure(style='Selected.TButton' if enabled else 'TButton')
            self.speaker_mutes[identity].configure(style='TButton' if enabled else 'Selected.TButton')
        if isinstance(mode, str):
            label = '全部原声' if mode == 'all' else '全部静音'
        else:
            count = sum(mode.enabled(identity) for identity in self.speaker_buttons)
            label = f'听 {count} 人 · 不听 {len(self.speaker_buttons) - count} 人'
            label += ' · 新人默认听' if mode.default_enabled else ' · 新人默认不听'
        self.mode_label.config(text='当前：' + label)

    def update_people(self, speakers, active=()):
        for speaker in speakers:
            identity = speaker["id"]
            if speaker.get('saved'):
                self.saved_ids.add(identity)
            if identity not in self.speaker_buttons:
                self.track_area.pack(fill='x', before=self.people_note, pady=(8, 0))
                card = tk.Frame(self.track_grid, bg='#eef2f6', highlightthickness=2,
                                highlightbackground='#cbd5e1', highlightcolor='#cbd5e1', padx=7, pady=5)
                position = len(self.speaker_cards)
                card.grid(row=position // 2, column=position % 2, sticky='w', padx=(0, 12), pady=4)
                label = ttk.Label(card, wraplength=380)
                self.speaker_names[identity] = tk.StringVar(value=speaker.get('name', f"说话人 {identity.split('_')[-1]}"))
                name_box = ttk.Entry(card, textvariable=self.speaker_names[identity], width=22)
                name_box.pack(anchor='w')
                name_box.bind('<FocusOut>', lambda _, key=identity: self.commit_name(key))
                name_box.bind('<Return>', lambda _, key=identity: self.commit_name(key))
                activity = ttk.Label(card, text='未在说话', foreground='#67758a')
                activity.pack(anchor='w')
                self.speaker_activity[identity] = activity
                label.pack(anchor='w')
                row = ttk.Frame(card)
                row.pack()
                button = ttk.Button(row, text='听他', command=lambda key=identity: self.set_person(key, True))
                button.pack(side='left', padx=(0, 4))
                mute = ttk.Button(row, text='不听他', command=lambda key=identity: self.set_person(key, False))
                mute.pack(side='left')
                if speaker.get('saved'):
                    ttk.Button(row, text='删除资料', command=lambda key=identity: self.delete_person(key)).pack(side='left', padx=4)
                self.speaker_buttons[identity] = button
                self.speaker_cards[identity] = card
                self.speaker_labels[identity] = label
                self.speaker_mutes[identity] = mute
            text = f"编号 {speaker.get('display_number', identity.split('_')[-1])}"
            if not speaker["ready"]:
                text += " · 确认中"
            if speaker.get('saved'):
                text += f" · 已保存 · 样本 {speaker.get('feature_samples', 0)} 段 · 追加学习 {speaker.get('learned_observations', 0)} 段"
                if not speaker.get('feature_samples'):
                    text += ' · 初始声纹已保存'
            self.speaker_labels[identity].config(text=text)
        if speakers:
            self.people_note.config(text="名字可直接编辑，回车保存。新人需跨片段确认，避免重复建人；同时最多分离两路人声。")
        else:
            self.people_note.config(text=('尚未识别到足够清晰的人声，请尝试更清晰的录音。' if self.file_mode
                                          else '尚未识别到足够清晰的人声。请查看下方输入电平，并确认麦克风选择。'))
        self.refresh_track_controls()

    def update_speaking(self, active=()):
        active = set(active) & self.speaker_cards.keys()
        for identity in self.active_speakers ^ active:
            if identity not in self.speaker_cards:
                continue
            speaking = identity in active
            color = '#16a34a' if speaking else '#cbd5e1'
            self.speaker_cards[identity].config(highlightbackground=color, highlightcolor=color)
            self.speaker_activity[identity].config(text='● 正在说话' if speaking else '未在说话',
                                                 foreground='#15803d' if speaking else '#67758a')
        self.active_speakers = active

    def commit_name(self, identity):
        if identity not in self.speaker_names:
            return True
        value = ' '.join(self.speaker_names[identity].get().split())[:40]
        self.speaker_names[identity].set(value or f"说话人 {identity.split('_')[-1]}")
        if identity in self.saved_ids:
            try:
                self.speaker_names[identity].set(self.profile_store.rename(identity, value))
            except Exception as exc:
                self.profile_note.config(text=f'名字未保存：{exc}')
                return False
        return True

    def save_window(self):
        try:
            self.selected_window_seconds()
            return True
        except ValueError as exc:
            self.status.config(text=str(exc))
            return False

    def selected_window_seconds(self):
        seconds = validate_window_seconds(self.window_choice.get())
        if self.profile_store is not None:
            self.profile_store.set_window_seconds(seconds)
        return seconds

    def start_live(self, automatic=False):
        if self.file_mode or not self.alive:
            return
        if self.pending_delete is not None:
            self.status.config(text='正在删除人物资料，请稍候再开始收音。')
            return
        try:
            window_seconds = self.selected_window_seconds()
            if self.profile_error:
                raise ValueError('人物资料无法加载：' + self.profile_error)
            if self.file_busy:
                raise ValueError("请等待文件处理结束，或先停止")
            if self.ai.get() and not self.backend:
                self.auto_start_pending = True
                self.load_models()
                return
            self.stop()
            self.prepare_audio_devices()
            input_id = self.selected_device(self.input_box, self.inputs)
            output_id = self.selected_device(self.output_box, self.outputs)
            self.file_packet = self.file_report = None
            self.reset_people(reset_selection=not automatic)
            generation = self.generation
            notify = lambda event: self.events.put({**event, "generation": generation})
            registry = SpeakerRegistry(store=self.profile_store) if self.ai.get() else None
            session = LiveSession(input_id, output_id, self.control, notify, self.backend if self.ai.get() else None,
                                  registry=registry, window_seconds=window_seconds)
            session.start()
            self.session = session
            self.start_button.config(state="disabled")
            self.ai_check.config(state="disabled")
            self.window_box.config(state='disabled')
            self.recognition_label.config(text=(f'识别进度：等待收满 {window_seconds:g} 秒音频'
                                                if self.ai.get() else '识别进度：原声模式，未启用人声识别'))
            self.status.config(text=f"正在收音 · AI 每 {window_seconds:g} 秒处理一块" if self.ai.get() else "正在收音 · 原声测试（未启用分人）")
            if not self.ai.get():
                self.people_note.config(text='当前是原声模式，不识别人。启用“自动识别人声并分离”后才会生成人物按钮。')
        except Exception as exc:
            self.auto_start_pending = False
            self.status.config(text=f'未能开始收音：{exc}。请检查设备后点“开始收音”重试。')
            if not automatic:
                messagebox.showerror("无法开始", str(exc))

    def stop(self):
        self.pending_device_start = None
        self.recognition_label.config(text='识别进度：已停止')
        self.update_speaking()
        self.auto_start_pending = False
        if self.startup_after is not None:
            self.root.after_cancel(self.startup_after)
            self.startup_after = None
        self.generation += 1
        self.file_cancel.set()
        if self.session:
            self.session.stop()
            # Keep reference until worker exits to prevent concurrent model inference.
            if not self.session.worker or not self.session.worker.is_alive():
                self.session = None
        if self.player:
            self.player.stop()
            self.player = None
        self.start_button.config(state="normal")
        self.ai_check.config(state="normal")
        self.window_box.config(state='disabled' if self.file_busy else 'normal')
        self.status.config(text="已停止音频输入与播放")

    def poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event["type"]
                if kind in ('devices', 'device_error'):
                    self.accept_device_scan(event)
                    continue
                if kind in ("file_done", "file_error"):
                    self.file_busy = False
                    self.file_button.config(state="normal")
                    self.window_box.config(state='normal')
                if "generation" in event and event["generation"] != self.generation:
                    continue
                if kind == "loaded":
                    self.backend = event["backend"]
                    self.backend_busy = False
                    self.model_status.config(text="已加载：Conv-TasNet + ECAPA + Silero · CPU · 权重已核验")
                    self.load_button.config(text="模型就绪")
                    if self.auto_start_pending:
                        self.start_live(automatic=True)
                    else:
                        self.status.config(text='模型就绪，可以导入音频。' if self.file_mode else '模型就绪，已停止收音。')
                elif kind == "load_error":
                    self.auto_start_pending = False
                    self.backend_busy = False
                    self.load_button.config(state="normal")
                    self.model_status.config(text="模型加载失败，可点击加载按钮重试。")
                    self.status.config(text=event["message"])
                elif kind == "progress":
                    if event.get('recognition_status'):
                        self.recognition_label.config(text='识别进度：' + event['recognition_status'])
                    self.update_people(event["speakers"], event["active"])
                    if self.file_mode and "fraction" in event:
                        self.progress["value"] = event["fraction"]
                    self.status.config(text=f"音频 {event['seconds']:.2f}s / 计算 {event['compute_seconds']:.2f}s / RTF {event['rtf']:.2f}（小于 1 才能跟上输入）")
                    if event.get('pending_speakers'):
                        self.status.config(text=self.status['text'] + ' · 新声音确认中，暂未新增人物')
                elif kind == "file_done":
                    self.file_packet, self.file_report = event["packet"], event["report"]
                    self.play_button.config(state="normal")
                    self.export_button.config(state="normal")
                    confirmed = sum(s['ready'] for s in self.file_report['speakers'])
                    self.status.config(text=f"处理完成：发现 {len(self.file_packet.tracks)} 人，其中 {confirmed} 人已确认；均可选择试听。")
                elif kind in ("error", "file_error"):
                    self.stop()
                    self.status.config(text=f"处理停止：{event['message']}")
        except Empty:
            pass
        if self.session:
            if self.session.cancel.is_set() and (not self.session.worker or not self.session.worker.is_alive()):
                self.session = None
            else:
                stats = self.session.stats()
                self.stats_label.config(text=f"输入电平 {stats['level']:.3f} · 输入丢块 {stats['input_drops']} · 输出丢块 {stats['output_drops']} · 缓冲 {stats['queue_seconds']:.1f}s\n软件块年龄 {stats['software_age']:.1f}s（非端到端延迟）· 空缓冲回调 {stats['empty_callbacks']}")
        self.finish_delete()
        self.finish_device_switch()
        source = self.session or self.player
        self.update_speaking(source.active_speakers() if source is not None else ())
        if self.alive:
            self.poll_after = self.root.after(100, self.poll)

    def close(self):
        if self.pending_delete is not None:
            self.status.config(text='正在等待音频处理结束并删除资料，请稍候再关闭。')
            return
        self.stop()
        self.save_window()
        if not self.save_names():
            return
        self.alive = False
        if self.device_after is not None:
            self.root.after_cancel(self.device_after)
        if self.poll_after is not None:
            self.root.after_cancel(self.poll_after)
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
