"""Live Tk interface and shared controls for the separate file-test application."""
from queue import Queue, Empty
from threading import Thread, Event
import tkinter as tk
from tkinter import ttk, messagebox

from .core import OutputControl
from .audio import devices, LiveSession, LIVE_WINDOW_SECONDS
from .models import ModelBackend


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
        self.source_devices = []
        self.manual_devices = {}
        self.alive = True
        self.generation = 0
        self._build()
        self.refresh_devices()
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
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)
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
        self.model_status = ttk.Label(controls, text="等待加载模型…", foreground="#67758a")
        self.model_status.pack(anchor="w", pady=(8, 0))
        timing_note = ('文件每 4 秒处理一块；处理完成后可循环试听、导出音轨。' if self.file_mode
                       else f'实时识别每 {LIVE_WINDOW_SECONDS:g} 秒处理一块；首次识别即可选择，仍有采样和计算延迟。')
        ttk.Label(controls, text=timing_note, foreground="#9b6500").pack(anchor="w", pady=(4, 0))

        people = ttk.LabelFrame(main, text="3  选择收听目标", padding=12)
        people.pack(fill="x", pady=5)
        self.people_row = ttk.Frame(people)
        self.people_row.pack(fill="x")
        ttk.Button(self.people_row, text="全部听（原声）", command=lambda: self.choose("all")).pack(side="left", padx=(0, 8))
        ttk.Button(self.people_row, text="全部静音", command=lambda: self.choose("mute")).pack(side="left", padx=(0, 8))
        self.track_area = ttk.Frame(people)
        self.track_canvas = tk.Canvas(self.track_area, height=112, highlightthickness=0, bg='#eef2f6')
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

    def refresh_devices(self):
        if self.session is not None or self.player is not None:
            self.status.config(text="请先停止收音和播放，再刷新设备。")
            return
        try:
            self.source_devices = devices(refresh=True)
            self.inputs = [d for d in self.source_devices if d["input"]]
            self.outputs = [d for d in self.source_devices if d["output"]]
            for direction, box, items in [('input', self.input_box, self.inputs),
                                          ('output', self.output_box, self.outputs)]:
                names = [d['name'] + ('（系统默认）' if d[f'default_{direction}'] else '')
                         for d in items]
                box["values"] = names
                key = self.manual_devices.get(direction)
                matches = [i for i, d in enumerate(items) if (d['api'], d['name']) == key]
                selected = matches[0] if len(matches) == 1 else next(
                    (i for i, d in enumerate(items) if d[f'default_{direction}']), None)
                if len(matches) != 1:
                    self.manual_devices.pop(direction, None)
                box.set('')
                if selected is not None:
                    box.current(selected)
            missing = [label for box, label in [(self.input_box, '输入'), (self.output_box, '输出')]
                       if box.current() < 0 and (not self.file_mode or label == '输出')]
            note = ('、'.join(missing) + '设备未选中：请连接设备后刷新，或手动选择。'
                    if missing else ('已选好播放设备。' if self.file_mode else '已选好输入和输出。连接或切换耳机后，请刷新设备。'))
            self.device_note.config(text=note)
        except Exception as exc:
            self.inputs, self.outputs = [], []
            for box in (self.input_box, self.output_box):
                box["values"] = []
                box.set("")
            self.device_note.config(text=f"设备枚举失败：{exc}")

    def remember_device(self, direction):
        box, items = ((self.input_box, self.inputs) if direction == 'input'
                      else (self.output_box, self.outputs))
        index = box.current()
        if 0 <= index < len(items):
            device = items[index]
            self.manual_devices[direction] = (device['api'], device['name'])

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

    def reset_people(self):
        for card in self.speaker_cards.values():
            card.destroy()
        self.speaker_buttons.clear()
        self.speaker_cards.clear()
        self.speaker_labels.clear()
        self.speaker_mutes.clear()
        self.speaker_names.clear()
        self.track_area.pack_forget()
        # A new source is a new identity session; a prior target must not carry over.
        self.choose("all")
        self.people_note.config(text="等待人声。首次识别即可选择，身份确认会继续进行。")

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
            if identity not in self.speaker_buttons:
                self.track_area.pack(fill='x', before=self.people_note, pady=(8, 0))
                card = ttk.Frame(self.track_grid)
                position = len(self.speaker_cards)
                card.grid(row=position // 2, column=position % 2, sticky='w', padx=(0, 12), pady=4)
                label = ttk.Label(card)
                self.speaker_names[identity] = tk.StringVar(value=f"说话人 {identity.split('_')[-1]}")
                name_box = ttk.Entry(card, textvariable=self.speaker_names[identity], width=22)
                name_box.pack(anchor='w')
                name_box.bind('<FocusOut>', lambda _, key=identity: self.commit_name(key))
                name_box.bind('<Return>', lambda _, key=identity: self.commit_name(key))
                label.pack(anchor='w')
                row = ttk.Frame(card)
                row.pack()
                button = ttk.Button(row, text='听他', command=lambda key=identity: self.set_person(key, True))
                button.pack(side='left', padx=(0, 4))
                mute = ttk.Button(row, text='不听他', command=lambda key=identity: self.set_person(key, False))
                mute.pack(side='left')
                self.speaker_buttons[identity] = button
                self.speaker_cards[identity] = card
                self.speaker_labels[identity] = label
                self.speaker_mutes[identity] = mute
            text = f"编号 {identity.split('_')[-1]}"
            text += " · 有声音" if identity in active else ""
            if not speaker["ready"]:
                text += " · 确认中"
            self.speaker_labels[identity].config(text=text)
        if speakers:
            self.people_note.config(text="名字可直接编辑，回车保存。绿色为当前选择；“确认中”可能误判，同时最多分离两路人声。")
        else:
            self.people_note.config(text=('尚未识别到足够清晰的人声，请尝试更清晰的录音。' if self.file_mode
                                          else '尚未识别到足够清晰的人声。请查看下方输入电平，并确认麦克风选择。'))
        self.refresh_track_controls()

    def commit_name(self, identity):
        if identity not in self.speaker_names:
            return
        value = ' '.join(self.speaker_names[identity].get().split())[:40]
        self.speaker_names[identity].set(value or f"说话人 {identity.split('_')[-1]}")

    def start_live(self, automatic=False):
        if self.file_mode or not self.alive:
            return
        try:
            if self.file_busy:
                raise ValueError("请等待文件处理结束，或先停止")
            if self.ai.get() and not self.backend:
                self.auto_start_pending = True
                self.load_models()
                return
            input_id = self.selected_device(self.input_box, self.inputs)
            output_id = self.selected_device(self.output_box, self.outputs)
            self.stop()
            if self.session is not None:
                raise ValueError("上一轮推理正在结束，请稍后开始")
            self.file_packet = self.file_report = None
            self.reset_people()
            generation = self.generation
            notify = lambda event: self.events.put({**event, "generation": generation})
            session = LiveSession(input_id, output_id, self.control, notify, self.backend if self.ai.get() else None)
            session.start()
            self.session = session
            self.start_button.config(state="disabled")
            self.ai_check.config(state="disabled")
            self.status.config(text=f"正在收音 · AI 每 {LIVE_WINDOW_SECONDS:g} 秒处理一块" if self.ai.get() else "正在收音 · 原声测试（未启用分人）")
            if not self.ai.get():
                self.people_note.config(text='当前是原声模式，不识别人。启用“自动识别人声并分离”后才会生成人物按钮。')
        except Exception as exc:
            self.auto_start_pending = False
            self.status.config(text=f'未能开始收音：{exc}。请检查设备后点“开始收音”重试。')
            if not automatic:
                messagebox.showerror("无法开始", str(exc))

    def stop(self):
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
        self.status.config(text="已停止音频输入与播放")

    def poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event["type"]
                if kind in ("file_done", "file_error"):
                    self.file_busy = False
                    self.file_button.config(state="normal")
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
                    self.update_people(event["speakers"], event["active"])
                    if self.file_mode and "fraction" in event:
                        self.progress["value"] = event["fraction"]
                    self.status.config(text=f"音频 {event['seconds']:.2f}s / 计算 {event['compute_seconds']:.2f}s / RTF {event['rtf']:.2f}（小于 1 才能跟上输入）")
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
            if self.session.cancel.is_set() and not self.session.worker.is_alive():
                self.session = None
            else:
                stats = self.session.stats()
                self.stats_label.config(text=f"输入电平 {stats['level']:.3f} · 输入丢块 {stats['input_drops']} · 输出丢块 {stats['output_drops']} · 缓冲 {stats['queue_seconds']:.1f}s\n软件块年龄 {stats['software_age']:.1f}s（非端到端延迟）· 空缓冲回调 {stats['empty_callbacks']}")
        if self.alive:
            self.poll_after = self.root.after(100, self.poll)

    def close(self):
        self.alive = False
        self.stop()
        if self.poll_after is not None:
            self.root.after_cancel(self.poll_after)
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
