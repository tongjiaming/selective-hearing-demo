"""Native Tk interface; safe to open without models or a microphone."""
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time

from .core import OutputControl
from .audio import devices, LiveSession, FilePlayer, process_file, export_result, LIVE_WINDOW_SECONDS
from .models import ModelBackend


class App:
    def __init__(self, root):
        self.root = root
        root.title("Selective Hearing · 选择听谁说话")
        root.geometry("1000x980+80+30")
        root.minsize(860, 900)
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
        self.source_devices = []
        self.manual_devices = {}
        self.alive = True
        self.generation = 0
        self._build()
        self.refresh_devices()
        self.root.after(100, self.poll)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

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
        ttk.Label(main, text="选择听谁说话", font=("Microsoft YaHei UI", 24, "bold")).pack(anchor="w")
        ttk.Label(main, text="Windows 本地实验版  ·  每人独立开关  ·  打开界面不会自动收音", foreground="#536378").pack(anchor="w", pady=(4, 14))

        source = ttk.LabelFrame(main, text="1  音频设备", padding=12)
        source.pack(fill="x", pady=5)
        source.columnconfigure(1, weight=1)
        ttk.Label(source, text="输入麦克风").grid(row=0, column=0, padx=(0, 12))
        self.input_box = ttk.Combobox(source, state="readonly", width=55)
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

        controls = ttk.LabelFrame(main, text="2  开始实验", padding=12)
        controls.pack(fill="x", pady=5)
        row = ttk.Frame(controls)
        row.pack(fill="x")
        self.load_button = ttk.Button(row, text="加载人声模型", command=self.load_models)
        self.load_button.pack(side="left", padx=(0, 8))
        self.ai = tk.BooleanVar(value=True)
        self.ai_check = ttk.Checkbutton(row, text="自动识别人声并分离", variable=self.ai)
        self.ai_check.pack(side="left", padx=6)
        self.start_button = ttk.Button(row, text="开始收音", command=self.start_live)
        self.start_button.pack(side="left", padx=8)
        ttk.Button(row, text="停止", command=self.stop).pack(side="left")
        self.model_status = ttk.Label(controls, text="模型未加载。可先取消勾选，仅测试原声与静音。", foreground="#67758a")
        self.model_status.pack(anchor="w", pady=(8, 0))
        ttk.Label(controls, text=f"实时识别每 {LIVE_WINDOW_SECONDS:g} 秒处理一块；首次识别即可选择，仍有采样和计算延迟。", foreground="#9b6500").pack(anchor="w", pady=(4, 0))

        people = ttk.LabelFrame(main, text="3  选择收听目标", padding=12)
        people.pack(fill="x", pady=5)
        self.people_row = ttk.Frame(people)
        self.people_row.pack(fill="x")
        ttk.Button(self.people_row, text="全部听（原声）", command=lambda: self.choose("all")).pack(side="left", padx=(0, 8))
        ttk.Button(self.people_row, text="全部静音", command=lambda: self.choose("mute")).pack(side="left", padx=(0, 8))
        self.track_area = ttk.Frame(people)
        self.track_canvas = tk.Canvas(self.track_area, height=80, highlightthickness=0, bg='#eef2f6')
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

        file_area = ttk.LabelFrame(main, text="没有耳机也能先试：混合录音", padding=12)
        file_area.pack(fill="x", pady=5)
        self.file_label = ttk.Label(file_area, text="导入 WAV / FLAC / OGG，处理后切换按钮试听同一时间轴。")
        self.file_label.pack(anchor="w", pady=(0, 8))
        file_row = ttk.Frame(file_area)
        file_row.pack(fill="x")
        self.file_button = ttk.Button(file_row, text="导入并处理", command=self.import_file)
        self.file_button.pack(side="left", padx=(0, 8))
        self.play_button = ttk.Button(file_row, text="循环试听", command=self.play_file, state="disabled")
        self.play_button.pack(side="left", padx=(0, 8))
        self.export_button = ttk.Button(file_row, text="导出音轨和报告", command=self.export, state="disabled")
        self.export_button.pack(side="left")
        self.progress = ttk.Progressbar(file_area, maximum=1)
        self.progress.pack(fill="x", pady=(10, 0))

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
                       if box.current() < 0]
            note = ('、'.join(missing) + '设备未选中：请连接设备后刷新，或手动选择。'
                    if missing else '已选好输入和输出。连接或切换耳机后，请刷新设备。')
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
            text = f"说话人 {identity.split('_')[-1]}"
            text += " · 有声音" if identity in active else ""
            if not speaker["ready"]:
                text += " · 确认中"
            self.speaker_labels[identity].config(text=text)
        if speakers:
            self.people_note.config(text="每人独立开关，绿色表示当前选择。“确认中”可能误判；同时最多分离两路人声，背景音暂不能独立分离。")
        else:
            self.people_note.config(text="尚未识别到足够清晰的人声。请查看下方输入电平，并确认麦克风选择。")
        self.refresh_track_controls()

    def start_live(self):
        try:
            if self.file_busy:
                raise ValueError("请等待文件处理结束，或先停止")
            if self.ai.get() and not self.backend:
                raise ValueError("请先加载人声模型，或取消勾选进行原声测试")
            input_id = self.selected_device(self.input_box, self.inputs)
            output_id = self.selected_device(self.output_box, self.outputs)
            self.stop()
            if self.session is not None:
                raise ValueError("上一轮推理正在结束，请稍后开始")
            self.file_packet = self.file_report = None
            self.play_button.config(state="disabled")
            self.export_button.config(state="disabled")
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
            messagebox.showerror("无法开始", str(exc))

    def stop(self):
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

    def import_file(self):
        if not self.backend:
            messagebox.showinfo("需要模型", "请先加载人声模型。")
            return
        if self.file_busy:
            return
        path = filedialog.askopenfilename(filetypes=[("音频", "*.wav *.flac *.ogg"), ("所有文件", "*.*")])
        if not path:
            return
        self.stop()
        if self.session:
            messagebox.showinfo("正在停止", "上一轮推理仍在结束，请稍后再导入。")
            return
        self.reset_people()
        self.file_busy = True
        self.file_cancel = Event()
        self.file_packet = self.file_report = None
        self.file_button.config(state="disabled")
        self.play_button.config(state="disabled")
        self.export_button.config(state="disabled")
        self.file_label.config(text=Path(path).name)
        self.status.config(text="正在处理混合录音…")
        generation = self.generation
        cancel = self.file_cancel
        def work():
            try:
                notify = lambda e: self.events.put({**e, "generation": generation})
                packet, report = process_file(path, self.backend, notify, cancel)
                self.events.put({"type": "file_done", "packet": packet, "report": report, "generation": generation})
            except Exception as exc:
                self.events.put({"type": "file_error", "message": str(exc), "generation": generation})
        Thread(target=work, daemon=True).start()

    def play_file(self):
        if self.file_packet is None:
            return
        try:
            device = self.selected_device(self.output_box, self.outputs)
            self.stop()
            self.player = FilePlayer(self.file_packet, device, self.control)
            self.player.start()
            self.status.config(text="正在循环试听已处理录音 · 切换按钮保持同一播放位置")
        except Exception as exc:
            if self.player is not None:
                self.player.stop()
                self.player = None
            messagebox.showerror("播放失败", str(exc))

    def export(self):
        if self.file_packet is None:
            return
        folder = filedialog.askdirectory(title="选择保存目录（在其内创建独立结果文件夹）")
        if folder:
            destination = Path(folder) / f"selective-hearing-{time.time_ns()}"
            export_result(self.file_packet, self.file_report, destination)
            self.status.config(text=f"已保存到 {destination}")

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
                elif kind == "load_error":
                    self.backend_busy = False
                    self.load_button.config(state="normal")
                    self.model_status.config(text="模型加载失败；原声测试仍可使用。")
                    self.status.config(text=event["message"])
                elif kind == "progress":
                    self.update_people(event["speakers"], event["active"])
                    if "fraction" in event:
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
            self.root.after(100, self.poll)

    def close(self):
        self.alive = False
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
