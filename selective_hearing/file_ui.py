"""Independent recording-test application. Never starts microphone capture."""
from pathlib import Path
from threading import Event, Thread
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .ui import App
from .core import SpeakerRegistry
from .audio import FilePlayer, process_file, export_result


class FileApp(App):
    def __init__(self, root):
        super().__init__(root, auto_start=False, file_mode=True)

    def build_file_controls(self, main):
        area = ttk.LabelFrame(main, text='4  导入音频文件', padding=12)
        area.pack(fill='x', pady=5)
        self.file_label = ttk.Label(area, text='导入 WAV / FLAC / OGG，处理后切换人物试听。')
        self.file_label.pack(anchor='w', pady=(0, 8))
        row = ttk.Frame(area)
        row.pack(fill='x')
        self.file_button = ttk.Button(row, text='导入并处理', command=self.import_file)
        self.file_button.pack(side='left', padx=(0, 8))
        self.play_button = ttk.Button(row, text='循环试听', command=self.play_file, state='disabled')
        self.play_button.pack(side='left', padx=(0, 8))
        self.export_button = ttk.Button(row, text='导出音轨和报告', command=self.export, state='disabled')
        self.export_button.pack(side='left')
        self.progress = ttk.Progressbar(area, maximum=1)
        self.progress.pack(fill='x', pady=(10, 0))

    def import_file(self):
        if self.pending_delete is not None:
            self.status.config(text='正在删除人物资料，请稍候再导入。')
            return
        if self.profile_error:
            self.status.config(text='人物资料无法加载：' + self.profile_error)
            return
        if not self.backend:
            messagebox.showinfo('需要模型', '请等待模型加载完成。')
            return
        if self.file_busy:
            return
        try:
            window_seconds = self.selected_window_seconds()
        except ValueError as exc:
            self.status.config(text=str(exc))
            return
        path = filedialog.askopenfilename(filetypes=[('音频', '*.wav *.flac *.ogg'), ('所有文件', '*.*')])
        if not path:
            return
        self.stop()
        try:
            self.reset_people()
        except Exception as exc:
            self.status.config(text=str(exc))
            return
        self.file_busy = True
        self.file_cancel = Event()
        self.file_packet = self.file_report = None
        self.file_button.config(state='disabled')
        self.window_box.config(state='disabled')
        self.play_button.config(state='disabled')
        self.export_button.config(state='disabled')
        self.file_label.config(text=Path(path).name)
        self.progress['value'] = 0
        self.status.config(text='正在处理混合录音…')
        generation, cancel = self.generation, self.file_cancel
        def work():
            try:
                notify = lambda e: self.events.put({**e, 'generation': generation})
                packet, report = process_file(path, self.backend, notify, cancel,
                                              registry=SpeakerRegistry(store=self.profile_store),
                                              window_seconds=window_seconds)
                self.events.put({'type': 'file_done', 'packet': packet, 'report': report, 'generation': generation})
            except Exception as exc:
                self.events.put({'type': 'file_error', 'message': str(exc), 'generation': generation})
        Thread(target=work, daemon=True).start()

    def play_file(self):
        if self.file_packet is None or self.pending_delete is not None:
            return
        try:
            device = self.selected_device(self.output_box, self.outputs)
            self.stop()
            self.player = FilePlayer(self.file_packet, device, self.control)
            self.player.start()
            self.status.config(text='正在循环试听录音 · 切换人物保持同一播放位置')
        except Exception as exc:
            if self.player is not None:
                self.player.stop()
                self.player = None
            messagebox.showerror('播放失败', str(exc))

    def named_report(self):
        if not self.save_names():
            raise ValueError('名字未能保存，请检查人物资料提示。')
        included = {speaker['id'] for speaker in self.file_report['speakers']}
        names = {identity: value.get() for identity, value in self.speaker_names.items() if identity in included}
        return {**self.file_report, 'display_names': names,
                'speakers': [{**speaker, 'name': names.get(speaker['id'], speaker['id'])}
                             for speaker in self.file_report['speakers']]}

    def export(self):
        if self.file_packet is None:
            return
        folder = filedialog.askdirectory(title='选择保存目录（在其内创建独立结果文件夹）')
        if folder:
            destination = Path(folder) / f'selective-hearing-{time.time_ns()}'
            export_result(self.file_packet, self.named_report(), destination)
            self.status.config(text=f'已保存到 {destination}')


def main():
    root = tk.Tk()
    FileApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
