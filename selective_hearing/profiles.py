"""Local speaker profiles. SQLite transactions; no audio or candidate storage."""
from contextlib import contextmanager
from pathlib import Path
import math
import sqlite3

import numpy as np

from .core import Speaker, unit
from .model_assets import ASSETS

MODEL_KEY = 'ecapa-fbank80-sentence-mean-v1:' + ASSETS['ecapa.ckpt']['sha256']


def default_profile_path(file_mode=False):
    # Shared user Documents path also works with this machine's desktop launcher.
    return Path.home() / 'Documents' / 'SelectiveHearing' / 'profiles' / (
        'file-speakers.sqlite3' if file_mode else 'live-speakers.sqlite3')


class ProfileStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('人物资料库损坏；请保留原文件并检查，未覆盖资料。')
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            expected = {'schema': '1', 'model': MODEL_KEY}
            for key, value in expected.items():
                db.execute('INSERT OR IGNORE INTO metadata VALUES (?, ?)', (key, value))
                if db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()[0] != value:
                    raise ValueError('人物资料版本或声纹模型不兼容，未覆盖原资料。')
            db.execute('''CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                vector BLOB NOT NULL,
                observations INTEGER NOT NULL CHECK(observations >= 1),
                seconds REAL NOT NULL CHECK(seconds >= 0),
                learned_vector BLOB,
                learned_seconds REAL NOT NULL DEFAULT 0,
                learned_observations INTEGER NOT NULL DEFAULT 0)''')
        self.load_speakers()  # Validate before the application starts capture.

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def number(identity):
        prefix, number = identity.rsplit('_', 1)
        if prefix != 'speaker' or not number.isdecimal() or int(number) < 1:
            raise ValueError('无效人物编号')
        return int(number)

    @staticmethod
    def vector(value):
        value = np.asarray(value, dtype='<f4')
        if value.shape != (192,):
            raise ValueError('声纹维度不兼容；原资料保持不变。')
        return unit(value)

    def load_speakers(self):
        with self.connection() as db:
            rows = db.execute('SELECT id, name, vector, observations, seconds, learned_vector, learned_seconds, learned_observations FROM speakers ORDER BY id').fetchall()
        result = []
        for number, name, blob, observations, seconds, learned_blob, learned_seconds, learned_observations in rows:
            if (number < 1 or not isinstance(name, str) or not name.strip() or len(name) > 40
                    or not isinstance(observations, int) or observations < 1
                    or not math.isfinite(seconds) or seconds < 0
                    or not math.isfinite(learned_seconds) or learned_seconds < 0
                    or not isinstance(learned_observations, int) or learned_observations < 0):
                raise ValueError('人物资料内容异常；原文件保持不变。')
            vector = self.vector(np.frombuffer(blob, dtype='<f4'))
            learned = self.vector(np.frombuffer(learned_blob, dtype='<f4')) if learned_blob is not None else None
            if ((learned is None) != (learned_observations == 0)
                    or (learned is not None and float(learned @ vector) < 0.65 - 1e-6)):
                raise ValueError('学习声纹与初始资料不一致；原文件保持不变。')
            result.append(Speaker(f'speaker_{number}', vector, observations, seconds,
                                  learned, learned_seconds, learned_observations))
        return result

    def summaries(self):
        with self.connection() as db:
            rows = db.execute('SELECT id, name, observations, seconds, learned_observations FROM speakers ORDER BY id').fetchall()
        return [{'id': f'speaker_{n}', 'name': name, 'observations': observations,
                 'seconds': seconds, 'ready': observations >= 2 and seconds >= 1.5, 'saved': True,
                 'learned_observations': learned}
                for n, name, observations, seconds, learned in rows]

    def create(self, vector, observations, seconds):
        vector = self.vector(vector)
        if observations < 1 or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('无效声纹统计')
        with self.connection() as db:
            cursor = db.execute('INSERT INTO speakers (name, vector, observations, seconds) VALUES (?, ?, ?, ?)',
                                ('说话人', vector.astype('<f4').tobytes(), observations, seconds))
            number = cursor.lastrowid
            db.execute('UPDATE speakers SET name=? WHERE id=?', (f'说话人 {number}', number))
        return Speaker(f'speaker_{number}', vector, observations, seconds)

    def record_match(self, speaker, vector, seconds, learn):
        # Serialize read/modify/write across processes without overwriting names.
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT observations, seconds, learned_vector, learned_seconds, learned_observations FROM speakers WHERE id=?',
                             (self.number(speaker.id),)).fetchone()
            if row is None:
                raise ValueError('人物资料已被删除，请停止并重新开始收音。')
            observations, total_seconds, blob, learned_seconds, learned_observations = row
            current = Speaker(speaker.id, speaker.vector, observations, total_seconds,
                              self.vector(np.frombuffer(blob, dtype='<f4')) if blob is not None else None,
                              learned_seconds, learned_observations)
            current.observe(vector, seconds, learn)
            db.execute('UPDATE speakers SET observations=?, seconds=?, learned_vector=?, learned_seconds=?, learned_observations=? WHERE id=?',
                       (current.observations, current.seconds,
                        current.learned_vector.astype('<f4').tobytes() if current.learned_vector is not None else None,
                        current.learned_seconds, current.learned_observations, self.number(speaker.id)))
        speaker.observations, speaker.seconds = current.observations, current.seconds
        speaker.learned_vector = current.learned_vector
        speaker.learned_seconds, speaker.learned_observations = current.learned_seconds, current.learned_observations

    def rename(self, identity, name):
        name = ' '.join(name.split())[:40] or f'说话人 {self.number(identity)}'
        with self.connection() as db:
            if db.execute('UPDATE speakers SET name=? WHERE id=?', (name, self.number(identity))).rowcount != 1:
                raise ValueError('人物资料不存在，名字未保存。')

    def delete(self, identity):
        with self.connection() as db:
            db.execute('PRAGMA secure_delete=ON')
            db.execute('DELETE FROM speakers WHERE id=?', (self.number(identity),))
