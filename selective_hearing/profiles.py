"""Local speaker profiles and bounded segment features; no waveform storage."""
from contextlib import contextmanager
from pathlib import Path
import math
import sqlite3

import numpy as np

from .core import Speaker, unit
from .feature_bank import VoiceSample, MAX_SAMPLES, MAX_PER_SPEAKER, MAX_PENDING
from .model_assets import ASSETS

MODEL_KEY = 'ecapa-fbank80-sentence-mean-v1:' + ASSETS['ecapa.ckpt']['sha256']


def validate_window_seconds(value):
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        raise ValueError('分析窗口请输入 1 到 10 之间的秒数。') from None
    if not math.isfinite(seconds) or not 1 <= seconds <= 10:
        raise ValueError('分析窗口请输入 1 到 10 之间的秒数。')
    return seconds


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
            db.execute('CREATE TABLE IF NOT EXISTS voice_samples ('
                       'id INTEGER PRIMARY KEY AUTOINCREMENT, vector BLOB NOT NULL, seconds REAL NOT NULL, '
                       'speaker_id INTEGER REFERENCES speakers(id) ON DELETE CASCADE, batch_id INTEGER NOT NULL DEFAULT 0)')
            if 'display_number' not in {row[1] for row in db.execute('PRAGMA table_info(speakers)')}:
                db.execute('ALTER TABLE speakers ADD COLUMN display_number INTEGER')
            # Migrate old profiles once, preserving custom names and internal IDs.
            for number, name in db.execute('SELECT id, name FROM speakers WHERE display_number IS NULL ORDER BY id').fetchall():
                display = self.next_display_number(db)
                db.execute('UPDATE speakers SET display_number=?, name=? WHERE id=?',
                           (display, f'说话人 {display}' if name == f'说话人 {number}' else name, number))
            db.execute('CREATE UNIQUE INDEX IF NOT EXISTS speaker_display_number ON speakers(display_number)')
        self.load_speakers()  # Validate before the application starts capture.
        self.load_samples()

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def next_display_number(db):
        used = {row[0] for row in db.execute('SELECT display_number FROM speakers')}
        number = 1
        while number in used:
            number += 1
        return number

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

    def load_samples(self):
        with self.connection() as db:
            rows = db.execute('SELECT id, vector, seconds, speaker_id, batch_id FROM voice_samples ORDER BY id DESC LIMIT ?',
                              (MAX_SAMPLES,)).fetchall()
        result = []
        for number, blob, seconds, speaker, batch_id in reversed(rows):
            if number < 1 or not math.isfinite(seconds) or seconds < .75:
                raise ValueError('人声样本资料异常，未覆盖原资料。')
            result.append(VoiceSample(number, self.vector(np.frombuffer(blob, dtype='<f4')),
                                     seconds, f'speaker_{speaker}' if speaker is not None else None, batch_id))
        return result

    def add_samples(self, entries):
        rows = [(self.vector(v).astype('<f4').tobytes(), seconds,
                 self.number(identity) if identity is not None else None) for v, seconds, identity in entries]
        if any(not math.isfinite(r[1]) or r[1] < .75 for r in rows):
            raise ValueError('无效人声样本')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            batch_id = db.execute('SELECT COALESCE(MAX(batch_id), 0) + 1 FROM voice_samples').fetchone()[0]
            db.executemany('INSERT INTO voice_samples (vector, seconds, speaker_id, batch_id) VALUES (?, ?, ?, ?)',
                           [(*row, batch_id) for row in rows])

    def correct_samples(self, changes):
        with self.connection() as db:
            # Compare-and-set avoids overwriting another app instance's later
            # correction. IDs are allocated by SQLite, never by each process.
            for number, previous, current in changes:
                db.execute('UPDATE voice_samples SET speaker_id=? WHERE id=? AND speaker_id IS ?',
                           (self.number(current), number, self.number(previous) if previous else None))
            # Bounded per-person neighborhoods prevent one talkative speaker
            # from crowding everybody else out of the global feature memory.
            db.execute('DELETE FROM voice_samples WHERE id IN ('
                       'SELECT id FROM (SELECT id, speaker_id, ROW_NUMBER() OVER '
                       '(PARTITION BY speaker_id ORDER BY id DESC) AS rank FROM voice_samples) '
                       'WHERE rank > CASE WHEN speaker_id IS NULL THEN ? ELSE ? END)',
                       (MAX_PENDING, MAX_PER_SPEAKER))
            db.execute('DELETE FROM voice_samples WHERE id NOT IN '
                       '(SELECT id FROM voice_samples ORDER BY id DESC LIMIT ?)', (MAX_SAMPLES,))

    def summaries(self):
        with self.connection() as db:
            rows = db.execute('SELECT id, name, observations, seconds, learned_observations, display_number FROM speakers ORDER BY id').fetchall()
            counts = dict(db.execute('SELECT speaker_id, COUNT(*) FROM voice_samples GROUP BY speaker_id'))
        return [{'id': f'speaker_{n}', 'name': name, 'observations': observations,
                 'seconds': seconds, 'ready': observations >= 2 and seconds >= 1.5, 'saved': True,
                 'learned_observations': learned, 'feature_samples': counts.get(n, 0), 'display_number': display}
                for n, name, observations, seconds, learned, display in rows]

    def window_seconds(self, default):
        with self.connection() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='window_seconds'").fetchone()
        return validate_window_seconds(row[0] if row else default)

    def set_window_seconds(self, value):
        seconds = validate_window_seconds(value)
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('window_seconds', ?)", (str(seconds),))
        return seconds

    def create(self, vector, observations, seconds):
        vector = self.vector(vector)
        if observations < 1 or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('无效声纹统计')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            display = self.next_display_number(db)
            cursor = db.execute('INSERT INTO speakers (name, vector, observations, seconds, display_number) VALUES (?, ?, ?, ?, ?)',
                                (f'说话人 {display}', vector.astype('<f4').tobytes(), observations, seconds, display))
            number = cursor.lastrowid
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
        name = ' '.join(name.split())[:40]
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT display_number FROM speakers WHERE id=?', (self.number(identity),)).fetchone()
            if row is None:
                raise ValueError('人物资料不存在，名字未保存。')
            name = name or f'说话人 {row[0]}'
            if db.execute('UPDATE speakers SET name=? WHERE id=?', (name, self.number(identity))).rowcount != 1:
                raise ValueError('人物资料不存在，名字未保存。')
        return name

    def delete(self, identity):
        with self.connection() as db:
            db.execute('PRAGMA secure_delete=ON')
            db.execute('DELETE FROM speakers WHERE id=?', (self.number(identity),))
