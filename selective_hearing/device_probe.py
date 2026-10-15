"""Fresh capability-only enumeration, isolated from the app's open audio streams."""
import json
from pathlib import Path
import subprocess
import sys

import sounddevice as sd


def stream_settings(device):
    info = sd.query_devices(device)
    if sd.query_hostapis(info['hostapi'])['name'] == 'Windows WASAPI':
        return sd.WasapiSettings(auto_convert=True)
    return None


def devices(refresh=False):
    if refresh:
        # PortAudio caches endpoints. Never reinitialize with streams open in
        # this process. The monitor instead calls us in a fresh helper process.
        if sd._initialized:
            sd._terminate()
        sd._initialize()
    apis = sd.query_hostapis()
    api_id = next((i for i, api in enumerate(apis) if api['name'] == 'Windows WASAPI'),
                  sd.default.hostapi)
    api = apis[api_id]
    result = []
    for i, device in enumerate(sd.query_devices()):
        if device['hostapi'] != api_id:
            continue
        item = {'id': i, 'name': device['name'], 'api': api['name'],
                'rate': int(device['default_samplerate'])}
        for direction, check in [('input', sd.check_input_settings),
                                 ('output', sd.check_output_settings)]:
            item[direction] = 0
            item[f'default_{direction}'] = i == api[f'default_{direction}_device']
            if not device[f'max_{direction}_channels']:
                continue
            try:
                check(device=i, channels=1, samplerate=16000, dtype='float32',
                      extra_settings=stream_settings(i))
            except (sd.PortAudioError, ValueError):
                continue
            item[direction] = device[f'max_{direction}_channels']
        if item['input'] or item['output']:
            result.append(item)
    return result


def catalog_signature(items):
    # Numeric PortAudio IDs belong to one process/initialization only.
    return sorted((d['api'], d['name'], d['input'], d['output'],
                   d['default_input'], d['default_output'], d.get('rate', 0)) for d in items)


def fresh_devices():
    completed = subprocess.run(
        [sys.executable, '-B', '-m', 'selective_hearing.device_probe'],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True, encoding='utf-8', timeout=10,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
    return json.loads(completed.stdout)


if __name__ == '__main__':
    # ASCII JSON also works when launched by pythonw on non-UTF-8 Windows.
    print(json.dumps(devices(), ensure_ascii=True))
