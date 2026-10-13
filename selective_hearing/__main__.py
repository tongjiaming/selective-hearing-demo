def main():
    import sys
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    parser = argparse.ArgumentParser(description="Selective Hearing 本地实验 Demo")
    parser.add_argument("--devices", action="store_true", help="列出设备，不开始收音")
    parser.add_argument("--file", help="离线处理音频，不播放")
    parser.add_argument("--output", default="results/offline", help="离线处理输出目录")
    args = parser.parse_args()
    if args.devices:
        from .audio import devices
        import json
        print(json.dumps(devices(), ensure_ascii=False, indent=2))
    elif args.file:
        from .models import ModelBackend
        from .audio import process_file, export_result
        backend = ModelBackend()
        packet, report = process_file(args.file, backend, lambda e: print(
            f"{e['fraction']:.0%}  RTF={e['rtf']:.2f}  {e['speakers']}", flush=True))
        export_result(packet, report, args.output)
        print(f"结果已保存：{args.output}")
    else:
        from .ui import main as gui
        gui()


if __name__ == "__main__":
    main()
