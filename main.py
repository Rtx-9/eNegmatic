"""eNegmatic - entry point.

    python main.py [files or folders...]
"""
import multiprocessing
import sys
import time

SPLASH_MIN_SECONDS = 1.6


def selftest(src: str, out_dir: str) -> int:
    """Headless build check:  main.py --selftest NEGATIVE OUT_DIR

    Decodes and converts a negative in-process (preview path), then exports it
    through the worker-process pool (export path). Exit code 0 = success.
    """
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    from enegmatic.autocrop import detect_frame
    from enegmatic.batch import ExportOptions, export_one
    from enegmatic.core import Settings, analyze, apply_geometry, detect_film_type, render
    from enegmatic.fileio import load_proxy, save_image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proxy, _, _ = load_proxy(src)
    angle, crop = detect_frame(proxy)
    s = Settings(straighten=angle, crop=crop)
    g = apply_geometry(proxy, s)
    s.film_type = detect_film_type(g)
    save_image(str(out / "preview.png"), render(g, s, analyze(g, s)), "png")
    jobs = [ExportOptions(fmt="jpeg", use_subfolder=False, out_dir=str(out), overwrite=True),
            ExportOptions(fmt="tiff", bits=16, color_space="prophoto", suffix="_prophoto",
                          use_subfolder=False, out_dir=str(out), overwrite=True)]
    with ProcessPoolExecutor(max_workers=1) as pool:
        results = [pool.submit(export_one, src, s.to_dict(), o.to_dict()).result() for o in jobs]
    lines = [f"film={s.film_type} angle={angle} crop={crop}"]
    lines += [f"export={written} error={err}" for _, written, err in results]
    (out / "selftest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if all(written and not err for _, written, err in results) else 1


def main():
    # Heavy imports happen after the splash is up, and only in the GUI process
    # (export worker processes spawned on Windows import this module too).
    from PySide6.QtCore import QSettings, QTimer
    from PySide6.QtWidgets import QApplication

    from enegmatic.splash import Splash, app_icon, load_brand_font
    from enegmatic.theme import system_theme

    if sys.platform == "win32":
        # Own taskbar identity, so Windows shows the eNegmatic icon rather than Python's.
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Rtx9.eNegmatic")
        except (AttributeError, OSError):
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("eNegmatic")
    app.setOrganizationName("eNegmatic")
    app.setStyle("Fusion")
    family = load_brand_font()
    app.setWindowIcon(app_icon(family))

    theme = QSettings("eNegmatic", "eNegmatic").value("theme", "system")
    dark = (system_theme() if theme not in ("dark", "light") else theme) == "dark"
    splash = Splash(family, dark)
    splash.show()
    splash.status("Loading imaging libraries…")
    t0 = time.monotonic()

    from enegmatic.mainwindow import MainWindow

    splash.status("Starting…")
    win = MainWindow()

    def reveal():
        win.show()
        win.raise_()
        splash.fade_to(win)
        if len(sys.argv) > 1:
            win.open_paths(sys.argv[1:])

    remaining = SPLASH_MIN_SECONDS - (time.monotonic() - t0)
    QTimer.singleShot(max(0, int(remaining * 1000)), reveal)
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) == 4 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2], sys.argv[3]))
    main()
