"""Capture real screenshots of the running web UI (evidence for the report).

    PYTHONPATH=. python scripts/screenshots.py      ->  reports/figures/ui_*.png

Starts the API in a separate process with a temporary database, drives it with headless Chromium
(Playwright) exactly like a user would, and saves what the browser actually renders.
"""
from __future__ import annotations

import http.client
import os
import subprocess
import sys
import tempfile
import time

from playwright.sync_api import sync_playwright

from src import config

PORT = 5078
URL = f"http://127.0.0.1:{PORT}/"
OUT = config.FIG_DIR


def wait_ready():
    for _ in range(100):
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1)
            c.request("GET", "/api/health")
            if c.getresponse().status == 200:
                return
        except OSError:
            time.sleep(0.15)
    raise RuntimeError("server did not start")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="bt60_shots_")
    code = ("from app.api import create_app; "
            f"create_app({{'DB_PATH': r'{tmp}/shots.db'}}).run(host='127.0.0.1', port={PORT}, threaded=True)")
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=config.ROOT,
                            env=dict(os.environ, PYTHONPATH=str(config.ROOT), PYTHONWARNINGS="ignore"),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready()
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1366, "height": 900})
            page.goto(URL)
            page.wait_for_selector("#module-list button")

            # 1. landing page, empty state
            page.screenshot(path=str(OUT / "ui_01_home.png"))

            # 2-4. one prediction per module (typical positive case, then a typical negative case for heart)
            for key, name in (("diabetes", "Diabetes"), ("heart", "Heart Disease"), ("breast", "Breast Cancer")):
                page.click(f'#module-list button[data-key="{key}"]')
                page.click("#btn-pos")
                page.click("#btn-predict")
                page.wait_for_selector(".risk-number")
                page.wait_for_timeout(700)                  # let the ruler animation finish
                page.evaluate("window.scrollTo(0, 0)")
                page.screenshot(path=str(OUT / f"ui_02_{key}_positive.png"), full_page=True)

            page.click('#module-list button[data-key="heart"]')
            page.click("#btn-neg")
            page.click("#btn-predict")
            page.wait_for_selector(".risk-number")
            page.wait_for_timeout(700)
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(OUT / "ui_03_heart_negative.png"), full_page=True)

            # 5. validation feedback: out-of-range glucose + blank required field
            page.click('#module-list button[data-key="diabetes"]')
            page.click("#btn-pos")
            page.fill("#f-Glucose", "9999")
            page.fill("#f-Age", "")
            page.click("#btn-predict")
            page.wait_for_selector(".field.invalid")
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(OUT / "ui_04_validation_errors.png"), full_page=True)

            # 6. optional field left blank -> warning shown
            page.click("#btn-pos")
            page.fill("#f-Insulin", "")
            page.click("#btn-predict")
            page.wait_for_selector(".warn")
            page.wait_for_timeout(700)
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(OUT / "ui_05_missing_value_warning.png"), full_page=True)

            # 7. prediction history (database-backed)
            page.click("#btn-history")
            page.wait_for_selector("#history-body tr")
            page.locator(".history").screenshot(path=str(OUT / "ui_06_history.png"))

            # 8. mobile layout
            mobile = browser.new_page(viewport={"width": 390, "height": 844})
            mobile.goto(URL)
            mobile.wait_for_selector("#module-list button")
            mobile.click("#btn-pos")
            mobile.click("#btn-predict")
            mobile.wait_for_selector(".risk-number")
            mobile.wait_for_timeout(700)
            mobile.evaluate("window.scrollTo(0, 0)")
            mobile.screenshot(path=str(OUT / "ui_07_mobile.png"), full_page=True)
            browser.close()
        print("screenshots written to", OUT)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
