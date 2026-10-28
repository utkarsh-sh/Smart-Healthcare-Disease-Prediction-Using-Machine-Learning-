"""Browser end-to-end tests: the real UI, driven by headless Chromium (Playwright).

These tests exist because two UI defects ("null" printed in labels and in the result panel) were invisible to
the API tests and were only caught by looking at a screenshot. They are skipped automatically when Playwright
or its Chromium build is not installed (`pip install playwright && playwright install chromium`).
"""
import http.client
import os
import socket
import subprocess
import sys
import time

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

from src import config


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    port = free_port()
    db = tmp_path_factory.mktemp("ui_db") / "ui.db"
    code = ("from app.api import create_app; "
            f"create_app({{'DB_PATH': r'{db}'}}).run(host='127.0.0.1', port={port}, threaded=True)")
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=config.ROOT,
                            env=dict(os.environ, PYTHONPATH=str(config.ROOT), PYTHONWARNINGS="ignore"),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                c.request("GET", "/api/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                time.sleep(0.15)
        else:
            pytest.fail("server did not start")
        yield f"http://127.0.0.1:{port}/"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as pw:
        try:
            b = pw.chromium.launch()
        except Exception as exc:                       # browser binary missing
            pytest.skip(f"Chromium not available: {exc}")
        yield b
        b.close()


@pytest.fixture()
def page(browser, base_url):
    p = browser.new_page(viewport={"width": 1366, "height": 900})
    p.goto(base_url)
    p.wait_for_selector("#module-list button")
    yield p
    p.close()


def predict_example(page, key, which="#btn-pos"):
    page.click(f'#module-list button[data-key="{key}"]')
    page.click(which)
    page.click("#btn-predict")
    page.wait_for_selector(".risk-number")


def test_modules_are_listed_in_registry_order(page):
    names = page.locator("#module-list strong").all_inner_texts()
    assert names == ["Diabetes", "Heart Disease", "Breast Cancer"]


@pytest.mark.parametrize("key", ["diabetes", "heart", "breast"])
def test_prediction_renders_without_stray_null_text(page, key):
    predict_example(page, key)
    assert page.locator(".risk-number").inner_text().endswith("%")
    assert page.locator(".risk-label").inner_text()
    assert page.locator(".factor").count() == 3
    for region in ("#fields", "#result", "#module-intro"):
        text = page.locator(region).inner_text().split()
        assert "null" not in text and "undefined" not in text, f"stray text in {region}"


def test_typical_positive_scores_higher_than_typical_negative_in_the_ui(page):
    predict_example(page, "heart", "#btn-pos")
    high = int(page.locator(".risk-number").inner_text().rstrip("%"))
    predict_example(page, "heart", "#btn-neg")
    low = int(page.locator(".risk-number").inner_text().rstrip("%"))
    assert high > low


def test_invalid_input_shows_field_errors_and_no_result(page):
    page.click('#module-list button[data-key="diabetes"]')
    page.click("#btn-pos")
    page.fill("#f-Glucose", "9999")
    page.fill("#f-Age", "")
    page.click("#btn-predict")
    page.wait_for_selector(".field.invalid")
    assert page.locator(".field.invalid").count() == 2
    assert "Must be between" in page.locator("#f-Glucose-err").inner_text()
    assert "Required" in page.locator("#f-Age-err").inner_text()
    assert page.locator(".risk-number").count() == 0


def test_blank_optional_field_produces_warning_and_prediction(page):
    page.click('#module-list button[data-key="diabetes"]')
    page.click("#btn-pos")
    page.fill("#f-Insulin", "")
    page.click("#btn-predict")
    page.wait_for_selector(".warn")
    assert "insulin" in page.locator(".warn").inner_text().lower()
    assert page.locator(".risk-number").count() == 1


def test_history_table_shows_the_prediction(page):
    predict_example(page, "breast")
    page.click("#btn-history")
    page.wait_for_selector("#history-body tr")
    assert "Breast Cancer" in page.locator("#history-body").inner_text()


def test_mobile_layout_has_no_horizontal_overflow(browser, base_url):
    p = browser.new_page(viewport={"width": 390, "height": 844})
    p.goto(base_url)
    p.wait_for_selector("#module-list button")
    p.click("#btn-pos")
    p.click("#btn-predict")
    p.wait_for_selector(".risk-number")
    assert p.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    p.close()
