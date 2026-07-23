"""Ad-hoc Playwright script to drive the deployed Render webapp end-to-end
and log progress/console/network activity, to diagnose the page-820 freeze.

Usage:
    python scripts/test_render_webapp.py <url> <pdf_path>
"""
import sys
import time
import json
from playwright.sync_api import sync_playwright


def main():
    url = sys.argv[1]
    pdf_path = sys.argv[2]
    log_path = "render_test_log.txt"

    events = []

    def log(msg):
        line = "[%.1fs] %s" % (time.time() - start, msg)
        print(line, flush=True)
        events.append(line)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("=== Render webapp test log ===\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.on("console", lambda msg: log("CONSOLE[%s]: %s" % (msg.type, msg.text)))
        page.on("pageerror", lambda exc: log("PAGEERROR: %s" % exc))
        page.on("requestfailed", lambda req: log("REQFAILED: %s %s (%s)" % (req.method, req.url, req.failure)))
        page.on("websocket", lambda ws: log("WEBSOCKET opened: %s" % ws.url))
        page.on("close", lambda: log("PAGE CLOSED"))

        def on_response(resp):
            if resp.status >= 400:
                log("HTTP %d: %s" % (resp.status, resp.url))

        page.on("response", on_response)

        log("Navigating to %s" % url)
        page.goto(url, timeout=120000, wait_until="domcontentloaded")
        # Render free tier cold start shows a generic "Application loading" title.
        for _ in range(15):
            if "loading" not in page.title().lower():
                break
            log("Still on Render cold-start loading screen, waiting...")
            page.wait_for_timeout(6000)
            page.reload(timeout=60000, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle", timeout=60000)
        log("Page loaded, title=%r" % page.title())

        page.screenshot(path="render_test_initial.png")
        log("Initial screenshot saved")

        # Find the file upload input and set the PDF file.
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(pdf_path)
        log("File uploaded: %s" % pdf_path)

        page.wait_for_timeout(2000)
        page.screenshot(path="render_test_after_upload.png")

        # Submit button label is "开 始 注 释" (spaced characters).
        submit_btn = page.get_by_role("button", name="注 释")
        if submit_btn.count() == 0:
            submit_btn = page.locator("button", has_text="注").first
        submit_btn.first.click()
        log("Clicked submit button")

        # Poll progress text periodically for up to ~60 minutes.
        last_text = None
        stall_start = None
        max_wait_s = 60 * 60
        poll_interval = 15
        elapsed = 0
        while elapsed < max_wait_s:
            page.wait_for_timeout(poll_interval * 1000)
            elapsed += poll_interval
            try:
                body_text = page.inner_text("body")
            except Exception as exc:
                log("Failed to read body text: %s" % exc)
                body_text = ""

            # Extract any progress-looking substring.
            snippet = None
            for marker in ["正在生成注释", "正在读取词汇", "%"]:
                idx = body_text.find(marker)
                if idx >= 0:
                    snippet = body_text[max(0, idx - 20): idx + 60]
                    break

            if snippet != last_text:
                log("PROGRESS: %s" % (snippet.replace("\n", " ") if snippet else "<no marker found>"))
                last_text = snippet
                stall_start = time.time()
            else:
                stalled_for = time.time() - stall_start if stall_start else 0
                if stalled_for > 0 and int(stalled_for) % 60 < poll_interval:
                    log("STILL AT: %s (stalled %.0fs)" % (snippet, stalled_for))

            if "error" in body_text.lower() or "错误" in body_text or "失败" in body_text:
                log("POSSIBLE ERROR TEXT DETECTED")
                page.screenshot(path="render_test_error.png")

            # Heuristic: consider done if a download link/output file appears.
            if "annotated" in body_text.lower() or "下载" in body_text:
                log("Looks like output may be ready.")
                page.screenshot(path="render_test_done.png")

            # Stop early if stalled for a very long time (30 min) with no change.
            if stall_start and (time.time() - stall_start) > 30 * 60:
                log("STALLED for 30+ minutes with no progress-text change. Stopping.")
                break

        page.screenshot(path="render_test_final.png")
        log("Final screenshot saved. Done polling.")
        browser.close()


if __name__ == "__main__":
    main()
