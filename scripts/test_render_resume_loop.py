"""Automated multi-cycle resume loop: repeatedly upload the latest partial
file, submit, wait for a result, and if still partial, download + re-upload
+ resubmit again -- until the book is fully done or max_cycles is hit.

This both maximizes the chance of naturally coinciding with one of Render's
periodic restarts (further validating the streaming-checkpoint fix) and
drives the real target book to full completion.

Usage:
    python scripts/test_render_resume_loop.py <url> <original_pdf_path> <starting_partial_pdf_path> [max_cycles]
"""
import os
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from playwright.sync_api import sync_playwright


def main():
    url = sys.argv[1]
    pdf_path = sys.argv[2]
    partial_path = sys.argv[3]
    max_cycles = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    log_path = "render_resume_loop_log.txt"
    download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_resume_downloads")
    os.makedirs(download_dir, exist_ok=True)

    import fitz  # noqa: E402

    start = time.time()

    def log(msg):
        line = "[%.1fs] %s" % (time.time() - start, msg)
        try:
            print(line, flush=True)
        except Exception:
            pass
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("=== Render resume-loop test log ===\n")

    def wait_for_status(page, timeout_s=900, poll_s=10, min_next_page=None):
        elapsed = 0
        last_seen = None
        while elapsed < timeout_s:
            page.wait_for_timeout(poll_s * 1000)
            elapsed += poll_s
            body = page.inner_text("body")
            if "已完成全部" in body:
                return "done", None
            if "本次只完成到第" in body:
                marker = "从第 "
                idx = body.find(marker)
                next_page = None
                if idx >= 0:
                    tail = body[idx + len(marker):idx + len(marker) + 10]
                    digits = "".join(ch for ch in tail if ch.isdigit())
                    next_page = int(digits) if digits else None
                if min_next_page is not None and (next_page is None or next_page <= min_next_page):
                    if last_seen != "stale":
                        log("Ignoring stale partial status (next_page=%s <= floor=%s)..."
                            % (next_page, min_next_page))
                        last_seen = "stale"
                else:
                    return "partial", next_page
            idx = body.find("已完成到第")
            if idx >= 0:
                snippet = body[max(0, idx - 2): idx + 30]
                if snippet != last_seen:
                    log("progress (checkpoint): %s" % snippet.replace("\n", " "))
                    last_seen = snippet
                continue
            idx = body.find("正在生成注释")
            if idx >= 0:
                snippet = body[idx: idx + 40]
                if snippet != last_seen:
                    log("progress: %s" % snippet.replace("\n", " "))
                    last_seen = snippet
        return "timeout", None

    current_partial = partial_path

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for cycle in range(1, max_cycles + 1):
            floor = len(fitz.open(current_partial))
            log("=== Cycle %d: resuming from floor=%d pages ===" % (cycle, floor))

            page = browser.new_page(accept_downloads=True)
            page.on("pageerror", lambda exc: log("PAGEERROR: %s" % exc))

            page.goto(url, timeout=120000, wait_until="domcontentloaded")
            for _ in range(20):
                if "loading" not in page.title().lower():
                    break
                log("Cold-start loading screen, waiting...")
                page.wait_for_timeout(6000)
                try:
                    page.reload(timeout=60000, wait_until="domcontentloaded")
                except Exception as exc:
                    log("Reload hiccup (%s), retrying via goto..." % exc)
                    try:
                        page.goto(url, timeout=60000, wait_until="domcontentloaded")
                    except Exception as exc2:
                        log("Goto also failed (%s)..." % exc2)
            page.wait_for_load_state("networkidle", timeout=60000)

            page.locator("#pdf-in input[type='file']").set_input_files(pdf_path)
            page.wait_for_timeout(1000)
            page.locator("#resume-in input[type='file']").set_input_files(current_partial)
            page.wait_for_timeout(1500)
            page.locator("#run-btn").click()
            log("Cycle %d: submitted" % cycle)

            status, next_page = wait_for_status(page, timeout_s=2400, poll_s=10, min_next_page=floor)
            log("Cycle %d result: status=%s next_page=%s" % (cycle, status, next_page))
            page.screenshot(path="render_resume_loop_cycle%d.png" % cycle)

            if status == "done":
                log("SUCCESS: book fully annotated after %d cycle(s)." % cycle)
                download_link = page.locator("#pdf-out a").first
                with page.expect_download(timeout=60000) as dl_info:
                    download_link.click()
                final_path = os.path.join(download_dir, "final_complete.pdf")
                dl_info.value.save_as(final_path)
                final_pages = len(fitz.open(final_path))
                log("Final file saved to %s (%d pages)" % (final_path, final_pages))
                page.close()
                break

            if status != "partial":
                log("Cycle %d did not finish cleanly (status=%s) -- stopping." % (cycle, status))
                page.close()
                break

            download_link = page.locator("#pdf-out a").first
            with page.expect_download(timeout=60000) as dl_info:
                download_link.click()
            new_partial = os.path.join(download_dir, "partial_cycle%d.pdf" % cycle)
            dl_info.value.save_as(new_partial)
            new_floor = len(fitz.open(new_partial))
            log("Cycle %d: downloaded new partial (%d pages)" % (cycle, new_floor))
            current_partial = new_partial
            page.close()
        else:
            log("Reached max_cycles=%d without finishing." % max_cycles)

        browser.close()


if __name__ == "__main__":
    main()
