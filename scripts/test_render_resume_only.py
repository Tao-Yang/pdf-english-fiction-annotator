"""Faster follow-up test: reuses the already-downloaded partial.pdf from a
previous run and only exercises the resume (Run2) step, to avoid waiting
~9 minutes for Run1 to reproduce a partial result again.

Usage:
    python scripts/test_render_resume_only.py <url> <original_pdf_path> <partial_pdf_path>
"""
import os
import sys
import time
import traceback

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
    log_path = "render_resume_only_log.txt"

    import fitz  # noqa: E402
    resume_floor = len(fitz.open(partial_path))

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
        fh.write("=== Render resume-only test log (floor=%d) ===\n" % resume_floor)

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
                        log("Ignoring stale partial status (next_page=%s <= floor=%s), "
                            "waiting for a fresh result..." % (next_page, min_next_page))
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

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(accept_downloads=True)
        page.on("pageerror", lambda exc: log("PAGEERROR: %s" % exc))

        log("Navigating to %s" % url)
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
                    log("Goto also failed (%s), waiting and continuing loop..." % exc2)
        page.wait_for_load_state("networkidle", timeout=60000)
        log("Page loaded, title=%r" % page.title())

        main_file_input = page.locator("#pdf-in input[type='file']")
        main_file_input.set_input_files(pdf_path)
        log("Original PDF uploaded: %s" % pdf_path)
        page.wait_for_timeout(1000)

        resume_file_input = page.locator("#resume-in input[type='file']")
        resume_file_input.set_input_files(partial_path)
        log("Resume file uploaded: %s (floor=%d pages)" % (partial_path, resume_floor))
        page.wait_for_timeout(1500)

        submit_btn = page.locator("#run-btn")
        submit_btn.click()
        log("Clicked submit")

        status, next_page = wait_for_status(page, timeout_s=2400, poll_s=10, min_next_page=resume_floor)
        log("Result: status=%s next_page=%s (floor was %d)" % (status, next_page, resume_floor))
        page.screenshot(path="render_resume_only_result.png")

        if status == "done":
            log("SUCCESS: resume continued correctly and finished the whole book.")
            download_link = page.locator("#pdf-out a").first
            with page.expect_download(timeout=60000) as dl_info:
                download_link.click()
            final_path = os.path.join(os.path.dirname(partial_path), "final_complete.pdf")
            dl_info.value.save_as(final_path)
            log("Final file saved to %s (%d pages)" % (final_path, len(fitz.open(final_path))))
        elif status == "partial":
            if next_page > resume_floor:
                log("SUCCESS: resume continued correctly, made real progress "
                    "(%d -> %d), still needs more cycles." % (resume_floor, next_page))
                download_link = page.locator("#pdf-out a").first
                with page.expect_download(timeout=60000) as dl_info:
                    download_link.click()
                new_partial_path = os.path.join(
                    os.path.dirname(partial_path), "partial_next%d.pdf" % next_page)
                dl_info.value.save_as(new_partial_path)
                log("Downloaded new partial to %s (%d pages)" % (
                    new_partial_path, len(fitz.open(new_partial_path))))
            else:
                log("BUG: next_page (%s) did not advance past floor (%d)." % (next_page, resume_floor))
        else:
            log("Did not finish cleanly (status=%s)." % status)

        browser.close()


if __name__ == "__main__":
    main()
