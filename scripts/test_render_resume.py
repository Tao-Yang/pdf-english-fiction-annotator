"""Playwright script to verify the full resume cycle on the deployed webapp:
upload -> wait for the time-budget to produce a partial result -> download the
partial PDF -> re-upload it into the "续传" (resume) file input -> submit again
-> confirm it continues (does not restart from page 1) and, ideally, reaches
full completion.

Usage:
    python scripts/test_render_resume.py <url> <pdf_path>
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright


def main():
    url = sys.argv[1]
    pdf_path = sys.argv[2]
    log_path = "render_resume_log.txt"
    download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_resume_downloads")
    os.makedirs(download_dir, exist_ok=True)

    start = time.time()

    def log(msg):
        line = "[%.1fs] %s" % (time.time() - start, msg)
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("=== Render resume-cycle test log ===\n")

    def wait_for_status(page, timeout_s=900, poll_s=10, min_next_page=None):
        """Poll body text until either a done or paused status message appears.

        If min_next_page is given, a "partial" match is only trusted once its
        next_page value is STRICTLY GREATER than min_next_page -- this guards
        against reading Gradio's stale leftover status text from a *previous*
        run before the new run has actually produced a fresh result.

        Returns ("done", None) or ("partial", next_page:int).
        """
        elapsed = 0
        last_seen = None
        while elapsed < timeout_s:
            page.wait_for_timeout(poll_s * 1000)
            elapsed += poll_s
            body = page.inner_text("body")
            if "已完成全部" in body:
                return "done", None
            if "本次只完成到第" in body:
                # Extract "...继续...从第 N 页继续" -> N
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
            # Log a light heartbeat with the current progress marker, if any.
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

        # --- Run 1: upload the original PDF -----------------------------
        main_file_input = page.locator("#pdf-in input[type='file']")
        main_file_input.set_input_files(pdf_path)
        log("Run1: file uploaded: %s" % pdf_path)
        page.wait_for_timeout(1500)

        submit_btn = page.locator("#run-btn")
        submit_btn.click()
        log("Run1: clicked submit")

        status, next_page = wait_for_status(page, timeout_s=900, poll_s=10)
        log("Run1 result: status=%s next_page=%s" % (status, next_page))
        page.screenshot(path="render_resume_run1_result.png")

        if status != "partial":
            log("Run1 did not produce a partial result (status=%s) -- stopping here." % status)
            browser.close()
            return

        # --- Download the partial file -----------------------------------
        download_link = page.locator("#pdf-out a").first
        with page.expect_download(timeout=60000) as dl_info:
            download_link.click()
        download = dl_info.value
        partial_path = os.path.join(download_dir, "partial.pdf")
        download.save_as(partial_path)
        log("Downloaded partial file to %s" % partial_path)

        import fitz  # noqa: E402
        partial_pages = len(fitz.open(partial_path))
        log("Partial file page count: %d (expected next_page-1=%s)" % (partial_pages, next_page))

        # --- Run 2: upload the partial file into the resume box, resubmit -
        resume_file_input = page.locator("#resume-in input[type='file']")
        resume_file_input.set_input_files(partial_path)
        log("Run2: resume file uploaded: %s" % partial_path)
        page.wait_for_timeout(1500)

        submit_btn.click()
        log("Run2: clicked submit")

        status2, next_page2 = wait_for_status(page, timeout_s=900, poll_s=10, min_next_page=next_page)
        log("Run2 result: status=%s next_page=%s" % (status2, next_page2))
        page.screenshot(path="render_resume_run2_result.png")

        if status2 == "done":
            log("SUCCESS: resume cycle completed the full book.")
        elif status2 == "partial":
            log("Run2 produced ANOTHER partial (expected if remaining pages still exceed the "
                "time budget) -- next_page2=%s. This is still correct behavior, just needs "
                "one more resume cycle." % next_page2)
        else:
            log("Run2 did not finish cleanly (status=%s)." % status2)

        browser.close()


if __name__ == "__main__":
    main()
