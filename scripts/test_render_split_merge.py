"""Ad-hoc Playwright script to verify the new split/merge tabs on the
deployed Render webapp, without running the (slow) full annotation
pipeline: split a real large PDF, download the zip, verify part page
counts, then merge those parts back and verify the merged page count
matches the original.

Usage:
    python scripts/test_render_split_merge.py <url> <pdf_path> [max_pages]
"""
import os
import sys
import time
import zipfile

import fitz
from playwright.sync_api import sync_playwright


def wait_for_download(page, log, label, out_elem_id, timeout_s=180):
    deadline = time.time() + timeout_s
    dl_link = None
    scope = page.locator("#%s" % out_elem_id)
    while time.time() < deadline:
        candidate = scope.locator('a[download]').last
        if candidate.count() > 0:
            dl_link = candidate
            break
        page.wait_for_timeout(2000)
    if dl_link is None:
        raise RuntimeError("%s output download link never appeared" % label)
    log("%s download link appeared, clicking..." % label)
    with page.expect_download(timeout=30000) as dl_info:
        dl_link.click(timeout=5000)
    return dl_info.value


def main():
    url = sys.argv[1]
    pdf_path = sys.argv[2]
    max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 400

    with fitz.open(pdf_path) as d:
        total_pages = len(d)
    print("Source PDF has %d pages, splitting at %d pages/part" % (total_pages, max_pages), flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "_split_merge_test")
    os.makedirs(out_dir, exist_ok=True)

    start = time.time()

    def log(msg):
        print("[%.1fs] %s" % (time.time() - start, msg), flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(accept_downloads=True)

        log("Navigating to %s" % url)
        page.goto(url, timeout=120000, wait_until="domcontentloaded")
        for _ in range(15):
            if "loading" not in page.title().lower():
                break
            log("Cold-start loading screen, waiting...")
            page.wait_for_timeout(6000)
            page.reload(timeout=60000, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle", timeout=60000)
        log("Page loaded, title=%r" % page.title())

        # --- Split tab ---
        page.get_by_role("tab", name="拆分大文件").click()
        log("Clicked split tab")
        page.wait_for_timeout(1000)

        split_file_input = page.locator('input[type="file"]').last
        split_file_input.set_input_files(pdf_path)
        log("Uploaded source PDF to split tab")

        max_pages_box = page.get_by_label("每份最多页数")
        max_pages_box.fill(str(max_pages))
        log("Set max pages = %d" % max_pages)

        page.get_by_role("button", name="拆 分").click()
        log("Clicked split button, waiting for zip output...")

        download = wait_for_download(page, log, "split", "split-zip-out")
        zip_path = os.path.join(out_dir, "split_result.zip")
        download.save_as(zip_path)
        log("Downloaded split zip to %s" % zip_path)

        parts_dir = os.path.join(out_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(parts_dir)
        part_paths = sorted(
            os.path.join(parts_dir, n) for n in os.listdir(parts_dir) if n.lower().endswith(".pdf")
        )
        log("Extracted %d parts: %s" % (len(part_paths), [os.path.basename(p) for p in part_paths]))

        part_page_counts = []
        for p in part_paths:
            with fitz.open(p) as d:
                part_page_counts.append(len(d))
        log("Part page counts: %s (sum=%d, expected total=%d)" % (part_page_counts, sum(part_page_counts), total_pages))
        assert sum(part_page_counts) == total_pages, "split page count mismatch"
        assert all(c <= max_pages for c in part_page_counts), "a part exceeds max_pages"
        log("SPLIT VERIFIED OK")

        # --- Merge tab ---
        page.get_by_role("tab", name="合并结果").click()
        log("Clicked merge tab")
        page.wait_for_timeout(1000)

        merge_file_input = page.locator('input[type="file"]').last
        merge_file_input.set_input_files(part_paths)
        log("Uploaded %d parts to merge tab" % len(part_paths))

        page.get_by_role("button", name="合 并").click()
        log("Clicked merge button, waiting for merged output...")

        download2 = wait_for_download(page, log, "merge", "merge-pdf-out")
        merged_path = os.path.join(out_dir, "merged_result.pdf")
        download2.save_as(merged_path)
        log("Downloaded merged PDF to %s" % merged_path)

        with fitz.open(merged_path) as d:
            merged_pages = len(d)
        log("Merged PDF has %d pages (expected %d)" % (merged_pages, total_pages))
        assert merged_pages == total_pages, "merged page count mismatch"
        log("MERGE VERIFIED OK")

        browser.close()

    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
