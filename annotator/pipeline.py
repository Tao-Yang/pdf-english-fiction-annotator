"""End-to-end annotation pipeline.

For each body page:

1. Extract plain text and pick words/phrases to annotate (``WordSelector``).
2. Locate each surface form's rectangle(s) with ``page.search_for``.
3. Create a new, wider page (original width + margin) and stamp the original
   page content onto it via ``show_pdf_page`` — this preserves the visual
   layout untouched while giving us blank space on the right.
4. Draw a green underline under each annotated word and place a rasterised
   Chinese label in the right margin, aligned to the word's baseline, using a
   simple top-down collision-avoidance layout.
5. Re-attach internal link annotations (``show_pdf_page`` copies pixels, not
   links) so the table of contents keeps working, and carry over the outline /
   bookmarks.

The output PDF has the same page count and navigation as the source.
"""

import gc
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from .config import AnnotationConfig
from .dictionary import Dictionary
from .literary_translation import (
    LiteraryPassage,
    LiteraryTranslator,
    TRANSLATOR_STYLES,
    detect_passages,
)
from .renderer import LabelRenderer
from .selector import Selected, WordSelector


@dataclass
class AnnotateResult:
    """Outcome of a (possibly time-boxed) ``annotate_pdf`` call.

    ``finished`` is ``False`` only when ``time_budget_seconds`` was given and
    the budget ran out before the last page was processed. In that case
    ``output_path`` still points at a valid, openable PDF covering pages
    ``[0, next_start_page)`` -- the caller can hand it to the user and, in a
    later call, pass it back as ``resume_from_path`` (with
    ``config.start_page = next_start_page``) to continue.
    """

    output_path: str
    finished: bool
    next_start_page: int
    total_pages: int


class _Placement:
    """A single margin annotation ready to draw: an ordinary one-line word
    gloss, or (when ``height``/``accent_rgb`` are given) a taller literary-
    translation block. ``selected`` is ``None`` for the latter.
    """

    __slots__ = ("selected", "word_rect", "png", "height", "accent_rgb")

    def __init__(
        self,
        selected: Optional[Selected],
        word_rect: fitz.Rect,
        png: bytes,
        height: Optional[float] = None,
        accent_rgb: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.selected = selected
        self.word_rect = word_rect
        self.png = png
        self.height = height
        self.accent_rgb = accent_rgb


# How many pages to accumulate in memory before flushing the in-progress
# output document to disk and reopening it. Without this, ``out`` (a
# fitz.Document holding every already-built page, including rasterised PNG
# labels) grows roughly linearly with book length and can exceed a few
# hundred MB on a full-length novel -- easily enough to OOM a
# memory-constrained host (e.g. Render's free 512MB tier), which looks to the
# user like the progress bar freezing partway through. Periodically saving
# to a temp file and reopening bounds memory to a roughly constant ceiling
# regardless of how many pages remain.
#
# Measured on the real 875-page target file (web-upload config: dense
# annotation starting at page 1, disk-backed SQLite dictionary): baseline
# settles around 220MB and grows ~1.5MB/page between checkpoints. 80 pages
# between checkpoints keeps the peak comfortably under 400MB (well inside a
# 512MB host) while halving checkpoint save/reopen overhead versus a smaller
# interval, since each checkpoint re-compacts the whole accumulated document.
_CHECKPOINT_EVERY_PAGES = 80

# A resume batch shorter than _CHECKPOINT_EVERY_PAGES (e.g. the last 46 pages
# of a book) would otherwise never reach a single checkpoint before finishing
# -- so a mid-flight platform restart during that final short batch silently
# drops the whole request with nothing delivered, same as if on_checkpoint
# didn't exist at all (confirmed in practice against the live Render app).
# Also checkpoint on elapsed wall-clock time since the last checkpoint (or
# start), whichever comes first, so short trailing batches are covered too.
_CHECKPOINT_EVERY_SECONDS = 60.0


def annotate_pdf(
    input_path: str,
    output_path: Optional[str] = None,
    config: Optional[AnnotationConfig] = None,
    report_path: Optional[str] = None,
    progress: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    resume_from_path: Optional[str] = None,
    time_budget_seconds: Optional[float] = None,
    on_checkpoint: Optional[Callable[[str, int, int], None]] = None,
) -> AnnotateResult:
    """Annotate ``input_path`` and write the result to ``output_path``.

    ``progress_cb``, if given, is called after every page as
    ``progress_cb(pno, total_pages)`` (0-based page index) so a long-running
    caller (e.g. the Gradio webapp) can surface real per-page progress
    instead of appearing to hang on a single big book.

    ``resume_from_path``, if given, is an already-annotated partial PDF from
    a previous, time-boxed call (see below) -- its pages are kept as-is and
    only pages from ``len(resume_from_path's pages)`` onward are (re)built.

    ``time_budget_seconds``, if given, stops the run soon after this many
    seconds have elapsed (checked once per page, so the actual overrun is at
    most one page's processing time) instead of running until every page is
    done. This exists for hosts that periodically restart a long-running
    process outside the application's control (observed on Render's free
    tier: roughly every 15-25 minutes, seemingly regardless of code or
    traffic -- most likely severe CPU-quota throttling causing the
    platform's own health check to time out). A single request annotating a
    several-hundred-page book can never reliably finish in one shot on such
    a host; budgeting well under the observed restart interval and handing
    back a valid partial PDF plus a precise resume point lets the caller
    (e.g. the webapp) turn "silently killed with nothing to show" into
    "download what's done so far, then continue".

    ``on_checkpoint``, if given, is called as ``on_checkpoint(ckpt_path, pno,
    total_pages)`` right after every periodic memory-bounding checkpoint save
    (see ``_CHECKPOINT_EVERY_PAGES`` and ``_CHECKPOINT_EVERY_SECONDS`` -- a
    checkpoint fires on whichever of the two is reached first, so a resume
    batch shorter than ``_CHECKPOINT_EVERY_PAGES`` still gets at least one)
    -- i.e. well before ``time_budget_seconds``
    would otherwise return a partial result. This exists because the same
    host that periodically restarts this process outside application control
    (see ``time_budget_seconds`` above) can do so at *any* point, including
    mid-request -- which silently drops the connection with nothing ever
    delivered to the caller, no matter how large ``time_budget_seconds`` is.
    A caller that copies ``ckpt_path`` elsewhere on every callback (the file
    is reused/overwritten roughly one more checkpoint interval later, so it
    must be copied promptly) and hands it to the user immediately -- e.g. by
    ``yield``-ing it as an incremental Gradio update -- bounds the worst-case
    lost progress to one checkpoint interval instead of the whole request.

    Returns an ``AnnotateResult``. When ``time_budget_seconds`` is ``None``
    (the default), the run always goes to completion and
    ``result.finished`` is always ``True``.
    """
    config = config or AnnotationConfig()
    if output_path is None:
        stem, _ext = os.path.splitext(input_path)
        output_path = "%s-annotated-%s.pdf" % (stem, config.cefr_level.upper())

    dictionary = Dictionary(config.ecdict_path, config.historical_glossary_path)
    selector = WordSelector(config, dictionary)
    renderer = LabelRenderer(config)
    # Optional "master translator" mode (see annotator.literary_translation):
    # off unless explicitly enabled, and a no-op if it can't reach the API
    # (missing key / network failure) -- never blocks the ordinary
    # word-gloss pipeline.
    literary_translator = (
        LiteraryTranslator(config) if config.enable_literary_translation else None
    )

    src = fitz.open(input_path)
    out = fitz.open(resume_from_path) if resume_from_path else fitz.open()
    # If resuming, these pages are already fully annotated (they came from a
    # previous, time-boxed call) -- skip rebuilding them and just append new
    # ones after them.
    resume_start = len(out) if resume_from_path else 0
    if resume_from_path:
        # A late-book resume file (e.g. 800+ already-annotated pages, tens
        # of MB of vector labels/underlines) measurably spikes RSS the
        # moment it's opened -- confirmed live on Render's free tier (512MB
        # limit) via its own memory-metrics API: idle ~160-190MB, jumping to
        # 400-520MB within ~1 minute of a resume request starting, then the
        # container gets OOM-killed before a single checkpoint has a chance
        # to fire (indistinguishable in the app's own logs from the
        # platform's unrelated periodic restarts). fitz.TOOLS.store_shrink
        # right after opening it evicts whatever MuPDF cached just parsing
        # the file, trimming peak memory before any new work begins.
        fitz.TOOLS.store_shrink(100)
        gc.collect()
    # Ping-pong temp files derived from output_path so concurrent runs (e.g.
    # multiple webapp requests, each with their own unique output_path) never
    # collide on the same checkpoint file.
    ckpt_paths = [output_path + ".ckpt0.tmp", output_path + ".ckpt1.tmp"]
    ckpt_toggle = 0
    pages_since_checkpoint = 0
    t_start = time.monotonic()
    t_last_checkpoint = t_start
    # On a resume, force a checkpoint after just the *first* newly-built
    # page instead of waiting for the normal 80-page/60s thresholds. This
    # bounds how long the process holds both the freshly-loaded, already
    # bulky ``out`` document *and* the newly annotated page in memory at
    # once to a single page, and the checkpoint's save+reopen cycle
    # (below) compacts ``out`` back down immediately afterwards --
    # directly targeting the early-resume memory spike described above.
    force_next_checkpoint = resume_from_path is not None

    stats = {
        "input": input_path,
        "output": output_path,
        "pages": len(src),
        "pages_annotated": 0,
        "annotations": 0,
        "cefr_level": config.cefr_level.upper(),
    }

    finished = True
    next_start_page = stats["pages"]

    try:
        for pno in range(len(src)):
            if pno < resume_start:
                continue
            src_page = src[pno]
            page_doc, annotated = _build_page(
                src_page, selector, renderer, config, pno,
                literary_translator=literary_translator,
            )
            # Import the finished page into the output document.
            out.insert_pdf(page_doc, from_page=0, to_page=0)
            page_doc.close()
            # Force MuPDF to release its internal store (font/image/glyph
            # cache) after every page. This native store is not fully
            # reclaimed by Document.close() alone and otherwise grows
            # across the many repeated fitz.open()/close() cycles in this
            # single long-running process -- observed in practice as the
            # progress bar going from a steady ~1-2s/page cadence to a
            # permanent-looking freeze around page ~830/875 despite no
            # single page being inherently slow. store_shrink(100) is cheap
            # (just evicts cached objects) relative to building a page, so
            # doing it every page bounds this cost to a per-process
            # constant regardless of book length.
            fitz.TOOLS.store_shrink(100)

            if annotated:
                stats["pages_annotated"] += 1
                stats["annotations"] += annotated
            if progress and (pno % 25 == 0 or pno == len(src) - 1):
                print(
                    "  page %d/%d  (annotated pages: %d, notes: %d)"
                    % (pno + 1, len(src), stats["pages_annotated"], stats["annotations"])
                )
            if progress_cb:
                progress_cb(pno, len(src))

            is_last_page = pno == len(src) - 1

            if (
                time_budget_seconds is not None
                and not is_last_page
                and (time.monotonic() - t_start) >= time_budget_seconds
            ):
                finished = False
                next_start_page = pno + 1
                break

            pages_since_checkpoint += 1
            should_checkpoint = (
                pages_since_checkpoint >= _CHECKPOINT_EVERY_PAGES
                or (
                    pages_since_checkpoint > 0
                    and (time.monotonic() - t_last_checkpoint) >= _CHECKPOINT_EVERY_SECONDS
                )
                or force_next_checkpoint
            )
            if should_checkpoint and not is_last_page:
                force_next_checkpoint = False
                # NOTE: intentionally *not* using ``garbage=4`` here (unlike
                # the final save below). ``garbage=4`` makes MuPDF do a full
                # duplicate-object scan/renumber across the whole document,
                # which is the most expensive garbage-collection level and
                # scales badly with page/object count -- and this call is
                # only an intermediate checkpoint (immediately reopened),
                # not the final deliverable, so a fully garbage-collected
                # file buys nothing here. On a CPU-throttled host (e.g.
                # Render's free tier, which grants only a fraction of one
                # vCPU), this made each checkpoint take long enough to look
                # exactly like the progress bar permanently freezing --
                # reproduced in practice at the very first two checkpoint
                # boundaries (page ~80 and ~160) on the real 875-page target
                # book. Dropping ``garbage`` (default 0, no object scan)
                # keeps just the compression (``deflate``) benefit while
                # making the save O(pages) instead of touching every object
                # for deduplication.
                ckpt_path = ckpt_paths[ckpt_toggle]
                out.save(ckpt_path, deflate=True)
                out.close()
                out = fitz.open(ckpt_path)
                # Reopening the checkpoint we *just* saved is exactly as
                # expensive, memory-wise, as the initial resume-file open
                # above (it's the same large, already-annotated document,
                # freshly re-parsed) -- confirmed live: this reopen alone
                # spiked RSS to ~450MB and triggered an OOM kill on Render's
                # free tier even with the resume-open shrink already in
                # place, immediately after the very first forced checkpoint
                # delivered successfully. Shrink here too, for the same
                # reason.
                fitz.TOOLS.store_shrink(100)
                gc.collect()
                ckpt_toggle = 1 - ckpt_toggle
                pages_since_checkpoint = 0
                t_last_checkpoint = time.monotonic()
                if on_checkpoint:
                    on_checkpoint(ckpt_path, pno, len(src))

        if finished:
            # Preserve bookmarks / outline.
            toc = src.get_toc(simple=False)
            if toc:
                out.set_toc(toc)
            out.save(output_path, garbage=4, deflate=True)
        else:
            # Partial result: this file will be resumed and re-saved fully
            # later, so skip the expensive garbage=4 dedup pass (same
            # reasoning as the periodic checkpoints above) -- just make sure
            # it is a valid, complete-so-far PDF the user can download now.
            out.save(output_path, deflate=True)
        out.close()
    finally:
        src.close()
        for ckpt_path in ckpt_paths:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    if literary_translator is not None:
        stats["literary_translations"] = (
            config.literary_max_total - literary_translator.remaining
        )

    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)

    if finished:
        print(
            "Done: %d/%d pages annotated, %d notes -> %s"
            % (stats["pages_annotated"], stats["pages"], stats["annotations"], output_path)
        )
    else:
        print(
            "Time budget reached: completed pages up to %d/%d -> partial result %s "
            "(resume from page %d)"
            % (next_start_page, stats["pages"], output_path, next_start_page + 1)
        )
    return AnnotateResult(
        output_path=output_path,
        finished=finished,
        next_start_page=next_start_page,
        total_pages=stats["pages"],
    )


# --- Parallel, chunked variant -------------------------------------------
#
# Splits the book into ``chunk_pages``-page slices and annotates each slice
# in its own worker process, then merges the finished slices back into one
# output PDF in original page order. Two independent benefits:
#
# 1. Speed: multiple chunks are annotated concurrently instead of one page
#    at a time in a single process (real parallelism, since NLTK/wordfreq
#    word-selection is plain CPU-bound Python and would not benefit from
#    threading due to the GIL).
# 2. Memory: each worker only ever holds a small (``chunk_pages``-page)
#    fitz.Document in memory, so there is no need for the sequential path's
#    mid-run checkpoint/reopen dance to bound memory on a long book.
#
# The trade-off is that every worker *process* pays its own one-time cost of
# loading NLTK data, wordfreq, and opening the dictionary (~150-200MB
# measured on the real target book) -- so ``max_workers`` should stay
# conservative on memory-constrained hosts (e.g. Render's free 512MB tier).
# ``_init_worker`` amortizes that cost across every chunk routed to a given
# worker process (built once, not once per chunk).

_worker_config: Optional[AnnotationConfig] = None
_worker_selector: Optional[WordSelector] = None
_worker_renderer: Optional[LabelRenderer] = None
_worker_literary_translator: Optional[LiteraryTranslator] = None


def _init_worker(config: AnnotationConfig) -> None:
    """Build the per-process Dictionary/Selector/Renderer, once.

    Used as the ``ProcessPoolExecutor`` initializer, so it always runs
    *inside* the worker process, after it has started, and only builds this
    process's own independent copy of the NLTK/wordfreq/dictionary state.

    Earlier this also pre-built the same state in the *parent* process
    before forking, hoping fork-based hosts (Linux) would let workers
    inherit it via copy-on-write instead of loading it themselves. That
    caused a worse regression: forking a live, multi-threaded server
    process (the webapp's request handler runs on one of several threads)
    can inherit half-acquired internal locks from other still-running
    threads, which then deadlock forever in the child with no exception
    raised -- observed as the progress bar getting stuck at the initial 40%
    marker with not even one chunk ever completing. Building fresh, per-
    process state only after the worker process has fully started (as done
    here) avoids that hazard; the memory cost is paid independently by each
    worker, which is why ``_safe_max_workers`` below budgets for the full
    per-worker cost rather than assuming any sharing.
    """
    global _worker_config, _worker_selector, _worker_renderer, _worker_literary_translator
    _worker_config = config
    dictionary = Dictionary(config.ecdict_path, config.historical_glossary_path)
    _worker_selector = WordSelector(config, dictionary)
    _worker_renderer = LabelRenderer(config)
    # Each worker process gets its own independent translation budget (see
    # ``config.literary_max_total``) -- with several concurrent workers the
    # effective run-wide cap is therefore an approximation (roughly
    # ``literary_max_total`` per worker, not per run), which is an
    # acceptable trade-off for an optional, opt-in enhancement given that
    # ``ProcessPoolExecutor`` workers don't share Python state to coordinate
    # a single global counter.
    _worker_literary_translator = (
        LiteraryTranslator(config) if config.enable_literary_translation else None
    )


# --- Memory-aware worker cap ---------------------------------------------
#
# Letting a caller (e.g. the webapp, via ANNOTATOR_MAX_WORKERS) request more
# workers than the host can actually afford leads to the whole container
# getting OOM-killed by the platform -- which looks to the user like the
# progress bar silently freezing partway through (the request never comes
# back because the process that was going to answer it no longer exists).
# Rather than trust the caller, clamp to a conservative estimate of what
# fits, assuming each worker independently pays the full NLTK/wordfreq/
# dictionary load cost (~150-200MB measured on the real target book).

_MEMORY_PER_WORKER_MB = 220
_BASELINE_MEMORY_MB = 150  # main process, before any worker is started
_MEMORY_SAFETY_MARGIN = 0.8  # only plan against 80% of the detected limit


def _detect_memory_limit_bytes() -> Optional[int]:
    """Best-effort container/host memory limit, or ``None`` if unknown.

    Checks cgroup v2, then cgroup v1 (how Docker/Render/HF Spaces/Railway
    enforce a container's memory limit on Linux), then falls back to total
    physical RAM reported by the OS.
    """
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 reports a huge sentinel (close to 2**63) when unlimited.
        if 0 < value < (1 << 62):
            return value
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
    if page_size > 0 and phys_pages > 0:
        return page_size * phys_pages
    return None


def _detect_cpu_quota() -> Optional[float]:
    """Best-effort container CPU quota (in whole-core units), or ``None``.

    Platforms like Render/Railway/HF Spaces commonly grant free/cheap tiers
    only a *fraction* of one vCPU (e.g. 0.1-0.5 cores) via a cgroup CPU
    quota, even though ``os.cpu_count()`` still reports the host's full
    core count. Spawning as many worker *processes* as ``os.cpu_count()``
    suggests, when the container is actually only entitled to a sliver of
    one core, causes those processes to fiercely contend for the same tiny
    slice of CPU time -- the resulting scheduling/throttling overhead can
    make the whole run dramatically slower than running sequentially,
    which looks to the user exactly like the progress bar freezing.
    """
    # cgroup v2: "$MAX $PERIOD" in microseconds, or "max $PERIOD" if unlimited.
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota_str, period_str = fh.read().split()
        if quota_str != "max":
            quota, period = int(quota_str), int(period_str)
            if period > 0:
                return quota / period
    except (OSError, ValueError):
        pass
    # cgroup v1: separate quota/period files; quota of -1 means unlimited.
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def _safe_max_workers(requested: int) -> int:
    """Clamp ``requested`` worker count to what the host can safely afford.

    Falls back to ``requested`` unchanged if neither the memory limit nor
    the CPU quota can be detected (e.g. non-Linux dev machines), since in
    that case we have no better information than what the caller asked for.
    """
    if requested <= 1:
        return max(1, requested)

    safe = requested

    limit_bytes = _detect_memory_limit_bytes()
    if limit_bytes is not None:
        limit_mb = limit_bytes / (1024 * 1024)
        budget_mb = limit_mb * _MEMORY_SAFETY_MARGIN - _BASELINE_MEMORY_MB
        safe = min(safe, 1 if budget_mb <= 0 else max(1, int(budget_mb // _MEMORY_PER_WORKER_MB)))

    cpu_quota = _detect_cpu_quota()
    if cpu_quota is not None:
        # A container with < 1 full core to itself gains nothing from
        # extra worker processes -- they'd just contend for the same
        # sliver of CPU time. Only allow >1 worker once the quota covers
        # at least that many whole cores.
        safe = min(safe, max(1, int(cpu_quota)))

    return max(1, safe)


def _annotate_chunk(
    args: Tuple[str, int, int, str]
) -> Tuple[int, int, int, int]:
    """Annotate pages ``[page_start, page_end)`` of ``input_path``.

    Writes the resulting slice to ``chunk_out_path`` and returns
    ``(page_start, page_end, pages_annotated, annotations)``. Runs inside a
    worker process; relies on the globals set by :func:`_init_worker`.
    """
    input_path, page_start, page_end, chunk_out_path = args
    config = _worker_config
    selector = _worker_selector
    renderer = _worker_renderer
    literary_translator = _worker_literary_translator

    src = fitz.open(input_path)
    out = fitz.open()
    pages_annotated = 0
    annotations = 0
    try:
        for pno in range(page_start, page_end):
            page_doc, annotated = _build_page(
                src[pno], selector, renderer, config, pno,
                literary_translator=literary_translator,
            )
            out.insert_pdf(page_doc, from_page=0, to_page=0)
            page_doc.close()
            # See the matching comment in annotate_pdf(): bound MuPDF's
            # native store growth across repeated open/close cycles even
            # within a single chunk/worker.
            fitz.TOOLS.store_shrink(100)
            if annotated:
                pages_annotated += 1
                annotations += annotated
        out.save(chunk_out_path, garbage=4, deflate=True)
    finally:
        out.close()
        src.close()
    return page_start, page_end, pages_annotated, annotations


def annotate_pdf_parallel(
    input_path: str,
    output_path: Optional[str] = None,
    config: Optional[AnnotationConfig] = None,
    report_path: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    chunk_pages: int = 10,
    max_workers: Optional[int] = None,
) -> str:
    """Annotate ``input_path`` by processing ``chunk_pages``-page slices
    concurrently across worker processes, then merging them back into a
    single ``output_path`` PDF in original page order.

    ``progress_cb(pages_done, total_pages)`` is called as each chunk
    finishes (coarser-grained than :func:`annotate_pdf`, which reports after
    every single page, since whole chunks complete at a time and may finish
    out of order).

    ``max_workers`` defaults to ``min(4, os.cpu_count() or 1)``, then is
    further clamped by :func:`_safe_max_workers` to whatever the host's
    detected memory limit (cgroup limit on Linux, else total RAM) can
    actually afford -- see that function for the reasoning. Each worker
    independently pays the NLTK/wordfreq/dictionary load cost (built lazily
    inside the worker process itself, by :func:`_init_worker`, once the
    ``ProcessPoolExecutor`` has actually started it) rather than sharing it
    with the parent; without this cap a caller requesting more workers than
    the host can hold risks the whole container getting OOM-killed mid-run
    (which looks like the progress bar silently freezing, since the process
    that would answer the request no longer exists).
    """
    config = config or AnnotationConfig()
    if output_path is None:
        stem, _ext = os.path.splitext(input_path)
        output_path = "%s-annotated-%s.pdf" % (stem, config.cefr_level.upper())
    if chunk_pages < 1:
        raise ValueError("chunk_pages must be >= 1")
    max_workers = max_workers or min(4, os.cpu_count() or 1)
    max_workers = _safe_max_workers(max_workers)

    src = fitz.open(input_path)
    total_pages = len(src)
    toc = src.get_toc(simple=False)
    src.close()

    chunk_ranges = [
        (start, min(start + chunk_pages, total_pages))
        for start in range(0, total_pages, chunk_pages)
    ]

    work_dir = tempfile.mkdtemp(prefix="annotator-chunks-")
    chunk_paths = {
        start: os.path.join(work_dir, "chunk-%06d-%06d.pdf" % (start, end))
        for start, end in chunk_ranges
    }

    stats = {
        "input": input_path,
        "output": output_path,
        "pages": total_pages,
        "pages_annotated": 0,
        "annotations": 0,
        "cefr_level": config.cefr_level.upper(),
    }

    try:
        pages_done = 0
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(config,),
                # Recycle each worker process after a bounded number of
                # chunks. Long-running worker processes accumulate leaked
                # native (C-level) state inside PyMuPDF/MuPDF across many
                # repeated fitz.open()/close() cycles -- font/colorspace
                # caches and similar internals are not fully released by
                # ``Document.close()``. On a long book with few workers
                # (e.g. a memory/CPU-constrained host clamped to
                # ``max_workers=1``, which then handles *every* chunk in a
                # single process), this leak compounds over hundreds of
                # chunks until the process becomes slow enough to look like
                # a hang to the caller -- observed in practice as the
                # progress bar freezing partway through a long book despite
                # each individual chunk being cheap. Restarting the worker
                # periodically bounds how much of that native state can
                # accumulate in any one process. ``max_tasks_per_child``
                # requires Python >= 3.11 (this app's Docker image pins
                # exactly that), so no availability check is needed here.
                #
                # IMPORTANT: this must run with the "spawn" (or
                # "forkserver") start method, never the POSIX default
                # "fork". When a worker recycles mid-run under "fork",
                # CPython replaces it by fork()-ing a new child *after* the
                # executor's background management thread is already
                # running, which is exactly the classic "fork a
                # multi-threaded process" deadlock hazard (see CPython
                # gh-90622 / gh-115634 -- an upstream fix for this specific
                # max_tasks_per_child + fork combination only landed in
                # very recent CPython, long after 3.11). We hit this in
                # production: the webapp hung permanently (not just
                # slowly) partway through a long book once
                # ``max_tasks_per_child`` started actually recycling a
                # worker, reproducing the exact silent-freeze symptom this
                # parameter was meant to fix in the first place. "spawn"
                # starts each worker via a fresh interpreter (no fork)
                # instead, which sidesteps that deadlock class entirely.
                # Each worker already pays its own NLTK/wordfreq/dictionary
                # load cost inside ``_init_worker`` regardless of start
                # method (no copy-on-write sharing is attempted), so
                # switching away from "fork" costs only slightly slower
                # process startup, not extra steady-state memory.
                mp_context=multiprocessing.get_context("spawn"),
                max_tasks_per_child=20,
            ) as pool:
                futures = {
                    pool.submit(
                        _annotate_chunk, (input_path, start, end, chunk_paths[start])
                    ): (start, end)
                    for start, end in chunk_ranges
                }
                for future in as_completed(futures):
                    start, end = futures[future]
                    _, _, pages_annotated, annotations = future.result()
                    stats["pages_annotated"] += pages_annotated
                    stats["annotations"] += annotations
                    pages_done += end - start
                    if progress_cb:
                        progress_cb(pages_done - 1, total_pages)
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "A worker process crashed (most likely out of memory) after "
                "annotating %d/%d pages. Try lowering ANNOTATOR_MAX_WORKERS "
                "and/or ANNOTATOR_CHUNK_PAGES." % (pages_done, total_pages)
            ) from exc

        # Merge chunks back in original page order.
        out = fitz.open()
        for start, end in chunk_ranges:
            chunk_doc = fitz.open(chunk_paths[start])
            out.insert_pdf(chunk_doc)
            chunk_doc.close()
        if toc:
            out.set_toc(toc)
        out.save(output_path, garbage=4, deflate=True)
        out.close()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)

    print(
        "Done: %d/%d pages annotated, %d notes -> %s"
        % (stats["pages_annotated"], stats["pages"], stats["annotations"], output_path)
    )
    return output_path


def _build_page(
    src_page: fitz.Page,
    selector: WordSelector,
    renderer: LabelRenderer,
    config: AnnotationConfig,
    pno: int,
    literary_translator: Optional[LiteraryTranslator] = None,
) -> Tuple[fitz.Document, int]:
    """Return a single-page document containing the widened, annotated page.

    The *document* (not the page) is returned so the caller holds a strong
    reference; ``page.parent`` is only a weak proxy and the page would be
    orphaned if the document were garbage-collected.
    """
    src_rect = src_page.rect
    new_w = src_rect.width + config.margin_width
    new_h = src_rect.height

    tmp = fitz.open()
    page = tmp.new_page(width=new_w, height=new_h)

    # Stamp the original page pixels onto the left region.
    page.show_pdf_page(
        fitz.Rect(0, 0, src_rect.width, src_rect.height),
        src_page.parent,
        src_page.number,
    )

    # Re-attach internal links so navigation keeps working.
    _copy_links(src_page, page)

    if pno < config.start_page:
        return tmp, 0

    text = src_page.get_text("text")
    selections = selector.select_from_text(text)
    placements: List[_Placement] = []
    if selections:
        placements.extend(_locate(src_page, selections, renderer, config))

    if literary_translator is not None and config.enable_literary_translation:
        passages = detect_passages(text, config)
        if passages:
            placements.extend(
                _locate_literary(
                    src_page, passages, literary_translator, renderer, config
                )
            )

    if not placements:
        return tmp, 0

    # Top-down order for stable collision avoidance across both word glosses
    # and (taller) literary-translation blocks.
    placements.sort(key=lambda p: p.word_rect.y0)
    _draw(page, placements, config, src_rect.width)
    return tmp, len(placements)


def _locate(
    src_page: fitz.Page,
    selections: List[Selected],
    renderer: LabelRenderer,
    config: AnnotationConfig,
) -> List[_Placement]:
    placements: List[_Placement] = []
    for sel in selections:
        rect = _search(src_page, sel.surface)
        if rect is None:
            continue
        label_w = config.margin_width - 8
        png = renderer.render(sel.gloss, label_w, config.label_height)
        placements.append(_Placement(sel, rect, png))
    # Top-down order for stable collision avoidance.
    placements.sort(key=lambda p: p.word_rect.y0)
    return placements


def _locate_literary(
    src_page: fitz.Page,
    passages: List[LiteraryPassage],
    translator: LiteraryTranslator,
    renderer: LabelRenderer,
    config: AnnotationConfig,
) -> List[_Placement]:
    """Translate and lay out detected long-sentence/poem passages.

    Bounded by ``config.literary_max_per_page`` (this page) and the
    translator's own remaining run-wide budget
    (``config.literary_max_total``). Any passage that can't be located on
    the page, or fails to translate, is silently skipped -- this is an
    additive enhancement, never a hard requirement.
    """
    placements: List[_Placement] = []
    used = 0
    for passage in passages:
        if used >= config.literary_max_per_page or translator.remaining <= 0:
            break
        rect = _search(src_page, passage.anchor_head)
        if rect is None:
            continue
        translation = translator.translate(passage)
        if not translation:
            continue
        header = "\u3010%s\u8bd1\u3011" % TRANSLATOR_STYLES[passage.translator_key]["label"]
        label_w = config.margin_width - 8
        height = max(
            config.label_height, renderer.measure_block(header, translation, label_w)
        )
        png = renderer.render_block(
            header,
            translation,
            label_w,
            height,
            config.literary_accent_hex,
            config.literary_box_hex,
        )
        placements.append(
            _Placement(
                None, rect, png, height=height, accent_rgb=config.literary_accent_rgb_pdf
            )
        )
        used += 1
    return placements


def _search(page: fitz.Page, surface: str) -> Optional[fitz.Rect]:
    for variant in _quote_variants(surface):
        hits = page.search_for(variant)
        if hits:
            return hits[0]
    return None


def _quote_variants(surface: str) -> List[str]:
    variants = [surface]
    if "'" in surface:
        variants.append(surface.replace("'", "\u2019"))
    if "\u2019" in surface:
        variants.append(surface.replace("\u2019", "'"))
    return variants


def _draw(
    page: fitz.Page,
    placements: List[_Placement],
    config: AnnotationConfig,
    content_width: float,
) -> None:
    margin_x = content_width + 4
    label_w = config.margin_width - 8
    last_bottom = -1e9

    for p in placements:
        rect = p.word_rect
        # Word glosses use the green scheme; literary-translation blocks
        # (``p.accent_rgb`` set) use the distinct burgundy accent so the two
        # annotation kinds are visually distinguishable at a glance.
        color = p.accent_rgb or config.green_rgb_pdf
        height = p.height or config.label_height
        # Underline under the source word/anchor.
        y = rect.y1 + 0.5
        page.draw_line(
            fitz.Point(rect.x0, y),
            fitz.Point(rect.x1, y),
            color=color,
            width=config.underline_width,
        )

        # Vertical position for the label: aligned to the word, pushed down to
        # avoid overlapping the previous label.
        top = max(rect.y0, last_bottom + config.label_gap)
        bottom = top + height
        if bottom > page.rect.height - 4:
            bottom = page.rect.height - 4
            top = bottom - height
        last_bottom = bottom

        box = fitz.Rect(margin_x, top, margin_x + label_w, bottom)

        # Restore the visual leader used by the original annotated edition.
        # A short elbow leaves the underline, then points to the vertical
        # centre of the margin label. Draw it before the PNG so its endpoint
        # tucks neatly underneath the label's edge marker.
        label_y = (top + bottom) / 2.0
        elbow_x = min(content_width - 2, max(rect.x1 + 4, content_width - 18))
        page.draw_polyline(
            [
                fitz.Point(rect.x1, y),
                fitz.Point(elbow_x, y),
                fitz.Point(content_width + 2, label_y),
                fitz.Point(margin_x + 1, label_y),
            ],
            color=color,
            width=config.leader_line_width,
            dashes="[2 2] 0",
            lineCap=1,
            lineJoin=1,
        )
        page.insert_image(box, stream=p.png, keep_proportion=False)


def _copy_links(src_page: fitz.Page, dst_page: fitz.Page) -> None:
    for link in src_page.get_links():
        payload: Dict = {"kind": link["kind"], "from": link["from"]}
        for key in ("page", "to", "zoom", "uri", "file", "nameddest"):
            if key in link:
                payload[key] = link[key]
        try:
            dst_page.insert_link(payload)
        except Exception:
            # Skip malformed links rather than aborting the whole run.
            continue
