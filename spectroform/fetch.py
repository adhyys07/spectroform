"""Fetch audio or video from a URL, then hand the local file to the pipeline.

Nothing here analyses a stream. The download lands in a cache folder and
`audio_path` is pointed at it, so every existing feature -- the analysis cache,
dispersion, bars, sections, the sound strip in the sequencer -- works on a
fetched track exactly as it does on a local one.

Two routes:
  direct   a plain http(s) link to a media file. urllib only, no dependency.
  extract  a page that embeds media, via yt-dlp if the user installs it.

The direct route is the one that always works. yt-dlp is opt-in and is never
bundled, because it is a moving target against the sites it reads.

You are responsible for having the right to use whatever you download, and a
site's terms of service may forbid it. The UI says so next to the field.
"""

import hashlib
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request

import bpy
from bpy.types import Operator

from .constants import AUDIO_EXTENSIONS, SUPPORTED_EXTENSIONS

# Only these. file:// would let a "URL" read anything on disk, and ftp:// adds
# a protocol surface for no benefit.
ALLOWED_SCHEMES = {"http", "https"}
CHUNK = 64 * 1024
TIMEOUT = 30

# Content-Type to extension. The URL path is preferred when it carries a
# usable extension; this is the fallback for links that end in a query string.
_TYPE_MAP = {
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/flac": ".flac", "audio/x-flac": ".flac",
    "audio/ogg": ".ogg", "application/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".m4a",
    "audio/aiff": ".aiff", "audio/x-aiff": ".aiff",
    "video/mp4": ".mp4", "video/webm": ".webm",
    "video/quicktime": ".mov", "video/x-matroska": ".mkv",
}

# Shared with the modal operator. A worker thread writes, the timer reads.
PROGRESS = {
    "active": False,
    "done": False,
    "cancel": False,
    "bytes": 0,
    "total": 0,
    "label": "",
    "error": "",
    "path": "",
}


def is_url(text):
    text = (text or "").strip()
    if "://" not in text:
        return False
    try:
        parts = urllib.parse.urlsplit(text)
    except ValueError:
        return False
    return parts.scheme.lower() in ALLOWED_SCHEMES and bool(parts.netloc)


def looks_like_media_file(url):
    """True when the path itself ends in something we can decode.

    Decides between the direct route and needing an extractor, without a
    network round trip.
    """
    path = urllib.parse.urlsplit(url).path
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def cache_name(url, ext):
    """Stable filename from the URL.

    Hashed rather than derived from the remote name on purpose: a server
    controls Content-Disposition, and letting it choose a path is how you get
    directory traversal. The hash also makes re-fetching a no-op.
    """
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    # A readable stem helps when looking in the folder, but it is sanitised to
    # a safe character set and never used for the directory part.
    stem = os.path.basename(urllib.parse.urlsplit(url).path) or "stream"
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(stem)[0])[:40]
    return "%s_%s%s" % (stem or "stream", digest, ext)


def cache_dir(props):
    folder = bpy.path.abspath(props.fetch_folder or "//downloads/")
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    return folder


def existing_download(url, props):
    """A previously fetched file for this URL, whatever extension it got."""
    folder = bpy.path.abspath(props.fetch_folder or "//downloads/")
    if not os.path.isdir(folder):
        return ""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    for name in sorted(os.listdir(folder)):
        if digest in name and os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
            return os.path.join(folder, name)
    return ""


# ---------------------------------------------------------------------------
# direct download
# ---------------------------------------------------------------------------

def _opener():
    # A generic client name. Never put the user's identity in an outgoing
    # header.
    return urllib.request.build_opener()


def download_direct(url, folder, max_mb=100, progress=PROGRESS):
    """Stream a media URL to `folder`. Returns the written path.

    Runs on a worker thread, so it touches `progress` and the filesystem only
    -- never bpy.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "Spectroform/1.0"})
    limit = int(max_mb) * 1024 * 1024

    with _opener().open(request, timeout=TIMEOUT) as response:
        ctype = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        ext = os.path.splitext(urllib.parse.urlsplit(url).path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            ext = _TYPE_MAP.get(ctype, "")
        if not ext:
            raise RuntimeError(
                "Could not tell what kind of media this is (Content-Type %r). "
                "Link directly to a file with an extension." % (ctype or "none"))

        total = int(response.headers.get("Content-Length") or 0)
        if total and total > limit:
            raise RuntimeError(
                "That file is %.2f MB, over the %s MB limit." %
                (total / 1048576.0, max_mb))
        progress["total"] = total

        path = os.path.join(folder, cache_name(url, ext))
        partial = path + ".part"
        written = 0
        try:
            with open(partial, "wb") as handle:
                while True:
                    if progress.get("cancel"):
                        raise RuntimeError("Cancelled")
                    block = response.read(CHUNK)
                    if not block:
                        break
                    written += len(block)
                    if written > limit:
                        raise RuntimeError(
                            "Download passed the %d MB limit." % max_mb)
                    handle.write(block)
                    progress["bytes"] = written
        except BaseException:
            # Never leave a half file where the cache lookup would find it and
            # hand it to the decoder.
            if os.path.exists(partial):
                os.remove(partial)
            raise

        if written == 0:
            os.remove(partial)
            raise RuntimeError("The server sent an empty response.")
        os.replace(partial, path)
        return path


# ---------------------------------------------------------------------------
# extractor route (optional dependency)
# ---------------------------------------------------------------------------

def ytdlp_available():
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        return False
    return True


def download_extract(url, folder, max_mb=100, progress=PROGRESS):
    """Pull the best audio-only stream via yt-dlp.

    Audio-only keeps the download small -- a video stream is many times the
    size and the pipeline only ever reads the audio track.
    """
    import yt_dlp

    def hook(status):
        if progress.get("cancel"):
            raise RuntimeError("Cancelled")
        if status.get("status") == "downloading":
            progress["bytes"] = int(status.get("downloaded_bytes") or 0)
            progress["total"] = int(status.get("total_bytes")
                                    or status.get("total_bytes_estimate") or 0)

    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    options = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(folder, "stream_%s.%%(ext)s" % digest),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "max_filesize": int(max_mb) * 1024 * 1024,
        # No post-processing: ffmpeg may not be on PATH, and aud reads m4a,
        # opus and webm audio as they come.
        "postprocessors": [],
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    path = ydl.prepare_filename(info)
    if not os.path.isfile(path):
        # The real extension is only known after the fact.
        for name in os.listdir(folder):
            if digest in name:
                path = os.path.join(folder, name)
                break
    if not os.path.isfile(path):
        raise RuntimeError("yt-dlp reported success but wrote no file.")
    return path


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

class AUDIOGEO_OT_fetch(Operator):
    bl_idname = "audiogeo.fetch"
    bl_label = "Fetch From Link"
    bl_description = ("Download the audio or video at the link into the cache "
                      "folder, then use it as the source")
    bl_options = {"REGISTER"}

    _timer = None
    _thread = None

    @classmethod
    def poll(cls, context):
        return (not PROGRESS["active"]
                and is_url(context.scene.audiogeo.media_url))

    def invoke(self, context, event):
        props = context.scene.audiogeo
        url = props.media_url.strip()

        if not is_url(url):
            self.report({"ERROR"}, "Enter an http or https link.")
            return {"CANCELLED"}

        cached = existing_download(url, props)
        if cached and not props.fetch_force:
            # Already have it; skip the network entirely.
            self._adopt(context, cached, cached=True)
            return {"FINISHED"}

        direct = looks_like_media_file(url)
        if not direct and not ytdlp_available():
            self.report(
                {"ERROR"},
                "That link is a page, not a media file. Install yt-dlp to read "
                "it, or paste a direct link to an audio or video file.")
            return {"CANCELLED"}

        try:
            folder = cache_dir(props)
        except OSError as exc:
            self.report({"ERROR"}, "Cannot create the cache folder: %s" % exc)
            return {"CANCELLED"}

        PROGRESS.update(active=True, done=False, cancel=False, bytes=0,
                        total=0, error="", path="",
                        label="Fetching" if direct else "Extracting")

        def work():
            try:
                fn = download_direct if direct else download_extract
                PROGRESS["path"] = fn(url, folder, props.fetch_max_mb, PROGRESS)
            except BaseException as exc:
                PROGRESS["error"] = str(exc) or type(exc).__name__
            finally:
                PROGRESS["done"] = True

        # A thread, not a blocking call: a 100 MB download would otherwise
        # freeze Blender with no way to cancel.
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(
            0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            PROGRESS["cancel"] = True
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        props = context.scene.audiogeo
        if not PROGRESS["done"]:
            total, got = PROGRESS["total"], PROGRESS["bytes"]
            if total:
                props.last_report = "%s %.0f%% (%.1f/%.1f MB)" % (
                    PROGRESS["label"], got * 100.0 / total,
                    got / 1048576.0, total / 1048576.0)
            else:
                props.last_report = "%s %.1f MB" % (
                    PROGRESS["label"], got / 1048576.0)
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        return self._finish(context)

    def _finish(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        error, path = PROGRESS["error"], PROGRESS["path"]
        PROGRESS.update(active=False, done=False, cancel=False)

        if error:
            context.scene.audiogeo.last_report = "Fetch failed: %s" % error
            self.report({"ERROR"}, context.scene.audiogeo.last_report)
            return {"CANCELLED"}

        self._adopt(context, path, cached=False)
        return {"FINISHED"}

    def _adopt(self, context, path, cached):
        props = context.scene.audiogeo
        # Relative when the .blend is saved, so a project folder stays portable.
        props.audio_path = (bpy.path.relpath(path) if bpy.data.is_saved
                            else path)
        size = os.path.getsize(path) / 1048576.0
        ext = os.path.splitext(path)[1].lower()
        props.last_report = "%s %s (%.1f MB)%s" % (
            "Using cached" if cached else "Fetched",
            os.path.basename(path), size,
            "" if ext in AUDIO_EXTENSIONS else " -- audio track of a video")
        self.report({"INFO"}, props.last_report)

        if props.fetch_auto_generate:
            bpy.ops.audiogeo.generate()


class AUDIOGEO_OT_fetch_clear_cache(Operator):
    bl_idname = "audiogeo.fetch_clear_cache"
    bl_label = "Clear Download Cache"
    bl_description = "Delete every file this add-on downloaded into the cache folder"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        folder = bpy.path.abspath(
            context.scene.audiogeo.fetch_folder or "//downloads/")
        return os.path.isdir(folder)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        props = context.scene.audiogeo
        folder = bpy.path.abspath(props.fetch_folder or "//downloads/")
        removed, freed = 0, 0
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            # Only files this module could have written: a recognised media
            # extension or a leftover .part. Never anything else the user put
            # in that folder.
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS and ext != ".part":
                continue
            try:
                freed += os.path.getsize(path)
                os.remove(path)
                removed += 1
            except OSError:
                pass
        props.last_report = "Removed %d file(s), freed %.1f MB" % (
            removed, freed / 1048576.0)
        self.report({"INFO"}, props.last_report)
        return {"FINISHED"}


class AUDIOGEO_OT_install_fetch_deps(Operator):
    bl_idname = "audiogeo.install_fetch_deps"
    bl_label = "Install yt-dlp"
    bl_description = ("Install yt-dlp into Blender's own Python so links to "
                      "pages can be read. You are responsible for having the "
                      "right to use what you download")
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        import subprocess
        import sys

        try:
            # sys.executable is Blender's bundled interpreter -- the one that
            # has to end up with the module, not the system Python.
            subprocess.check_call([sys.executable, "-m", "ensurepip"])
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "--upgrade", "yt-dlp"])
        except Exception as exc:
            self.report({"ERROR"}, "Install failed: %s" % exc)
            return {"CANCELLED"}

        context.scene.audiogeo.last_report = (
            "yt-dlp installed -- restart Blender to enable page links")
        self.report({"INFO"}, context.scene.audiogeo.last_report)
        return {"FINISHED"}


classes = (
    AUDIOGEO_OT_fetch,
    AUDIOGEO_OT_fetch_clear_cache,
    AUDIOGEO_OT_install_fetch_deps,
)
