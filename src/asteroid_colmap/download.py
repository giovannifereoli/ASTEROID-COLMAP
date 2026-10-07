"""Download calibrated Dawn FC FITS images and their PDS3 labels from the PDS SBN archive."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from . import __version__
from .config import Dataset
from .pds import parse_listing

log = logging.getLogger(__name__)

# e.g. FC21B0003112_11205060002F1C.FIT -> filter 1
FC_FILE = re.compile(r"^FC2\w+_\d{11}F(\d)[A-Z]\.FIT$", re.IGNORECASE)


@dataclass(frozen=True)
class RemoteImage:
    """One FITS image and its detached PDS3 label in the remote archive.

    Attributes
    ----------
    subdir : str
        Archive sub-directory (observation sequence).
    fit, lbl : str
        File names of the image and of its label.
    filter : int
        FC filter number, parsed from the file name.
    base_url : str
        Dataset base URL; ends with ``/``.
    """
    subdir: str
    fit: str
    lbl: str
    filter: int
    base_url: str

    def url(self, name: str) -> str:
        """Full URL of a file in this image's sub-directory.

        Parameters
        ----------
        name : str
            File name, usually :attr:`fit` or :attr:`lbl`.

        Returns
        -------
        str
        """
        return f"{self.base_url}{self.subdir}/{name}"


def make_session() -> requests.Session:
    """HTTP session with an identifying User-Agent and automatic retries.

    HTTPS requests are retried up to 5 times, with exponential back-off (factor 1 s), on
    connection errors and on status 429, 500, 502, 503 and 504. The pool keeps up to 16
    connections, enough for the parallel downloads.

    Returns
    -------
    requests.Session
    """
    session = requests.Session()
    session.headers["User-Agent"] = f"asteroid-colmap/{__version__} (+https://github.com/colmap/colmap)"
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    return session


def list_images(
    dataset: Dataset,
    filters: tuple[int, ...] | None = None,
    subdirs: tuple[str, ...] | None = None,
    session: requests.Session | None = None,
) -> list[RemoteImage]:
    """List the remote FITS images, with labels, of the selected filters.

    Parameters
    ----------
    dataset : Dataset
        Archive location, default sub-directories and default filters.
    filters : tuple of int, optional
        FC filter numbers; default ``dataset.default_filters``.
    subdirs : tuple of str, optional
        Sub-directories to list; default ``dataset.subdirs``.
    session : requests.Session, optional
        Session to reuse; default :func:`make_session`.

    Returns
    -------
    list of RemoteImage
        By sub-directory, then by file name (time order within a sequence). Images
        without a label are skipped with a warning.

    Raises
    ------
    requests.HTTPError
        If a directory listing cannot be fetched.
    """
    filters = filters or dataset.default_filters
    session = session or make_session()
    images: list[RemoteImage] = []
    for sub in subdirs or dataset.subdirs:
        resp = session.get(f"{dataset.base_url}{sub}/", timeout=60)
        resp.raise_for_status()
        names = set(parse_listing(resp.text))
        for name in sorted(names):
            m = FC_FILE.match(name)
            if not m or int(m.group(1)) not in filters:
                continue
            lbl = name[:-4] + ".LBL"
            if lbl not in names:
                log.warning("no label for %s/%s, skipped", sub, name)
                continue
            images.append(RemoteImage(sub, name, lbl, int(m.group(1)), dataset.base_url))
    return images


def select_evenly(items: list, n: int | None) -> list:
    """Pick ``n`` items spread evenly over a list, including the first.

    Parameters
    ----------
    items : list
        Items in order.
    n : int or None
        Number to keep; ``None``, 0 or ``n >= len(items)`` keeps everything.

    Returns
    -------
    list
        New list of the selected items in their original order; for ``n > 1`` it also
        includes the last item.
    """
    if not n or n >= len(items):
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


def _fetch(session: requests.Session, url: str, dest: Path) -> bool:
    """Stream ``url`` to ``dest`` through a ``.part`` file that is renamed when complete.

    Parameters
    ----------
    session : requests.Session
        HTTP session.
    url : str
        Source URL.
    dest : pathlib.Path
        Target file; nothing is downloaded if it exists and is not empty.

    Returns
    -------
    bool
        True if the file was downloaded, False if it was already there.

    Raises
    ------
    OSError
        If fewer bytes than ``Content-Length`` arrived.
    requests.HTTPError
        On an HTTP error status.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return False
    tmp = dest.with_name(dest.name + ".part")
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        expected = int(resp.headers.get("Content-Length") or 0)
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    if expected and tmp.stat().st_size != expected:
        tmp.unlink()
        raise OSError(f"truncated download: {url}")
    tmp.replace(dest)
    return True


def download(
    dataset: Dataset,
    raw_dir: Path,
    filters: tuple[int, ...] | None = None,
    subdirs: tuple[str, ...] | None = None,
    max_images: int | None = None,
    workers: int = 6,
) -> list[Path]:
    """Download FITS + LBL pairs into ``raw_dir/<subdir>/``.

    Already-complete files are skipped, so an interrupted download can simply be re-run.

    Parameters
    ----------
    dataset : Dataset
        Archive to download from.
    raw_dir : pathlib.Path
        Destination root.
    filters, subdirs : tuple, optional
        See :func:`list_images`.
    max_images : int, optional
        Keep only this many images, spread evenly over the listing (:func:`select_evenly`).
    workers : int
        Number of parallel download threads.

    Returns
    -------
    list of pathlib.Path
        Label paths of the selected images.

    Raises
    ------
    RuntimeError
        If no image matches the filters and sub-directories.
    """
    session = make_session()
    remote = select_evenly(list_images(dataset, filters, subdirs, session), max_images)
    if not remote:
        raise RuntimeError("no images matched the requested filters/sub-directories")
    jobs = []
    for img in remote:
        out = Path(raw_dir) / img.subdir
        out.mkdir(parents=True, exist_ok=True)
        jobs += [(img.url(img.fit), out / img.fit), (img.url(img.lbl), out / img.lbl)]
    log.info("%d images (%d files) from %s", len(remote), len(jobs), dataset.base_url)
    fetched = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, session, url, dest): dest for url, dest in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="download", unit="file"):
            fetched += fut.result()
    log.info("downloaded %d new files, %d already present", fetched, len(jobs) - fetched)
    return [Path(raw_dir) / img.subdir / img.lbl for img in remote]
