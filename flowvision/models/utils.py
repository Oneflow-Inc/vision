import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile
import tarfile
import warnings
import logging
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from tqdm import tqdm
from typing import Optional

import oneflow as flow

HASH_REGEX = re.compile(r"([a-f0-9]*)_")


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    """
    Modified from https://github.com/facebookresearch/iopath/blob/main/iopath/common/file_io.py
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.
    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:
        1) $FLOWVISION_CACHE, if set
        2) otherwise ~/.oneflow/flowvision_cache
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            os.getenv("FLOWVISION_CACHE", "~/.oneflow/flowvision_cache")
        )
    try:
        os.makedirs(cache_dir, exist_ok=True)
        assert os.access(cache_dir, os.W_OK)
    except (OSError, AssertionError):
        tmp_dir = os.path.join(tempfile.gettempdir(), "flowvision_cache")
        logger = logging.getLogger(__name__)
        logger.warning(f"{cache_dir} is not accessible! Using {tmp_dir} instead!")
        cache_dir = tmp_dir
    return cache_dir


def _is_legacy_tar_format(filename):
    return tarfile.is_tarfile(filename)


def _legacy_tar_load(filename, model_dir, map_location, delete_tar_file=True):
    with tarfile.open(filename) as f:
        members = f.getnames()
        extracted_name = members[0]
        extracted_file = os.path.join(model_dir, extracted_name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        f.extractall(model_dir)
    if delete_tar_file:
        os.remove(filename)
    return flow.load(extracted_file)


def _is_legacy_zip_format(filename):
    return zipfile.is_zipfile(filename)


def _legacy_zip_load(filename, model_dir, map_location, delete_zip_file=True):
    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        extracted_name = members[0].filename
        extracted_file = os.path.join(model_dir, extracted_name)
        if not os.path.exists(extracted_file):
            os.mkdir(extracted_file)
        f.extractall(model_dir)
    if delete_zip_file and os.path.exists(filename):
        os.remove(filename)
    return flow.load(extracted_file, map_location)


def load_state_dict_from_url(
    url,
    model_dir=None,
    map_location=None,
    progress=True,
    check_hash=False,
    file_name=None,
    delete_file=True,
):
    r"""Loads the OneFlow serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see flow.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: ``True``
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: ``False``
        file_name (string, optional): name for the downloaded file. Filename from `url` will be used if not set
        delete_file (bool, optional): delete downloaded `.zip` file or `.tar.gz` file after unzipping them.
    """

    try:
        model_dir = get_cache_dir(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    # if already download the weight, directly return loaded state_dict
    pretrained_weight_dir = os.path.join(model_dir, filename.split(".")[0])
    if os.path.exists(pretrained_weight_dir):
        state_dict = flow.load(pretrained_weight_dir)
        return state_dict

    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location, delete_file)
    elif _is_legacy_tar_format(cached_file):
        return _legacy_tar_load(cached_file, model_dir, map_location, delete_file)
    else:
        state_dict = flow.load(cached_file)
        return state_dict


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: ``None``
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: ``True``
    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    # TODO: if there is a backend requirements, we should add headers in Request module
    req = Request(url)
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest
                    )
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
