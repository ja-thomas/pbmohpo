import pytest

from pbmohpo.archive import *


def test_archive():
    archive = Archive()
    assert type(archive) == Archive
