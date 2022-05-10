"""
Test for directory_utils functions.
"""

import pytest, os
from utils.directory_utils import mkdir_p

# Directories to be created and removed after testing
pathsToBeTested = ["", "testPath", "parentDir"]

def test_mkdir_p_emptyPath():
    emptyPath = pathsToBeTested[0]
    with pytest.raises(OSError):
        mkdir_p(emptyPath)

def test_mkdir_p_correctPath():
    correctPath = pathsToBeTested[1]
    mkdir_p(correctPath)

def test_mkdir_p_correctPathWithParent():
    correctPathWithParent = pathsToBeTested[2] + "/" + pathsToBeTested[1]
    mkdir_p(correctPathWithParent)

def test_clean_directories():
    os.system("rm -rf {} {}".format(pathsToBeTested[1], pathsToBeTested[2]))