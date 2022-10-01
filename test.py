import os, pathlib
from tabnanny import verbose
import pytest

os.chdir( pathlib.Path.cwd() / 'tests' )

pytest.main()