from contextlib import contextmanager
import os
from tempfile import NamedTemporaryFile


@contextmanager
def temporary_file(suffix=""):
    """Yield a writable temporary filename that is deleted on context exit.
    Parameters
    ----------
    suffix : string, optional
        The suffix for the file.
    Examples
    --------
    >>> import numpy as np
    >>> from practical_python_and_opencv_case_studies.utils import io
    >>> with temporary_file('.tif') as tempfile:
    ...     im = np.arange(25, dtype=np.uint8).reshape((5, 5))
    ...     io.imsave(tempfile, im)
    ...     assert np.all(io.imread(tempfile) == im)
    """
    tempfile_stream = NamedTemporaryFile(suffix=suffix, delete=False)
    tempfile = tempfile_stream.name
    tempfile_stream.close()
    yield tempfile
    os.remove(tempfile)
