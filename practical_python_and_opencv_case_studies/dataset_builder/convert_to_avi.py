import glob
import itertools
import sys

import cv2
import ffmpy
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from selenium import webdriver

from practical_python_and_opencv_case_studies.dataset_builder import (
    constants,
    label_data,
    utils,
)


def convert_mp4_to_avi():
    movies = glob.glob(f"{constants.movies_path}/*.mp4")
    top_k = 10
    for p in range(top_k):
        print("\r%i/%i" % (p, top_k), end="")
        ff = ffmpy.FFmpeg(
            inputs={np.random.choice(movies): None},
            outputs={f"{constants.videos_folder}/video%d.avi" % p: None},
        )
        ff.run()


if __name__ == "__main__":
    # Catch exception
    import sys

    from IPython.core import ultratb
    from IPython.core.debugger import Tracer  # noqa
    # from bpython.bpdb.debugger import BPdb
    import bpdb
    import bpython

    # sys.excepthook = ultratb.FormattedTB(
    #     mode="Verbose", color_scheme="Linux", call_pdb=True, ostream=sys.__stdout__, debugger_cls=bpdb.BPdb
    # )
    sys.excepthook = ultratb.FormattedTB(
        mode="Verbose", color_scheme="Linux", call_pdb=True, ostream=sys.__stdout__
    )

    convert_mp4_to_avi()
