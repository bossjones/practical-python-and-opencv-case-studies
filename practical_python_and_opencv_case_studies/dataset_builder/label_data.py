# pylint:disable=no-member
# python -m practical_python_and_opencv_case_studies.dataset_builder.label_data
import glob
import os
import pathlib
from random import shuffle
import time

import cv2
import ffmpy
import keras
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from prompt_toolkit.completion import Completer, WordCompleter, merge_completers
# from prompt_toolkit.eventloop.defaults import create_event_loop
from prompt_toolkit.eventloop.inputhook import (
    new_eventloop_with_inputhook,
    set_eventloop_with_inputhook,
)
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession, input_dialog, prompt
from selenium import webdriver

# import train
from practical_python_and_opencv_case_studies.dataset_builder import constants, train

# pic_size = 64
pic_size = 80
# plt.ioff()

kb = KeyBindings()


@kb.add("c-space")
def _(event):
    """
    Start auto completion. If the menu is showing already, select the next
    completion.
    """
    b = event.app.current_buffer
    if b.complete_state:
        b.complete_next()
    else:
        b.start_completion(select_first=False)


def get_character_name(name: str):
    """
    Get the character name from just a part of it, comparing to saved characters
    :param name: part of the character name
    :return: full name
    """
    chars = [
        pathlib.Path(f"{k}") for k in glob.glob(f"{constants.characters_folder}/*")
    ]
    char_name = [
        f"{k.stem}" for k in chars if name.lower().replace(" ", "_") in f"{k.stem}"
    ]
    print(f" [chars] = {chars}")
    print(f" [char_name] = {char_name}")
    if len(char_name) > 0:
        return char_name[0]
    else:
        print("FAKE NAME")
        return "ERROR"


# Explicitly spinning the Event Loop for matplot


def slow_loop(N: int, img_axes: matplotlib.image.AxesImage):
    for j in range(N):
        time.sleep(0.1)  # to simulate some work
        img_axes.figure.canvas.flush_events()


# # https://github.com/prompt-toolkit/python-prompt-toolkit/blob/5aea076692a304ec2bf8ad18fc59d4885cc462b1/examples/prompts/inputhook.py
# # https://matplotlib.org/stable/users/interactive_guide.html#input-hook-integration
# def inputhook(context):
#     """
#     When the eventloop of prompt-toolkit is idle, call this inputhook.
#     This will run the GTK main loop until the file descriptor
#     `context.fileno()` becomes ready.
#     :param context: An `InputHookContext` instance.
#     """

#     def _main_quit(*a, **kw):
#         gtk.main_quit()
#         return False

#     gobject.io_add_watch(context.fileno(), gobject.IO_IN, _main_quit)
#     gtk.main()

# SOURCE: https://github.com/GamestonkTerminal/GamestonkTerminal/blob/e7e49538b03e6271e1709c5229f99b5c6f4b494d/gamestonk_terminal/menu.py
def inputhook(inputhook_contex):
    while not inputhook_contex.input_is_ready():
        # print("not inputhook_contex.input_is_ready")
        try:
            # Run the GUI event loop for interval seconds.
            # If there is an active figure, it will be updated and displayed before the pause, and the GUI event loop (if any) will run during the pause.
            # This can be used for crude animation. For more complex animation use matplotlib.animation.
            # If there is no active figure, sleep for interval seconds instead.
            pyplot.pause(0.5)
            # img_axes.figure.canvas.flush_events()
        # pylint: disable=unused-variable
        except Exception:  # noqa: F841
            continue
    return False


def labelized_data(to_shuffle=False, interactive=False):
    """
    Interactive labeling data with the possibility to crop the picture shown : full picture,
    left part, right part. Manually labeling data from .avi videos in the same folder. Analzying
    frame (randomly chosen) of each video and then save the picture into the right character
    folder.
    :param interactive: boolean to label from terminal
    """
    movies = glob.glob(f"{constants.videos_folder}/*.avi")
    if to_shuffle:
        shuffle(movies)
    for fname in movies[::-1]:
        try:
            m, s = np.random.randint(0, 3), np.random.randint(0, 59)
            print(f"fname = {fname}")
            cap = cv2.VideoCapture(fname)  # video_name is the video being called
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(1, fps * (m * 60 + s))  # Where frame_no is the frame you want
            i = 0
            while True:
                i += 1
                ret, frame = cap.read()  # Read the frame
                # Resizing HD pictures (we don't need HD)
                if np.min(frame.shape[:2]) > 900:
                    frame = cv2.resize(
                        frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
                    )
                if i % np.random.randint(100, 250) == 0:
                    if interactive:
                        plt.ion()
                    img_axes = plt.imshow(frame)
                    img_axes.figure.canvas.flush_events()
                    # slow_loop(100, img_axes)
                    plt.show()
                    # where = input("Where is the character ?[No,Right,Left,Full] ")
                    # where = prompt(
                    #     title="Where is the character ?", text="Please type one of the following [No,Right,Left,Full] :", completer=constants.yes_no_completer
                    # ).run()
                    # input()
                    where = prompt(
                        message="Where is the character ? Please type one of the following [No,Right,Left,Full] :",
                        completer=constants.yes_no_completer,
                        complete_while_typing=True,
                        key_bindings=kb,
                        # inputhook=inputhook
                        # complete_in_thread=True
                    )
                    # set_eventloop_with_inputhook(inputhook)
                    if where.lower() == "stop":
                        # os.remove(fname)
                        raise

                    elif where.lower() in ["left", "l"]:
                        plt.close()
                        img_axes = plt.imshow(frame[:, : int(frame.shape[1] / 2)])
                        img_axes.figure.canvas.flush_events()
                        plt.show()
                        # name = input("Name ?[Name or No] ")
                        # name = input("Name ?[Name or No] ")
                        # name = prompt(
                        #     title="Name ?", text="Please type one of the following [Name or No] :", completer=constants.name_completer
                        # ).run()
                        name = prompt(
                            message="Name ? Please type one of the following [Name or No] :",
                            completer=constants.name_completer,
                            complete_while_typing=True,
                            key_bindings=kb,
                            # complete_in_thread=True
                        )
                        plt.close()
                        if name.lower() not in ["no", "n", ""]:
                            name_char = get_character_name(name)
                            name_new_pic = "pic_{:04d}.jpg".format(
                                len(
                                    glob.glob(
                                        f"{constants.characters_folder}/%s/*"
                                        % name_char
                                    )
                                )
                            )
                            title = f"{constants.characters_folder}/%s/%s" % (
                                name_char,
                                name_new_pic,
                            )
                            cv2.imwrite(title, frame[:, : int(frame.shape[1] / 2)])
                            print("Saved at %s" % title)
                            print(
                                "%s : %d photos labeled"
                                % (
                                    name_char,
                                    len(
                                        glob.glob(
                                            f"{constants.characters_folder}/%s/*"
                                            % name_char
                                        )
                                    ),
                                )
                            )

                    elif where.lower() in ["right", "r"]:
                        plt.close()
                        img_axes = plt.imshow(frame[:, int(frame.shape[1] / 2) :])
                        img_axes.figure.canvas.flush_events()
                        plt.show()
                        # name = input("Name ?[Name or No] ")
                        name = prompt(
                            message="Name ? Please type one of the following [Name or No] :",
                            completer=constants.name_completer,
                            complete_while_typing=True,
                            key_bindings=kb,
                            # complete_in_thread=True
                        )
                        plt.close()
                        if name.lower() not in ["no", "n", ""]:
                            name_char = get_character_name(name)
                            name_new_pic = "pic_{:04d}.jpg".format(
                                len(
                                    glob.glob(
                                        f"{constants.characters_folder}/%s/*"
                                        % name_char
                                    )
                                )
                            )
                            title = f"{constants.characters_folder}/%s/%s" % (
                                name_char,
                                name_new_pic,
                            )
                            cv2.imwrite(title, frame[:, int(frame.shape[1] / 2) :])
                            print("Saved at %s" % title)
                            print(
                                "%s : %d photos labeled"
                                % (
                                    name_char,
                                    len(
                                        glob.glob(
                                            f"{constants.characters_folder}/%s/*"
                                            % name_char
                                        )
                                    ),
                                )
                            )

                    elif where.lower() in ["full", "f"]:
                        # name = input("Name ?[Name or No] ")
                        name = prompt(
                            message="Name ? Please type one of the following [Name or No] :",
                            completer=constants.name_completer,
                            complete_while_typing=True,
                            key_bindings=kb,
                            # complete_in_thread=True
                        )
                        plt.close()
                        if name.lower() not in ["no", "n", ""]:
                            name_char = get_character_name(name)
                            name_new_pic = "pic_{:04d}.jpg".format(
                                len(
                                    glob.glob(
                                        f"{constants.characters_folder}/%s/*"
                                        % name_char
                                    )
                                )
                            )
                            title = f"{constants.characters_folder}/%s/%s" % (
                                name_char,
                                name_new_pic,
                            )
                            cv2.imwrite(title, frame)
                            print("Saved at %s" % title)
                            print(
                                "%s : %d photos labeled"
                                % (
                                    name_char,
                                    len(
                                        glob.glob(
                                            f"{constants.characters_folder}/%s/*"
                                            % name_char
                                        )
                                    ),
                                )
                            )
        except Exception as e:
            if e == KeyboardInterrupt:
                return
            else:
                continue


def generate_pic_from_videos():
    """
    Randomly generate pictures from videos : get the full picture, the right part, the left part.
    So, three pictures are saved for each analyzed frame (chosen randomly).
    """
    for k, fname in enumerate(glob.glob(f"{constants.videos_folder}/*.avi")):
        m, s = np.random.randint(0, 3), np.random.randint(0, 59)
        cap = cv2.VideoCapture(fname)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(1, fps * (m * 60 + s))  # Where frame_no is the frame you want
        i = 0
        while i < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            try:
                i += 1
                ret, frame = cap.read()  # Read the frame
                if i % np.random.randint(400, 700) == 0:
                    pics = {
                        "pic_%s_r_%d_%d.jpg"
                        % (
                            fname.split("/")[1].split(".")[0],
                            i,
                            np.random.randint(10000),
                        ): frame[:, : int(frame.shape[1] / 2)],
                        "pic_%s_l_%d_%d.jpg"
                        % (
                            fname.split("/")[1].split(".")[0],
                            i,
                            np.random.randint(10000),
                        ): frame[:, int(frame.shape[1] / 2) :],
                        "pic_%s_f_%d_%d.jpg"
                        % (
                            fname.split("/")[1].split(".")[0],
                            i,
                            np.random.randint(10000),
                        ): frame,
                    }
                    for name, img in pics.items():
                        cv2.imwrite(
                            f"{constants.dataset_folder}/autogenerate/" + name, img
                        )
            except:
                pass
        print(
            "\r%d/%d" % (k + 1, len(glob.glob(f"{constants.videos_folder}/*.avi"))),
            end="",
        )


def classify_pics():
    """
    Use a Keras saved model to classify pictures and move them into the right character folder.
    """
    l = glob.glob(f"{constants.dataset_folder}/autogenerate/*.jpg")
    model = train.load_model_from_checkpoint(
        f"{constants.path_to_best_model}", six_conv=True
    )
    d = len(l)
    for i, p in enumerate(l):
        img = cv2.imread(p)
        img = cv2.resize(img, (pic_size, pic_size)).astype("float32") / 255.0
        a = model.predict(img.reshape((-1, pic_size, pic_size, 3)), verbose=0)[0]
        if np.max(a) > 0.6:
            char = constants.map_characters[np.argmax(a)]
            os.rename(
                p,
                f"{constants.dataset_folder}/autogenerate/%s/%s"
                % (char, p.split("/")[2]),
            )
        else:
            # os.remove(p)
            print(f"would remove {p}. skipping...")
        print("\r%d/%d" % (i + 1, d), end="")


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

    labelized_data(interactive=True)
    # _name = get_character_name("captain")
