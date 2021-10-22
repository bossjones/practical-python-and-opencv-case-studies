import glob
import os
import pathlib

from prompt_toolkit.completion import Completer, WordCompleter, merge_completers

# mkdir -p ~/dev/bossjones/practical-python-and-opencv-case-studies/deeplearning_data/{autogenerate,characters,models}/{captain_kobayashi,eiko_yamano,izana_shinatose,kouichi_tsuruuchi,lalah_hiyama,mozuku_kunato,nagate_tanikaze,norio_kunato,ochiai,samari_ittan,shizuka_hoshijiro,yuhata_midorikawa}/{edited,non_filtered,pic_video}

ROOT_DIR = os.path.dirname(__file__)

movies_path = "/Users/malcolm/Downloads/farming/anime/knights_of_sidonia"
dataset_folder = "/Users/malcolm/dev/bossjones/practical-python-and-opencv-case-studies/deeplearning_data"

videos_folder = f"{dataset_folder}/videos"

# map_characters = {
#     0: "captain_kobayashi",
#     1: "lalah_hiyama",
#     2: "izana_shinatose",
#     3: "kouichi_tsuruuchi",
#     4: "eiko_yamano",
#     5: "mozuku_kunato",
#     6: "nagate_tanikaze",
#     7: "norio_kunato",
#     8: "ochiai",
#     9: "samari_ittan",
#     10: "shizuka_hoshijiro",
#     11: "yuhata_midorikawa",
# }

map_characters = {
    0: "annie_leonhart",
    1: "armin_arlert",
    2:"connie_springer",
    3:"eren_yeager",
    4: "erwin_smith",
    5: "hanji_zoe",
    6: "historia_reiss",
    7: "jean_kirschtein",
    8: "levi_ackerman",
    9: "mikasa_ackerman",
    10: "reiner_braun"
}

# characters_folder = f"{movies_path}/characters"
characters_folder = f"{dataset_folder}/characters"

# Best size of images
IMG_SIZE = (80, 80)
# Since we don't require color in our images, set this to 1, grayscale
channels = 1


# pic_size = 64
pic_size = 80
batch_size = 32
# epochs = 200
epochs = 200
num_classes = len(map_characters)
pictures_per_class = 1000
test_size = 0.15

# ./models/weights.best_6conv2.hdf5
# filepath = f"{ROOT_DIR}/weights_6conv_%s.hdf5" % time.strftime("%d%m/%Y")
path_to_best_model = f"{ROOT_DIR}/models/weights.best_6conv2.hdf5"

chars_list_from_folder = [
    pathlib.Path(f"{k}").stem for k in glob.glob(f"{characters_folder}/*")
]
chars_list_from_folder.append("no")

name_completer = WordCompleter(
    chars_list_from_folder,
    ignore_case=True,
)

yes_no_completer = WordCompleter(
    ["No", "Right", "Left", "Full", "Stop"], ignore_case=True
)
