# mkdir -p ~/dev/bossjones/practical-python-and-opencv-case-studies/deeplearning_data/{autogenerate,characters,models}/{captain_kobayashi,eiko_yamano,izana_shinatose,kouichi_tsuruuchi,lalah_hiyama,mozuku_kunato,nagate_tanikaze,norio_kunato,ochiai,samari_ittan,shizuka_hoshijiro,yuhata_midorikawa}/{edited,non_filtered,pic_video}


movies_path = "/Users/malcolm/Downloads/knights_of_sidonia"
dataset_folder = "/Users/malcolm/dev/bossjones/practical-python-and-opencv-case-studies/deeplearning_data"

map_characters = {
    0: "captain_kobayashi",
    1: "lalah_hiyama",
    # 2: "izana_shinatose",
    # 3: "kouichi_tsuruuchi",
    # 4: "eiko_yamano",
    # 5: "mozuku_kunato",
    # 6: "nagate_tanikaze",
    # 7: "norio_kunato",
    # 8: "ochiai",
    # 9: "samari_ittan",
    # 10: "shizuka_hoshijiro",
    # 11: "yuhata_midorikawa",
}

# characters_folder = f"{movies_path}/characters"
characters_folder = f"{dataset_folder}/characters"

# Best size of images
IMG_SIZE = (80, 80)
# Since we don't require color in our images, set this to 1, grayscale
channels = 1
