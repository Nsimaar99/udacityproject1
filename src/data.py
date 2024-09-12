import os
import requests
from pathlib import Path
import tarfile

def setup_flower_data(data_dir='./flowers'):
    """
    Downloads and extracts the flower dataset if it is not already set up.

    Args:
        data_dir (str): Path to the dataset directory. Default is './flowers'.
    """
    # using pathlib.Path for handling PosixPath
    FLOWERS_DIR = Path(data_dir)

    # checking if the dataset already exists
    if not FLOWERS_DIR.is_dir():
        # creating directory
        FLOWERS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Directory created: ./{FLOWERS_DIR}")

        print() # for readability

        # tarball path
        TARBALL = FLOWERS_DIR / "flower_data.tar.gz"

        # downloading and writing the tarball to './flowers' directory
        print(f"[INFO] Downloading the file 'flower_data.tar.gz' to ./{FLOWERS_DIR}")
        request = requests.get('https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz')
        with open(TARBALL, "wb") as file_ref:
            file_ref.write(request.content)
            print(f"[INFO] 'flower_data.tar.gz' saved to ./{FLOWERS_DIR}")

        print() # for readability

        # extracting the downloaded tarball
        print(f"[INFO] Extracting the downloaded tarball to ./{FLOWERS_DIR}")
        with tarfile.open(TARBALL, "r") as tar_ref:
            tar_ref.extractall(FLOWERS_DIR)
            print(f"[INFO] 'flower_data.tar.gz' extracted successfully to ./{FLOWERS_DIR}")

        print() # for readability

        # using os.remove to delete the downloaded tarball
        print("[INFO] Deleting the tarball to save space.")
        os.remove(TARBALL)
    else:
        print(f"[INFO] Dataset already set up at ./{FLOWERS_DIR}")

import json

def create_flower_name_mapping(filename='cat_to_name.json'):
    """
    Creates a JSON file mapping flower class indices to their names.

    Args:
        filename (str): Name of the output JSON file. Default is 'cat_to_name.json'.
    """
    data = {
        "21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster",
        "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy",
        "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly",
        "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist",
        "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower",
        "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation",
        "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone",
        "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow",
        "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid",
        "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia",
        "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura",
        "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium",
        "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily",
        "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william",
        "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon",
        "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula",
        "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower",
        "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple",
        "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus",
        "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily",
        "46": "wallflower", "77": "passion flower", "51": "petunia"
    }

    with open(filename, 'w') as file:
        json.dump(data, file)
        print(f"[INFO] Class-to-name mapping saved as '{filename}'")


