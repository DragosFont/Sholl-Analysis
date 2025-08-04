import numpy as np
import matplotlib.pyplot as plt
from aicsimageio import AICSImage
from pathlib import Path

def display_all_slices_separate_windows_rgb_mapping(file_path: str):
    img = AICSImage(file_path)
    data = img.get_image_data("TCZYX", S=0)  # shape: (T, C, Z, Y, X)

    colors = {
        0: (0, 0, 1),  # Albastru - neuroni
        1: (0, 1, 0),  # Verde - citoplasmă
        2: (1, 0, 0),  # Roșu - dendrite
    }

    for t in range(data.shape[0]):
        for c in range(data.shape[1]):
            for z in range(data.shape[2]):
                slice_2d = data[t, c, z]

                # Normalizare și conversie în RGB
                rgb = np.zeros((*slice_2d.shape, 3), dtype=np.float32)
                if slice_2d.max() > 0:
                    norm_slice = slice_2d / slice_2d.max()
                else:
                    norm_slice = slice_2d

                for i in range(3):
                    rgb[..., i] = norm_slice * colors.get(c, (1, 1, 1))[i]

                plt.figure()
                plt.imshow(rgb)
                plt.title(f"{Path(file_path).name} - T={t}, C={c}, Z={z}")
                plt.axis("off")
                plt.show(block=False)
