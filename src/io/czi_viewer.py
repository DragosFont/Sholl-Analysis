import matplotlib.pyplot as plt
from aicsimageio import AICSImage

def display_slices_like_fiji(path):
    img = AICSImage(path)
    data = img.get_image_data("TCZYX")

    T, C, Z, Y, X = data.shape
    print(f"Dimensiuni: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")

    for t in range(T):
        for c in range(C):
            for z in range(Z):
                slice_img = data[t, c, z, :, :]
                fig, ax = plt.subplots()
                ax.imshow(slice_img, cmap='gray')
                ax.set_title(f"T={t}, C={c}, Z={z} din {path}")
                ax.axis('off')
                plt.show(block=False)  # deschide separat
