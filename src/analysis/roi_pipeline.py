import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from aicsimageio import AICSImage
from skimage import filters, measure, morphology
from skimage.io import imsave
from src.io.sholl import sholl_analysis

def process_roi_from_mip(file_path: str, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Încarcă imaginea .czi și generează MIP pe axa Z
    img = AICSImage(file_path)
    data = img.get_image_data("CZYX", T=0, S=0)
    mip = data.max(axis=1)

    # Generează imagine RGB
    rgb = np.zeros((*mip.shape[1:], 3), dtype=np.float32)
    for c, color in enumerate([(0, 0, 1), (0, 1, 0), (1, 0, 0)]):  # B, G, R
        norm = mip[c] / mip[c].max() if mip[c].max() > 0 else mip[c]
        for i in range(3):
            rgb[..., i] += norm * color[i]
    imsave(os.path.join(output_dir, "full_rgb.png"), (rgb * 255).astype(np.uint8))

    # Selectează ROI manual
    roi_coords = []

    def onselect(eclick, erelease):
        if roi_coords:
            return  # Permite doar o selecție
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        roi_coords.append((min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)))
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, color="red", linewidth=2))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'enter' and roi_coords:
            plt.close()

    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_title("Selectează ROI pe imaginea RGB (click + drag, ENTER pt confirmare)")
    toggle_selector = RectangleSelector(ax, onselect, useblit=True)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not roi_coords:
        print("❌ Nu ai selectat niciun ROI. Încearcă din nou.")
        return

    # Extrage coordonatele ROI
    y1, y2, x1, x2 = roi_coords[0]
    roi_red = mip[2, y1:y2, x1:x2]
    roi_green = mip[1, y1:y2, x1:x2]
    roi_blue = mip[0, y1:y2, x1:x2]

    # Salvează imagine ROI RGB
    roi_rgb = np.stack([
        roi_blue / roi_blue.max(),
        roi_green / roi_green.max(),
        roi_red / roi_red.max()
    ], axis=-1)
    imsave(os.path.join(output_dir, "roi_rgb.png"), (roi_rgb * 255).astype(np.uint8))

    # Salvează imagine ROI pe canalul verde
    fig, ax = plt.subplots()
    ax.imshow(roi_green, cmap="Greens")
    ax.set_title("ROI pe canalul VERDE")
    plt.savefig(os.path.join(output_dir, "roi_green.png"))
    plt.close(fig)

    # Salvează imagine ROI pe canalul albastru
    fig, ax = plt.subplots()
    ax.imshow(roi_blue, cmap="Blues")
    ax.set_title("Verifică forma pe canalul ALBASTRU")
    plt.savefig(os.path.join(output_dir, "roi_blue.png"))
    plt.close(fig)

    # Salvează imagine ROI pe canalul roșu
    fig, ax = plt.subplots()
    ax.imshow(roi_red, cmap="Reds")
    ax.set_title("Canal ROȘU pentru analiză")
    plt.savefig(os.path.join(output_dir, "roi_red.png"))
    plt.close(fig)

    # Binarizare și detecție contur
    red_norm = roi_red / roi_red.max()
    threshold = filters.threshold_otsu(red_norm)
    binary = red_norm > threshold
    binary = morphology.remove_small_objects(binary, min_size=64)

    binary_path = os.path.join(output_dir, "roi_binary.tif")
    imsave(binary_path, (binary * 255).astype(np.uint8))

    # Salvează contururi
    contours = measure.find_contours(binary, 0.5)
    fig, ax = plt.subplots()
    ax.imshow(binary, cmap="gray")
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="red")
    ax.set_title("Contur ROI roșu")
    plt.savefig(os.path.join(output_dir, "roi_contours.png"))
    plt.close(fig)

    # Analiza Sholl
    if binary.sum() == 0:
        print("❌ Imaginea binarizată este goală. Nu se poate face analiza Sholl.")
    else:
        print("✅ Lansez analiza Sholl...")
        sholl_analysis(
            image_path=binary_path,
            step_size=5,
            max_radius=250,
            save_path=os.path.join(output_dir, "sholl_analysis.png")
        )
        print("✅ Analiza Sholl salvată cu succes.")
