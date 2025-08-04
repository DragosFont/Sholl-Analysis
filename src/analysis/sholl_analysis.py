# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from skimage import io, measure
# # from skimage.morphology import skeletonize
# #
# #
# # def sholl_analysis(image=None, image_path: str = None, soma_coords: tuple = None,
# #                    step_size: int = 10, max_radius: int = 300, save_path: str = None):
# #     if image is None:
# #         if image_path is None:
# #             raise ValueError("Trebuie să oferi o imagine (image) sau un path (image_path).")
# #         image = io.imread(image_path)
# #
# #     binary = image > 0
# #
# #     if np.sum(binary) == 0:
# #         print("⚠️ Imagine binară goală – se întrerupe analiza Sholl.")
# #         return 0, 0
# #
# #     if soma_coords is None:
# #         labeled_img = measure.label(binary)
# #         regions = measure.regionprops(labeled_img)
# #         if not regions:
# #             print("⚠️ Nu a fost detectată nicio regiune pentru soma.")
# #             return 0, 0
# #         largest_region = max(regions, key=lambda r: r.area)
# #         soma_coords = tuple(map(int, largest_region.centroid[::-1]))
# #
# #     skeleton = skeletonize(binary)
# #     radii = np.arange(0, max_radius, step_size)
# #     intersections = []
# #
# #     for r in radii:
# #         circle = _create_circle_mask(binary.shape, center=soma_coords, radius=r)
# #         overlap = skeleton & circle
# #         intersections.append(np.count_nonzero(overlap))
# #
# #     if len(intersections) > 1:
# #         degree = min(20, len(intersections) - 1)
# #         coeffs = np.polyfit(radii, intersections, degree)
# #         poly_fit = np.poly1d(coeffs)
# #         radii_fine = np.linspace(min(radii), max(radii), 500)
# #         intersections_fine = poly_fit(radii_fine)
# #     else:
# #         radii_fine = radii
# #         intersections_fine = intersections
# #
# #     fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# #     axs[0].imshow(binary, cmap="gray")
# #     axs[0].scatter(*soma_coords[::-1], c='red', s=40)
# #     axs[0].set_title("Segmentare + Soma")
# #     axs[1].plot(radii, intersections, 'o', label='Date brute')
# #     axs[1].plot(radii_fine, intersections_fine, '-', label='Fit polinomial', color='blue')
# #     axs[1].set_title("Sholl Analysis")
# #     axs[1].set_xlabel("Rază (pixeli)")
# #     axs[1].set_ylabel("Nr. Intersecții")
# #     axs[1].legend()
# #     axs[1].grid(True)
# #     plt.tight_layout()
# #
# #     if save_path:
# #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
# #         plt.savefig(save_path)
# #     else:
# #         plt.show()
# #
# #     return max(intersections), sum(intersections)
# #
# #
# # def _create_circle_mask(shape, center, radius):
# #     Y, X = np.ogrid[:shape[0], :shape[1]]
# #     dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
# #     return (dist >= radius) & (dist < radius + 1)
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from skimage import io, measure
# from skimage.morphology import skeletonize
# from scipy.integrate import simps
# from scipy.stats import linregress
#
# def sholl_analysis(image=None, image_path: str = None, soma_coords: tuple = None,
#                    step_size: int = 10, max_radius: int = 300, save_path: str = None):
#     if image is None:
#         if image_path is None:
#             raise ValueError("Trebuie să oferi o imagine (image) sau un path (image_path).")
#         image = io.imread(image_path)
#
#     binary = image > 0
#
#     if soma_coords is None:
#         labeled_img = measure.label(binary)
#         regions = measure.regionprops(labeled_img)
#         if not regions:
#             raise ValueError("Nu s-a detectat nicio regiune pentru soma.")
#         largest_region = max(regions, key=lambda r: r.area)
#         soma_coords = tuple(map(int, largest_region.centroid[::-1]))
#
#     skeleton = skeletonize(binary)
#     radii = np.arange(0, max_radius, step_size)
#     intersections = []
#
#     for r in radii:
#         circle = _create_circle_mask(binary.shape, center=soma_coords, radius=r)
#         overlap = skeleton & circle
#         intersections.append(np.count_nonzero(overlap))
#
#     # Parametri cantitativi:
#     peak_number = max(intersections)
#     radius_at_peak = radii[np.argmax(intersections)]
#     auc = simps(intersections, radii)
#
#     # Regressie pe partea descrescătoare
#     peak_index = np.argmax(intersections)
#     desc_radii = radii[peak_index:]
#     desc_values = intersections[peak_index:]
#     if len(desc_radii) >= 2:
#         slope, *_ = linregress(desc_radii, desc_values)
#     else:
#         slope = 0  # fallback
#
#     # Plot dual: imagine + grafic
#     fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#     axs[0].imshow(binary, cmap="gray")
#     axs[0].scatter(*soma_coords[::-1], c='red', s=40)
#     axs[0].set_title("Segmentare + Soma")
#
#     axs[1].plot(radii, intersections, 'o', label='Date brute')
#     axs[1].set_title("Sholl Analysis")
#     axs[1].set_xlabel("Rază (pixeli)")
#     axs[1].set_ylabel("Nr. Intersecții")
#     axs[1].grid(True)
#     axs[1].legend()
#
#     plt.tight_layout()
#
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path)
#     else:
#         plt.show()
#
#     return {
#         "peak_number": peak_number,
#         "radius_at_peak": radius_at_peak,
#         "auc": auc,
#         "slope": slope
#     }
#
# def _create_circle_mask(shape, center, radius):
#     Y, X = np.ogrid[:shape[0], :shape[1]]
#     dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
#     return (dist >= radius) & (dist < radius + 1)
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology
from skimage.morphology import skeletonize


def create_circle_mask(shape, center, radius):
    """Creează o mască circulară pentru o rază specifică."""
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return (dist >= radius) & (dist < radius + 1)


def empty_result():
    """Returnează rezultat gol pentru cazurile de eroare."""
    return {
        "peak_number": 0,
        "radius_at_peak": 0,
        "auc": 0,
        "slope": 0
    }


def sholl_analysis(image=None, image_path: str = None, soma_coords: tuple = None,
                   step_size: int = 10, max_radius: int = 300, save_path: str = None):
    """
    Analiza Sholl cu centrul personalizat din soma.

    Args:
        image: imaginea binară (numpy array)
        image_path: calea către imaginea binară
        soma_coords: coordonatele centrului soma-ului (x, y)
        step_size: pasul pentru cercurile concentrice
        max_radius: raza maximă pentru analiză
        save_path: calea pentru salvarea graficului

    Returns:
        dict cu rezultatele analizei
    """
    # Încarcă imaginea dacă nu e dată direct
    if image is None:
        if image_path is None:
            raise ValueError("Trebuie să oferi o imagine (image) sau un path (image_path).")
        image = io.imread(image_path)

    binary = image > 0

    if np.sum(binary) > 0:
        print("Imagine validă – există pixeli nenuli.")
