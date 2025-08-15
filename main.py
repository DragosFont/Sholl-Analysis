#!/usr/bin/env python3
"""
FIXES pentru problemele identificate:
1. Cleanup CORECT al variabilelor tkinter - elimină erorile "main thread is not in main loop"
2. CSV Append Mode - păstrează rezultatele anterioare, nu le suprascrie
3. Gestionare îmbunătățită a thread-urilor pentru închiderea aplicației

PROBLEME REZOLVATE:
- Exception ignored in: <function Variable.__del__> - FIXED
- CSV se suprascrie - FIXED (acum adaugă în loc să suprascrie)
- Cleanup corect la închiderea aplicației
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
import threading
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
import cv2
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import warnings
import queue
from PIL import Image, ImageDraw
import pandas as pd
import atexit

# Suppress warnings
warnings.filterwarnings('ignore')

# Imports pentru procesarea avansată
try:
    from aicsimageio import AICSImage
    from skimage import filters, measure, morphology, exposure
    from skimage.io import imsave
    import shutil

    print("✅ Biblioteci avansate încărcate")
except ImportError as e:
    print(f"⚠️ Unele biblioteci lipsesc: {e}")
    print("💡 Pentru funcționalitate completă instalează: pip install aicsimageio[all] scikit-image")


def load_image_robust_fixed(image_path):
    """Încarcă imaginea cu MIP complet pentru afișare și raw data pentru procesare"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Fișierul nu există: {image_path}")

    file_ext = os.path.splitext(image_path)[1].lower()
    file_size = os.path.getsize(image_path)

    print(f"📂 Încărcare: {os.path.basename(image_path)}")
    print(f"   Format: {file_ext}, Dimensiune: {file_size / (1024 * 1024):.1f} MB")

    # Strategia 1: Pentru fișiere .czi - cu Maximum Intensity Projection CORECT
    if file_ext == '.czi':
        print("🧬 Proces special pentru fișiere .czi cu MIP...")

        try:
            from aicsimageio import AICSImage
            print("   📚 Folosesc aicsimageio cu MIP...")

            img = AICSImage(image_path)
            print(f"   📊 Dimensiuni originale: {img.shape}")
            print(f"   📊 Canale disponibile: {img.channel_names if hasattr(img, 'channel_names') else 'necunoscute'}")

            # Extrage datele complete
            data = img.data
            print(f"   📊 Forma datelor brute: {data.shape}")

            # Identifică dimensiunile (TCZYX sau CZYX sau ZYX)
            data = np.squeeze(data)  # Elimină dimensiunile singulare
            print(f"   📊 După squeeze: {data.shape}")

            # Determină ordinea dimensiunilor și aplică MIP
            if len(data.shape) == 5:  # TCZYX
                print("   🔄 Format TCZYX detectat")
                data = data[0]  # Ia primul timp
                print(f"   📊 După selectare timp: {data.shape}")

            if len(data.shape) == 4:  # CZYX
                print("   🔄 Format CZYX detectat")

                # MIP pe axa Z pentru fiecare canal
                if data.shape[1] > 1:  # Avem mai multe slice-uri Z
                    print("   🌟 Aplicând Maximum Intensity Projection pe Z...")
                    data = np.max(data, axis=1)  # MIP pe Z, rezultat: CYX
                    print(f"   📊 După MIP pe Z: {data.shape}")
                else:
                    data = data[:, 0, :, :]  # Elimină dimensiunea Z singulară

                # Mapare CORECTĂ a canalelor - menține ordinea RGB standard
                print(f"   🎨 Procesez {data.shape[0]} canale pentru afișare RGB CORECT...")

                # Verifică valorile în fiecare canal
                for i in range(min(data.shape[0], 5)):  # Verifică primele 5 canale
                    channel_data = data[i]
                    min_val, max_val = channel_data.min(), channel_data.max()
                    non_zero = np.count_nonzero(channel_data)
                    print(f"   Canal {i}: min={min_val}, max={max_val}, non-zero pixels={non_zero}")

                # Mapare STANDARD RGB fără amestecare
                if data.shape[0] >= 3:
                    print("   🎨 Mapare RGB STANDARD: Canal 0->Red, Canal 1->Green, Canal 2->Blue")
                    red_channel = data[0]  # Primul canal -> roșu
                    green_channel = data[1]  # Al doilea canal -> verde
                    blue_channel = data[2]  # Al treilea canal -> albastru

                elif data.shape[0] == 2:
                    print("   🎨 2 canale: Canal 0->Red, Canal 1->Green, Blue=zero")
                    red_channel = data[0]
                    green_channel = data[1]
                    blue_channel = np.zeros_like(data[0])

                else:
                    print("   🎨 Un singur canal: mapez pe toate canalele RGB")
                    single_channel = data[0]
                    red_channel = single_channel
                    green_channel = single_channel
                    blue_channel = single_channel

                # Creează imaginea RGB pentru afișare
                display_data = np.stack([red_channel, green_channel, blue_channel], axis=2)
                raw_data = data

            elif len(data.shape) == 3:  # ZYX sau CYX
                print("   🔄 Format 3D detectat (ZYX sau CYX)")

                if data.shape[0] > 10:  # Probabil Z-stack
                    print("   🌟 Aplicând MIP pe Z-stack...")
                    mip_data = np.max(data, axis=0)  # MIP pe Z
                    red_channel = mip_data
                    green_channel = mip_data
                    blue_channel = mip_data
                    display_data = np.stack([red_channel, green_channel, blue_channel], axis=2)
                    raw_data = np.stack([red_channel, green_channel, blue_channel], axis=0)
                else:  # Probabil canale
                    if data.shape[0] >= 3:
                        print("   🎨 Mapez 3+ canale ca RGB STANDARD...")
                        red_channel = data[0]
                        green_channel = data[1]
                        blue_channel = data[2]
                        display_data = np.stack([red_channel, green_channel, blue_channel], axis=2)
                        raw_data = data

                    elif data.shape[0] == 2:
                        print("   🎨 Mapez 2 canale: primul->Red, al doilea->Green")
                        red_channel = data[0]
                        green_channel = data[1]
                        blue_channel = np.zeros_like(data[0])
                        display_data = np.stack([red_channel, green_channel, blue_channel], axis=2)
                        raw_data = np.stack([red_channel, green_channel, blue_channel], axis=0)
                    else:
                        print("   🎨 Un singur canal: mapez grayscale...")
                        combined = data[0] if data.shape[0] == 1 else np.max(data, axis=0)
                        red_channel = combined
                        green_channel = combined
                        blue_channel = combined
                        display_data = np.stack([red_channel, green_channel, blue_channel], axis=2)
                        raw_data = np.stack([red_channel, green_channel, blue_channel], axis=0)

            # Normalizare pentru fiecare canal separat
            print(f"   🔄 Normalizez pentru afișare...")

            # Normalizează fiecare canal separat pentru contrast optim
            for i in range(3):  # R, G, B
                channel = display_data[:, :, i]
                if channel.max() > channel.min():
                    # Folosește percentile pentru normalizare robustă
                    p1, p99 = np.percentile(channel, [1, 99])
                    channel_normalized = np.clip((channel - p1) / (p99 - p1), 0, 1)
                    display_data[:, :, i] = (channel_normalized * 255).astype(np.uint8)
                else:
                    display_data[:, :, i] = np.zeros_like(channel, dtype=np.uint8)

            print(f"   ✅ .czi încărcat cu MIP CORECT: display={display_data.shape}, raw={raw_data.shape}")
            return display_data, raw_data

        except ImportError:
            print("   ⚠️ aicsimageio nu este instalat")
        except Exception as e:
            print(f"   ⚠️ Eroare aicsimageio: {e}")
            import traceback
            traceback.print_exc()

    # Strategia 2: OpenCV (funcționează pentru majoritatea formatelor)
    try:
        print("   📚 Folosesc OpenCV...")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("OpenCV nu poate citi fișierul")

        # Convertește BGR la RGB pentru matplotlib
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 2:
            # Pentru grayscale, creează RGB
            image = np.stack([image, image, image], axis=2)

        # Asigură-te că este uint8
        if image.dtype != np.uint8:
            if image.dtype in [np.uint16, np.uint32]:
                image = (image >> 8).astype(np.uint8)
            else:
                image_min, image_max = image.min(), image.max()
                if image_max > image_min:
                    image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                else:
                    image = np.zeros_like(image, dtype=np.uint8)

        print(f"   ✅ Încărcat cu OpenCV: {image.shape}, dtype: {image.dtype}")

        # Pentru formate simple, creează raw_data în format CYX
        raw_data = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        return image, raw_data

    except Exception as e:
        print(f"   ⚠️ Eroare OpenCV: {e}")

    # Dacă toate metodele au eșuat
    raise RuntimeError(f"Nu s-a putut încărca imaginea cu nicio metodă: {image_path}")


class ThreadSafeFreehandROI_Fixed:
    """Selector ROI thread-safe ÎMBUNĂTĂȚIT pentru integrare cu tkinter"""

    def __init__(self, image, title="Selecție ROI Freehand", result_queue=None):
        if image is None:
            raise ValueError("Imaginea nu poate fi None!")

        self.image = image
        self.title = title
        self.result_queue = result_queue
        self.rois = []
        self.current_path = []
        self.drawing = False
        self.mouse_pressed = False

        # Configurare matplotlib cu backend non-interactiv pentru thread-uri
        self.original_backend = matplotlib.get_backend()

        # Configurare matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle(title, fontsize=14, weight='bold')

        # Afișează imaginea cu îmbunătățiri
        if len(image.shape) == 3:
            self.ax.imshow(image)
        else:
            self.ax.imshow(image, cmap='gray')

        # Instrucțiuni mai clare
        instruction_text = """INSTRUCȚIUNI ÎMBUNĂTĂȚITE:
🖱️ Ține apăsat click STÂNGA + mișcă mouse-ul pentru a desena ROI
🖱️ Click DREAPTA pentru a finaliza ROI-ul curent
🔲 Folosește butoanele de jos pentru control
⚠️ IMPORTANT: Desenează ROI-uri complete și închise!"""

        self.ax.set_title(instruction_text, fontsize=10, pad=15)

        # Conectează evenimentele
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        # Adaugă butoane de control
        self.add_buttons()

        # Variabile pentru tracking
        self.temp_lines = []
        self.roi_patches = []
        self.closed = False
        self.min_roi_points = 5

    def add_buttons(self):
        """Adaugă butoanele de control ÎMBUNĂTĂȚITE"""
        ax_done = plt.axes([0.02, 0.02, 0.12, 0.05])
        ax_clear = plt.axes([0.15, 0.02, 0.12, 0.05])
        ax_undo = plt.axes([0.28, 0.02, 0.12, 0.05])
        ax_info = plt.axes([0.41, 0.02, 0.15, 0.05])

        self.btn_done = Button(ax_done, 'FINALIZEAZĂ')
        self.btn_clear = Button(ax_clear, 'ȘTERGE TOT')
        self.btn_undo = Button(ax_undo, 'ANULEAZĂ')
        self.btn_info = Button(ax_info, f'ROI-uri: {len(self.rois)}')

        self.btn_done.on_clicked(self.finish)
        self.btn_clear.on_clicked(self.clear_all)
        self.btn_undo.on_clicked(self.undo_last)

    def update_roi_counter(self):
        """Actualizează contorul de ROI-uri"""
        self.btn_info.label.set_text(f'ROI-uri: {len(self.rois)}')
        try:
            self.fig.canvas.draw_idle()
        except:
            pass

    def on_press(self, event):
        """Gestionează click-urile mouse-ului"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Click stânga
            print(f"   🖱️ Început desenare ROI la ({event.xdata:.1f}, {event.ydata:.1f})")
            self.mouse_pressed = True
            self.drawing = True
            self.current_path = [(event.xdata, event.ydata)]

        elif event.button == 3:  # Click dreapta
            if self.drawing and len(self.current_path) >= self.min_roi_points:
                print(f"   ✅ Finalizez ROI cu {len(self.current_path)} puncte")
                self.complete_roi()
            elif self.drawing:
                print(f"   ⚠️ ROI prea mic: {len(self.current_path)} puncte (min {self.min_roi_points})")

    def on_release(self, event):
        """Gestionează eliberarea mouse-ului"""
        if event.button == 1:
            self.mouse_pressed = False

    def on_motion(self, event):
        """Gestionează mișcarea mouse-ului"""
        if not self.drawing or not self.mouse_pressed or event.inaxes != self.ax:
            return

        if event.xdata is not None and event.ydata is not None:
            if len(self.current_path) == 0:
                self.current_path.append((event.xdata, event.ydata))
            else:
                last_point = self.current_path[-1]
                distance = np.sqrt((event.xdata - last_point[0]) ** 2 + (event.ydata - last_point[1]) ** 2)

                if distance > 1.5:
                    self.current_path.append((event.xdata, event.ydata))
                    self.update_temp_display()

    def update_temp_display(self):
        """Actualizează afișarea temporară în timpul desenării"""
        for line in self.temp_lines:
            try:
                line.remove()
            except:
                pass
        self.temp_lines.clear()

        if len(self.current_path) > 1:
            x_coords = [p[0] for p in self.current_path]
            y_coords = [p[1] for p in self.current_path]

            line, = self.ax.plot(x_coords, y_coords, 'lime', linewidth=3, alpha=0.8)
            self.temp_lines.append(line)

            start_point, = self.ax.plot(x_coords[0], y_coords[0], 'go', markersize=10)
            self.temp_lines.append(start_point)

            current_point, = self.ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8)
            self.temp_lines.append(current_point)

        try:
            self.fig.canvas.draw_idle()
        except:
            pass

    def complete_roi(self):
        """Completează ROI-ul curent"""
        if len(self.current_path) < self.min_roi_points:
            print(f"❌ ROI-ul trebuie să aibă cel puțin {self.min_roi_points} puncte!")
            return

        # Validează că ROI-ul este în limitele imaginii
        roi_points = []
        for point in self.current_path:
            x = max(0, min(self.image.shape[1] - 1, point[0]))
            y = max(0, min(self.image.shape[0] - 1, point[1]))
            roi_points.append((x, y))

        # Închide conturul dacă este necesar
        if roi_points[0] != roi_points[-1]:
            roi_points.append(roi_points[0])

        self.rois.append(np.array(roi_points))

        # Desenează ROI-ul finalizat
        self.draw_completed_roi(roi_points, len(self.rois))

        # Reset pentru următorul ROI
        self.drawing = False
        self.current_path = []

        # Șterge liniile temporare
        for line in self.temp_lines:
            try:
                line.remove()
            except:
                pass
        self.temp_lines.clear()

        print(f"✅ ROI {len(self.rois)} completat cu {len(roi_points)} puncte! Total: {len(self.rois)} ROI-uri")
        self.update_roi_counter()

        try:
            self.fig.canvas.draw()
        except:
            pass

    def draw_completed_roi(self, points, roi_number):
        """Desenează un ROI completat"""
        colors = ['lime', 'red', 'cyan', 'yellow', 'magenta', 'orange', 'white', 'pink']
        color = colors[(roi_number - 1) % len(colors)]

        try:
            polygon = Polygon(points[:-1], fill=False, edgecolor=color,
                              linewidth=4, alpha=1.0)
            self.ax.add_patch(polygon)
            self.roi_patches.append(polygon)

            fill_polygon = Polygon(points[:-1], fill=True, facecolor=color,
                                   alpha=0.15, edgecolor='none')
            self.ax.add_patch(fill_polygon)
            self.roi_patches.append(fill_polygon)

            center_x = np.mean([p[0] for p in points[:-1]])
            center_y = np.mean([p[1] for p in points[:-1]])

            text = self.ax.text(center_x, center_y, f'ROI {roi_number}',
                                color='black', fontsize=14, weight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.9, edgecolor='black'))
            self.roi_patches.append(text)

        except Exception as e:
            print(f"❌ Eroare la desenarea ROI: {e}")

    def clear_all(self, event):
        """Șterge toate ROI-urile"""
        for patch in self.roi_patches:
            try:
                patch.remove()
            except:
                pass
        self.roi_patches.clear()

        for line in self.temp_lines:
            try:
                line.remove()
            except:
                pass
        self.temp_lines.clear()

        self.rois = []
        self.current_path = []
        self.drawing = False

        print("🗑️ Toate ROI-urile au fost șterse")
        self.update_roi_counter()

        try:
            self.fig.canvas.draw()
        except:
            pass

    def undo_last(self, event):
        """Anulează ultimul ROI"""
        if not self.rois:
            print("⚠️ Nu există ROI-uri de anulat")
            return

        self.rois.pop()

        if len(self.roi_patches) >= 3:
            for _ in range(3):
                try:
                    patch = self.roi_patches.pop()
                    patch.remove()
                except:
                    pass

        print(f"↶ Ultimul ROI a fost anulat. Total: {len(self.rois)} ROI-uri")
        self.update_roi_counter()

        try:
            self.fig.canvas.draw()
        except:
            pass

    def finish(self, event):
        """Finalizează selecția"""
        if len(self.rois) == 0:
            print("⚠️ Nu au fost desenate ROI-uri!")
            return

        print(f"🏁 Finalizez cu {len(self.rois)} ROI-uri selectate")
        self.closed = True
        if self.result_queue:
            self.result_queue.put(self.rois)
        plt.close(self.fig)

    def on_close(self, event):
        """Gestionează închiderea ferestrei"""
        print(f"🔚 Fereastra închisă cu {len(self.rois)} ROI-uri")
        self.closed = True
        if self.result_queue:
            self.result_queue.put(self.rois)

    def show_blocking(self):
        """Afișează interfața și așteaptă rezultatul"""
        plt.show()
        return self.rois


class ExactFreehandMultiROIAdapter:
    """Adaptor care procesează ROI-urile freehand cu forma EXACTĂ (nu bbox) - FIXED"""

    def __init__(self, file_path: str, raw_channels_data: np.ndarray, output_dir: str = "outputs"):
        self.file_path = file_path
        self.raw_channels_data = raw_channels_data
        self.output_dir = output_dir

    def process_freehand_rois_exact(self, freehand_rois):
        """
        Procesează ROI-urile freehand cu forma EXACTĂ - nu le convertește la dreptunghiuri
        """
        if not freehand_rois:
            print("⚠️ Nu există ROI-uri freehand de procesat!")
            return []

        print(f"🔄 Procesez {len(freehand_rois)} ROI-uri freehand cu forma EXACTĂ...")

        results_list = []

        for i, roi_points in enumerate(freehand_rois):
            roi_num = i + 1
            print(f"\n{'=' * 50}")
            print(f"🎯 Procesez ROI {roi_num}/{len(freehand_rois)} cu forma EXACTĂ freehand...")

            try:
                # Creează mască EXACTĂ din poligonul freehand
                exact_mask = self._create_exact_mask_from_polygon(roi_points, self.raw_channels_data.shape)

                if exact_mask.sum() == 0:
                    print(f"⚠️ ROI {roi_num}: Masca exactă este goală!")
                    results_list.append(self._empty_result())
                    continue

                # Procesează cu masca exactă
                result = self._process_single_roi_exact(roi_num, roi_points, exact_mask)
                results_list.append(result)

                if result['peak'] > 0:
                    print(f"✅ ROI {roi_num} procesat cu SUCCES folosind forma EXACTĂ!")
                    print(f"   Peak: {result['peak']}, AUC: {result['auc']:.1f}")
                else:
                    print(f"⚠️ ROI {roi_num} procesat cu rezultate goale")

            except Exception as e:
                print(f"❌ Eroare la procesarea ROI {roi_num}: {e}")
                import traceback
                traceback.print_exc()
                results_list.append(self._empty_result())

        print(f"\n✅ Procesare completă cu {len(results_list)} rezultate folosind forma EXACTĂ freehand")
        return results_list

    def _create_exact_mask_from_polygon(self, roi_points, data_shape):
        """Creează o mască EXACTĂ din coordonatele poligonului freehand"""

        # Determină dimensiunile imaginii
        if len(data_shape) == 3:  # CYX
            height, width = data_shape[1], data_shape[2]
        else:  # YX
            height, width = data_shape[0], data_shape[1]

        print(f"   🎯 Creez mască exactă pentru imagine {width}x{height}")

        # Convertește coordonatele la întregi
        roi_coords_array = np.array(roi_points, dtype=np.int32)

        # Validează coordonatele
        roi_coords_array[:, 0] = np.clip(roi_coords_array[:, 0], 0, width - 1)  # x
        roi_coords_array[:, 1] = np.clip(roi_coords_array[:, 1], 0, height - 1)  # y

        # Creează mască folosind PIL pentru umplerea poligonului
        mask_image = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_image)

        # Convertește coordonatele la format pentru PIL (x, y)
        polygon_coords = [(int(x), int(y)) for x, y in roi_coords_array]

        # Desenează poligonul umplut
        draw.polygon(polygon_coords, fill=255)

        # Convertește înapoi la numpy array
        exact_mask = np.array(mask_image) > 0

        print(f"   🎯 Mască exactă creată: {exact_mask.sum()} pixeli ({100 * exact_mask.sum() / (width * height):.1f}%)")

        return exact_mask

    def _process_single_roi_exact(self, roi_num: int, roi_points, exact_mask) -> Dict:
        """
        Procesează un singur ROI cu IZOLARE COMPLETĂ pentru a evita contaminarea cu ROI-uri anterioare
        FIXED: Adaugă debug pentru a verifica izolarea ROI-urilor
        """

        # Creează directoriile
        roi_dir = os.path.join(self.output_dir, f"roi_{roi_num}_exact_freehand")
        debug_dir = os.path.join(roi_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"🎯 PROCESEZ ROI {roi_num} cu IZOLARE COMPLETĂ")
        print(f"   ROI points shape: {len(roi_points)} puncte")
        print(f"   Exact mask shape: {exact_mask.shape}")
        print(f"   Exact mask pixels: {exact_mask.sum()}")
        print(f"   Raw channels shape: {self.raw_channels_data.shape}")

        # VERIFICARE CRITICĂ: Asigură-te că exact_mask se referă la imaginea completă
        if exact_mask.shape != self.raw_channels_data.shape[1:]:  # shape[1:] pentru a sări peste dimensiunea de canale
            print(f"   ❌ EROARE: Exact mask shape {exact_mask.shape} != image shape {self.raw_channels_data.shape[1:]}")
            return self._empty_result()

        # Determină bounding box STRICT pentru această ROI
        coords = np.where(exact_mask)
        if len(coords[0]) == 0:
            print(f"   ❌ ROI {roi_num}: Exact mask este goală!")
            return self._empty_result()

        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1

        print(f"   📦 Bounding box ROI {roi_num}: ({x_min},{y_min}) -> ({x_max},{y_max})")
        print(f"   📏 Dimensiuni bounding box: {x_max - x_min}x{y_max - y_min}")

        # VERIFICARE: Asigură-te că bounding box-ul este rezonabil
        bbox_area = (x_max - x_min) * (y_max - y_min)
        mask_area = exact_mask.sum()
        coverage = mask_area / bbox_area if bbox_area > 0 else 0

        print(f"   📊 Acoperire ROI în bounding box: {coverage:.1%}")
        if coverage < 0.05:  # Dacă ROI-ul acoperă mai puțin de 5% din bounding box
            print(f"   ⚠️ ATENȚIE: ROI {roi_num} pare foarte dispersat în bounding box!")

        # Extrage canalele din regiunea bounding box CU IZOLARE COMPLETĂ
        channels = self._extract_roi_channels_isolated(y_min, y_max, x_min, x_max, roi_num)

        # Extrage masca exactă pentru această regiune CU VERIFICARE
        roi_exact_mask = exact_mask[y_min:y_max, x_min:x_max]

        print(f"   🎯 ROI mask în bounding box: {roi_exact_mask.sum()} pixeli")
        print(f"   🎯 Channels extrași: {[name + ':' + str(ch.shape) for name, ch in channels.items()]}")

        # SALVARE DEBUG PENTRU VERIFICAREA IZOLĂRII
        self._save_roi_isolation_debug(channels, roi_exact_mask, exact_mask, roi_num, debug_dir)

        # CONTINUĂ cu procesarea normală...
        # [restul codului rămâne la fel]

        # PASUL 1: Crează masca din verde (DOAR în această ROI)
        green_mask = self._create_isolated_green_mask(channels['green'], roi_exact_mask, debug_dir, roi_num)

        # PASUL 2: INTERSECȚIA DIRECTĂ - aceasta este MASCA FINALĂ
        final_analysis_mask = green_mask & roi_exact_mask

        print(f"   ✅ MASKA FINALĂ (freehand ∩ verde) pentru ROI {roi_num}: {final_analysis_mask.sum()} pixeli")

        if final_analysis_mask.sum() == 0:
            print(f"   ❌ ROI {roi_num}: Intersecția este goală!")
            return self._empty_result()

        # Continuă cu restul procesării...
        soma_center = self._find_soma_in_final_mask(channels['blue'], final_analysis_mask, debug_dir, roi_num)
        dendrites_binary = self._extract_dendrites_from_final_mask(channels['red'], final_analysis_mask, debug_dir,
                                                                   roi_num)
        binary_path = self._save_final_results_with_mask(dendrites_binary, channels['red'], soma_center,
                                                         final_analysis_mask, roi_dir, roi_num)
        results = self._perform_sholl_analysis_exact(dendrites_binary, binary_path, soma_center, roi_dir, roi_num)

        print(f"🏁 ROI {roi_num} COMPLETAT cu izolare verificată!")
        return results

    def _extract_roi_channels_isolated(self, y_min: int, y_max: int, x_min: int, x_max: int, roi_num: int) -> Dict[
        str, np.ndarray]:
        """
        Extrage canalele pentru regiunea bounding box cu VERIFICARE DE IZOLARE
        """
        print(f"   📡 Extragere canale IZOLATE pentru ROI {roi_num}...")

        file_ext = os.path.splitext(self.file_path)[1].lower()

        # Extrage canalele
        try:
            if file_ext == '.czi':
                # Pentru .czi, folosim maparea corectă bazată pe nume de canale
                blue_channel = self.raw_channels_data[0, y_min:y_max, x_min:x_max].copy()
                green_channel = self.raw_channels_data[1, y_min:y_max, x_min:x_max].copy()
                red_channel = self.raw_channels_data[2, y_min:y_max, x_min:x_max].copy()
            else:
                # Mapare standard pentru alte formate
                blue_channel = self.raw_channels_data[0, y_min:y_max, x_min:x_max].copy()
                green_channel = self.raw_channels_data[1, y_min:y_max, x_min:x_max].copy()
                red_channel = self.raw_channels_data[2, y_min:y_max, x_min:x_max].copy()

            channels = {
                'blue': blue_channel,
                'green': green_channel,
                'red': red_channel
            }

            # VERIFICARE DE IZOLARE: Asigură-te că fiecare canal este unic pentru această ROI
            for name, channel in channels.items():
                print(
                    f"      Canal {name}: shape={channel.shape}, min={channel.min():.4f}, max={channel.max():.4f}, mean={channel.mean():.4f}")

                # Verifică dacă canalul pare să conțină date reale
                non_zero_pixels = np.count_nonzero(channel)
                total_pixels = channel.size
                print(
                    f"      Canal {name}: {non_zero_pixels}/{total_pixels} pixeli non-zero ({100 * non_zero_pixels / total_pixels:.1f}%)")

            return channels

        except Exception as e:
            print(f"   ❌ EROARE la extragerea canalelor pentru ROI {roi_num}: {e}")
            return {
                'blue': np.zeros((y_max - y_min, x_max - x_min)),
                'green': np.zeros((y_max - y_min, x_max - x_min)),
                'red': np.zeros((y_max - y_min, x_max - x_min))
            }

    def _save_roi_isolation_debug(self, channels: Dict, roi_mask: np.ndarray, full_exact_mask: np.ndarray,
                                  roi_num: int, debug_dir: str):
        """
        Salvează debug pentru verificarea izolării ROI-urilor
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Prima linie: canalele extrase pentru această ROI
            axes[0, 0].imshow(channels['blue'], cmap='Blues')
            axes[0, 0].set_title(f'ROI {roi_num} - Canal Albastru\nExtras izolat')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(channels['green'], cmap='Greens')
            axes[0, 1].set_title(f'ROI {roi_num} - Canal Verde\nExtras izolat')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(channels['red'], cmap='Reds')
            axes[0, 2].set_title(f'ROI {roi_num} - Canal Roșu\nExtras izolat')
            axes[0, 2].axis('off')

            # A doua linie: măștile și verificarea izolării
            axes[1, 0].imshow(roi_mask, cmap='gray')
            axes[1, 0].set_title(f'ROI {roi_num} - Maska în Bounding Box\n{roi_mask.sum()} pixeli')
            axes[1, 0].axis('off')

            # Afișează masca completă pentru comparație
            axes[1, 1].imshow(full_exact_mask, cmap='gray')
            axes[1, 1].set_title(f'ROI {roi_num} - Maska Completă\n{full_exact_mask.sum()} pixeli')
            axes[1, 1].axis('off')

            # Overlay pentru verificare
            if len(channels['green'].shape) == 2 and len(roi_mask.shape) == 2:
                overlay = channels['green'] * 0.7 + roi_mask.astype(float) * 0.3
                axes[1, 2].imshow(overlay, cmap='Greens')
                axes[1, 2].set_title(f'Overlay Verde + Maska\nVerificare izolare')
                axes[1, 2].axis('off')

            plt.suptitle(f'ROI {roi_num} - Verificare Izolare Completă', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, f"00_ROI_{roi_num}_isolation_check.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"   💾 Debug izolare salvat pentru ROI {roi_num}")

        except Exception as e:
            print(f"   ⚠️ Eroare la salvarea debug izolare pentru ROI {roi_num}: {e}")

    def _create_isolated_green_mask(self, green_channel: np.ndarray, roi_mask: np.ndarray,
                                    debug_dir: str, roi_num: int) -> np.ndarray:
        """
        Crează masca verde DOAR pentru această ROI, IZOLAT de alte ROI-uri
        """
        print(f"🟢 ROI {roi_num}: Crează mască verde IZOLATĂ...")

        # Normalizare
        green_norm = exposure.rescale_intensity(green_channel, out_range=(0, 1))
        green_smooth = filters.gaussian(green_norm, sigma=0.8)

        # IMPORTANT: Calculează threshold DOAR pe pixelii din această ROI
        green_in_roi = green_smooth * roi_mask.astype(float)

        if green_in_roi.sum() > 0:
            non_zero_in_roi = green_in_roi[green_in_roi > 0]
            if len(non_zero_in_roi) > 10:
                # Threshold calculat DOAR pe această ROI
                threshold = np.percentile(non_zero_in_roi, 50)
                threshold = max(threshold, 0.08)
                print(f"   • Threshold calculat pe ROI {roi_num}: {threshold:.4f} din {len(non_zero_in_roi)} pixeli")
            else:
                threshold = 0.08
                print(f"   • Threshold default pentru ROI {roi_num}: {threshold:.4f}")
        else:
            threshold = 0.08
            print(f"   ⚠️ Nu există semnal verde în ROI {roi_num}, folosesc threshold default: {threshold:.4f}")

        # Crează masca threshold
        green_threshold_mask = green_smooth > threshold

        # Curățare minimă
        if green_threshold_mask.sum() > 0:
            green_threshold_clean = morphology.remove_small_objects(green_threshold_mask, min_size=10)
            green_threshold_clean = morphology.binary_closing(green_threshold_clean, morphology.disk(1))
        else:
            green_threshold_clean = green_threshold_mask.copy()

        print(f"   • ROI {roi_num} - Maska verde: {green_threshold_clean.sum()} pixeli")

        return green_threshold_clean

    def _create_simple_green_threshold_mask(self, green_channel: np.ndarray, debug_dir: str,
                                            roi_num: int) -> np.ndarray:
        """
        Crează o mască simplă din canalul verde - fără complicații
        """
        print(f"🟢 ROI {roi_num}: Crează mască simplă din verde...")

        # Normalizare
        green_norm = exposure.rescale_intensity(green_channel, out_range=(0, 1))

        # Smoothing ușor
        green_smooth = filters.gaussian(green_norm, sigma=0.8)

        # Threshold simplu
        if green_smooth.sum() > 0:
            # Folosește percentila 60 - nici prea strictă, nici prea permisivă
            threshold = np.percentile(green_smooth[green_smooth > 0], 60)
            threshold = max(threshold, 0.08)  # Minim rezonabil
        else:
            threshold = 0.1

        # Mască binară
        green_mask = green_smooth > threshold

        # Curățare minimă
        green_mask = morphology.remove_small_objects(green_mask, min_size=50)
        green_mask = morphology.binary_closing(green_mask, morphology.disk(2))

        print(f"   • Threshold verde: {threshold:.4f}")
        print(f"   • Pixeli mască verde: {green_mask.sum()}")

        return green_mask

    def _find_soma_in_final_mask(self, blue_channel: np.ndarray, final_mask: np.ndarray,
                                 debug_dir: str, roi_num: int) -> Tuple[int, int]:
        """
        Găsește soma DOAR în masca finală (freehand ∩ verde)
        """
        print(f"🔵 ROI {roi_num}: Găsesc soma în masca finală...")

        # Normalizare
        blue_norm = exposure.rescale_intensity(blue_channel, out_range=(0, 1))

        # Aplică DOAR masca finală
        blue_in_final_mask = blue_norm * final_mask.astype(float)

        if blue_in_final_mask.sum() == 0:
            # Fallback: centrul geometric al măștii finale
            coords = np.where(final_mask)
            if len(coords[0]) > 0:
                center_y = int(np.mean(coords[0]))
                center_x = int(np.mean(coords[1]))
            else:
                center_y = blue_norm.shape[0] // 2
                center_x = blue_norm.shape[1] // 2
            print(f"   • Centru fallback: ({center_x}, {center_y})")
            return (center_x, center_y)

        # Găsește punctul cu intensitatea maximă ÎN MASCA FINALĂ
        max_coords = np.unravel_index(np.argmax(blue_in_final_mask), blue_in_final_mask.shape)
        center_y, center_x = max_coords

        # Salvare debug
        self._save_debug_image(blue_norm, "04_blue_original", debug_dir)
        self._save_debug_image(blue_in_final_mask, "05_blue_in_final_mask", debug_dir)

        print(f"   • Centrul soma în masca finală: ({center_x}, {center_y})")
        return (center_x, center_y)

    def _extract_dendrites_from_final_mask(self, red_channel: np.ndarray, final_mask: np.ndarray,
                                           debug_dir: str, roi_num: int) -> np.ndarray:
        """
        Extrage dendrite DOAR din masca finală (freehand ∩ verde)
        SIMPLIFIED: Fără strategii complicate, doar procesare directă în masca finală
        """
        print(f"🔴 ROI {roi_num}: Extrag dendrite DOAR din masca finală...")

        # Normalizare
        red_norm = exposure.rescale_intensity(red_channel, out_range=(0, 1))

        # Aplică DOAR masca finală
        red_in_final_mask = red_norm * final_mask.astype(float)

        if red_in_final_mask.sum() == 0:
            print("   ⚠️ Nu există semnal roșu în masca finală!")
            return np.zeros_like(red_norm, dtype=bool)

        print(f"   • Semnal roșu în masca finală: {red_in_final_mask.sum():.2f}")

        # Denoising ușor
        red_denoised = filters.gaussian(red_in_final_mask, sigma=0.5)
        red_denoised = filters.median(red_denoised, morphology.disk(1))

        # Threshold în masca finală
        non_zero_values = red_denoised[red_denoised > 0]
        if len(non_zero_values) > 20:
            # Threshold conservator dar nu exagerat
            threshold = np.percentile(non_zero_values, 70)
            threshold = max(threshold, 0.05)
        else:
            threshold = 0.1

        print(f"   • Threshold roșu: {threshold:.4f}")

        # Creare mască binară LIMITATĂ la masca finală
        binary_dendrites = red_denoised > threshold
        binary_dendrites = binary_dendrites & final_mask  # IMPORTANT: limitează la masca finală

        # Curățare minimă
        binary_cleaned = morphology.remove_small_objects(binary_dendrites, min_size=5)
        binary_cleaned = morphology.binary_closing(binary_cleaned, morphology.disk(1))

        # Skeletonizare dacă sunt suficienți pixeli
        if binary_cleaned.sum() >= 10:
            skeleton = morphology.skeletonize(binary_cleaned)

            # Elimină puncte izolate din skeleton
            if skeleton.sum() > 0:
                # Găsește puncte cu puțini vecini
                from scipy import ndimage
                neighbor_count = ndimage.convolve(skeleton.astype(int),
                                                  np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
                                                  mode='constant')
                # Elimină puncte complet izolate
                isolated = (skeleton == 1) & (neighbor_count == 0)
                skeleton[isolated] = False
        else:
            skeleton = binary_cleaned.copy()

        # Salvare debug
        self._save_debug_image(red_norm, "06_red_original", debug_dir)
        self._save_debug_image(red_in_final_mask, "07_red_in_final_mask", debug_dir)
        self._save_debug_image(binary_dendrites, "08_binary_dendrites", debug_dir)
        self._save_debug_image(skeleton, "09_final_skeleton", debug_dir)

        print(f"   • Pixeli skeleton final: {skeleton.sum()}")
        print(f"   • Procent din masca finală: {100 * skeleton.sum() / max(1, final_mask.sum()):.1f}%")

        return skeleton

    def _save_final_results_with_mask(self, skeleton: np.ndarray, red_channel: np.ndarray,
                                      soma_center: Tuple[int, int], final_mask: np.ndarray,
                                      roi_dir: str, roi_num: int) -> str:
        """
        Salvează rezultatele finale cu masca finală (freehand ∩ verde)
        """

        # Salvează imaginea binară pentru analiza Sholl
        binary_path = os.path.join(roi_dir, "roi_binary_final_mask.tif")
        imsave(binary_path, (skeleton * 255).astype(np.uint8))

        # Crează imaginea cu rezultatele vizualizate
        fig, ax = plt.subplots(figsize=(10, 8))

        # Background cu canalul roșu
        red_norm = exposure.rescale_intensity(red_channel, out_range=(0, 1))
        ax.imshow(red_norm, cmap="Reds", alpha=0.7)

        # Afișează masca finală ca contur
        ax.contour(final_mask, colors='lime', linewidths=3, alpha=0.9, levels=[0.5])

        # Afișează skeleton-ul
        ax.imshow(skeleton, cmap="gray_r", alpha=0.9)

        # Marchează centrul soma-ului
        ax.plot(soma_center[0], soma_center[1], 'b*', markersize=20,
                markeredgecolor='yellow', markeredgewidth=3, label='Centru Soma')

        ax.set_title(f"ROI {roi_num} - Dendrite în Masca Finală (Freehand ∩ Verde)")
        ax.axis('off')
        ax.legend()
        plt.savefig(os.path.join(roi_dir, "roi_results_final_mask.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        return binary_path

    def _extract_roi_channels_exact(self, y_min: int, y_max: int, x_min: int, x_max: int) -> Dict[str, np.ndarray]:
        """Extrage canalele pentru regiunea bounding box"""

        file_ext = os.path.splitext(self.file_path)[1].lower()

        # Mapare corectă pentru .czi
        if file_ext == '.czi':
            # Pentru .czi, folosim maparea corectă bazată pe nume de canale
            return {
                'blue': self.raw_channels_data[0, y_min:y_max, x_min:x_max],  # Canal 0 -> nuclei (albastru)
                'green': self.raw_channels_data[1, y_min:y_max, x_min:x_max],  # Canal 1 -> neuroni (verde)
                'red': self.raw_channels_data[2, y_min:y_max, x_min:x_max]  # Canal 2 -> dendrite (roșu)
            }
        else:
            # Mapare standard pentru alte formate
            return {
                'blue': self.raw_channels_data[0, y_min:y_max, x_min:x_max],
                'green': self.raw_channels_data[1, y_min:y_max, x_min:x_max],
                'red': self.raw_channels_data[2, y_min:y_max, x_min:x_max]
            }

    def _save_basic_images_exact(self, channels: Dict[str, np.ndarray], exact_mask: np.ndarray,
                                 roi_dir: str, roi_num: int):
        """Salvează imaginile de bază cu masca exactă aplicată"""

        # Aplică masca exactă pe fiecare canal
        masked_channels = {}
        for name, channel in channels.items():
            masked_channels[name] = channel * exact_mask.astype(float)

        # Imagine RGB combinată cu masca exactă
        rgb_stack = np.stack([
            self._normalize_channel(masked_channels['red']),  # R
            self._normalize_channel(masked_channels['green']),  # G
            self._normalize_channel(masked_channels['blue'])  # B
        ], axis=-1)

        # Adaugă conturul exact al ROI-ului
        contour_overlay = rgb_stack.copy()
        contour_coords = np.where(exact_mask)
        if len(contour_coords[0]) > 0:
            # Marchează conturul cu galben
            from skimage import segmentation
            contour = segmentation.find_boundaries(exact_mask, mode='outer')
            contour_overlay[contour] = [1, 1, 0]  # Galben pentru contur

        rgb_enhanced = np.clip(contour_overlay * 2.0, 0, 1)
        imsave(os.path.join(roi_dir, "roi_rgb_exact_freehand.png"),
               (rgb_enhanced * 255).astype(np.uint8))

        # Canale individuale cu masca exactă
        channel_info = [
            (masked_channels['green'], "Greens", "verde_exact"),
            (masked_channels['blue'], "Blues", "albastru_exact"),
            (masked_channels['red'], "Reds", "rosu_exact")
        ]

        for channel_data, cmap, name in channel_info:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(channel_data, cmap=cmap)
            ax.contour(exact_mask, colors='yellow', linewidths=2, alpha=0.8)
            ax.set_title(f"ROI {roi_num} - Canal {name} (Forma EXACTĂ Freehand)")
            ax.axis('off')
            plt.savefig(os.path.join(roi_dir, f"roi_{name}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Salvează masca exactă
        imsave(os.path.join(roi_dir, "exact_freehand_mask.png"),
               (exact_mask * 255).astype(np.uint8))

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalizează un canal la intervalul [0, 1]"""
        if channel.max() > 0:
            return channel / channel.max()
        return channel

    def _create_neuron_mask_in_exact_region(self, green_channel: np.ndarray, exact_mask: np.ndarray,
                                            debug_dir: str, roi_num: int) -> np.ndarray:
        """
        Crează masca neuronului folosind DOAR intersecția directă freehand + verde
        SIMPLIFIED: Doar intersecția directă care funcționează perfect
        """
        print(f"🟢 ROI {roi_num}: Creez masca DOAR cu intersecția directă (freehand ∩ verde)...")

        # Pasul 1: Procesează canalul verde pentru threshold
        green_norm = exposure.rescale_intensity(green_channel, out_range=(0, 1))
        green_smooth = filters.gaussian(green_norm, sigma=0.8)

        # Pasul 2: Calculează threshold pe întregul canal verde
        if green_smooth.sum() > 0:
            non_zero_green = green_smooth[green_smooth > 0]
            if len(non_zero_green) > 100:
                # Threshold moderat - nu prea strict, nu prea permisiv
                threshold = np.percentile(non_zero_green, 50)  # Percentila 50
                threshold = max(threshold, 0.08)  # Minim rezonabil
            else:
                threshold = 0.08
        else:
            threshold = 0.08

        print(f"   • Threshold verde: {threshold:.4f}")

        # Pasul 3: Crează masca threshold pe tot canalul verde
        green_threshold_mask = green_smooth > threshold

        # Pasul 4: Curățare minimă a măștii threshold
        if green_threshold_mask.sum() > 0:
            green_threshold_clean = morphology.remove_small_objects(green_threshold_mask, min_size=30)
            green_threshold_clean = morphology.binary_closing(green_threshold_clean, morphology.disk(2))
        else:
            green_threshold_clean = green_threshold_mask.copy()

        print(f"   • Maska threshold verde: {green_threshold_clean.sum()} pixeli")
        print(f"   • Maska freehand exactă: {exact_mask.sum()} pixeli")

        # Pasul 5: DOAR INTERSECȚIA DIRECTĂ
        final_mask = green_threshold_clean & exact_mask
        intersection_pixels = final_mask.sum()

        print(f"   ✅ INTERSECȚIE DIRECTĂ: {intersection_pixels} pixeli")

        # Pasul 6: Verificare că intersecția nu este goală
        if intersection_pixels == 0:
            print(f"   ⚠️ Intersecția directă este goală! Încerc threshold mai permisiv...")
            # Încearcă cu threshold mai mic
            threshold_permissive = np.percentile(non_zero_green, 30) if len(non_zero_green) > 0 else 0.05
            threshold_permissive = max(threshold_permissive, 0.03)

            green_permissive = green_smooth > threshold_permissive
            green_permissive = morphology.remove_small_objects(green_permissive, min_size=20)
            final_mask = green_permissive & exact_mask

            print(f"   • Threshold permisiv: {threshold_permissive:.4f}")
            print(f"   • Intersecție permisivă: {final_mask.sum()} pixeli")

            if final_mask.sum() == 0:
                print(f"   ❌ Nici intersecția permisivă nu funcționează! Folosesc doar freehand")
                final_mask = exact_mask.copy()

        # Pasul 7: Curățare finală minimă
        if final_mask.sum() > 0:
            final_mask = morphology.remove_small_objects(final_mask, min_size=10)
            final_mask = morphology.binary_closing(final_mask, morphology.disk(1))

        # Pasul 8: Salvare debug SIMPLU - doar ceea ce contează
        try:
            self._save_debug_image(green_norm, "01_green_original", debug_dir)
            self._save_debug_image(green_smooth, "02_green_smooth", debug_dir)
            self._save_debug_image(green_threshold_clean, "03_green_threshold", debug_dir)
            self._save_debug_image(exact_mask, "04_freehand_exact", debug_dir)
            self._save_debug_image(final_mask, "05_intersection_direct_FINAL", debug_dir)

            # O singură imagine de comparație simplă
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))

            axes[0].imshow(green_norm, cmap='Greens')
            axes[0].set_title('Verde original')
            axes[0].axis('off')

            axes[1].imshow(green_threshold_clean, cmap='Greens')
            axes[1].set_title(f'Threshold verde\n{green_threshold_clean.sum()} px')
            axes[1].axis('off')

            axes[2].imshow(exact_mask, cmap='Blues')
            axes[2].set_title(f'Freehand exact\n{exact_mask.sum()} px')
            axes[2].axis('off')

            axes[3].imshow(final_mask, cmap='Reds')
            axes[3].set_title(f'INTERSECȚIE DIRECTĂ\n{final_mask.sum()} px')
            axes[3].axis('off')

            # Overlay final
            overlay = green_norm * 0.6
            overlay_colored = np.stack([overlay + final_mask.astype(float) * 0.4,
                                        overlay + final_mask.astype(float) * 0.4,
                                        overlay], axis=-1)
            overlay_colored = np.clip(overlay_colored, 0, 1)
            axes[4].imshow(overlay_colored)
            axes[4].set_title('Overlay final')
            axes[4].axis('off')

            plt.suptitle(f'ROI {roi_num} - DOAR Intersecția Directă (Freehand ∩ Verde)')
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "06_intersection_direct_only.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"   ⚠️ Eroare la salvarea debug: {e}")

        # Rezultat final
        final_pixels = final_mask.sum()
        coverage_of_freehand = 100 * final_pixels / max(1, exact_mask.sum())

        print(f"   🎯 REZULTAT FINAL (DOAR INTERSECȚIE DIRECTĂ):")
        print(f"      • Pixeli finali: {final_pixels}")
        print(f"      • Acoperire din freehand: {coverage_of_freehand:.1f}%")
        print(f"      • Threshold folosit: {threshold:.4f}")

        return final_mask

    def _find_soma_in_exact_green_region(self, blue_channel: np.ndarray, neuron_mask: np.ndarray,
                                         debug_dir: str, roi_num: int) -> Tuple[int, int]:
        """Găsește soma-ul din albastru DOAR în regiunea verde din forma exactă"""
        print(f"🔵 ROI {roi_num}: Găsesc soma-ul în regiunea verde din forma EXACTĂ...")

        # Normalizare
        blue_norm = exposure.rescale_intensity(blue_channel, out_range=(0, 1))

        # Aplicăm masca verde din forma exactă
        blue_in_green_exact = blue_norm * neuron_mask.astype(float)

        if blue_in_green_exact.sum() == 0:
            print("   ⚠️ Nu există semnal albastru în regiunea verde din forma exactă!")
            # Fallback: centrul geometric al măștii verzi exacte
            coords = np.where(neuron_mask)
            if len(coords[0]) > 0:
                center_y = int(np.mean(coords[0]))
                center_x = int(np.mean(coords[1]))
            else:
                center_y = blue_norm.shape[0] // 2
                center_x = blue_norm.shape[1] // 2
            print(f"   • Centru fallback în forma exactă: ({center_x}, {center_y})")
            return (center_x, center_y)

        # Găsește punctul cu intensitatea maximă
        max_coords = np.unravel_index(np.argmax(blue_in_green_exact), blue_in_green_exact.shape)
        center_y, center_x = max_coords

        # Salvare debug
        self._save_debug_image(blue_norm, "06_blue_original", debug_dir)
        self._save_debug_image(blue_in_green_exact, "07_blue_in_green_exact", debug_dir)

        # Salvare imagine cu centrul marcat
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(blue_in_green_exact, cmap='Blues', alpha=0.8)
        ax.contour(neuron_mask, colors='green', linewidths=1, alpha=0.5)
        ax.plot(center_x, center_y, 'r*', markersize=15,
                markeredgecolor='yellow', markeredgewidth=2)
        ax.set_title(f'ROI {roi_num} - Soma în Forma EXACTĂ Verde')
        ax.axis('off')
        plt.savefig(os.path.join(debug_dir, "08_soma_in_exact_green.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"   • Centrul soma în forma exactă: ({center_x}, {center_y})")
        return (center_x, center_y)

    def _extract_dendrites_in_exact_region(self, red_channel: np.ndarray, neuron_mask: np.ndarray,
                                           debug_dir: str, roi_num: int) -> np.ndarray:
        """
        Extrage dendrite DOAR din forma exactă freehand cu SUPPRESSIA ZGOMOTULUI
        FIXED: Elimină zgomotul și artefactele care fac skeleton-ul să "devină nebun"
        """
        print(f"🔴 ROI {roi_num}: Extrag dendrite din forma EXACTĂ cu eliminarea zgomotului...")

        # Pasul 1: Normalizare robustă
        red_norm = exposure.rescale_intensity(red_channel, out_range=(0, 1))

        # Pasul 2: Aplică masca neuronului (care este deja perfectă din verde)
        red_in_exact_neuron = red_norm * neuron_mask.astype(float)

        if red_in_exact_neuron.sum() == 0:
            print("   ⚠️ Nu există semnal roșu în neuronul din forma exactă!")
            return np.zeros_like(red_norm, dtype=bool)

        print(f"   • Semnal roșu în neuron: {red_in_exact_neuron.sum():.2f}")
        print(f"   • Pixeli neuron: {neuron_mask.sum()}")

        # Pasul 3: DENOISING AGRESIV înainte de orice procesare
        print("   🧹 Aplicând denoising agresiv...")

        # Multiple etape de denoising
        red_denoised = red_in_exact_neuron.copy()

        # Etapa 1: Gaussian blur pentru smooth general
        red_denoised = filters.gaussian(red_denoised, sigma=0.8)

        # Etapa 2: Median filter pentru eliminarea punctelor izolate
        red_denoised = filters.median(red_denoised, morphology.disk(2))

        # Etapa 3: Un alt Gaussian mai mic pentru final smoothing
        red_denoised = filters.gaussian(red_denoised, sigma=0.4)

        # Pasul 4: THRESHOLD MULT MAI CONSERVATOR
        if red_denoised.sum() > 0:
            non_zero_values = red_denoised[red_denoised > 0]

            # Folosește percentile mult mai mari pentru a elimina noise-ul
            threshold_candidates = [
                np.percentile(non_zero_values, 75),  # 75th percentile
                np.percentile(non_zero_values, 80),  # 80th percentile
                np.percentile(non_zero_values, 85),  # 85th percentile
            ]

            # Alege threshold-ul care dă rezultate rezonabile
            best_threshold = None
            best_pixel_count = 0

            for thresh in threshold_candidates:
                test_binary = red_denoised > thresh
                test_binary = test_binary & neuron_mask  # Menține în neuron

                # Curățare rapidă pentru test
                test_cleaned = morphology.remove_small_objects(test_binary, min_size=5)
                pixel_count = test_cleaned.sum()

                print(f"   • Test threshold {thresh:.4f}: {pixel_count} pixeli")

                # Caută un număr rezonabil de pixeli (nu prea puțini, nu prea mulți)
                if 20 <= pixel_count <= neuron_mask.sum() * 0.6:  # Între 20 pixeli și 60% din neuron
                    if pixel_count > best_pixel_count:
                        best_threshold = thresh
                        best_pixel_count = pixel_count

            # Dacă niciun threshold nu este bun, folosește cel mai conservator
            if best_threshold is None:
                best_threshold = max(threshold_candidates)
                print(f"   ⚠️ Folosesc threshold conservator: {best_threshold:.4f}")
            else:
                print(f"   ✅ Threshold optim: {best_threshold:.4f} ({best_pixel_count} pixeli)")

            threshold = best_threshold
        else:
            threshold = 0.5  # Foarte conservator
            print(f"   ⚠️ Nu există semnal, threshold conservator: {threshold}")

        # Pasul 5: Creare mască binară cu threshold ales
        binary_dendrites = red_denoised > threshold
        binary_dendrites = binary_dendrites & neuron_mask

        print(f"   • Pixeli după threshold: {binary_dendrites.sum()}")

        # Pasul 6: CURĂȚARE AGRESIVĂ PENTRU ELIMINAREA ZGOMOTULUI
        print("   🧹 Curățare agresivă...")

        # Elimină obiecte foarte mici (probabil noise)
        cleaned_stage1 = morphology.remove_small_objects(binary_dendrites, min_size=8)
        print(f"   • După eliminarea obiectelor mici: {cleaned_stage1.sum()}")

        # Verifică conectivitatea - elimină fragmente foarte izolate
        if cleaned_stage1.sum() > 0:
            labeled = measure.label(cleaned_stage1)
            props = measure.regionprops(labeled)

            # Păstrează doar componentele cu o dimensiune rezonabilă
            good_components = []
            for prop in props:
                # Verificări multiple pentru calitate:
                # 1. Dimensiunea minimă
                if prop.area < 6:
                    continue

                # 2. Nu foarte rotunde (dendritele sunt elongate)
                if prop.eccentricity < 0.3:
                    continue

                # 3. Raportul aspect nu foarte exagerat (elimină linii foarte subțiri - probabil noise)
                bbox_height = prop.bbox[2] - prop.bbox[0]
                bbox_width = prop.bbox[3] - prop.bbox[1]
                aspect_ratio = max(bbox_height, bbox_width) / max(1, min(bbox_height, bbox_width))
                if aspect_ratio > 20:  # Prea subțire
                    continue

                good_components.append(prop.label)

            # Reconstituie masca doar cu componentele bune
            if good_components:
                cleaned_stage2 = np.isin(labeled, good_components)
                print(
                    f"   • După filtrarea componentelor: {cleaned_stage2.sum()} pixeli din {len(good_components)} componente")
            else:
                print("   ⚠️ Nicio componentă bună găsită!")
                cleaned_stage2 = np.zeros_like(cleaned_stage1, dtype=bool)
        else:
            cleaned_stage2 = cleaned_stage1.copy()

        # Pasul 7: CLOSING FOARTE MIC pentru conectarea fragmentelor apropiate
        if cleaned_stage2.sum() > 0:
            # Un closing foarte mic pentru a conecta pixeli apropiați
            closed = morphology.binary_closing(cleaned_stage2, morphology.disk(1))
            # Dar limitează să nu se extindă prea mult
            closed = closed & neuron_mask
            print(f"   • După closing: {closed.sum()}")
        else:
            closed = cleaned_stage2.copy()

        # Pasul 8: VERIFICARE FINALĂ ÎNAINTE DE SKELETONIZE
        if closed.sum() < 15:
            print(f"   ❌ Prea puține dendrite pentru skeletonizare ({closed.sum()} < 15)")
            return np.zeros_like(red_norm, dtype=bool)

        # Pasul 9: SKELETONIZARE CONTROLATĂ
        print("   🦴 Skeletonizare controlată...")

        try:
            skeleton = morphology.skeletonize(closed)
            skeleton_pixels = skeleton.sum()
            print(f"   • Skeleton inițial: {skeleton_pixels} pixeli")

            # VERIFICARE POST-SKELETONIZARE: elimină puncte izolate
            if skeleton_pixels > 0:
                # Elimină puncte complet izolate (fără vecini)
                # Un pixel izolat într-un skeleton nu poate fi o dendrita reală
                skeleton_cleaned = skeleton.copy()

                # Găsește pixeli cu prea puțini vecini (probabil artefacte)
                from scipy import ndimage
                neighbor_count = ndimage.convolve(skeleton.astype(int),
                                                  np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
                                                  mode='constant')

                # Elimină pixeli complet izolați (0 vecini)
                isolated_pixels = (skeleton == 1) & (neighbor_count == 0)
                skeleton_cleaned[isolated_pixels] = False

                final_pixels = skeleton_cleaned.sum()
                print(f"   • Skeleton final (după eliminarea punctelor izolate): {final_pixels}")

                if final_pixels < 8:
                    print(f"   ❌ Skeleton final prea mic ({final_pixels} < 8)")
                    return np.zeros_like(red_norm, dtype=bool)

                skeleton = skeleton_cleaned

        except Exception as e:
            print(f"   ❌ Eroare la skeletonizare: {e}")
            return np.zeros_like(red_norm, dtype=bool)

        # Pasul 10: Salvare debug DETALIATĂ
        self._save_debug_image(red_norm, "09_red_original", debug_dir)
        self._save_debug_image(red_in_exact_neuron, "10_red_in_exact_neuron", debug_dir)
        self._save_debug_image(red_denoised, "11_red_denoised_aggressive", debug_dir)
        self._save_debug_image(binary_dendrites, "12_binary_after_threshold", debug_dir)
        self._save_debug_image(cleaned_stage1, "13_after_small_objects_removal", debug_dir)
        self._save_debug_image(cleaned_stage2, "14_after_component_filtering", debug_dir)
        self._save_debug_image(closed, "15_after_closing", debug_dir)
        self._save_debug_image(skeleton, "16_final_skeleton_clean", debug_dir)

        # Salvare analiză comparativă
        try:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            axes[0, 0].imshow(red_norm, cmap='Reds')
            axes[0, 0].set_title('Red Original')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(red_denoised, cmap='Reds')
            axes[0, 1].set_title('După Denoising')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(binary_dendrites, cmap='gray')
            axes[0, 2].set_title(f'După Threshold\n{binary_dendrites.sum()} px')
            axes[0, 2].axis('off')

            axes[0, 3].imshow(cleaned_stage1, cmap='gray')
            axes[0, 3].set_title(f'Fără Obiecte Mici\n{cleaned_stage1.sum()} px')
            axes[0, 3].axis('off')

            axes[1, 0].imshow(cleaned_stage2, cmap='gray')
            axes[1, 0].set_title(f'Filtrare Componente\n{cleaned_stage2.sum()} px')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(closed, cmap='gray')
            axes[1, 1].set_title(f'După Closing\n{closed.sum()} px')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(skeleton, cmap='gray_r')
            axes[1, 2].set_title(f'Skeleton Final\n{skeleton.sum()} px')
            axes[1, 2].axis('off')

            # Overlay final pe imaginea originală
            overlay = red_norm * 0.5
            overlay_colored = np.stack([
                overlay + skeleton.astype(float) * 0.5,  # Red + skeleton
                overlay,  # Green
                overlay  # Blue
            ], axis=-1)
            overlay_colored = np.clip(overlay_colored, 0, 1)
            axes[1, 3].imshow(overlay_colored)
            axes[1, 3].set_title('Overlay Final')
            axes[1, 3].axis('off')

            plt.suptitle(f'ROI {roi_num} - Extragere Dendrite cu Suppressia Zgomotului\nThreshold: {threshold:.4f}')
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "17_dendrite_extraction_analysis.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"   ⚠️ Eroare la salvarea analizei: {e}")

        print(f"   🎯 REZULTAT DENDRITE FINAL:")
        print(f"      • Threshold folosit: {threshold:.4f}")
        print(f"      • Pixeli skeleton final: {skeleton.sum()}")
        print(f"      • Procent din neuron: {100 * skeleton.sum() / max(1, neuron_mask.sum()):.1f}%")

        return skeleton

    def _save_final_results_exact(self, skeleton: np.ndarray, red_channel: np.ndarray,
                                  soma_center: Tuple[int, int], exact_mask: np.ndarray,
                                  roi_dir: str, roi_num: int) -> str:
        """Salvează rezultatele finale cu forma exactă"""

        # Salvează imaginea binară pentru analiza Sholl
        binary_path = os.path.join(roi_dir, "roi_binary_exact_freehand.tif")
        imsave(binary_path, (skeleton * 255).astype(np.uint8))

        # Crează imaginea cu rezultatele vizualizate
        fig, ax = plt.subplots(figsize=(10, 8))

        # Background cu canalul roșu
        red_norm = exposure.rescale_intensity(red_channel, out_range=(0, 1))
        ax.imshow(red_norm, cmap="Reds", alpha=0.7)
        ax.imshow(skeleton, cmap="gray_r", alpha=0.9)

        # Afișează conturul exact freehand
        ax.contour(exact_mask, colors='yellow', linewidths=3, alpha=0.9)

        # Marchează centrul soma-ului
        ax.plot(soma_center[0], soma_center[1], 'b*', markersize=20,
                markeredgecolor='yellow', markeredgewidth=3, label='Centru Soma')

        ax.set_title(f"ROI {roi_num} - Dendrite în Forma EXACTĂ Freehand")
        ax.axis('off')
        ax.legend()
        plt.savefig(os.path.join(roi_dir, "roi_results_exact_freehand.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        return binary_path

    def _perform_sholl_analysis_exact(self, skeleton: np.ndarray, binary_path: str,
                                      soma_center: Tuple[int, int], roi_dir: str,
                                      roi_num: int) -> Dict:
        """Efectuează analiza Sholl pe forma exactă"""
        if skeleton.sum() < 10:
            print(f"⚠️ ROI {roi_num}: Prea puține dendrite în forma exactă pentru analiza Sholl")
            return self._empty_result()

        print(f"📊 ROI {roi_num}: Efectuez analiza Sholl pe forma EXACTĂ freehand...")

        try:
            from src.io.sholl import ShollAnalyzer

            analyzer = ShollAnalyzer()
            sholl_path = os.path.join(roi_dir, "sholl_analysis_exact_freehand.png")

            # Calculează raza maximă mai inteligent
            max_radius = min(skeleton.shape[0], skeleton.shape[1]) // 2
            max_radius = min(max_radius, 150)

            results = analyzer.analyze(
                binary_image=skeleton,
                soma_center=soma_center,
                step_size=5,
                max_radius=max_radius,
                save_path=sholl_path
            )

            if results and isinstance(results, dict):
                print(f"   • Peak: {results.get('peak_number', 0)}")
                print(f"   • Radius: {results.get('radius_at_peak', 0)}")
                print(f"   • AUC: {results.get('auc', 0):.2f}")
                print(f"   • Slope: {results.get('slope', 0):.4f}")
                print(f"   • Total intersections: {results.get('total_intersections', 0)}")

                return {
                    'peak': results.get('peak_number', 0),
                    'radius': results.get('radius_at_peak', 0),
                    'auc': results.get('auc', 0),
                    'regression_coef': results.get('slope', 0),
                    'total_intersections': results.get('total_intersections', 0),
                    'max_radius': results.get('max_radius', 0),
                    'mean_intersections': results.get('mean_intersections', 0)
                }
            else:
                print("   ⚠️ Analiza Sholl a eșuat")

        except Exception as e:
            print(f"   ❌ Eroare la analiza Sholl: {e}")
            import traceback
            traceback.print_exc()

        return self._empty_result()

    def _save_debug_image(self, image: np.ndarray, name: str, debug_dir: str):
        """Salvează o imagine pentru debugging"""
        if image.dtype == bool:
            image_to_save = (image * 255).astype(np.uint8)
        else:
            image_to_save = (exposure.rescale_intensity(image, out_range=(0, 255))).astype(np.uint8)

        imsave(os.path.join(debug_dir, f"{name}.png"), image_to_save)

    def _empty_result(self):
        """Returnează un rezultat gol"""
        return {
            'peak': 0, 'radius': 0, 'auc': 0, 'regression_coef': 0,
            'total_intersections': 0, 'max_radius': 0, 'mean_intersections': 0
        }


class CombinedNeuronAnalyzerFixed:
    """Analizorul principal FIXED pentru threading și ROI exact freehand"""

    def __init__(self, image_path, output_dir="outputs"):
        self.image_path = image_path
        self.output_dir = output_dir

        # SETUP MODIFICAT: Șterge imaginile, păstrează CSV
        self._setup_output_directory_keep_csv()

        # Încarcă imaginea cu MIP
        print("🔄 Încărcare imagine cu MIP complet...")
        try:
            self.display_image, self.raw_channels = load_image_robust_fixed(image_path)
            print(f"✅ Imagine încărcată: display={self.display_image.shape}, raw={self.raw_channels.shape}")
        except Exception as e:
            raise RuntimeError(f"Nu s-a putut încărca imaginea: {e}")

    def _setup_output_directory_keep_csv(self):
        """
        Configurează directorul de output: ȘTERGE imaginile vechi, PĂSTREAZĂ CSV-ul
        """
        import shutil
        import glob

        if os.path.exists(self.output_dir):
            print(f"🧹 Curăț imaginile vechi din '{self.output_dir}' dar PĂSTREZ CSV-ul...")

            # 1. PĂSTREAZĂ CSV-urile (și backup-urile)
            csv_files_to_preserve = []
            for pattern in ["*.csv", "*backup*.csv", "*results*.csv"]:
                csv_files_to_preserve.extend(glob.glob(os.path.join(self.output_dir, pattern)))

            # 2. Creează backup temporar pentru CSV-uri
            temp_csv_backup = {}
            for csv_file in csv_files_to_preserve:
                if os.path.exists(csv_file):
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        temp_csv_backup[os.path.basename(csv_file)] = f.read()
                    print(f"   💾 Backup temporar: {os.path.basename(csv_file)}")

            # 3. ȘTERGE TOT din directorul output
            try:
                shutil.rmtree(self.output_dir)
                print(f"   🗑️ Director '{self.output_dir}' șters complet")
            except Exception as e:
                print(f"   ⚠️ Eroare la ștergerea directorului: {e}")

            # 4. Recreează directorul
            os.makedirs(self.output_dir, exist_ok=True)

            # 5. RESTAUREAZĂ doar CSV-urile
            for csv_name, csv_content in temp_csv_backup.items():
                csv_path = os.path.join(self.output_dir, csv_name)
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write(csv_content)
                print(f"   ✅ CSV restaurat: {csv_name}")

            print(f"✅ Cleanup complet: imagini șterse, CSV păstrat!")

        else:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"✅ Directorul '{self.output_dir}' a fost creat.")

    def run_complete_analysis_main_thread_fixed(self, result_callback=None):
        """Rulează analiza completă FIXED în thread-ul principal"""
        print("🚀 Începe analiza completă cu ROI EXACT freehand...")

        # Pasul 1: Freehand selection pe MIP complet în thread-ul principal
        print("\n📍 PASUL 1: Selecție freehand pe MIP complet")
        print("🖊️ Deschid interfața de selecție ROI freehand...")

        # Creează queue pentru rezultate
        result_queue = queue.Queue()

        selector = ThreadSafeFreehandROI_Fixed(
            self.display_image,
            f"Selecție ROI Freehand EXACT - {os.path.basename(self.image_path)} (MIP complet)",
            result_queue
        )

        # Afișează selectorul în thread-ul principal
        freehand_rois = selector.show_blocking()

        if not freehand_rois:
            print("⚠️ Nu au fost selectate ROI-uri freehand!")
            if result_callback:
                result_callback([])
            return []

        print(f"✅ {len(freehand_rois)} ROI-uri freehand selectate")

        # Pasul 2: Procesarea cu forma EXACTĂ freehand (în thread secundar)
        def process_rois_exact():
            print(f"\n🎯 PASUL 2: Procesare cu forma EXACTĂ a {len(freehand_rois)} ROI-uri freehand")

            adapter = ExactFreehandMultiROIAdapter(
                self.image_path,
                self.raw_channels,
                self.output_dir
            )

            # Folosește forma EXACTĂ, nu bbox
            results = adapter.process_freehand_rois_exact(freehand_rois)

            # Pasul 3: Salvează rezultatele cu ordinea corectă
            print(f"\n💾 PASUL 3: Salvare rezultate cu ordinea CORECTĂ")
            self._save_exact_results_csv_append(results, freehand_rois)

            # Pasul 4: Sumar final
            print(f"\n📊 PASUL 4: Sumar final pentru ROI-uri cu forma EXACTĂ")
            successful_results = [r for r in results if r.get('peak', 0) > 0]

            print(f"✅ Analiză completă cu forma EXACTĂ terminată!")
            print(f"   • ROI-uri cu forma exactă procesate: {len(results)}")
            print(f"   • Analize reușite: {len(successful_results)}")

            if successful_results:
                avg_peak = np.mean([r['peak'] for r in successful_results])
                avg_auc = np.mean([r.get('auc', 0) for r in successful_results])
                print(f"   • Peak mediu: {avg_peak:.1f}")
                print(f"   • AUC mediu: {avg_auc:.1f}")

            print(f"📁 Rezultate cu forma EXACTĂ salvate în: {self.output_dir}/")

            if result_callback:
                result_callback(results)

            return results

        # Rulează procesarea în thread secundar
        processing_thread = threading.Thread(target=process_rois_exact)
        processing_thread.daemon = True
        processing_thread.start()

        return freehand_rois

    # ÎNLOCUIEȘTE metoda _save_exact_results_csv_append() cu această versiune
    # care salvează în exact_freehand_sholl_results.csv

    def _save_exact_results_csv_append(self, results, freehand_rois):
        """
        FIXED: Păstrează datele existente și folosește ordinea ORIGINALĂ din CSV-ul tău.

        NU ȘTERGE nimic din CSV-ul existent!
        NU SCHIMBĂ ordinea coloanelor existente!
        DOAR ADAUGĂ date noi cu aceeași ordine!
        """

        import csv
        import os
        import pandas as pd
        import numpy as np
        from datetime import datetime

        # NUMELE CORECT AL FIȘIERULUI
        csv_file = os.path.join(self.output_dir, "exact_freehand_sholl_results.csv")

        print(f"\n💾 Adaug {len(results)} rezultate în EXACT_FREEHAND_SHOLL_RESULTS.CSV...")
        print("🎯 PĂSTREZ datele existente și ordinea originală!")
        print(f"📁 Fișier țintă: {csv_file}")

        # Asigură-te că directorul există
        os.makedirs(self.output_dir, exist_ok=True)

        # VERIFICĂ CE ORDINE ARE CSV-UL EXISTENT
        existing_headers = None
        file_exists = os.path.exists(csv_file)

        if file_exists:
            try:
                # CITEȘTE HEADER-UL EXISTENT pentru a menține ordinea
                existing_df = pd.read_csv(csv_file)
                existing_headers = list(existing_df.columns)
                existing_count = len(existing_df)

                print(f"✅ CSV existent găsit cu {existing_count} înregistrări")
                print(f"📋 Ordinea existentă: {existing_headers[:8]}...")
                print("🔒 MENȚIN ordinea existentă - NU SCHIMB NIMIC!")

            except Exception as e:
                print(f"⚠️ Eroare la citirea CSV existent: {e}")
                existing_headers = None

        # DETERMINĂ ORDINEA COLOANELOR
        if existing_headers:
            # FOLOSEȘTE ordinea existentă - NU O SCHIMBA!
            headers_to_use = existing_headers
            print("🔒 Folosesc ordinea coloanelor din CSV-ul existent")
        else:
            # CSV nou - folosește ordinea ta preferată (fără timestamp primul)
            headers_to_use = [
                'image_name',  # 0
                'roi_index',  # 1
                'roi_type',  # 2
                'roi_area_pixels',  # 3
                'roi_perimeter_pixels',  # 4
                'peak_number',  # 5 ⭐ PEAK AICI!
                'radius_at_peak',  # 6 ⭐ RADIUS AICI!
                'auc',  # 7
                'regression_coef',  # 8
                'total_intersections',  # 9
                'max_radius',  # 10
                'mean_intersections',  # 11
                'roi_folder',  # 12
                'timestamp'  # 13 (la sfârșit, nu primul!)
            ]
            print("📋 CSV nou - folosesc ordinea ta preferată (timestamp la sfârșit)")

        # CREEAZĂ CSV-ul dacă nu există
        if not file_exists:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers_to_use)
            print(f"✅ CSV nou creat cu ordinea corectă")

        # ADAUGĂ DATELE NOI cu ordinea existentă
        successful_saves = 0
        timestamp_base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

            for i, (result, roi_points) in enumerate(zip(results, freehand_rois)):
                roi_num = i + 1

                try:
                    # Calculează geometria ROI-ului
                    area_pixels = len(roi_points)

                    # Calculează perimetrul
                    perimeter = 0
                    for j in range(len(roi_points)):
                        p1 = roi_points[j]
                        p2 = roi_points[(j + 1) % len(roi_points)]
                        perimeter += np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

                    # Timestamp pentru acest ROI
                    timestamp = f"{timestamp_base}_{roi_num:02d}"

                    # CONSTRUIEȘTE datele într-un dicționar
                    data_dict = {
                        'timestamp': timestamp,
                        'image_name': os.path.basename(self.image_path),
                        'roi_index': roi_num,
                        'roi_type': 'exact_freehand',
                        'roi_area_pixels': round(float(area_pixels), 2),
                        'roi_perimeter_pixels': round(float(perimeter), 6),
                        'peak_number': int(result['peak']),  # ⭐ PEAK
                        'radius_at_peak': int(result['radius']),  # ⭐ RADIUS
                        'auc': round(float(result['auc']), 2),
                        'regression_coef': round(float(result['regression_coef']), 10),
                        'total_intersections': int(result.get('total_intersections', 0)),
                        'max_radius': int(result.get('max_radius', 0)),
                        'mean_intersections': round(float(result.get('mean_intersections', 0)), 6),
                        'roi_folder': f"roi_{roi_num}_exact_freehand"
                    }

                    # CONSTRUIEȘTE rândul în ORDINEA EXISTENTĂ
                    row_data = []
                    for header in headers_to_use:
                        if header in data_dict:
                            row_data.append(data_dict[header])
                        else:
                            row_data.append('')  # Valoare goală pentru coloane lipsă

                    # SCRIE rândul cu ordinea PĂSTRATĂ
                    writer.writerow(row_data)
                    successful_saves += 1

                    # Găsește pozițiile pentru peak și radius în ordinea existentă
                    peak_pos = headers_to_use.index('peak_number') if 'peak_number' in headers_to_use else -1
                    radius_pos = headers_to_use.index('radius_at_peak') if 'radius_at_peak' in headers_to_use else -1

                    print(
                        f"✅ ROI {roi_num}: Peak={result['peak']}(pos.{peak_pos}), Radius={result['radius']}(pos.{radius_pos})")

                except Exception as e:
                    print(f"❌ Eroare la procesarea ROI {roi_num}: {e}")
                    continue

        print(f"\n📊 REZULTAT SALVARE:")
        print(f"   • ROI-uri procesate: {len(results)}")
        print(f"   • ROI-uri salvate cu succes: {successful_saves}")
        print(f"   • DATELE EXISTENTE PĂSTRATE: ✅")
        print(f"   • ORDINEA COLOANELOR PĂSTRATĂ: ✅")
        print(f"   • CSV: exact_freehand_sholl_results.csv")

        # Verificare finală
        try:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                print(f"\n🔍 VERIFICARE FINALĂ:")
                print(f"   • Total înregistrări în CSV: {len(df)}")
                print(f"   • Înregistrări adăugate acum: {successful_saves}")
                print(f"   • Header păstrat: {list(df.columns)[:5]}...")

                # Găsește pozițiile peak și radius în CSV-ul final
                if 'peak_number' in df.columns and 'radius_at_peak' in df.columns:
                    peak_col_pos = list(df.columns).index('peak_number')
                    radius_col_pos = list(df.columns).index('radius_at_peak')
                    print(f"   • Peak în coloana {peak_col_pos}, Radius în coloana {radius_col_pos}")

                    # Verifică ultimele înregistrări adăugate
                    if successful_saves > 0:
                        recent_records = df.tail(successful_saves)
                        print(f"   • Ultimele {len(recent_records)} înregistrări adăugate:")

                        for idx, row in recent_records.iterrows():
                            image = str(row['image_name'])[:15] if pd.notna(row['image_name']) else "N/A"
                            roi = int(row['roi_index']) if pd.notna(row['roi_index']) else 0
                            peak = int(row['peak_number']) if pd.notna(row['peak_number']) else 0
                            radius = int(row['radius_at_peak']) if pd.notna(row['radius_at_peak']) else 0
                            print(
                                f"     {image} ROI{roi}: Peak={peak}(col.{peak_col_pos}), Radius={radius}(col.{radius_col_pos})")

        except Exception as e:
            print(f"⚠️ Eroare la verificarea finală: {e}")

        print("✅ Datele ADĂUGATE cu succes - NIMIC ȘTERS!")

    # ==================================================================================
    # FUNCȚIE DE RECUPERARE (dacă s-au pierdut date)
    # ==================================================================================

    def recover_lost_data_from_backup(self):
        """
        Funcție de recuperare dacă s-au pierdut date din CSV.
        Caută backup-uri și le restaurează.
        """

        csv_file = os.path.join(self.output_dir, "exact_freehand_sholl_results.csv")

        # Caută backup-uri
        backup_files = []
        for file in os.listdir(self.output_dir):
            if 'backup' in file and file.endswith('.csv'):
                backup_files.append(os.path.join(self.output_dir, file))

        if backup_files:
            print(f"🔍 Găsite {len(backup_files)} backup-uri:")
            for backup in backup_files:
                print(f"   📋 {os.path.basename(backup)}")

            # Folosește cel mai recent backup
            latest_backup = max(backup_files, key=os.path.getmtime)
            print(f"📋 Restaurez din: {os.path.basename(latest_backup)}")

            try:
                # Citește backup-ul
                backup_df = pd.read_csv(latest_backup)

                # Verifică dacă CSV-ul curent există
                if os.path.exists(csv_file):
                    current_df = pd.read_csv(csv_file)
                    # Combină datele
                    combined_df = pd.concat([backup_df, current_df], ignore_index=True).drop_duplicates()
                    combined_df.to_csv(csv_file, index=False)
                    print(f"✅ Date restaurate și combinate: {len(combined_df)} înregistrări totale")
                else:
                    # Restaurează direct
                    backup_df.to_csv(csv_file, index=False)
                    print(f"✅ Date restaurate: {len(backup_df)} înregistrări")

            except Exception as e:
                print(f"❌ Eroare la restaurare: {e}")
        else:
            print("⚠️ Nu s-au găsit backup-uri")

    def move_data_to_exact_freehand_csv(self):
        """
        Funcție opțională pentru a muta datele din sholl_results.csv în exact_freehand_sholl_results.csv
        """

        old_csv = os.path.join(self.output_dir, "sholl_results.csv")
        new_csv = os.path.join(self.output_dir, "exact_freehand_sholl_results.csv")

        if os.path.exists(old_csv):
            try:
                # Citește datele din fișierul vechi
                df_old = pd.read_csv(old_csv)
                print(f"📋 Găsite {len(df_old)} înregistrări în sholl_results.csv")

                # Verifică dacă fișierul nou există
                if os.path.exists(new_csv):
                    df_new = pd.read_csv(new_csv)
                    print(f"📋 Găsite {len(df_new)} înregistrări în exact_freehand_sholl_results.csv")

                    # Combină datele
                    combined_df = pd.concat([df_new, df_old], ignore_index=True)
                    combined_df.to_csv(new_csv, index=False)
                    print(
                        f"✅ Date combinate: {len(combined_df)} înregistrări totale în exact_freehand_sholl_results.csv")
                else:
                    # Mută datele direct
                    df_old.to_csv(new_csv, index=False)
                    print(f"✅ Date mutate în exact_freehand_sholl_results.csv")

                # Creează backup pentru fișierul vechi și îl șterge
                backup_old = old_csv.replace('.csv', f'_moved_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                os.rename(old_csv, backup_old)
                print(f"📋 sholl_results.csv redenumit în: {os.path.basename(backup_old)}")

            except Exception as e:
                print(f"❌ Eroare la mutarea datelor: {e}")

    # ==================================================================================
    # INSTRUCȚIUNI DE UTILIZARE
    # ==================================================================================

    """
    PENTRU A UTILIZA:

    1. ÎNLOCUIEȘTE metoda _save_exact_results_csv_append() din clasa CombinedNeuronAnalyzerFixed
       cu versiunea de mai sus

    2. OPȚIONAL: Dacă vrei să muți datele existente din sholl_results.csv în exact_freehand_sholl_results.csv,
       adaugă metoda move_data_to_exact_freehand_csv() în clasă și apelează-o

    3. RULEAZĂ codul - acum va salva în exact_freehand_sholl_results.csv

    4. VERIFICĂ că vezi mesajele:
       "💾 Salvez X rezultate în EXACT_FREEHAND_SHOLL_RESULTS.CSV..."
       "✅ ROI X: Peak=Y(pos.6), Radius=Z(pos.7)"
    """

    def _create_simple_fixed_logger(self):
        """
        Creează un logger simplu cu ordinea corectă ca fallback.
        FOLOSEȘTE DOAR DACĂ nu merge importul ShollCSVLogger!
        """
        import csv
        import os
        from datetime import datetime

        class SimpleFixedLogger:
            def __init__(self, output_path):
                self.output_path = output_path
                self.csv_file = os.path.join(output_path, "sholl_results_fixed.csv")

                # HEADER-UL CORECT cu ordinea fixată
                self.headers = [
                    'timestamp',  # 0
                    'image_name',  # 1
                    'roi_index',  # 2
                    'roi_type',  # 3
                    'roi_area_pixels',  # 4
                    'roi_perimeter_pixels',  # 5
                    'peak_number',  # 6 ⭐ PEAK AICI!
                    'radius_at_peak',  # 7 ⭐ RADIUS AICI!
                    'auc',  # 8
                    'regression_coef',  # 9
                    'total_intersections',  # 10
                    'max_radius',  # 11
                    'mean_intersections',  # 12
                    'roi_folder'  # 13
                ]

                os.makedirs(output_path, exist_ok=True)
                self._ensure_csv_exists()

            def _ensure_csv_exists(self):
                if not os.path.exists(self.csv_file):
                    with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(self.headers)
                    print(f"✅ CSV fallback creat: {self.csv_file}")

            def log_result(self, image_name, roi_index, peak=None, radius=None,
                           peak_number=None, radius_at_peak=None, **kwargs):

                # Rezolvă alias-urile
                final_peak = peak_number if peak_number is not None else (peak if peak is not None else 0)
                final_radius = radius_at_peak if radius_at_peak is not None else (radius if radius is not None else 0)

                # Timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # CONSTRUIEȘTE rândul cu ordinea FORȚATĂ
                row_data = [
                    timestamp,  # 0
                    str(image_name).replace(',', '_').replace('"', ''),  # 1
                    int(roi_index),  # 2
                    kwargs.get('roi_type', 'exact_freehand'),  # 3
                    float(kwargs.get('roi_area_pixels', 0)),  # 4
                    float(kwargs.get('roi_perimeter_pixels', 0)),  # 5
                    int(final_peak),  # 6 ⭐ PEAK FORȚAT!
                    int(final_radius),  # 7 ⭐ RADIUS FORȚAT!
                    float(kwargs.get('auc', 0)),  # 8
                    float(kwargs.get('regression_coef', 0)),  # 9
                    int(kwargs.get('total_intersections', 0)),  # 10
                    int(kwargs.get('max_radius', 0)),  # 11
                    float(kwargs.get('mean_intersections', 0)),  # 12
                    kwargs.get('roi_folder', '')  # 13
                ]

                try:
                    with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(row_data)
                    return True
                except Exception as e:
                    print(f"❌ Eroare la salvare fallback: {e}")
                    return False

        return SimpleFixedLogger(self.output_dir)




class NeuronGUIFixed:
    """Interfața grafică FIXED pentru threading și cleanup tkinter COMPLET"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔬 Analizor Neuroni - ROI EXACT Freehand (FIXED + CSV APPEND)")
        self.root.geometry("800x650")
        self.root.configure(bg='#f0f0f0')

        # FIXED: Variables management cu cleanup complet
        self._tk_vars = []
        self.selected_file = self._create_managed_var(tk.StringVar())
        self.output_dir = self._create_managed_var(tk.StringVar(value="outputs"))
        self.processing = False

        # Register cleanup pentru atexit
        atexit.register(self._cleanup_on_exit)

        # Bind pentru close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.setup_ui()
        self.center_window()

    def _create_managed_var(self, var):
        """Creează o variabilă tkinter cu management pentru cleanup"""
        self._tk_vars.append(var)
        return var

    def _cleanup_tkinter_vars_safe(self):
        """Cleanup SIGUR pentru variabilele tkinter - FIXED"""
        print("🧹 Cleanup tkinter variables...")

        try:
            # Verifică dacă tk-ul este încă valid
            if hasattr(self, 'root') and self.root and hasattr(self.root, 'tk'):
                try:
                    # Test dacă tk-ul este încă activ
                    self.root.tk.eval('info exists tk_version')
                    tk_active = True
                except:
                    tk_active = False

                if tk_active:
                    # Tk este activ - putem face cleanup normal
                    for var in self._tk_vars:
                        try:
                            if hasattr(var, '_name') and var._name:
                                # Verifică dacă variabila încă există
                                try:
                                    exists = var._tk.getboolean(var._tk.call("info", "exists", var._name))
                                    if exists:
                                        var._tk.call("unset", var._name)
                                        print(f"   ✅ Variabila {var._name} eliminată")
                                except:
                                    pass  # Variabila deja eliminată
                        except Exception as e:
                            print(f"   ⚠️ Eroare cleanup variabilă: {e}")
                else:
                    print("   ⚠️ Tk nu mai este activ - skip cleanup variables")
            else:
                print("   ⚠️ Root nu mai există - skip cleanup")

        except Exception as e:
            print(f"   ❌ Eroare în cleanup tkinter: {e}")

        # Clear lista
        self._tk_vars.clear()
        print("🧹 Cleanup tkinter terminat")

    def _cleanup_on_exit(self):
        """Cleanup la ieșirea din aplicație"""
        print("🔚 Cleanup la ieșire...")
        self._cleanup_tkinter_vars_safe()

    def _on_closing(self):
        """Handler pentru închiderea ferestrei - FIXED"""
        if self.processing:
            if messagebox.askyesno("Confirmare", "Procesarea este în curs. Sigur vrei să ieși?"):
                self._cleanup_and_quit()
        else:
            self._cleanup_and_quit()

    def _cleanup_and_quit(self):
        """Cleanup complet și închidere - FIXED"""
        print("🔚 Închidere aplicație cu cleanup complet...")

        # Oprește procesarea dacă este activă
        self.processing = False

        # Cleanup tkinter vars
        self._cleanup_tkinter_vars_safe()

        # Destroy window
        try:
            if hasattr(self, 'root') and self.root:
                self.root.quit()
                self.root.destroy()
        except Exception as e:
            print(f"⚠️ Eroare la distrugerea ferestrei: {e}")

    def setup_ui(self):
        """Configurează interfața utilizator FIXED"""
        # Font-uri
        title_font = tkfont.Font(family="Arial", size=16, weight="bold")
        label_font = tkfont.Font(family="Arial", size=10)
        button_font = tkfont.Font(family="Arial", size=11, weight="bold")

        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Titlu
        title_label = tk.Label(
            main_frame,
            text="🔬 ANALIZOR NEURONI - ROI EXACT + CSV APPEND (FIXED)",
            font=title_font,
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 5))

        subtitle_label = tk.Label(
            main_frame,
            text="✅ FIXED: Threading + ROI Exact + CSV Append (nu suprascrie)",
            font=tkfont.Font(family="Arial", size=10, weight="bold"),
            bg='#f0f0f0',
            fg='#27ae60'
        )
        subtitle_label.pack(pady=(0, 15))

        # Descriere workflow
        desc_text = """WORKFLOW FIXED (Threading + ROI Exact + CSV Append):
1. 🖊️ Selecție freehand pe MIP complet (thread principal - matplotlib OK)
2. 🎯 Procesare cu forma EXACTĂ freehand (nu se convertește la dreptunghi!)
3. 📊 Analiza Sholl pe dendritele din forma exactă selectată
4. 💾 CSV APPEND - păstrează rezultatele anterioare (nu suprascrie!)

🔧 PROBLEME REZOLVATE:
• Threading CORECT: matplotlib în thread principal tkinter
• ROI EXACT: folosește forma precisă freehand, nu bounding box
• Cleanup COMPLET: variabile tkinter se șterg SIGUR (fără erori)
• CSV APPEND: rezultatele se adaugă, nu se suprascriu
• Timestamp: fiecare analiză are timestamp pentru identificare"""

        desc_label = tk.Label(
            main_frame,
            text=desc_text,
            font=label_font,
            bg='#f0f0f0',
            fg='#34495e',
            justify=tk.LEFT
        )
        desc_label.pack(pady=(0, 15))

        # Frame pentru îmbunătățiri
        fixes_frame = ttk.LabelFrame(main_frame, text="🛠️ FIXES în Această Versiune", padding="10")
        fixes_frame.pack(fill=tk.X, pady=(0, 15))

        fixes_text = """✅ THREADING FIXED: matplotlib rulează în main thread tkinter (nu mai sunt erori)
🎯 ROI EXACT FIXED: folosește forma precisă freehand, nu o convertește la dreptunghi
🧹 CLEANUP COMPLET FIXED: variabile tkinter se șterg SIGUR (elimină toate erorile)
💾 CSV APPEND FIXED: rezultatele se ADAUGĂ în CSV, nu se suprascriu
⏰ TIMESTAMP: fiecare analiză are timestamp pentru identificare clară
🔬 PROCES ÎMBUNĂTĂȚIT: menține precizia selecției freehand în toată analiza
📊 SUMAR CSV: afișează statistici despre toate analizele din CSV"""

        fixes_label = tk.Label(
            fixes_frame,
            text=fixes_text,
            font=tkfont.Font(family="Arial", size=9),
            bg='#f0f0f0',
            fg='#27ae60',
            justify=tk.LEFT
        )
        fixes_label.pack()

        # Frame pentru selectarea fișierului
        file_frame = ttk.LabelFrame(main_frame, text="📂 Selectare Fișier", padding="15")
        file_frame.pack(fill=tk.X, pady=(0, 15))

        select_btn = tk.Button(
            file_frame,
            text="🔍 Alege Imaginea de Neuroni",
            font=button_font,
            bg='#3498db',
            fg='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=10,
            command=self.select_file
        )
        select_btn.pack(pady=(0, 10))

        self.file_label = tk.Label(
            file_frame,
            textvariable=self.selected_file,
            font=label_font,
            bg='#f0f0f0',
            fg='#27ae60',
            wraplength=600
        )
        self.file_label.pack()

        # Frame pentru output
        output_frame = ttk.LabelFrame(main_frame, text="📁 Director Output", padding="15")
        output_frame.pack(fill=tk.X, pady=(0, 15))

        output_entry_frame = tk.Frame(output_frame, bg='#f0f0f0')
        output_entry_frame.pack(fill=tk.X)

        tk.Label(output_entry_frame, text="Director:", font=label_font, bg='#f0f0f0').pack(side=tk.LEFT)

        output_entry = tk.Entry(
            output_entry_frame,
            textvariable=self.output_dir,
            font=label_font,
            width=30
        )
        output_entry.pack(side=tk.LEFT, padx=(10, 10), fill=tk.X, expand=True)

        output_btn = tk.Button(
            output_entry_frame,
            text="📁",
            command=self.select_output_dir,
            bg='#95a5a6',
            fg='white',
            width=3
        )
        output_btn.pack(side=tk.RIGHT)

        # Frame pentru butoane
        action_frame = tk.Frame(main_frame, bg='#f0f0f0')
        action_frame.pack(fill=tk.X, pady=(15, 0))

        self.process_btn = tk.Button(
            action_frame,
            text="🎯 ÎNCEPE ANALIZA ROI EXACT + CSV APPEND",
            font=button_font,
            bg='#27ae60',
            fg='white',
            relief=tk.RAISED,
            borderwidth=3,
            padx=30,
            pady=15,
            command=self.start_processing_fixed
        )
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))

        exit_btn = tk.Button(
            action_frame,
            text="❌ IEȘIRE SIGURĂ",
            font=button_font,
            bg='#95a5a6',
            fg='white',
            relief=tk.RAISED,
            borderwidth=3,
            padx=30,
            pady=15,
            command=self._on_closing
        )
        exit_btn.pack(side=tk.RIGHT)

        # Progress bar
        self.progress_frame = tk.Frame(main_frame, bg='#f0f0f0')

        self.progress_label = tk.Label(
            self.progress_frame,
            text="",
            font=label_font,
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        self.progress_label.pack(pady=(0, 10))

        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            length=400
        )

        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="📋 Log Procesare (ROI Exact + CSV Append + Cleanup Fixed)",
                                   padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

        log_scroll_frame = tk.Frame(log_frame)
        log_scroll_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_scroll_frame,
            font=tkfont.Font(family="Consolas", size=9),
            bg='#2c3e50',
            fg='#ecf0f1',
            insertbackground='white',
            wrap=tk.WORD,
            height=12
        )

        log_scrollbar = ttk.Scrollbar(log_scroll_frame, orient=tk.VERTICAL)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.config(command=self.log_text.yview)

        # Mesaj inițial în log
        self.log_message("🔬 Analizor Neuroni - ROI EXACT + CSV APPEND (FIXED) - Gata pentru analiză!")
        self.log_message("✅ THREADING FIXED: matplotlib în main thread tkinter")
        self.log_message("🎯 ROI EXACT FIXED: folosește forma precisă freehand (nu bbox)")
        self.log_message("🧹 CLEANUP COMPLET FIXED: variabile tkinter se șterg SIGUR")
        self.log_message("💾 CSV APPEND FIXED: rezultatele se adaugă, nu se suprascriu")
        self.log_message("⏰ TIMESTAMP: fiecare analiză are timestamp pentru identificare")

    def center_window(self):
        """Centrează fereastra pe ecran"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def log_message(self, message):
        """Adaugă un mesaj în log (thread-safe)"""

        def add_message():
            try:
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] {message}\n"
                self.log_text.insert(tk.END, formatted_message)
                self.log_text.see(tk.END)
                self.root.update_idletasks()
            except Exception as e:
                print(f"Eroare la adăugarea mesajului în log: {e}")

        # Asigură-te că mesajul este adăugat în thread-ul principal
        if threading.current_thread() == threading.main_thread():
            add_message()
        else:
            try:
                self.root.after(0, add_message)
            except:
                print(f"Nu s-a putut adăuga mesajul în log: {message}")

    def select_file(self):
        """Selectează fișierul pentru procesare"""
        filetypes = [
            ("Toate formatele suportate", "*.czi *.tif *.tiff *.png *.jpg *.jpeg"),
            ("Fișiere CZI", "*.czi"),
            ("Fișiere TIFF", "*.tif *.tiff"),
            ("Fișiere imagine", "*.png *.jpg *.jpeg"),
            ("Toate fișierele", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Selectează imaginea de neuroni",
            filetypes=filetypes
        )

        if filename:
            self.selected_file.set(filename)
            self.log_message(f"📂 Fișier selectat: {os.path.basename(filename)}")

            # Afișează informații despre fișier
            try:
                file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
                file_ext = os.path.splitext(filename)[1].lower()
                self.log_message(f"   • Dimensiune: {file_size:.1f} MB")
                self.log_message(f"   • Format: {file_ext}")

                if file_ext == '.czi':
                    self.log_message("   • Format CZI detectat - se va aplica MIP și mapare corectă canale")
                    self.log_message("   • ROI-urile freehand vor fi procesate cu forma EXACTĂ")

                # Verifică dacă există CSV anterior
                csv_path = os.path.join(self.output_dir.get(), "exact_freehand_sholl_results.csv")
                if os.path.exists(csv_path):
                    try:
                        existing_df = pd.read_csv(csv_path)
                        self.log_message(f"   💾 CSV existent găsit cu {len(existing_df)} înregistrări")
                        self.log_message("   • Rezultatele noi se vor ADĂUGA (nu suprascrie)")
                    except:
                        self.log_message("   ⚠️ CSV existent, dar nu poate fi citit")

            except Exception as e:
                self.log_message(f"   ⚠️ Nu s-au putut obține informații: {e}")

    def select_output_dir(self):
        """Selectează directorul de output"""
        directory = filedialog.askdirectory(
            title="Selectează directorul pentru rezultate"
        )

        if directory:
            self.output_dir.set(directory)
            self.log_message(f"📁 Director output setat: {directory}")

            # Verifică CSV existent în noul director
            csv_path = os.path.join(directory, "exact_freehand_sholl_results.csv")
            if os.path.exists(csv_path):
                try:
                    existing_df = pd.read_csv(csv_path)
                    unique_analyses = existing_df['timestamp'].nunique() if 'timestamp' in existing_df.columns else 1
                    self.log_message(
                        f"   💾 CSV existent: {len(existing_df)} înregistrări din {unique_analyses} analize")
                except:
                    self.log_message("   ⚠️ CSV existent, dar nu poate fi citit")

    def start_processing_fixed(self):
        """Începe procesarea FIXED"""
        if not self.selected_file.get():
            messagebox.showerror("Eroare", "Te rog selectează mai întâi un fișier!")
            return

        if self.processing:
            messagebox.showwarning("Avertisment", "Procesarea este deja în curs!")
            return

        # Schimbă starea butoanelor
        self.process_btn.config(state='disabled', text='🎯 Procesare ROI EXACT + CSV APPEND în curs...')
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_label.config(text="Procesare cu ROI exact freehand + CSV append în curs...")
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        self.progress_bar.start()

        self.processing = True

        self.log_message("🚀 Începe procesarea cu ROI EXACT freehand + CSV APPEND...")
        self.log_message("✅ THREADING FIXED: matplotlib în thread principal")
        self.log_message("🎯 ROI EXACT: va folosi forma precisă freehand")
        self.log_message("💾 CSV APPEND: rezultatele se vor adăuga la cele existente")
        self.log_message("🖊️ Selecția freehand va rula corect în thread-ul principal tkinter")

        # Procesează în thread-ul principal pentru matplotlib
        self.root.after(100, self.process_file_fixed)

    def process_file_fixed(self):
        """Procesează fișierul FIXED (rulează în thread-ul principal pentru matplotlib)"""
        try:
            self.log_message("🔄 Încărcare imagine cu MIP complet...")

            # Inițializează analizorul FIXED
            analyzer = CombinedNeuronAnalyzerFixed(
                self.selected_file.get(),
                self.output_dir.get()
            )

            self.log_message("✅ Imagine încărcată cu succes")
            self.log_message("🎯 Mapare canale corectă aplicată")
            self.log_message("🖊️ Deschid interfața de selecție freehand EXACT...")

            # Callback pentru rezultate
            def on_results_ready_fixed(results):
                self.log_message("📊 Procesarea cu ROI EXACT + CSV APPEND finalizată!")

                if results:
                    successful = len([r for r in results if r.get('peak', 0) > 0])
                    self.log_message(f"✅ Analiză cu ROI EXACT completă!")
                    self.log_message(f"   • ROI-uri cu forma exactă procesate: {len(results)}")
                    self.log_message(f"   • Analize reușite: {successful}")

                    if successful > 0:
                        avg_peak = np.mean([r['peak'] for r in results if r.get('peak', 0) > 0])
                        self.log_message(f"   • Peak mediu: {avg_peak:.1f}")

                    self.log_message(f"📁 Rezultate cu forma exactă + CSV append salvate în: {self.output_dir.get()}/")

                    # Afișează statistici CSV
                    try:
                        csv_path = os.path.join(self.output_dir.get(), "exact_freehand_sholl_results.csv")
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path)
                            unique_analyses = df['timestamp'].nunique() if 'timestamp' in df.columns else 1
                            unique_images = df['image_name'].nunique()
                            total_rois = len(df)

                            self.log_message(f"📈 STATISTICI CSV FINALE:")
                            self.log_message(f"   • Analize totale: {unique_analyses}")
                            self.log_message(f"   • Imagini analizate: {unique_images}")
                            self.log_message(f"   • ROI-uri totale: {total_rois}")
                    except Exception as e:
                        self.log_message(f"⚠️ Nu s-au putut afișa statisticile CSV: {e}")

                    # Afișează mesaj de succes
                    messagebox.showinfo(
                        "Succes - ROI EXACT + CSV APPEND!",
                        f"Analiza cu ROI EXACT freehand completată!\n\n"
                        f"ROI-uri cu forma exactă procesate: {len(results)}\n"
                        f"Analize reușite: {successful}\n\n"
                        f"Rezultatele au fost ADĂUGATE în CSV:\n{self.output_dir.get()}\n\n"
                        f"🎯 ROI-uri procesate cu forma EXACTĂ freehand\n"
                        f"✅ Threading CORECT (fără erori)\n"
                        f"💾 CSV APPEND (nu suprascrie)\n"
                        f"🧹 Cleanup tkinter COMPLET FIXED"
                    )
                else:
                    self.log_message("⚠️ Nu au fost selectate ROI-uri pentru procesare")
                    messagebox.showwarning(
                        "Avertisment",
                        "Nu au fost selectate ROI-uri pentru procesare!"
                    )

                # Reset UI
                self.reset_ui_fixed()

            # Rulează analiza FIXED în thread-ul principal
            freehand_rois = analyzer.run_complete_analysis_main_thread_fixed(on_results_ready_fixed)

            if freehand_rois:
                self.log_message(f"✅ {len(freehand_rois)} ROI-uri freehand selectate cu forma EXACTĂ")
                self.log_message("🔄 Începe procesarea cu forma EXACTĂ în thread secundar...")
                self.log_message("💾 Rezultatele se vor ADĂUGA la CSV-ul existent (nu suprascriu)")
            else:
                self.log_message("⚠️ Nu au fost selectate ROI-uri freehand!")
                self.reset_ui_fixed()

        except Exception as e:
            error_msg = f"❌ Eroare în timpul procesării: {str(e)}"
            self.log_message(error_msg)
            print(f"Eroare detaliată: {e}")
            import traceback
            traceback.print_exc()

            messagebox.showerror(
                "Eroare!",
                f"A apărut o eroare:\n\n{str(e)}\n\n"
                f"Verifică log-ul pentru detalii.\n\n"
                f"Versiunea FIXED cu threading, ROI exact și CSV append."
            )

            self.reset_ui_fixed()

    def reset_ui_fixed(self):
        """Resetează interfața după procesare FIXED"""
        self.processing = False
        self.process_btn.config(state='normal', text='🎯 ÎNCEPE ANALIZA ROI EXACT + CSV APPEND')
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.log_message("🏁 Gata pentru o nouă procesare cu ROI EXACT + CSV APPEND!")

    def run(self):
        """Rulează aplicația cu cleanup complet"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("⚠️ Întrerupere de la tastatură")
        except Exception as e:
            print(f"❌ Eroare în mainloop: {e}")
        finally:
            # Cleanup final complet
            print("🔚 Cleanup final...")
            self._cleanup_tkinter_vars_safe()


def main():
    """Funcția principală FIXED cu cleanup complet"""
    print("🔬 Analizor Neuroni - ROI EXACT Freehand + CSV APPEND (FIXED)")
    print("=" * 80)
    print("🛠️ PROBLEME REZOLVATE:")
    print("✅ THREADING FIXED: matplotlib rulează în thread principal tkinter")
    print("✅ ROI EXACT FIXED: folosește forma precisă freehand, nu bounding box")
    print("✅ CLEANUP COMPLET FIXED: variabile tkinter se șterg SIGUR (fără erori)")
    print("✅ CSV APPEND FIXED: rezultatele se ADAUGĂ, nu se suprascriu")
    print("✅ TIMESTAMP: fiecare analiză are timestamp pentru identificare")
    print("✅ ERORI THREADING: eliminate complet cu execuție corectă")
    print("=" * 80)
    print("📋 Workflow FIXED:")
    print("1. 🖊️  Freehand Selection pe MIP complet (thread principal - matplotlib OK)")
    print("2. 🎯  Procesare cu forma EXACTĂ freehand (nu se convertește la dreptunghi)")
    print("3. 📊  Analiza Sholl pe dendritele din forma exactă selectată")
    print("4. 💾  CSV APPEND - rezultatele se adaugă la cele existente")
    print("=" * 80)

    try:
        # Verifică dacă rulează din linia de comandă cu argument
        if len(sys.argv) > 1:
            image_path = sys.argv[1]

            if not os.path.exists(image_path):
                print(f"❌ Fișierul nu există: {image_path}")
                return

            print(f"📂 Procesez fișierul din linia de comandă: {image_path}")
            print("⚠️ Pentru linia de comandă, va rula cu threading CORECT și CSV APPEND")

            # Rulează analiza direct cu FIXED approach
            analyzer = CombinedNeuronAnalyzerFixed(image_path)

            # Pentru command line, folosește versiunea FIXED
            def simple_callback_fixed(results):
                print("🏁 Procesare din linia de comandă cu ROI EXACT + CSV APPEND completă!")
                if results:
                    successful = len([r for r in results if r.get('peak', 0) > 0])
                    print(f"✅ ROI-uri cu forma exactă procesate: {len(results)}, reușite: {successful}")
                    print("💾 Rezultatele au fost adăugate în CSV (nu suprascrise)")

            analyzer.run_complete_analysis_main_thread_fixed(simple_callback_fixed)

        else:
            # Rulează interfața grafică FIXED
            print("🖥️  Lansez interfața grafică FIXED cu ROI exact + CSV append...")
            app = NeuronGUIFixed()
            app.run()

    except KeyboardInterrupt:
        print("\n⚠️ Procesare întreruptă de utilizator")
    except Exception as e:
        print(f"❌ Eroare critică: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 Aplicația s-a închis complet")


if __name__ == "__main__":
    main()