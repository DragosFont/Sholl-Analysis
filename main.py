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
        """Procesează un singur ROI folosind masca exactă"""

        # Creează directoriile
        roi_dir = os.path.join(self.output_dir, f"roi_{roi_num}_exact_freehand")
        debug_dir = os.path.join(roi_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Determină bounding box pentru extragerea eficientă
        coords = np.where(exact_mask)
        if len(coords[0]) == 0:
            return self._empty_result()

        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1

        print(f"   🎯 Bounding box: ({x_min},{y_min}) -> ({x_max},{y_max})")
        print(f"   🎯 Dar folosesc forma EXACTĂ freehand pentru analiză!")

        # Extrage canalele din regiunea bounding box
        channels = self._extract_roi_channels_exact(y_min, y_max, x_min, x_max)

        # Extrage masca exactă pentru această regiune
        roi_exact_mask = exact_mask[y_min:y_max, x_min:x_max]

        # Salvează imaginile de bază cu masca exactă aplicată
        self._save_basic_images_exact(channels, roi_exact_mask, roi_dir, roi_num)

        # Pasul 1: Crează masca neuronului din canalul verde DOAR în zona exactă freehand
        neuron_mask = self._create_neuron_mask_in_exact_region(
            channels['green'], roi_exact_mask, debug_dir, roi_num)

        if neuron_mask.sum() == 0:
            print(f"⚠️ ROI {roi_num}: Nu s-a detectat neuron în forma exactă freehand!")
            return self._empty_result()

        # Pasul 2: Găsește soma-ul din canalul albastru DOAR în regiunea verde din forma exactă
        soma_center = self._find_soma_in_exact_green_region(
            channels['blue'], neuron_mask, debug_dir, roi_num)

        # Pasul 3: Extrage dendritele din forma exactă freehand
        dendrites_binary = self._extract_dendrites_in_exact_region(
            channels['red'], neuron_mask, debug_dir, roi_num)

        # Pasul 4: Salvează rezultatele finale cu forma exactă
        binary_path = self._save_final_results_exact(
            dendrites_binary, channels['red'], soma_center, roi_exact_mask, roi_dir, roi_num)

        # Pasul 5: Analiza Sholl pe forma exactă
        results = self._perform_sholl_analysis_exact(
            dendrites_binary, binary_path, soma_center, roi_dir, roi_num)

        return results

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
        """Crează masca neuronului DOAR în forma exactă freehand"""
        print(f"🟢 ROI {roi_num}: Creez masca neuronului în forma EXACTĂ freehand...")

        # Aplică masca exactă pe canalul verde
        green_in_exact = green_channel * exact_mask.astype(float)

        if green_in_exact.sum() == 0:
            print("   ⚠️ Nu există semnal verde în forma exactă freehand!")
            return np.zeros_like(exact_mask, dtype=bool)

        # Normalizare și denoising ușor
        green_norm = exposure.rescale_intensity(green_in_exact, out_range=(0, 1))
        green_smooth = filters.gaussian(green_norm, sigma=0.8)

        # Threshold mai permisiv - folosim percentila 65
        if green_smooth.sum() > 0:
            threshold = np.percentile(green_smooth[green_smooth > 0], 65)
            threshold = max(threshold, 0.1)
        else:
            threshold = 0.1

        # Creare mască inițială
        initial_mask = green_smooth > threshold

        # IMPORTANT: Aplică masca exactă freehand
        initial_mask = initial_mask & exact_mask

        # Curățare minimă
        cleaned_mask = morphology.remove_small_objects(initial_mask, min_size=50)

        # Aplicăm closing foarte mic
        final_mask = morphology.binary_closing(cleaned_mask, morphology.disk(2))

        # IMPORTANT: Asigură-te că rămâne în forma exactă
        final_mask = final_mask & exact_mask

        # Salvare debug
        self._save_debug_image(green_norm, "01_green_in_exact", debug_dir)
        self._save_debug_image(green_smooth, "02_green_smooth", debug_dir)
        self._save_debug_image(initial_mask, "03_initial_mask_exact", debug_dir)
        self._save_debug_image(final_mask, "04_final_mask_exact", debug_dir)
        self._save_debug_image(exact_mask, "05_exact_freehand_mask", debug_dir)

        print(f"   • Threshold: {threshold:.4f}")
        print(f"   • Pixeli în masca exactă neuron: {final_mask.sum()}")
        print(f"   • Procent din forma exactă: {100 * final_mask.sum() / max(1, exact_mask.sum()):.1f}%")

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
        """Extrage dendrite DOAR din forma exactă freehand"""
        print(f"🔴 ROI {roi_num}: Extrag dendrite din forma EXACTĂ freehand...")

        # Normalizare
        red_norm = exposure.rescale_intensity(red_channel, out_range=(0, 1))

        # Aplicăm masca neuronului din forma exactă
        red_in_exact_neuron = red_norm * neuron_mask.astype(float)

        if red_in_exact_neuron.sum() == 0:
            print("   ⚠️ Nu există semnal roșu în neuronul din forma exactă!")
            return np.zeros_like(red_norm, dtype=bool)

        # Enhancement ușor
        red_enhanced = filters.unsharp_mask(red_in_exact_neuron, radius=1, amount=1.2)
        red_enhanced = np.clip(red_enhanced, 0, 1)

        # Denoising foarte ușor
        red_denoised = filters.median(red_enhanced, morphology.disk(1))

        # Threshold foarte jos pentru a prinde structuri fine
        if red_denoised.sum() > 0:
            threshold = np.percentile(red_denoised[red_denoised > 0], 15)
            threshold = max(threshold, 0.01)
        else:
            threshold = 0.01

        # Creare mască binară
        binary_dendrites = red_denoised > threshold

        # IMPORTANT: Aplică masca neuronului din forma exactă
        binary_dendrites = binary_dendrites & neuron_mask

        # Curățare minimă
        cleaned = morphology.remove_small_objects(binary_dendrites, min_size=3)

        # Scheletizare pentru a obține linii fine
        if cleaned.sum() > 0:
            skeleton = morphology.skeletonize(cleaned)
        else:
            skeleton = cleaned.copy()

        # Salvare debug
        self._save_debug_image(red_norm, "09_red_original", debug_dir)
        self._save_debug_image(red_in_exact_neuron, "10_red_in_exact_neuron", debug_dir)
        self._save_debug_image(binary_dendrites, "11_binary_dendrites_exact", debug_dir)
        self._save_debug_image(skeleton, "12_final_skeleton_exact", debug_dir)

        print(f"   • Threshold: {threshold:.4f}")
        print(f"   • Pixeli în skeleton final exact: {skeleton.sum()}")

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

        # Încarcă imaginea cu MIP
        print("🔄 Încărcare imagine cu MIP complet...")
        try:
            self.display_image, self.raw_channels = load_image_robust_fixed(image_path)
            print(f"✅ Imagine încărcată: display={self.display_image.shape}, raw={self.raw_channels.shape}")
        except Exception as e:
            raise RuntimeError(f"Nu s-a putut încărca imaginea: {e}")

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