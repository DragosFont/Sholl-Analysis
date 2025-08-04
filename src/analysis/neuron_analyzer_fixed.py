#!/usr/bin/env python3
"""
FIX pentru Analizor neuroni combinat:
1. Corectează afișarea canalului verde în MIP
2. Rezolvă problema cu ROI-urile care nu se procesează
3. Îmbunătățește gestionarea canalelor pentru fișiere .czi
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
    """
    Încarcă imaginea cu MIP complet pentru afișare și raw data pentru procesare
    FIX: Corectează afișarea canalului verde și gestionarea canalelor
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Fișierul nu există: {image_path}")

    file_ext = os.path.splitext(image_path)[1].lower()
    file_size = os.path.getsize(image_path)

    print(f"📂 Încărcare: {os.path.basename(image_path)}")
    print(f"   Format: {file_ext}, Dimensiune: {file_size / (1024 * 1024):.1f} MB")

    # Strategia 1: Pentru fișiere .czi - cu Maximum Intensity Projection ÎMBUNĂTĂȚIT
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

                # ÎMBUNĂTĂȚIRE: Gestionare mai bună a canalelor pentru afișare
                print(f"   🎨 Procesez {data.shape[0]} canale pentru afișare...")

                # Verifică valorile în fiecare canal
                for i in range(min(data.shape[0], 5)):  # Verifică primele 5 canale
                    channel_data = data[i]
                    min_val, max_val = channel_data.min(), channel_data.max()
                    non_zero = np.count_nonzero(channel_data)
                    print(f"   Canal {i}: min={min_val}, max={max_val}, non-zero pixels={non_zero}")

                # FIX: Mapare optimizată pentru afișarea RGB cu accent pe verde
                if data.shape[0] >= 3:
                    # Identifică canalul cu cea mai mare activitate (probabil verde)
                    channel_activity = []
                    for i in range(data.shape[0]):
                        activity = np.sum(data[i] > np.percentile(data[i], 95))  # Pixels peste 95th percentile
                        channel_activity.append(activity)

                    print(f"   🔍 Activitate pe canale: {channel_activity}")

                    # Sortează canalele după activitate
                    sorted_channels = sorted(enumerate(channel_activity), key=lambda x: x[1], reverse=True)
                    print(f"   🏆 Canale sortate după activitate: {sorted_channels}")

                    # Mapare optimizată: cel mai activ canal pe verde, următoarele pe roșu și albastru
                    if len(sorted_channels) >= 3:
                        green_idx = sorted_channels[0][0]  # Cel mai activ -> verde
                        red_idx = sorted_channels[1][0]  # Al doilea -> roșu
                        blue_idx = sorted_channels[2][0]  # Al treilea -> albastru

                        print(f"   🎨 Mapare canale: R={red_idx}, G={green_idx}, B={blue_idx}")

                        red_channel = data[red_idx]
                        green_channel = data[green_idx]
                        blue_channel = data[blue_idx]
                    else:
                        # Fallback pentru mai puține canale
                        red_channel = data[0] if data.shape[0] > 0 else np.zeros_like(data[0])
                        green_channel = data[1] if data.shape[0] > 1 else np.zeros_like(data[0])
                        blue_channel = data[2] if data.shape[0] > 2 else np.zeros_like(data[0])

                elif data.shape[0] == 2:
                    # 2 canale: primul pe verde (cel mai important), al doilea pe roșu
                    print("   🎨 2 canale: mapez primul pe verde, al doilea pe roșu...")
                    red_channel = data[1]
                    green_channel = data[0]  # FIX: primul canal pe verde
                    blue_channel = np.zeros_like(data[0])

                else:
                    # Un singur canal: pune-l pe verde pentru vizibilitate maximă
                    print("   🎨 Un singur canal: mapez pe verde pentru vizibilitate...")
                    single_channel = data[0]
                    red_channel = np.zeros_like(single_channel)
                    green_channel = single_channel  # FIX: pune pe verde
                    blue_channel = np.zeros_like(single_channel)

                # Creează imaginea RGB pentru afișare
                display_data = np.stack([red_channel, green_channel, blue_channel], axis=2)

                # Păstrează datele raw în format CYX pentru Multi ROI Processor
                raw_data = data

            elif len(data.shape) == 3:  # ZYX sau CYX
                print("   🔄 Format 3D detectat (ZYX sau CYX)")

                if data.shape[0] > 10:  # Probabil Z-stack
                    print("   🌟 Aplicând MIP pe Z-stack...")
                    mip_data = np.max(data, axis=0)  # MIP pe Z
                    # FIX: pune datele pe verde pentru vizibilitate
                    red_channel = np.zeros_like(mip_data)
                    green_channel = mip_data
                    blue_channel = np.zeros_like(mip_data)
                    display_data = np.stack([red_channel, green_channel, blue_channel], axis=2)
                    raw_data = np.stack([red_channel, green_channel, blue_channel], axis=0)  # CYX format
                else:  # Probabil canale
                    if data.shape[0] >= 3:
                        print("   🎨 Reorganizez canalele ca RGB cu accent pe verde...")
                        # Similar cu logica de mai sus pentru maparea canalelor
                        channel_activity = []
                        for i in range(data.shape[0]):
                            activity = np.sum(data[i] > np.percentile(data[i], 95))
                            channel_activity.append(activity)

                        sorted_channels = sorted(enumerate(channel_activity), key=lambda x: x[1], reverse=True)

                        green_idx = sorted_channels[0][0]  # Cel mai activ pe verde
                        red_idx = sorted_channels[1][0] if len(sorted_channels) > 1 else 0
                        blue_idx = sorted_channels[2][0] if len(sorted_channels) > 2 else 0

                        red_channel = data[red_idx]
                        green_channel = data[green_idx]
                        blue_channel = data[blue_idx]

                        display_data = np.stack([red_channel, green_channel, blue_channel], axis=2)
                        raw_data = data

                    elif data.shape[0] == 2:
                        print("   🎨 Combin 2 canale cu primul pe verde...")
                        red_channel = data[1]
                        green_channel = data[0]  # FIX: primul canal pe verde
                        blue_channel = np.zeros_like(data[0])
                        display_data = np.stack([red_channel, green_channel, blue_channel], axis=2)
                        raw_data = np.stack([red_channel, green_channel, blue_channel], axis=0)
                    else:
                        print("   🌟 Un singur canal pe verde...")
                        combined = data[0] if data.shape[0] == 1 else np.max(data, axis=0)
                        red_channel = np.zeros_like(combined)
                        green_channel = combined  # FIX: pe verde
                        blue_channel = np.zeros_like(combined)
                        display_data = np.stack([red_channel, green_channel, blue_channel], axis=2)
                        raw_data = np.stack([red_channel, green_channel, blue_channel], axis=0)

            # ÎMBUNĂTĂȚIRE: Normalizare mai bună pentru fiecare canal separat
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

            print(f"   ✅ .czi încărcat cu MIP ÎMBUNĂTĂȚIT: display={display_data.shape}, raw={raw_data.shape}")

            # Verifică dacă verde este vizibil
            green_pixels = np.count_nonzero(display_data[:, :, 1])  # Canalul verde
            print(f"   🌿 Pixeli verzi vizibili: {green_pixels}")

            return display_data, raw_data

        except ImportError:
            print("   ⚠️ aicsimageio nu este instalat")
        except Exception as e:
            print(f"   ⚠️ Eroare aicsimageio: {e}")
            import traceback
            traceback.print_exc()

        # Dacă metodele speciale .czi au eșuat, continuă cu metodele generice
        print("   ⚠️ Metodele specializate .czi au eșuat, încerc metodele generice...")

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
            # FIX: Pentru grayscale, pune pe verde
            green_channel = image
            red_channel = np.zeros_like(image)
            blue_channel = np.zeros_like(image)
            image = np.stack([red_channel, green_channel, blue_channel], axis=2)

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

        # ÎMBUNĂTĂȚIRE: Instrucțiuni mai clare
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
        self.min_roi_points = 5  # FIX: Minimum puncte pentru ROI valid

    def add_buttons(self):
        """Adaugă butoanele de control ÎMBUNĂTĂȚITE"""
        # Poziții butoane
        ax_done = plt.axes([0.02, 0.02, 0.12, 0.05])
        ax_clear = plt.axes([0.15, 0.02, 0.12, 0.05])
        ax_undo = plt.axes([0.28, 0.02, 0.12, 0.05])
        ax_info = plt.axes([0.41, 0.02, 0.15, 0.05])

        # Creează butoanele
        self.btn_done = Button(ax_done, 'FINALIZEAZĂ')
        self.btn_clear = Button(ax_clear, 'ȘTERGE TOT')
        self.btn_undo = Button(ax_undo, 'ANULEAZĂ')
        self.btn_info = Button(ax_info, f'ROI-uri: {len(self.rois)}')

        # Conectează funcțiile
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
        """Gestionează click-urile mouse-ului ÎMBUNĂTĂȚIT"""
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
        """Gestionează mișcarea mouse-ului ÎMBUNĂTĂȚIT"""
        if not self.drawing or not self.mouse_pressed or event.inaxes != self.ax:
            return

        if event.xdata is not None and event.ydata is not None:
            # Adaugă punctul dacă este suficient de departe de ultimul
            if len(self.current_path) == 0:
                self.current_path.append((event.xdata, event.ydata))
            else:
                last_point = self.current_path[-1]
                distance = np.sqrt((event.xdata - last_point[0]) ** 2 + (event.ydata - last_point[1]) ** 2)

                if distance > 1.5:  # FIX: Prag mai mic pentru mai multe puncte
                    self.current_path.append((event.xdata, event.ydata))
                    self.update_temp_display()

    def update_temp_display(self):
        """Actualizează afișarea temporară în timpul desenării"""
        # Șterge liniile temporare
        for line in self.temp_lines:
            try:
                line.remove()
            except:
                pass
        self.temp_lines.clear()

        if len(self.current_path) > 1:
            x_coords = [p[0] for p in self.current_path]
            y_coords = [p[1] for p in self.current_path]

            # Linia principală
            line, = self.ax.plot(x_coords, y_coords, 'lime', linewidth=3, alpha=0.8)  # FIX: Verde lime vizibil
            self.temp_lines.append(line)

            # Marchează punctul de start
            start_point, = self.ax.plot(x_coords[0], y_coords[0], 'go', markersize=10)
            self.temp_lines.append(start_point)

            # Marchează punctul curent
            current_point, = self.ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8)
            self.temp_lines.append(current_point)

        try:
            self.fig.canvas.draw_idle()
        except:
            pass

    def complete_roi(self):
        """Completează ROI-ul curent ÎMBUNĂTĂȚIT"""
        if len(self.current_path) < self.min_roi_points:
            print(f"❌ ROI-ul trebuie să aibă cel puțin {self.min_roi_points} puncte!")
            return

        # FIX: Validează că ROI-ul este în limitele imaginii
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
        """Desenează un ROI completat ÎMBUNĂTĂȚIT"""
        colors = ['lime', 'red', 'cyan', 'yellow', 'magenta', 'orange', 'white', 'pink']  # FIX: Culori mai vizibile
        color = colors[(roi_number - 1) % len(colors)]

        try:
            # Conturul mai gros și mai vizibil
            polygon = Polygon(points[:-1], fill=False, edgecolor=color,
                              linewidth=4, alpha=1.0)  # FIX: Mai vizibil
            self.ax.add_patch(polygon)
            self.roi_patches.append(polygon)

            # Umplerea semi-transparentă
            fill_polygon = Polygon(points[:-1], fill=True, facecolor=color,
                                   alpha=0.15, edgecolor='none')
            self.ax.add_patch(fill_polygon)
            self.roi_patches.append(fill_polygon)

            # Eticheta mai vizibilă
            center_x = np.mean([p[0] for p in points[:-1]])
            center_y = np.mean([p[1] for p in points[:-1]])

            text = self.ax.text(center_x, center_y, f'ROI {roi_number}',
                                color='black', fontsize=14, weight='bold',  # FIX: Text negru mai vizibil
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

        # Șterge ultimele 3 patch-uri (contur + umplere + text)
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
        """Finalizează selecția ÎMBUNĂTĂȚIT"""
        if len(self.rois) == 0:
            print("⚠️ Nu au fost desenate ROI-uri!")
            # Nu închide fereastra dacă nu sunt ROI-uri
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


# FIX: Funcție pentru înlocuirea funcțiilor în codul principal
def apply_fixes_to_main_analyzer():
    """
    Aplică fix-urile în clasa principală CombinedNeuronAnalyzer
    """
    print("🔧 Aplicând fix-urile pentru:")
    print("   ✅ Afișarea corectă a canalului verde")
    print("   ✅ Procesarea corectă a ROI-urilor")
    print("   ✅ Interfață îmbunătățită de selecție freehand")
    print("   ✅ Validare ROI-uri și feedback mai bun")

    return {
        'load_image_robust': load_image_robust_fixed,
        'ThreadSafeFreehandROI': ThreadSafeFreehandROI_Fixed
    }


if __name__ == "__main__":
    print("🔧 FIX pentru Analizor Neuroni - Canalul Verde și ROI-uri")
    print("=" * 60)
    print("PROBLEME REZOLVATE:")
    print("1. 🌿 Canalul verde nu se afișa corect")
    print("2. ❌ ROI-urile nu se procesau ('Nu au fost selectate ROI-uri')")
    print("3. 🎨 Maparea optimizată a canalelor pentru vizibilitate")
    print("4. 🖱️ Interfață îmbunătățită pentru selecție freehand")
    print("5. ✅ Validare ROI-uri și feedback mai bun utilizatorului")
    print("=" * 60)

    # Demonstrează cum să aplici fix-urile
    fixes = apply_fixes_to_main_analyzer()
    print("🚀 Fix-urile sunt gata de aplicat în codul principal!")
    print("💡 Pentru a folosi fix-urile, înlocuiește în codul principal:")
    print("   • load_image_robust -> load_image_robust_fixed")
    print("   • ThreadSafeFreehandROI -> ThreadSafeFreehandROI_Fixed")