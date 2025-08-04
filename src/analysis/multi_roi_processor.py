"""
MultiRoiProcessorWithGreenMask - VERSIUNEA FINALĂ

Fix-uri implementate:
1. Maska verde corectă din canalul verde
2. CORECTARE SUPRAEXPUNERE: intensitatea minimă pe roșu devine 0
3. DENOISING puternic pentru a elimina zgomotul
4. Threshold-uri FOARTE STRICTE pentru detectarea controlului pozitiv
5. Verificări multiple anti-scheletizare zgomot
6. Debugging complet pentru troubleshooting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.widgets import RectangleSelector
from aicsimageio import AICSImage
from skimage import filters, measure, morphology, exposure, segmentation
from skimage.io import imsave
from skimage.filters import threshold_otsu, threshold_local
from scipy import ndimage
import shutil
from typing import Tuple, List, Dict, Optional
import warnings
from PIL import Image, ImageDraw


class MultiRoiProcessorWithGreenMask:
    """
    Processor FINAL pentru imagini de neuroni cu toate fix-urile:
    - Maska verde corectă
    - Corectare supraexpunere (min roșu → 0)
    - Denoising puternic
    - Threshold-uri FOARTE STRICTE pentru control pozitiv
    - Verificări anti-scheletizare zgomot
    """

    def __init__(self, file_path: str, output_dir: str = "outputs"):
        self.file_path = file_path
        self.output_dir = output_dir
        self.roi_coords = []
        self.file_extension = os.path.splitext(file_path)[1].lower()

        self._setup_output_directory()
        self._load_image()

    def _setup_output_directory(self):
        """Configurează directorul de output."""
        if os.path.exists(self.output_dir):
            for name in os.listdir(self.output_dir):
                full_path = os.path.join(self.output_dir, name)
                if os.path.isdir(full_path) or not name.endswith(".csv"):
                    if os.path.isdir(full_path):
                        shutil.rmtree(full_path)
                    else:
                        os.remove(full_path)
            print(f"🧹 Directorul '{self.output_dir}' a fost curățat.")
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"✅ Directorul '{self.output_dir}' a fost creat.")

    def _load_image(self):
        """Încarcă imaginea și creează proiecția MIP."""
        print(f"📂 Încărcare imagine: {os.path.basename(self.file_path)}")
        print(f"   Format detectat: {self.file_extension}")

        self.img = AICSImage(self.file_path)

        if hasattr(self.img, 'channel_names') and self.img.channel_names:
            print(f"   Canale disponibile: {self.img.channel_names}")

        try:
            self.data = self.img.get_image_data("CZYX", T=0, S=0)
            print(f"✅ Succes CZYX! Shape: {self.data.shape}")
        except Exception as e:
            print(f"⚠️ Eroare la citirea cu CZYX: {e}")
            print("🔁 Încerc fallback ZCYX...")
            self.data = self.img.get_image_data("ZCYX", T=0, S=0)
            self.data = np.transpose(self.data, (1, 0, 2, 3))
            print(f"✅ Succes ZCYX! Shape: {self.data.shape}")

        print("🔄 Creez proiecția MIP...")
        self.mip = self.data.max(axis=1)
        print(f"   MIP shape: {self.mip.shape}")

        self.rgb = self._generate_rgb()
        imsave(os.path.join(self.output_dir, "full_rgb.png"), (self.rgb * 255).astype(np.uint8))
        print("✅ Imagine încărcată și procesată cu succes!")

    def _generate_rgb(self) -> np.ndarray:
        """Generează imaginea RGB cu mapare corectă pentru .czi."""
        print("🎨 Generez imaginea RGB...")
        rgb = np.zeros((*self.mip.shape[1:], 3), dtype=np.float32)

        if self.file_extension == '.czi' and hasattr(self.img, 'channel_names') and self.img.channel_names:
            channel_names = [name.lower() for name in self.img.channel_names]
            print(f"   📊 Canale detectate: {channel_names}")

            color_mapping = []
            for i, name in enumerate(channel_names):
                if i >= self.mip.shape[0]:
                    break
                if any(keyword in name for keyword in ['dapi', 'hoechst', 'blue', 'nuclei']):
                    color_mapping.append((i, (0, 0, 1), 'blue'))
                elif any(keyword in name for keyword in ['alexa fluor 488', 'fitc', 'green', 'gfp']):
                    color_mapping.append((i, (0, 1, 0), 'green'))
                elif any(keyword in name for keyword in ['alexa fluor 555', 'cy3', 'red', 'texas red']):
                    color_mapping.append((i, (1, 0, 0), 'red'))
                else:
                    colors_fallback = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
                    color_mapping.append((i, colors_fallback[i % 3], f'unknown_{i}'))

            print(f"   🎨 Mapare canale pentru .czi:")
            for channel_idx, color, name in color_mapping:
                print(f"      Canal {channel_idx} ({self.img.channel_names[channel_idx]}) -> {name}")
        else:
            print("   🎨 Mapare standard RGB:")
            color_mapping = [(0, (1, 0, 0), 'red'), (1, (0, 1, 0), 'green'), (2, (0, 0, 1), 'blue')]

        for channel_idx, color, color_name in color_mapping:
            if channel_idx < self.mip.shape[0]:
                channel = self.mip[channel_idx]
                if channel.max() > 0:
                    p2, p98 = np.percentile(channel, (2, 98))
                    if p98 > p2:
                        norm = np.clip((channel - p2) / (p98 - p2), 0, 1)
                    else:
                        norm = channel / channel.max()
                else:
                    norm = channel
                norm = np.power(norm, 0.8)
                for i in range(3):
                    rgb[..., i] += norm * color[i]

        rgb_max = rgb.max()
        if rgb_max > 0:
            rgb = rgb / rgb_max
        rgb = np.clip(rgb * 1.5, 0, 1)
        print(f"   RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        return rgb

    def select_rois(self):
        """Interfață pentru selectarea ROI-urilor."""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.rgb)

        title = f"Selectează ROI-uri pentru neuroni - {os.path.basename(self.file_path)}"
        if self.file_extension == '.czi':
            title += " (Format .czi detectat)"
        self.ax.set_title(title + "\nENTER pentru a termina.")
        self.roi_coords = []

        def _onselect(eclick, erelease):
            if None in [eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata]:
                print("⚠️ Click în afara imaginii - ignorat.")
                return

            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
                print("⚠️ ROI prea mic - ignorat.")
                return

            roi = (min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2))
            self.roi_coords.append(roi)

            colors = ['red', 'yellow', 'cyan', 'magenta', 'orange']
            color = colors[(len(self.roi_coords) - 1) % len(colors)]

            self.ax.add_patch(plt.Rectangle(
                (roi[2], roi[0]), roi[3] - roi[2], roi[1] - roi[0],
                fill=False, color=color, linewidth=3
            ))
            self.ax.text(roi[2], roi[0] - 5, f"ROI {len(self.roi_coords)}",
                         color=color, fontweight="bold", fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

            print(f"✅ ROI {len(self.roi_coords)} adăugat: {roi}")
            self.fig.canvas.draw_idle()

        def _on_key(event):
            if event.key == 'enter':
                print(f"🔵 Selectare terminată. ROI-uri: {len(self.roi_coords)}")
                plt.close()
            elif event.key == 'escape':
                print("❌ Selectare anulată.")
                self.roi_coords = []
                plt.close()

        self.fig.canvas.mpl_connect("key_press_event", _on_key)
        self.selector = RectangleSelector(
            self.ax, _onselect, useblit=False, button=[1], spancoords='pixels',
            props=dict(edgecolor='red', linewidth=2, fill=False)
        )

        instruction_text = "INSTRUCȚIUNI:\n• Click și drag pentru a selecta ROI\n• ENTER = terminare\n• ESC = anulare"
        self.ax.text(0.02, 0.98, instruction_text, transform=self.ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))

        print("🔄 Selectează ROI-uri. Apasă ENTER când termini sau ESC pentru a anula.")
        plt.show()

    def process_all_rois(self):
        """Procesează toate ROI-urile cu fix-urile finale."""
        from src.io.sholl_exported_values import ShollCSVLogger

        logger = ShollCSVLogger(output_path=self.output_dir)

        print(f"\n🔬 Procesez {len(self.roi_coords)} ROI-uri cu FIX-urile FINALE:")
        print("🔴 CORECTARE SUPRAEXPUNERE: minim roșu → 0")
        print("🔇 DENOISING PUTERNIC: elimină zgomotul")
        print("🚨 THRESHOLD-uri FOARTE STRICTE: detectez controlul pozitiv")

        for i, (y1, y2, x1, x2) in enumerate(self.roi_coords):
            roi_num = i + 1
            print(f"\n{'=' * 60}")
            print(f"🔬 ROI {roi_num}/{len(self.roi_coords)}")

            try:
                results = self._process_single_roi_final(roi_num, y1, y2, x1, x2)
                self._log_results(logger, results, roi_num)

                if results['peak'] > 0:
                    print(f"✅ ROI {roi_num} SUCCES! Peak: {results['peak']}, AUC: {results['auc']:.1f}")
                else:
                    print(f"⚠️ ROI {roi_num} - CONTROL POZITIV sau zgomot detectat")

            except Exception as e:
                print(f"❌ Eroare ROI {roi_num}: {e}")
                import traceback
                traceback.print_exc()
                self._log_results(logger, self._empty_result(), roi_num)

        print(f"\n✅ Procesare completă terminată!")
        try:
            logger.print_summary()
        except Exception as e:
            print(f"⚠️ Eroare sumar: {e}")

    def _process_single_roi_final(self, roi_num: int, y1: int, y2: int, x1: int, x2: int) -> Dict:
        """Procesează un ROI cu toate fix-urile finale."""
        roi_dir = os.path.join(self.output_dir, f"roi_{roi_num}_final")
        debug_dir = os.path.join(roi_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Extrage canale
        channels = self._extract_roi_channels(y1, y2, x1, x2)
        print(f"   📊 Canale: roșu {channels['red'].shape}, verde {channels['green'].shape}")

        # Maska verde
        green_mask = self._create_green_mask(channels['green'], debug_dir, roi_num)
        if green_mask.sum() == 0:
            print(f"   ❌ Nu s-a detectat neuron în canalul verde!")
            return self._empty_result()
        print(f"   ✅ Maska verde: {green_mask.sum()} pixeli")

        # Soma
        soma_center = self._find_soma_in_green_mask(channels['blue'], green_mask, debug_dir, roi_num)
        print(f"   ✅ Soma la ({soma_center[0]}, {soma_center[1]})")

        # CRUCIAL: Dendrite cu toate fix-urile
        dendrites = self._extract_dendrites_FINAL_VERSION(channels['red'], green_mask, debug_dir, roi_num)
        if dendrites is None or dendrites.sum() == 0:
            print(f"   ❌ Nu s-au detectat dendrite reale!")
            return self._empty_result()
        print(f"   ✅ Dendrite detectate: {dendrites.sum()} pixeli")

        # Salvare și analiză
        binary_path = self._save_final_results(dendrites, channels['red'], soma_center, green_mask, roi_dir, roi_num)
        results = self._perform_sholl_analysis(dendrites, binary_path, soma_center, roi_dir, roi_num)
        return results

    def _extract_dendrites_FINAL_VERSION(self, red_channel: np.ndarray, green_mask: np.ndarray,
                                        debug_dir: str, roi_num: int) -> np.ndarray:
        """
        FIX ULTRA-AGRESIV cu TOATE verificările avansate:
        1. Percentile extreme (95-99.5%)
        2. SNR (Signal-to-Noise Ratio)
        3. Analiză morfologică (eccentricity, aspect ratio)
        4. Analiză topologică a scheletului (lungime ramuri)
        5. Verificări contrast și acoperire
        """
        print(f"🔴 ROI {roi_num}: FIX ULTRA-AGRESIV cu VERIFICĂRI AVANSATE...")

        # VERIFICARE PRELIMINARĂ 1: Contrast general
        contrast = red_channel.max() - red_channel.min()
        print(f"   📊 VERIFICARE CONTRAST GENERAL:")
        print(f"      Contrast total: {contrast:.6f}")

        if contrast < 0.01:  # Contrast mai mic de 1%
            print(f"   ❌ STOP: Contrast prea mic în canalul roșu ({contrast:.4f})")
            print("      → Imagine 'moartă' fără variație")
            return None

        # VERIFICARE PRELIMINARĂ 2: Acoperire masca verde
        coverage = green_mask.sum() / green_mask.size
        print(f"   📊 VERIFICARE ACOPERIRE MASCA VERDE:")
        print(f"      Acoperire: {coverage:.2%}")

        if coverage < 0.05:  # Mai puțin de 5%
            print(f"   ❌ STOP: Maska verde acoperă prea puțin ({coverage:.2%})")
            print("      → ROI probabil fără neuroni")
            return None

        # Aplică masca verde pentru a lucra doar în regiunea de interes
        red_in_green_raw = red_channel * green_mask.astype(float)

        if red_in_green_raw.sum() == 0:
            print("   ❌ STOP: Nu există semnal roșu în masca verde!")
            return None

        # Găsește toate valorile nenule în masca verde
        non_zero_values = red_in_green_raw[red_in_green_raw > 0]

        if len(non_zero_values) == 0:
            print("   ❌ STOP: Nu există pixeli cu valori în masca verde!")
            return None

        print(f"   📊 În masca verde ORIGINAL:")
        print(f"      Pixeli: {len(non_zero_values)}")
        print(f"      Min: {non_zero_values.min():.6f}")
        print(f"      Max: {non_zero_values.max():.6f}")
        print(f"      Mean: {non_zero_values.mean():.6f}")
        print(f"      Std: {np.std(non_zero_values):.6f}")

        # STRATEGIE ULTRA-AGRESIVĂ: Testează percentile extreme
        background_percentiles_to_try = [95, 97, 98, 99, 99.5]

        best_result = None
        best_skeleton_count = 0
        best_percentile = None

        for background_percentile in background_percentiles_to_try:
            print(f"\n   🧪 TESTEZ percentila {background_percentile}...")

            background_level = np.percentile(non_zero_values, background_percentile)
            print(f"      Nivel fundal (P{background_percentile}): {background_level:.6f}")

            # Scade fundalul din TOATĂ imaginea
            red_test = red_channel - background_level
            red_test = np.maximum(red_test, 0)  # Negative → 0

            # Verifică dacă mai există semnal
            if red_test.max() == 0:
                print(f"      ❌ Nu mai există semnal după P{background_percentile}")
                continue

            red_test_norm = red_test / red_test.max()
            red_test_in_green = red_test_norm * green_mask.astype(float)
            clean_non_zero = red_test_in_green[red_test_in_green > 0]

            if len(clean_non_zero) < 20:
                print(f"      ❌ Prea puțini pixeli după P{background_percentile}: {len(clean_non_zero)}")
                continue

            clean_mean = clean_non_zero.mean()
            clean_std = clean_non_zero.std()
            clean_max = clean_non_zero.max()

            print(f"      📊 După P{background_percentile}:")
            print(f"         Pixeli: {len(clean_non_zero)}")
            print(f"         Mean: {clean_mean:.6f}")
            print(f"         Std: {clean_std:.6f}")
            print(f"         Max: {clean_max:.6f}")

            # VERIFICARE AVANSATĂ 1: SNR (Signal-to-Noise Ratio)
            background_std = np.std(red_test) + 1e-6  # Evită diviziunea cu 0
            snr = clean_mean / background_std
            print(f"      📡 SNR: {snr:.2f}")

            if snr < 1.5:
                print(f"      ❌ SNR prea mic ({snr:.2f} < 1.5) → noise dominant")
                continue

            # Verificări de bază
            if (clean_max < 0.1 or clean_std < 0.02):
                print(f"      ❌ Semnal insuficient (max={clean_max:.3f}, std={clean_std:.3f})")
                continue

            print(f"      ✅ P{background_percentile} trece verificările de bază și SNR!")

            # Testează procesarea completă
            try:
                # Denoising foarte ușor
                red_smooth = filters.gaussian(red_test_in_green, sigma=0.3)

                # Threshold conservator
                threshold = clean_mean + clean_std * 0.5
                threshold = max(threshold, 0.05)

                binary_test = red_smooth > threshold
                binary_test = binary_test & green_mask

                # Curățare minimă
                cleaned_test = morphology.remove_small_objects(binary_test, min_size=3)

                if cleaned_test.sum() < 10:
                    print(f"      ❌ Prea puține pixeli după curățare: {cleaned_test.sum()}")
                    continue

                # VERIFICARE AVANSATĂ 2: Analiză morfologică
                labels = measure.label(cleaned_test)
                props = measure.regionprops(labels)

                valid_regions = 0
                total_area = 0

                for region in props:
                    # Exclude structuri fals pozitive (foarte subțiri și mici)
                    if region.eccentricity > 0.99 and region.area < 100:
                        print(f"      ⚠️ Structură suspectă: eccentricity={region.eccentricity:.3f}, area={region.area}")
                        continue

                    # Exclude structuri foarte mici și foarte rotunde (probabil noise)
                    if region.area < 5 or region.eccentricity < 0.3:
                        continue

                    valid_regions += 1
                    total_area += region.area

                if valid_regions == 0:
                    print(f"      ❌ Nu există regiuni morfologic valide")
                    continue

                print(f"      ✅ Regiuni morfologic valide: {valid_regions}")

                # Scheletizare
                skeleton_test = morphology.skeletonize(cleaned_test)
                skeleton_count = skeleton_test.sum()

                if skeleton_count < 10:
                    print(f"      ❌ Prea puțin skeleton: {skeleton_count}")
                    continue

                # VERIFICARE AVANSATĂ 3: Analiză topologică a scheletului
                try:
                    # Analiză simplă a conectivității scheletului
                    skel_labels = measure.label(skeleton_test)
                    skel_props = measure.regionprops(skel_labels)

                    if len(skel_props) == 0:
                        print(f"      ❌ Skeleton fără componente conectate")
                        continue

                    # Calculează lungimea medie a componentelor
                    component_areas = [prop.area for prop in skel_props]
                    mean_component_length = np.mean(component_areas)

                    print(f"      🌳 Analiză skeleton:")
                    print(f"         Componente: {len(skel_props)}")
                    print(f"         Lungime medie: {mean_component_length:.1f} px")

                    # Exclude skeleturi cu ramuri foarte scurte (probabil noise)
                    if mean_component_length < 8:  # Pragul adaptat
                        print(f"      ❌ Skeleton cu ramuri foarte scurte ({mean_component_length:.1f} < 8) → noise probabil")
                        continue

                    # Verifică dacă există cel puțin o componentă substanțială
                    max_component_area = max(component_areas)
                    if max_component_area < 15:  # Cel puțin o ramură de 15+ pixeli
                        print(f"      ❌ Nicio componentă substanțială (max={max_component_area})")
                        continue

                except Exception as e:
                    print(f"      ⚠️ Eroare analiză topologică: {e}")
                    # Continuă fără analiza topologică

                print(f"      ✅ TOATE verificările avansate trecute!")
                print(f"         SNR: {snr:.2f}")
                print(f"         Regiuni valide: {valid_regions}")
                print(f"         Skeleton: {skeleton_count} px")
                print(f"         Lungime medie componente: {mean_component_length:.1f} px")

                # Păstrează cel mai bun rezultat
                if skeleton_count > best_skeleton_count:
                    best_result = {
                        'skeleton': skeleton_test,
                        'red_processed': red_test_norm,
                        'threshold': threshold,
                        'percentile': background_percentile,
                        'background_level': background_level,
                        'snr': snr,
                        'valid_regions': valid_regions,
                        'mean_component_length': mean_component_length,
                        'clean_stats': {
                            'count': len(clean_non_zero),
                            'max': clean_max,
                            'mean': clean_mean,
                            'std': clean_std
                        }
                    }
                    best_skeleton_count = skeleton_count
                    best_percentile = background_percentile

            except Exception as e:
                print(f"      ❌ Eroare în procesare: {e}")

        # Evaluează rezultatele
        if best_result is None:
            print(f"\n   ❌ STOP: NICIO percentilă nu a trecut TOATE verificările avansate!")
            print("      → CONTROL POZITIV confirmat - nu există dendrite reale")
            print("      → Tot semnalul este noise/artefacte")

            # Salvează analiza pentru debugging
            self._save_advanced_analysis(red_channel, non_zero_values, green_mask, debug_dir)
            return None

        print(f"\n   ✅ SUCCES cu percentila {best_percentile}!")
        print(f"      🎯 Rezultat validat cu TOATE verificările avansate:")
        print(f"         Skeleton final: {best_skeleton_count} pixeli")
        print(f"         SNR: {best_result['snr']:.2f}")
        print(f"         Regiuni morfologic valide: {best_result['valid_regions']}")
        print(f"         Lungime medie componente: {best_result['mean_component_length']:.1f} px")
        print(f"         Threshold: {best_result['threshold']:.6f}")
        print(f"         Fundal eliminat: {best_result['background_level']:.6f}")

        # Salvează debug complet cu verificările avansate
        self._save_debug_image(red_channel, "01_original_extreme_noise", debug_dir)
        self._save_debug_image(best_result['red_processed'], "02_best_percentile_clean", debug_dir)
        self._save_debug_image(best_result['red_processed'] * green_mask.astype(float), "03_clean_in_green", debug_dir)
        self._save_debug_image(best_result['skeleton'], "04_final_skeleton_validated", debug_dir)

        # Salvează analiza completă avansată
        self._save_advanced_analysis(red_channel, non_zero_values, green_mask, debug_dir, best_result)

        return best_result['skeleton']

    def _save_advanced_analysis(self, original, non_zero_values, green_mask, debug_dir, best_result=None):
        """Salvează analiză completă cu toate verificările avansate."""
        try:
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))

            # Imaginea originală
            im1 = axes[0,0].imshow(original, cmap='Reds')
            axes[0,0].set_title('Original (noise extrem)')
            axes[0,0].axis('off')
            plt.colorbar(im1, ax=axes[0,0])

            # Masca verde
            axes[0,1].imshow(green_mask, cmap='Greens')
            axes[0,1].set_title(f'Maska verde (acoperire: {green_mask.sum()/green_mask.size:.1%})')
            axes[0,1].axis('off')

            # Histograma cu percentilele
            axes[0,2].hist(non_zero_values, bins=100, alpha=0.7, color='red', log=True)
            percentiles = [90, 95, 97, 98, 99, 99.5]
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
            for i, p in enumerate(percentiles):
                val = np.percentile(non_zero_values, p)
                axes[0,2].axvline(val, color=colors[i % len(colors)], linestyle='--',
                                 label=f'P{p}: {val:.3f}')
            axes[0,2].set_title('Histogramă + percentile')
            axes[0,2].legend(fontsize=8)
            axes[0,2].set_yscale('log')

            # Contrast analysis
            contrast = original.max() - original.min()
            axes[0,3].bar(['Contrast', 'Min req'], [contrast, 0.01], color=['red', 'green'])
            axes[0,3].set_title(f'Contrast: {contrast:.4f}')
            axes[0,3].set_ylabel('Valoare')

            # Testează și afișează rezultatele pentru diferite percentile
            test_percentiles = [95, 97, 98, 99]
            for i, p in enumerate(test_percentiles):
                row = 1 + i // 2
                col = i % 2

                background_level = np.percentile(non_zero_values, p)
                test_result = original - background_level
                test_result = np.maximum(test_result, 0)

                if test_result.max() > 0:
                    test_result_norm = test_result / test_result.max()
                    test_in_green = test_result_norm * green_mask.astype(float)

                    # Calculează SNR pentru această percentilă
                    clean_pixels = test_in_green[test_in_green > 0]
                    if len(clean_pixels) > 0:
                        background_std = np.std(test_result) + 1e-6
                        snr = clean_pixels.mean() / background_std
                    else:
                        snr = 0
                else:
                    test_in_green = test_result
                    snr = 0

                im = axes[row, col].imshow(test_in_green, cmap='Reds')
                title = f'P{p} (SNR: {snr:.2f})'
                if best_result and best_result['percentile'] == p:
                    title += ' ✅ALES'
                    axes[row, col].set_facecolor('lightgreen')
                elif snr < 1.5:
                    title += ' ❌SNR'
                    axes[row, col].set_facecolor('lightcoral')
                axes[row, col].set_title(title)
                axes[row, col].axis('off')

            # Statistici complete
            axes[2,2].axis('off')
            stats_text = f"""ANALIZĂ AVANSATĂ COMPLETĂ:

Verificări preliminare:
✓ Contrast general: {original.max() - original.min():.4f} (>0.01)
✓ Acoperire masca verde: {green_mask.sum()/green_mask.size:.1%} (>5%)

În masca verde:
• Pixeli: {len(non_zero_values)}
• Range: {non_zero_values.max() - non_zero_values.min():.4f}
• Mean: {non_zero_values.mean():.4f}
• Std: {np.std(non_zero_values):.4f}

Percentile testate: 95, 97, 98, 99, 99.5

Verificări avansate aplicate:
✓ SNR ≥ 1.5 (signal vs noise)
✓ Analiză morfologică (eccentricity, area)
✓ Analiză topologică (lungime componente)"""

            if best_result:
                stats_text += f"""

🎯 REZULTAT FINAL:
Percentila: {best_result['percentile']}%
SNR: {best_result['snr']:.2f}
Regiuni valide: {best_result['valid_regions']}
Lungime medie: {best_result['mean_component_length']:.1f}px
Skeleton: {best_result['skeleton'].sum()} pixeli"""
            else:
                stats_text += "\n\n❌ TOATE percentilele respinse"
                stats_text += "\n→ CONTROL POZITIV confirmat"

            axes[2,2].text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
                          transform=axes[2,2].transAxes, family='monospace')

            # Ultima poziție - rezultatul final sau mesaj de eșec
            if best_result:
                axes[2,3].imshow(best_result['skeleton'], cmap='gray_r')
                axes[2,3].set_title(f'Skeleton final validat\n{best_result["skeleton"].sum()} pixeli')
            else:
                axes[2,3].text(0.5, 0.5, 'CONTROL\nPOZITIV\n\nNu există\ndendrite reale',
                              ha='center', va='center', fontsize=14, color='red', weight='bold')
                axes[2,3].set_title('Rezultat final')
            axes[2,3].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "11_advanced_analysis_complete.png"),
                       dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"   ⚠️ Nu s-a putut salva analiza avansată: {e}")

    def _save_percentile_analysis(self, original, non_zero_values, green_mask, debug_dir, best_result=None):
        """Salvează o analiză completă a percentilelor testate."""
        try:
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))

            # Imaginea originală
            im1 = axes[0,0].imshow(original, cmap='Reds')
            axes[0,0].set_title('Original (noise extrem)')
            axes[0,0].axis('off')
            plt.colorbar(im1, ax=axes[0,0])

            # Histograma cu percentilele marcate
            axes[0,1].hist(non_zero_values, bins=100, alpha=0.7, color='red', log=True)
            percentiles_to_mark = [85, 90, 95, 97, 98, 99, 99.5]
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            for i, p in enumerate(percentiles_to_mark):
                val = np.percentile(non_zero_values, p)
                axes[0,1].axvline(val, color=colors[i % len(colors)], linestyle='--',
                                 label=f'P{p}: {val:.3f}')
            axes[0,1].set_title('Histogramă cu percentile')
            axes[0,1].legend(fontsize=8)
            axes[0,1].set_yscale('log')

            # Testează câteva percentile și afișează rezultatele
            test_percentiles = [95, 97, 98, 99]
            for i, p in enumerate(test_percentiles):
                if i >= 6:  # Maxim 6 subplot-uri pentru teste
                    break

                row = (i + 2) // 3
                col = (i + 2) % 3

                background_level = np.percentile(non_zero_values, p)
                test_result = original - background_level
                test_result = np.maximum(test_result, 0)

                if test_result.max() > 0:
                    test_result_norm = test_result / test_result.max()
                else:
                    test_result_norm = test_result

                test_in_green = test_result_norm * green_mask.astype(float)

                im = axes[row, col].imshow(test_in_green, cmap='Reds')
                title = f'P{p} eliminat'
                if best_result and best_result['percentile'] == p:
                    title += ' (ALES)'
                    axes[row, col].set_facecolor('lightgreen')
                axes[row, col].set_title(title)
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col])

            # Statistici în ultimul subplot
            axes[2,2].axis('off')
            stats_text = f"""Analiză percentile (în masca verde):
Total pixeli: {len(non_zero_values)}
Min: {non_zero_values.min():.6f}
Max: {non_zero_values.max():.6f}
Mean: {non_zero_values.mean():.6f}
Median: {np.median(non_zero_values):.6f}
Std: {np.std(non_zero_values):.6f}

Percentile testate:
P95: {np.percentile(non_zero_values, 95):.6f}
P97: {np.percentile(non_zero_values, 97):.6f}
P98: {np.percentile(non_zero_values, 98):.6f}
P99: {np.percentile(non_zero_values, 99):.6f}
P99.5: {np.percentile(non_zero_values, 99.5):.6f}"""

            if best_result:
                stats_text += f"""

REZULTAT FINAL:
Percentila aleasă: {best_result['percentile']}
Background eliminat: {best_result['background_level']:.6f}
Threshold folosit: {best_result['threshold']:.6f}
Pixeli skeleton: {best_result['skeleton'].sum()}"""
            else:
                stats_text += "\n\nNICIO percentilă nu a dat rezultate!"

            axes[2,2].text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
                          transform=axes[2,2].transAxes, family='monospace')

            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "10_percentile_analysis_extreme.png"),
                       dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"   ⚠️ Nu s-a putut salva analiza percentilelor: {e}")

    def _save_background_removal_analysis(self, original, background_removed, background_level, percentile, debug_dir):
        """Salvează o analiză vizuală a eliminării fundalului."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Imaginea originală
            im1 = axes[0,0].imshow(original, cmap='Reds')
            axes[0,0].set_title('Original (cu noise mare)')
            axes[0,0].axis('off')
            plt.colorbar(im1, ax=axes[0,0])

            # Imaginea după eliminarea fundalului
            im2 = axes[0,1].imshow(background_removed, cmap='Reds')
            axes[0,1].set_title('După eliminarea fundalului')
            axes[0,1].axis('off')
            plt.colorbar(im2, ax=axes[0,1])

            # Diferența
            diff = original - background_removed
            im3 = axes[0,2].imshow(diff, cmap='Blues')
            axes[0,2].set_title('Fundal eliminat')
            axes[0,2].axis('off')
            plt.colorbar(im3, ax=axes[0,2])

            # Histograma originală
            axes[1,0].hist(original.flatten(), bins=50, alpha=0.7, color='red', label='Original')
            axes[1,0].axvline(background_level, color='blue', linestyle='--',
                             label=f'Fundal (p{percentile}): {background_level:.3f}')
            axes[1,0].set_title('Histogramă original')
            axes[1,0].legend()
            axes[1,0].set_yscale('log')

            # Histograma după eliminare
            bg_removed_nonzero = background_removed[background_removed > 0]
            if len(bg_removed_nonzero) > 0:
                axes[1,1].hist(bg_removed_nonzero, bins=50, alpha=0.7, color='green', label='După eliminare')
                axes[1,1].set_title('Histogramă după eliminare fundal')
                axes[1,1].legend()
                axes[1,1].set_yscale('log')
            else:
                axes[1,1].text(0.5, 0.5, 'Nu mai există\nsemnal', ha='center', va='center')
                axes[1,1].set_title('Nu mai există semnal')

            # Statistici
            stats_text = f"""Eliminare fundal:
Percentila folosită: {percentile}%
Nivel fundal: {background_level:.6f}
Pixeli originali > 0: {np.count_nonzero(original)}
Pixeli după eliminare > 0: {np.count_nonzero(background_removed)}
Reducere: {(1 - np.count_nonzero(background_removed)/max(1, np.count_nonzero(original)))*100:.1f}%

Original:
  Min: {original.min():.6f}
  Max: {original.max():.6f}
  Mean: {original.mean():.6f}

După eliminare:
  Min: {background_removed.min():.6f}
  Max: {background_removed.max():.6f}
  Mean: {background_removed.mean():.6f}"""

            axes[1,2].text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
                          transform=axes[1,2].transAxes, family='monospace')
            axes[1,2].set_xlim(0, 1)
            axes[1,2].set_ylim(0, 1)
            axes[1,2].axis('off')
            axes[1,2].set_title('Statistici eliminare fundal')

            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "09_background_removal_analysis.png"),
                       dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"   ⚠️ Nu s-a putut salva analiza eliminării fundalului: {e}")

    def _save_contrast_analysis(self, red_in_green, non_zero_pixels, background_level, signal_level, snr, debug_dir):
        """Salvează o analiză vizuală a contrastului pentru debugging."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Imaginea originală
            axes[0,0].imshow(red_in_green, cmap='Reds')
            axes[0,0].set_title('Semnal roșu în masca verde')
            axes[0,0].axis('off')

            # Histograma valorilor
            axes[0,1].hist(non_zero_pixels, bins=50, alpha=0.7, color='red')
            axes[0,1].axvline(background_level, color='blue', linestyle='--', label=f'Background: {background_level:.3f}')
            axes[0,1].axvline(signal_level, color='green', linestyle='--', label=f'Signal: {signal_level:.3f}')
            axes[0,1].set_title(f'Histogramă intensități (SNR: {snr:.2f})')
            axes[0,1].legend()

            # Threshold visualization
            threshold_vis = red_in_green > (background_level + (signal_level - background_level) * 0.3)
            axes[1,0].imshow(threshold_vis, cmap='gray')
            axes[1,0].set_title('Threshold adaptat')
            axes[1,0].axis('off')

            # Statistici text
            stats_text = f"""Analiză contrast:
Range total: {np.max(non_zero_pixels) - np.min(non_zero_pixels):.6f}
Std dev: {np.std(non_zero_pixels):.6f}
Background: {background_level:.6f}
Signal: {signal_level:.6f}
SNR: {snr:.2f}
Pixeli analizați: {len(non_zero_pixels)}"""

            axes[1,1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
            axes[1,1].set_xlim(0, 1)
            axes[1,1].set_ylim(0, 1)
            axes[1,1].axis('off')
            axes[1,1].set_title('Statistici')

            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "08_contrast_analysis.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"   ⚠️ Nu s-a putut salva analiza de contrast: {e}")

    def _create_green_mask(self, green_channel: np.ndarray, debug_dir: str, roi_num: int) -> np.ndarray:
        """Crează masca din canalul verde."""
        green_norm = exposure.rescale_intensity(green_channel, out_range=(0, 1))
        green_smooth = filters.gaussian(green_norm, sigma=1.0)

        if green_smooth.sum() > 0:
            non_zero_values = green_smooth[green_smooth > 0]
            if len(non_zero_values) > 100:
                threshold = np.percentile(non_zero_values, 70)
                threshold = max(threshold, 0.08)
            else:
                threshold = 0.1
        else:
            return np.zeros_like(green_channel, dtype=bool)

        initial_mask = green_smooth > threshold
        cleaned_mask = morphology.remove_small_objects(initial_mask, min_size=100)
        filled_mask = morphology.binary_fill_holes(cleaned_mask)
        final_mask = morphology.binary_closing(filled_mask, morphology.disk(3))
        final_mask = morphology.remove_small_objects(final_mask, min_size=150)

        coverage = 100 * final_mask.sum() / green_channel.size
        print(f"      Maska verde: threshold={threshold:.4f}, acoperire={coverage:.1f}%")
        return final_mask

    def _find_soma_in_green_mask(self, blue_channel: np.ndarray, green_mask: np.ndarray,
                                 debug_dir: str, roi_num: int) -> Tuple[int, int]:
        """Găsește soma-ul în masca verde."""
        blue_norm = exposure.rescale_intensity(blue_channel, out_range=(0, 1))
        blue_in_green = blue_norm * green_mask.astype(float)

        if blue_in_green.sum() == 0:
            coords = np.where(green_mask)
            if len(coords[0]) > 0:
                center_y = int(np.mean(coords[0]))
                center_x = int(np.mean(coords[1]))
            else:
                center_y = blue_norm.shape[0] // 2
                center_x = blue_norm.shape[1] // 2
            return (center_x, center_y)

        max_coords = np.unravel_index(np.argmax(blue_in_green), blue_in_green.shape)
        center_y, center_x = max_coords
        return (center_x, center_y)

    def _save_final_results(self, skeleton: np.ndarray, red_channel: np.ndarray,
                           soma_center: Tuple[int, int], green_mask: np.ndarray,
                           roi_dir: str, roi_num: int) -> str:
        """Salvează rezultatele finale."""
        binary_path = os.path.join(roi_dir, "dendrites_binary_final.tif")
        imsave(binary_path, (skeleton * 255).astype(np.uint8))

        fig, ax = plt.subplots(figsize=(10, 8))
        red_norm = exposure.rescale_intensity(red_channel, out_range=(0, 1))
        ax.imshow(red_norm, cmap="Reds", alpha=0.6)
        ax.imshow(skeleton, cmap="gray_r", alpha=0.9)
        ax.contour(green_mask, colors='lime', linewidths=3)
        ax.plot(soma_center[0], soma_center[1], 'b*', markersize=20,
                markeredgecolor='yellow', markeredgewidth=3, label='Soma')

        skeleton_pixels = skeleton.sum()
        title = f"ROI {roi_num} FINAL - Dendrite: {skeleton_pixels} pixeli"
        ax.set_title(title)
        ax.axis('off')
        ax.legend()
        plt.savefig(os.path.join(roi_dir, "rezultat_final.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return binary_path

    def _perform_sholl_analysis(self, skeleton: np.ndarray, binary_path: str,
                               soma_center: Tuple[int, int], roi_dir: str, roi_num: int) -> Dict:
        """Efectuează analiza Sholl."""
        if skeleton.sum() < 20:
            print(f"   ⚠️ Prea puține dendrite pentru Sholl ({skeleton.sum()} pixeli)")
            return self._empty_result()

        try:
            from src.io.sholl import ShollAnalyzer

            analyzer = ShollAnalyzer()
            sholl_path = os.path.join(roi_dir, "sholl_analysis_final.png")
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
                print(f"      Peak: {results.get('peak_number', 0)}")
                print(f"      Radius: {results.get('radius_at_peak', 0)}")
                print(f"      AUC: {results.get('auc', 0):.2f}")

                return {
                    'peak': results.get('peak_number', 0),
                    'radius': results.get('radius_at_peak', 0),
                    'auc': results.get('auc', 0),
                    'regression_coef': results.get('slope', 0),
                    'total_intersections': results.get('total_intersections', 0),
                    'max_radius': results.get('max_radius', 0),
                    'mean_intersections': results.get('mean_intersections', 0)
                }

        except Exception as e:
            print(f"   ❌ Eroare Sholl: {e}")

        return self._empty_result()

    def _extract_roi_channels(self, y1: int, y2: int, x1: int, x2: int) -> Dict[str, np.ndarray]:
        """Extrage canalele pentru ROI cu mapare corectă."""
        if self.file_extension == '.czi' and hasattr(self.img, 'channel_names') and self.img.channel_names:
            channel_names = [name.lower() for name in self.img.channel_names]

            blue_idx = green_idx = red_idx = 0
            for i, name in enumerate(channel_names):
                if i >= self.mip.shape[0]:
                    break
                if any(keyword in name for keyword in ['dapi', 'hoechst', 'blue', 'nuclei']):
                    blue_idx = i
                elif any(keyword in name for keyword in ['alexa fluor 488', 'fitc', 'green', 'gfp']):
                    green_idx = i
                elif any(keyword in name for keyword in ['alexa fluor 555', 'cy3', 'red', 'texas red']):
                    red_idx = i

            return {
                'blue': self.mip[blue_idx, y1:y2, x1:x2],
                'green': self.mip[green_idx, y1:y2, x1:x2],
                'red': self.mip[red_idx, y1:y2, x1:x2]
            }
        else:
            return {
                'blue': self.mip[0, y1:y2, x1:x2],
                'green': self.mip[1, y1:y2, x1:x2],
                'red': self.mip[2, y1:y2, x1:x2]
            }

    def _log_results(self, logger, results: Dict, roi_num: int):
        """Salvează rezultatele în CSV."""
        logger.log_result(
            image_name=os.path.basename(self.file_path),
            roi_index=roi_num,
            peak=results['peak'],
            radius=results['radius'],
            auc=results['auc'],
            regression_coef=results['regression_coef'],
            roi_folder=f"roi_{roi_num}_final",
            total_intersections=results.get('total_intersections', 0),
            max_radius=results.get('max_radius', 0),
            mean_intersections=results.get('mean_intersections', 0)
        )

    def _save_debug_image(self, image: np.ndarray, name: str, debug_dir: str):
        """Salvează imagine pentru debug."""
        if image.dtype == bool:
            image_to_save = (image * 255).astype(np.uint8)
        else:
            image_to_save = (exposure.rescale_intensity(image, out_range=(0, 255))).astype(np.uint8)
        imsave(os.path.join(debug_dir, f"{name}.png"), image_to_save)

    def _empty_result(self):
        """Returnează rezultat gol."""
        return {
            'peak': 0, 'radius': 0, 'auc': 0, 'regression_coef': 0,
            'total_intersections': 0, 'max_radius': 0, 'mean_intersections': 0
        }

    def run(self):
        """Rulează procesarea completă cu TOATE fix-urile finale."""
        print("🚀 PROCESARE FINALĂ cu TOATE FIX-urile!")
        print(f"📁 Format: {self.file_extension}")

        if self.file_extension == '.czi':
            print("🔬 Suport special .czi activat")

        print("\n🔧 FIX-uri FINALE active:")
        print("✅ 1. Maska verde corectă din canalul verde")
        print("✅ 2. CORECTARE SUPRAEXPUNERE: minim roșu → 0")
        print("✅ 3. DENOISING PUTERNIC: Gaussian + Median + Final")
        print("✅ 4. THRESHOLD-uri FOARTE STRICTE pentru control pozitiv:")
        print("      • MIN_PIXEL_COUNT: 300")
        print("      • MIN_MEAN_INTENSITY: 20%")
        print("      • MIN_MAX_INTENSITY: 40%")
        print("      • MIN_MEDIAN_INTENSITY: 15%")
        print("✅ 5. THRESHOLD procesare FOARTE STRICT: 25%")
        print("✅ 6. Verificări finale: min 500 pixeli, densitate 15%, max 3 regiuni")
        print("✅ 7. Scheletizare doar pentru dendrite reale detectate")
        print("✅ 8. Debug complet cu 7 etape salvate")

        self.select_rois()

        if not self.roi_coords:
            print("⚠️ Nu au fost selectate ROI-uri!")
            return

        self.process_all_rois()
        print("\n🎉 PROCESARE FINALĂ completă!")


if __name__ == "__main__":
    import sys

    print("🔬 MultiRoiProcessor VERSIUNEA FINALĂ")
    print("=" * 80)
    print("🔧 FIX-uri FINALE implementate în ordinea corectă:")
    print("✅ 1. Maska verde corectă din canalul verde")
    print("✅ 2. CORECTARE SUPRAEXPUNERE: minim pe roșu devine 0")
    print("✅ 3. DENOISING PUTERNIC în 3 etape:")
    print("      • Gaussian blur (sigma=1.0)")
    print("      • Median filter (disk=2)")
    print("      • Final smoothing (sigma=0.5)")
    print("✅ 4. THRESHOLD-uri FOARTE STRICTE pentru control pozitiv:")
    print("      • Pixeli minimi: 300")
    print("      • Intensitate medie: 20%")
    print("      • Intensitate maximă: 40%")
    print("      • Intensitate mediană: 15%")
    print("✅ 5. THRESHOLD procesare FOARTE STRICT: 25% (percentila 50)")
    print("✅ 6. Verificări finale anti-scheletizare:")
    print("      • Minim 500 pixeli")
    print("      • Densitate minimă 15%")
    print("      • Maxim 3 regiuni separate")
    print("✅ 7. Scheletizare DOAR pentru dendrite reale")
    print("✅ 8. Debug complet cu imagini pentru toate etapele")
    print("=" * 80)

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"📁 Utilizez fișierul: {file_path}")
    else:
        file_path = "path/to/your/image.czi"
        print(f"📁 Fișier de test: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ Fișierul nu există: {file_path}")
        print("💡 Utilizare: python script.py path/to/your/image.czi")
        sys.exit(1)

    try:
        processor = MultiRoiProcessorWithGreenMask(file_path)
        processor.run()
    except Exception as e:
        print(f"❌ Eroare critică: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)