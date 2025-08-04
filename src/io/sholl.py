#!/usr/bin/env python3
"""
Implementare completă a analizei Sholl pentru neuroni.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from typing import Tuple, Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


class ShollAnalyzer:
    """Analizor pentru analiza Sholl a neuronilor."""

    def __init__(self):
        self.results = {}

    def analyze(self, binary_image: np.ndarray, soma_center: Tuple[int, int],
                step_size: int = 5, max_radius: int = None,
                save_path: str = None) -> Dict:
        """
        Efectuează analiza Sholl completă.

        Args:
            binary_image: Imaginea binară cu dendritele
            soma_center: Coordonatele centrului soma (x, y)
            step_size: Pasul pentru razele concentrice
            max_radius: Raza maximă de analiză
            save_path: Path pentru salvarea graficului

        Returns:
            Dict cu rezultatele analizei Sholl
        """
        print("🔬 Încep analiza Sholl...")

        # Validare input
        if binary_image.sum() == 0:
            print("⚠️ Imaginea binară este goală!")
            return self._empty_results()

        # Setare parametri
        if max_radius is None:
            max_radius = min(binary_image.shape) // 2

        # Calculează intersecțiile
        radii, intersections = self._calculate_intersections(
            binary_image, soma_center, step_size, max_radius
        )

        if len(radii) == 0 or np.sum(intersections) == 0:
            print("⚠️ Nu s-au găsit intersecții!")
            return self._empty_results()

        # Calculează metrici
        results = self._calculate_metrics(radii, intersections)

        # Generează graficul
        if save_path:
            self._create_plot(radii, intersections, results, save_path)

        print(f"✅ Analiza Sholl completă!")
        print(f"   • Intersecții totale: {np.sum(intersections)}")
        print(f"   • Peak: {results['peak_number']} la raza {results['radius_at_peak']}")
        print(f"   • AUC: {results['auc']:.2f}")

        return results

    def _calculate_intersections(self, binary_image: np.ndarray,
                                 soma_center: Tuple[int, int],
                                 step_size: int, max_radius: int) -> Tuple[List[int], List[int]]:
        """Calculează intersecțiile pentru fiecare rază."""
        cx, cy = soma_center
        radii = list(range(step_size, max_radius, step_size))
        intersections = []

        for radius in radii:
            # Creează cercul
            circle_points = self._generate_circle_points(cx, cy, radius)

            # Numără intersecțiile
            intersection_count = 0
            for x, y in circle_points:
                if (0 <= y < binary_image.shape[0] and
                        0 <= x < binary_image.shape[1] and
                        binary_image[y, x]):
                    intersection_count += 1

            intersections.append(intersection_count)

        return radii, intersections

    def _generate_circle_points(self, cx: int, cy: int, radius: int) -> List[Tuple[int, int]]:
        """Generează punctele de pe circumferința unui cerc."""
        points = []

        # Folosim algoritmul Bresenham pentru cerc
        x = 0
        y = radius
        d = 3 - 2 * radius

        while y >= x:
            # Adaugă punctele pentru toate octantele
            octant_points = [
                (cx + x, cy + y), (cx - x, cy + y),
                (cx + x, cy - y), (cx - x, cy - y),
                (cx + y, cy + x), (cx - y, cy + x),
                (cx + y, cy - x), (cx - y, cy - x)
            ]

            points.extend(octant_points)

            if d < 0:
                d = d + 4 * x + 6
            else:
                d = d + 4 * (x - y) + 10
                y -= 1
            x += 1

        # Elimină duplicatele
        return list(set(points))

    def _calculate_metrics(self, radii: List[int], intersections: List[int]) -> Dict:
        """Calculează metricii analizei Sholl."""
        results = {}

        # Peak și radius la peak
        if len(intersections) > 0:
            peak_idx = np.argmax(intersections)
            results['peak_number'] = intersections[peak_idx]
            results['radius_at_peak'] = radii[peak_idx]
        else:
            results['peak_number'] = 0
            results['radius_at_peak'] = 0

        # Area Under Curve (AUC)
        if len(radii) > 1:
            results['auc'] = np.trapz(intersections, radii)
        else:
            results['auc'] = 0

        # Regression coefficient (slope)
        results['slope'] = self._calculate_regression_slope(radii, intersections)

        # Metrici adiționale
        results['total_intersections'] = np.sum(intersections)
        results['max_radius'] = max(radii) if radii else 0
        results['mean_intersections'] = np.mean(intersections) if intersections else 0

        return results

    def _calculate_regression_slope(self, radii: List[int], intersections: List[int]) -> float:
        """Calculează coeficientul de regresie liniară."""
        if len(radii) < 2:
            return 0.0

        try:
            # Regresie liniară simplă
            x = np.array(radii)
            y = np.array(intersections)

            # Eliminăm valorile zero pentru log-linear regression
            mask = y > 0
            if np.sum(mask) < 2:
                return 0.0

            x_filtered = x[mask]
            y_filtered = y[mask]

            # Log-linear regression pentru Sholl
            log_y = np.log(y_filtered)

            # Calculăm slope-ul
            n = len(x_filtered)
            sum_x = np.sum(x_filtered)
            sum_y = np.sum(log_y)
            sum_xy = np.sum(x_filtered * log_y)
            sum_x2 = np.sum(x_filtered ** 2)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

            return slope

        except Exception as e:
            print(f"⚠️ Eroare la calculul slope-ului: {e}")
            return 0.0

    def _create_plot(self, radii: List[int], intersections: List[int],
                     results: Dict, save_path: str):
        """Creează și salvează graficul analizei Sholl."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Graficul principal
        ax1.plot(radii, intersections, 'b-o', linewidth=2, markersize=4)
        ax1.axhline(y=results['peak_number'], color='r', linestyle='--', alpha=0.7)
        ax1.axvline(x=results['radius_at_peak'], color='r', linestyle='--', alpha=0.7)
        ax1.scatter([results['radius_at_peak']], [results['peak_number']],
                    color='red', s=100, zorder=5)

        ax1.set_xlabel('Raza (pixeli)')
        ax1.set_ylabel('Număr Intersecții')
        ax1.set_title('Analiza Sholl - Profil de Intersecții')
        ax1.grid(True, alpha=0.3)

        # Adaugă text cu metrici
        textstr = f'Peak: {results["peak_number"]}\n'
        textstr += f'Raza Peak: {results["radius_at_peak"]}\n'
        textstr += f'AUC: {results["auc"]:.1f}\n'
        textstr += f'Slope: {results["slope"]:.4f}'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # Graficul log pentru vizualizarea slope-ului
        if len(radii) > 0 and max(intersections) > 0:
            mask = np.array(intersections) > 0
            if np.sum(mask) > 1:
                x_log = np.array(radii)[mask]
                y_log = np.array(intersections)[mask]

                ax2.semilogy(x_log, y_log, 'g-o', linewidth=2, markersize=4)
                ax2.set_xlabel('Raza (pixeli)')
                ax2.set_ylabel('Log(Intersecții)')
                ax2.set_title('Analiza Sholl - Scala Logaritmică')
                ax2.grid(True, alpha=0.3)

                # Adaugă linia de regresie
                if len(x_log) > 1:
                    try:
                        z = np.polyfit(x_log, np.log(y_log), 1)
                        p = np.poly1d(z)
                        ax2.plot(x_log, np.exp(p(x_log)), "r--", alpha=0.8,
                                 label=f'Slope: {z[0]:.4f}')
                        ax2.legend()
                    except:
                        pass

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Grafic salvat: {save_path}")

    def _empty_results(self) -> Dict:
        """Returnează rezultate goale."""
        return {
            'peak_number': 0,
            'radius_at_peak': 0,
            'auc': 0,
            'slope': 0,
            'total_intersections': 0,
            'max_radius': 0,
            'mean_intersections': 0
        }


# Funcție de compatibilitate cu codul existent
def sholl_analysis(image_path: str = None, binary_image: np.ndarray = None,
                   soma_coords: Tuple[int, int] = None, step_size: int = 5,
                   max_radius: int = 250, save_path: str = None) -> Dict:
    """
    Funcție wrapper pentru compatibilitate cu codul existent.
    """
    if binary_image is None and image_path:
        from skimage.io import imread
        binary_image = imread(image_path)
        if len(binary_image.shape) > 2:
            binary_image = binary_image[:, :, 0]  # Ia primul canal
        binary_image = binary_image > 0  # Convertește la boolean

    if binary_image is None or soma_coords is None:
        print("⚠️ Parametri insuficienți pentru analiza Sholl")
        return {'peak_number': 0, 'radius_at_peak': 0, 'auc': 0, 'slope': 0}

    analyzer = ShollAnalyzer()
    return analyzer.analyze(
        binary_image=binary_image,
        soma_center=soma_coords,
        step_size=step_size,
        max_radius=max_radius,
        save_path=save_path
    )


if __name__ == "__main__":
    # Test cu o imagine simulată
    print("🧪 Test analiza Sholl...")

    # Creează o imagine de test
    test_image = np.zeros((200, 200), dtype=bool)

    # Adaugă niște linii radiale simulate (dendrite)
    center = (100, 100)
    for angle in np.linspace(0, 2 * np.pi, 8):
        x_end = int(center[0] + 80 * np.cos(angle))
        y_end = int(center[1] + 80 * np.sin(angle))

        # Desenează linia
        from skimage.draw import line

        rr, cc = line(center[1], center[0], y_end, x_end)
        test_image[rr, cc] = True

    # Rulează analiza
    analyzer = ShollAnalyzer()
    results = analyzer.analyze(
        binary_image=test_image,
        soma_center=center,
        step_size=5,
        max_radius=100,
        save_path="test_sholl.png"
    )

    print("📊 Rezultate test:")
    for key, value in results.items():
        print(f"   • {key}: {value}")  # CORECTAT: Am eliminat paranteza în plus