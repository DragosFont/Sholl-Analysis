#!/usr/bin/env python3
"""
ShollCSVLogger DEFINITIV - înlocuiește src/io/sholl_exported_values.py

ACEASTĂ VERSIUNE:
✅ NU modifică niciodată CSV-ul existent
✅ NU face "reparări" automate care strică datele
✅ NU schimbă ordinea coloanelor
✅ NU creează backup-uri automate
✅ Păstrează poziția 6 pentru peak, poziția 7 pentru radius
✅ Scriere simplă și sigură, fără conflicte

ÎNLOCUIEȘTE COMPLET fișierul existent cu această versiune!
"""

import os
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any


class ShollCSVLogger:
    """
    CSV Logger FINAL STABIL pentru rezultatele analizei Sholl.

    PRINCIPII FERME:
    - NU modifică NICIODATĂ structura CSV existentă
    - NU "repară" sau "detectează" automat probleme
    - NU creează backup-uri care pot strica datele
    - DOAR adaugă rânduri noi în ordinea stabilită
    - Peak ÎNTOTDEAUNA în poziția 6, Radius în poziția 7
    """

    def __init__(self, output_path: str = "outputs"):
        self.output_path = output_path
        self.csv_file = os.path.join(output_path, "sholl_results.csv")

        # HEADER DEFINITIV - NICIODATĂ să nu fie schimbat!
        self.headers = [
            'timestamp',  # 0
            'image_name',  # 1
            'roi_index',  # 2
            'roi_type',  # 3
            'roi_area_pixels',  # 4
            'roi_perimeter_pixels',  # 5
            'peak_number',  # 6  ⭐ PEAK - poziția garantată
            'radius_at_peak',  # 7  ⭐ RADIUS - poziția garantată
            'auc',  # 8
            'regression_coef',  # 9
            'total_intersections',  # 10
            'max_radius',  # 11
            'mean_intersections',  # 12
            'roi_folder'  # 13
        ]

        # Asigură că directorul există
        os.makedirs(output_path, exist_ok=True)

        # Inițializează CSV DOAR dacă nu există
        self._ensure_csv_exists_simple()

    def _ensure_csv_exists_simple(self):
        """
        Creează CSV DOAR dacă nu există deloc.
        NU modifică, NU repară, NU schimbă nimic din cel existent!
        """
        if not os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
                print(f"✅ CSV nou creat: {self.csv_file}")
            except Exception as e:
                print(f"❌ Nu s-a putut crea CSV: {e}")
                raise
        else:
            # CSV există - nu îl atinge!
            try:
                df = pd.read_csv(self.csv_file, nrows=1)  # citește doar prima linie pentru test
                print(f"✅ CSV existent găsit - nu se modifică")
            except Exception as e:
                print(f"⚠️ CSV existent nu poate fi citit: {e}")

    def log_result(self, image_name: str, roi_index: int,
                   peak: int = 0, radius: int = 0,
                   peak_number: int = None, radius_at_peak: int = None,
                   auc: float = 0.0, regression_coef: float = 0.0,
                   roi_folder: str = "", **kwargs) -> bool:
        """
        Adaugă UN rând în CSV cu valorile în ordinea EXACTĂ.

        Acceptă atât peak/radius cât și peak_number/radius_at_peak pentru compatibilitate.

        Args:
            image_name: Numele imaginii
            roi_index: Indexul ROI-ului
            peak/peak_number: Numărul peak-ului (va fi în poziția 6)
            radius/radius_at_peak: Raza la peak (va fi în poziția 7)
            auc: Area Under Curve
            regression_coef: Coeficientul de regresie
            roi_folder: Folder-ul ROI-ului
            **kwargs: Alte valori opționale

        Returns:
            bool: True dacă salvarea a reușit
        """

        # Rezolvă valorile pentru peak și radius (compatibilitate)
        final_peak = peak_number if peak_number is not None else peak
        final_radius = radius_at_peak if radius_at_peak is not None else radius

        # Construiește rândul în ordinea EXACTĂ din self.headers
        row = [
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),  # 0: timestamp
            str(image_name).replace(',', '_'),  # 1: image_name
            int(roi_index),  # 2: roi_index
            kwargs.get('roi_type', 'processed'),  # 3: roi_type
            float(kwargs.get('roi_area_pixels', 0)),  # 4: roi_area_pixels
            float(kwargs.get('roi_perimeter_pixels', 0)),  # 5: roi_perimeter_pixels
            int(final_peak),  # 6: peak_number ⭐
            int(final_radius),  # 7: radius_at_peak ⭐
            float(auc),  # 8: auc
            float(regression_coef),  # 9: regression_coef
            int(kwargs.get('total_intersections', 0)),  # 10: total_intersections
            int(kwargs.get('max_radius', 0)),  # 11: max_radius
            float(kwargs.get('mean_intersections', 0.0)),  # 12: mean_intersections
            str(roi_folder)  # 13: roi_folder
        ]

        # Verificare de siguranță
        if len(row) != len(self.headers):
            print(f"❌ EROARE: {len(row)} valori vs {len(self.headers)} coloane!")
            return False

        # Scrie în CSV - operația cea mai simplă posibilă
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"✅ Salvat ROI {roi_index}: Peak={final_peak}(col.6), Radius={final_radius}(col.7), AUC={auc:.1f}")
            return True

        except Exception as e:
            print(f"❌ Eroare la salvarea în CSV: {e}")
            return False

    def print_summary(self):
        """
        Afișează un sumar al rezultatelor fără să modifice nimic.
        """
        if not os.path.exists(self.csv_file):
            print("📊 Nu există fișier CSV pentru sumar.")
            return

        try:
            df = pd.read_csv(self.csv_file)
        except Exception as e:
            print(f"📊 Nu s-a putut citi CSV pentru sumar: {e}")
            return

        if len(df) == 0:
            print("📊 CSV-ul este gol.")
            return

        print(f"\n📊 SUMAR REZULTATE SHOLL:")
        print(f"   • Total înregistrări: {len(df)}")
        print(f"   • Fișier: {self.csv_file}")

        # Găsește coloanele pentru peak și radius
        header_list = list(df.columns)
        peak_col_idx = None
        radius_col_idx = None

        for i, col_name in enumerate(header_list):
            if 'peak' in col_name.lower():
                peak_col_idx = i
            if 'radius' in col_name.lower():
                radius_col_idx = i

        if peak_col_idx is not None and radius_col_idx is not None:
            print(f"   • Peak în coloana {peak_col_idx}: {header_list[peak_col_idx]}")
            print(f"   • Radius în coloana {radius_col_idx}: {header_list[radius_col_idx]}")

            # Statistici simple
            successful_analyses = len(df[df.iloc[:, peak_col_idx] > 0])
            print(f"   • Analize reușite: {successful_analyses}/{len(df)}")

            if successful_analyses > 0:
                success_df = df[df.iloc[:, peak_col_idx] > 0]
                avg_peak = success_df.iloc[:, peak_col_idx].mean()
                avg_radius = success_df.iloc[:, radius_col_idx].mean()
                print(f"   • Peak mediu: {avg_peak:.1f}")
                print(f"   • Raza medie: {avg_radius:.1f}")
        else:
            print("   ⚠️ Nu s-au găsit coloanele pentru peak/radius")

    def get_results_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Returnează DataFrame cu rezultatele, fără modificări.

        Returns:
            pd.DataFrame sau None dacă nu se poate citi
        """
        if not os.path.exists(self.csv_file):
            return None

        try:
            return pd.read_csv(self.csv_file)
        except Exception as e:
            print(f"❌ Nu s-a putut citi CSV: {e}")
            return None


# Pentru compatibilitate cu codul existent care importă diferit
class ShollResultsLogger(ShollCSVLogger):
    """Alias pentru compatibilitate."""
    pass


def create_logger(output_path: str = "outputs") -> ShollCSVLogger:
    """Factory function pentru crearea logger-ului."""
    return ShollCSVLogger(output_path)


# Funcție de test pentru verificarea că totul funcționează
def test_logger():
    """Test rapid pentru verificarea funcționării."""
    print("🧪 Test ShollCSVLogger...")

    logger = ShollCSVLogger("test_output")

    # Test salvare
    success = logger.log_result(
        image_name="test.czi",
        roi_index=1,
        peak=25,
        radius=150,
        auc=1250.5,
        regression_coef=0.0123,
        roi_folder="roi_1"
    )

    if success:
        print("✅ Test reușit!")
        logger.print_summary()

        # Cleanup test
        import shutil
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
            print("🧹 Cleanup test terminat")
    else:
        print("❌ Test eșuat!")


if __name__ == "__main__":
    print("📋 ShollCSVLogger DEFINITIV - fără modificări automate")
    print("🎯 Peak în poziția 6, Radius în poziția 7 - GARANTAT")
    print("🔒 NU modifică CSV-ul existent - DOAR adaugă date noi")

    # Rulează test
    test_logger()