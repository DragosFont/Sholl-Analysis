#!/usr/bin/env python3
"""
ShollCSVLogger STABIL - înlocuiește src/io/sholl_exported_values.py

ACEASTĂ VERSIUNE:
✅ NU modifică niciodată CSV-ul existent
✅ Peak ÎNTOTDEAUNA în poziția 6, Radius în poziția 7  
✅ Fără funcții de "reparare" care strică datele
✅ Fără import-uri conflictuale
✅ Scriere simplă și sigură
"""

import os
import csv
import pandas as pd
from datetime import datetime
from typing import Optional


class ShollCSVLogger:
    """CSV Logger STABIL - nu modifică niciodată structura existentă."""

    def __init__(self, output_path: str = "outputs"):
        self.output_path = output_path
        self.csv_file = os.path.join(output_path, "sholl_results.csv")

        # HEADER DEFINITIV - niciodată să nu se schimbe!
        self.headers = [
            'timestamp', 'image_name', 'roi_index', 'roi_type',
            'roi_area_pixels', 'roi_perimeter_pixels',
            'peak_number',        # poziția 6 ⭐
            'radius_at_peak',     # poziția 7 ⭐  
            'auc', 'regression_coef', 'total_intersections',
            'max_radius', 'mean_intersections', 'roi_folder'
        ]

        os.makedirs(output_path, exist_ok=True)
        self._ensure_csv_exists_simple()

    def _ensure_csv_exists_simple(self):
        """Creează CSV DOAR dacă nu există - NU modifică pe cel existent."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_result(self, image_name: str, roi_index: int, 
                   peak: int = 0, radius: int = 0,
                   peak_number: int = None, radius_at_peak: int = None,
                   **kwargs) -> bool:
        """Adaugă rând în CSV cu poziții fixe pentru peak(6) și radius(7)."""

        final_peak = peak_number if peak_number is not None else peak
        final_radius = radius_at_peak if radius_at_peak is not None else radius

        row = [
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            str(image_name), int(roi_index), 
            kwargs.get('roi_type', 'processed'),
            float(kwargs.get('roi_area_pixels', 0)),
            float(kwargs.get('roi_perimeter_pixels', 0)),
            int(final_peak),      # poziția 6 ⭐
            int(final_radius),    # poziția 7 ⭐
            float(kwargs.get('auc', 0)),
            float(kwargs.get('regression_coef', 0)),
            int(kwargs.get('total_intersections', 0)),
            int(kwargs.get('max_radius', 0)),
            float(kwargs.get('mean_intersections', 0)),
            kwargs.get('roi_folder', '')
        ]

        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            return True
        except Exception as e:
            print(f"❌ Eroare salvare: {e}")
            return False

    def print_summary(self):
        """Sumar fără modificări."""
        if not os.path.exists(self.csv_file):
            print("📊 Nu există CSV.")
            return

        try:
            df = pd.read_csv(self.csv_file)
            print(f"📊 Total înregistrări: {len(df)}")
            if len(df) > 0:
                successful = len(df[df.iloc[:, 6] > 0])  # peak în poziția 6
                print(f"✅ Analize reușite: {successful}/{len(df)}")
        except Exception as e:
            print(f"❌ Eroare citire: {e}")
