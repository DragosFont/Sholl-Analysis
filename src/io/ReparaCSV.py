#!/usr/bin/env python3
"""
Script pentru repararea CSV-ului Sholl - versiune îmbunătățită
Păstrează toate datele valide și elimină doar liniile problematice
"""

import os
import csv
import shutil
from datetime import datetime


def repair_sholl_csv_fixed(csv_path="exact_freehand_sholl_results.csv"):
    """
    Repară fișierul CSV păstrând toate datele valide.

    Args:
        csv_path: Calea către fișierul CSV de reparat
    """

    if not os.path.exists(csv_path):
        print(f"❌ Fișierul {csv_path} nu există!")
        print("📁 Fișiere disponibile în directorul curent:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"   - {f}")
        return

    print(f"🔧 Încep repararea fișierului: {csv_path}")

    # Creează backup
    backup_file = csv_path.replace('.csv', '_backup_before_repair.csv')
    shutil.copy2(csv_path, backup_file)
    print(f"💾 Backup creat: {backup_file}")

    # Header-ul corect
    correct_headers = [
        'image_name',
        'roi_index',
        'roi_type',
        'roi_area_pixels',
        'roi_perimeter_pixels',
        'peak_number',
        'radius_at_peak',
        'auc',
        'regression_coef',
        'total_intersections',
        'max_radius',
        'mean_intersections',
        'roi_folder',
        'timestamp'
    ]

    # Citește toate liniile
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"📄 Citite {len(lines)} linii din fișier")

    # Procesează liniile
    valid_data = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        parts = line.split(',')

        # Verifică dacă este o linie validă cu date complete
        if len(parts) >= 13:
            # Verifică dacă prima coloană conține numele imaginii (.czi)
            if '.czi' in str(parts[0]):
                # Extrage primele 14 coloane (inclusiv timestamp dacă există)
                data_row = []
                for j in range(min(14, len(parts))):
                    data_row.append(parts[j].strip().replace('"', ''))

                # Dacă nu avem toate cele 14 coloane, completează cu valori goale
                while len(data_row) < 14:
                    data_row.append('')

                # Verifică dacă este o linie cu date complete (nu doar header duplicat)
                if (data_row[1] and data_row[1] != 'roi_index' and
                        data_row[2] and data_row[2] != 'roi_type'):
                    valid_data.append(data_row)
                    print(f"✅ Procesată linia {i + 1}: {data_row[0]} ROI {data_row[1]}")

            # Verifică liniile care încep cu timestamp și au datele deplasate
            elif ('2025-07-28' in str(parts[0]) and len(parts) >= 15 and
                  '.czi' in str(parts[1])):
                # Liniile care încep cu timestamp, apoi image_name
                data_row = [
                    parts[1].strip().replace('"', ''),  # image_name
                    parts[2].strip().replace('"', ''),  # roi_index
                    parts[3].strip().replace('"', ''),  # roi_type
                    parts[4].strip().replace('"', ''),  # roi_area_pixels
                    parts[5].strip().replace('"', ''),  # roi_perimeter_pixels
                    parts[6].strip().replace('"', ''),  # peak_number
                    parts[7].strip().replace('"', ''),  # radius_at_peak
                    parts[8].strip().replace('"', ''),  # auc
                    parts[9].strip().replace('"', ''),  # regression_coef
                    parts[10].strip().replace('"', ''),  # total_intersections
                    parts[11].strip().replace('"', ''),  # max_radius
                    parts[12].strip().replace('"', ''),  # mean_intersections
                    parts[13].strip().replace('"', ''),  # roi_folder
                    parts[0].strip().replace('"', '')  # timestamp
                ]

                # Verifică dacă sunt date valide
                if (data_row[1] and data_row[1] != 'roi_index' and
                        data_row[2] and data_row[2] != 'roi_type'):
                    valid_data.append(data_row)
                    print(f"✅ Procesată linia cu timestamp {i + 1}: {data_row[0]} ROI {data_row[1]}")

        # Oprește procesarea când întâlnești linii cu prea multe coloane goale
        # (acestea sunt probabil liniile problematice de la sfârșit)
        empty_count = sum(1 for part in parts if not part.strip())
        if empty_count > len(parts) // 2:  # Dacă mai mult de jumătate sunt goale
            print(f"⚠️ Opresc procesarea la linia {i + 1} (prea multe coloane goale)")
            break

    if not valid_data:
        print("❌ Nu s-au găsit date valide pentru procesare!")
        return

    print(f"📊 S-au găsit {len(valid_data)} înregistrări valide")

    # Scrie fișierul reparat
    repaired_file = csv_path.replace('.csv', '_repaired.csv')

    with open(repaired_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Scrie header-ul
        writer.writerow(correct_headers)
        # Scrie datele
        for data_row in valid_data:
            writer.writerow(data_row)

    print(f"\n🎉 SUCCESS!")
    print(f"✅ CSV reparat salvat ca: {repaired_file}")
    print(f"📊 Procesate {len(valid_data)} înregistrări")
    print(f"💾 Backup original: {backup_file}")

    # Înlocuiește fișierul original cu cel reparat
    replace_original = input(f"\n❓ Vrei să înlocuiești fișierul original cu cel reparat? (y/N): ")
    if replace_original.lower() in ['y', 'yes', 'da']:
        shutil.copy2(repaired_file, csv_path)
        print(f"✅ Fișierul original a fost înlocuit cu cel reparat!")

    # Afișează preview
    print(f"\n📋 PREVIEW - Primele 5 înregistrări:")
    print("-" * 80)
    for i, data in enumerate(valid_data[:5]):
        print(f"{i + 1}. {data[0]} ROI{data[1]}: Peak={data[5]}, AUC={data[7]}")

    if len(valid_data) > 5:
        print(f"... și încă {len(valid_data) - 5} înregistrări")

    return repaired_file


def main():
    print("🔧 SHOLL CSV REPAIR TOOL - VERSIUNE ÎMBUNĂTĂȚITĂ")
    print("=" * 60)

    # Caută fișiere CSV în mai multe directoare
    search_paths = [
        '.',  # directorul curent
        '..',  # directorul părinte
        '../..',  # cu două nivele mai sus
        os.path.join('..', '..', 'outputs'),  # în outputs
        'outputs'  # în outputs local
    ]

    all_csv_files = []

    for path in search_paths:
        if os.path.exists(path):
            try:
                files_in_path = [f for f in os.listdir(path) if f.endswith('.csv')]
                for f in files_in_path:
                    full_path = os.path.join(path, f)
                    all_csv_files.append((f, full_path, path))
            except PermissionError:
                continue

    if not all_csv_files:
        print("❌ Nu s-au găsit fișiere CSV în directoarele căutate!")
        print("🔍 Directoare căutate:")
        for path in search_paths:
            abs_path = os.path.abspath(path)
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"   {exists} {abs_path}")
        return

    print("📁 Fișiere CSV găsite:")
    for i, (filename, full_path, rel_path) in enumerate(all_csv_files, 1):
        print(f"   {i}. {filename} (în {rel_path})")

    # Caută automat fișierul target
    target_file = None
    target_path = None

    for filename, full_path, rel_path in all_csv_files:
        if 'exact_freehand' in filename.lower() or 'sholl' in filename.lower():
            target_file = filename
            target_path = full_path
            print(f"\n🎯 Fișier detectat automat: {filename} (în {rel_path})")
            break

    if target_file:
        proceed = input("Continui cu acest fișier? (Y/n): ")
        if proceed.lower() not in ['n', 'no', 'nu']:
            repair_sholl_csv_fixed(target_path)
        else:
            # Lasă utilizatorul să aleagă
            try:
                choice = int(input("Alege numărul fișierului de reparat: ")) - 1
                if 0 <= choice < len(all_csv_files):
                    selected_file = all_csv_files[choice][1]  # full_path
                    repair_sholl_csv_fixed(selected_file)
                else:
                    print("❌ Alegere invalidă!")
            except ValueError:
                print("❌ Te rog introdu un număr valid!")
    else:
        # Lasă utilizatorul să aleagă
        try:
            choice = int(input("Alege numărul fișierului de reparat: ")) - 1
            if 0 <= choice < len(all_csv_files):
                selected_file = all_csv_files[choice][1]  # full_path
                repair_sholl_csv_fixed(selected_file)
            else:
                print("❌ Alegere invalidă!")
        except ValueError:
            print("❌ Te rog introdu un număr valid!")


if __name__ == "__main__":
    main()