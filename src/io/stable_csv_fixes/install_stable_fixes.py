#!/usr/bin/env python3
"""
Script de instalare automată pentru fix-urile stabile CSV
"""

import os
import shutil
from datetime import datetime

def install_stable_fixes():
    """Instalează fix-urile stabile în proiect."""

    print("🔧 INSTALARE FIX-URI STABILE CSV")
    print("=" * 50)

    # Căutați fișierul original
    original_file = None
    possible_locations = [
        "src/io/sholl_exported_values.py",
        "io/sholl_exported_values.py", 
        "sholl_exported_values.py"
    ]

    for location in possible_locations:
        if os.path.exists(location):
            original_file = location
            break

    if original_file:
        # Creează backup
        backup_path = original_file + f".backup_stable_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(original_file, backup_path)
        print(f"📋 Backup original: {backup_path}")

        # Înlocuiește cu versiunea stabilă
        stable_source = "stable_csv_fixes/sholl_exported_values_stable.py"
        if os.path.exists(stable_source):
            shutil.copy2(stable_source, original_file)
            print(f"✅ Înlocuit: {original_file}")
        else:
            print(f"❌ Nu s-a găsit sursa stabilă: {stable_source}")
    else:
        print("⚠️ Nu s-a găsit fișierul original de înlocuit")

    print("🎉 Instalare completă!")

if __name__ == "__main__":
    install_stable_fixes()
