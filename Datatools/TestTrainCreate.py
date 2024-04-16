import os
import shutil
import random

def split_data(source_dir, train_dir, test_dir, test_size=0.2):
    try:
        # Sikre, at destinationsmapperne eksisterer
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Hent alle filnavne fra data-mappen
        all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

        # Bland listen og lav et tilfældigt split
        random.shuffle(all_files)
        split_point = int((1 - test_size) * len(all_files))

        # Fordel filerne i train og test mapper
        train_files = all_files[:split_point]
        test_files = all_files[split_point:]

        # Funktion til at kopiere filer til den angivne mappe
        def copy_files(files, destination):
            for f in files:
                shutil.copy(os.path.join(source_dir, f), os.path.join(destination, f))

        # Kopier filerne
        copy_files(train_files, train_dir)
        copy_files(test_files, test_dir)

        print(f'{len(train_files)} filer blev kopieret til træningsmappen.')
        print(f'{len(test_files)} filer blev kopieret til testmappen.')
    except FileNotFoundError as fnf_error:
        print(f"Fejl: {fnf_error}")
    except Exception as e:
        print(f"En uventet fejl opstod: {e}")

# Stier til mapperne
data_dir = 'Data/KD hele plader'
train_dir = 'Data/KD train plader'
test_dir = 'Data/KD test plader'

# Kald funktionen
split_data(data_dir, train_dir, test_dir)
