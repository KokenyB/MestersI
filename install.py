import sys
import subprocess

REQUIRED_PACKAGES = [
    "torch", "transformers", "diffusers", "requests", "Pillow", "torchaudio", "librosa"
]

def check_and_install_packages():
    """Ellenőrzi a fő csomagok telepítettségét, és javasolja a telepítést."""
    missing_packages = []
    
    print("--- 🛠️ Függőségek ellenőrzése ---")
    
    try:
        installed_packages_output = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format", "freeze"]).decode()
        installed_packages = {line.split('==')[0].lower() for line in installed_packages_output.split('\n') if line}
        
        for package in REQUIRED_PACKAGES:
            if package.lower() not in installed_packages:
                missing_packages.append(package)

    except subprocess.CalledProcessError:
        print("Hiba: Nem sikerült a pip csomagkezelő futtatása.")
        print("Kérjük, ellenőrizze a Python/pip telepítést.")
        return False
    
    if missing_packages:
        print(f"Hiányzó csomagok: {', '.join(missing_packages)}")
        print("\n*** Kérjük, futtassa a telepítést! ***")
        print("1. Hozza létre a 'requirements.txt' fájlt.")
        print(f"2. Futtassa a terminálban: pip install -r requirements.txt")
        print("\nA fő program ('main.py') csak ezután fog működni!")
        return False
    else:
        print("Minden szükséges Python csomag telepítve van.")
        print("Az install.py kész. Futtathatja a 'main.py' fájlt.")
        return True

if __name__ == "__main__":
    check_and_install_packages()