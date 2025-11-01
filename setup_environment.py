"""
BrainGraphNet - Environment Setup Script
Automated installation and verification of dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    else:
        print("✅ Python version is compatible")

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directory Structure")
    
    dirs = [
        "data/raw/synthetic",
        "data/raw/HCP",
        "data/raw/custom",
        "data/processed",
        "data/splits",
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/logs/tensorboard",
        "outputs/predictions",
        "outputs/figures"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    print("This may take 5-10 minutes...")
    print("Installing core packages...")
    
    try:
        # Install PyTorch (CPU version)
        print("\n[1/3] Installing PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch==2.0.1", "--index-url", "https://download.pytorch.org/whl/cpu",
            "--quiet"
        ])
        print("✅ PyTorch installed")
        
        # Install PyTorch Geometric dependencies
        print("\n[2/3] Installing PyTorch Geometric dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch-scatter", "torch-sparse", "torch-geometric",
            "-f", "https://data.pyg.org/whl/torch-2.0.1+cpu.html",
            "--quiet"
        ])
        print("✅ PyTorch Geometric installed")
        
        # Install remaining requirements
        print("\n[3/3] Installing remaining packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
            "--quiet"
        ])
        print("✅ All packages installed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR during installation: {e}")
        print("\nTry manual installation:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

def verify_installation():
    """Verify that all key packages are importable"""
    print_header("Verifying Installation")
    
    packages = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("nilearn", "Nilearn"),
        ("networkx", "NetworkX"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML")
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - FAILED")
            all_ok = False
    
    if not all_ok:
        print("\n❌ Some packages failed to import")
        print("Try reinstalling: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✅ All packages verified successfully!")

def check_dataset():
    """Check if datasets are available"""
    print_header("Checking Datasets")
    
    synthetic_exists = Path("data/raw/synthetic").exists()
    hcp_exists = Path("data/raw/HCP").exists() and any(Path("data/raw/HCP").iterdir())
    
    if synthetic_exists:
        print("✅ Synthetic data directory ready")
    
    if hcp_exists:
        print("✅ HCP data detected")
    else:
        print("ℹ️  HCP data not found (optional)")
        print("   Download from: https://zenodo.org/records/6770120")
        print("   Extract to: data/raw/HCP/")
    
    print("\n💡 Synthetic data will be generated automatically if no dataset is found")

def display_next_steps():
    """Display next steps for the user"""
    print_header("Setup Complete! 🎉")
    
    print("""
Next Steps:

1. Quick Test (5 minutes):
   python train.py --config config.yaml --epochs 10

2. Full Training (15-20 minutes):
   python train.py --config config.yaml

3. Explore Notebooks:
   jupyter notebook notebooks/01_data_exploration.ipynb

4. View Configuration:
   cat config.yaml

5. Get Help:
   python train.py --help

Documentation: See README.md for detailed usage

Happy brain network modeling! 🧠
""")

def main():
    """Main setup function"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║                                                        ║
    ║           🧠  BrainGraphNet Setup  🧠                  ║
    ║                                                        ║
    ║     Dynamic GNNs for Brain Connectivity Evolution     ║
    ║                                                        ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    try:
        check_python_version()
        create_directories()
        install_dependencies()
        verify_installation()
        check_dataset()
        display_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
