"""
Setup script to copy ABI file to the correct location
"""

import shutil
from pathlib import Path

def setup_abi():
    """Copy ABI file from artifacts to python directory"""
    
    # Paths
    project_root = Path(__file__).parent.parent
    abi_source = project_root / "blockchain" / "artifacts" / "contracts" / "MediChainFL.sol" / "abi.json"
    abi_dest = project_root / "blockchain" / "python" / "abi.json"
    
    print("🔧 Setting up ABI file...")
    print(f"   Source: {abi_source}")
    print(f"   Destination: {abi_dest}")
    
    if not abi_source.exists():
        print("\n❌ Error: ABI file not found!")
        print("   Please compile the contract first:")
        print("   cd blockchain && npx hardhat compile")
        return False
    
    # Create destination directory if needed
    abi_dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy file
    shutil.copy(abi_source, abi_dest)
    
    print("\n✅ ABI file copied successfully!")
    return True


if __name__ == "__main__":
    setup_abi()
