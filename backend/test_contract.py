"""Quick test to verify contract is deployed correctly"""
import sys
from pathlib import Path

# Add blockchain python directory to path
blockchain_path = Path(__file__).parent.parent / "blockchain" / "python"
sys.path.insert(0, str(blockchain_path))

from blockchain_client import contract, w3

print(f"Connected: {w3.is_connected()}")
print(f"Contract address: {contract.address}")

# Try to get code at the address
code = w3.eth.get_code(contract.address)
print(f"Code length: {len(code)} bytes")

if len(code) <= 2:
    print("❌ No contract deployed at this address!")
else:
    print("✅ Contract is deployed")

# Try calling getLogsCount
try:
    count = contract.functions.getLogsCount().call()
    print(f"✅ getLogsCount() returned: {count}")
except Exception as e:
    print(f"❌ getLogsCount() failed: {e}")

# Try calling totalLogs
try:
    total = contract.functions.totalLogs().call()
    print(f"✅ totalLogs() returned: {total}")
except Exception as e:
    print(f"❌ totalLogs() failed: {e}")

# List all functions
print("\nAvailable functions:")
for func in contract.functions:
    print(f"  - {func}")
