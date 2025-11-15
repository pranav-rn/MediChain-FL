from web3 import Web3
import json
from pathlib import Path

RPC = "http://127.0.0.1:8545"
PROXY_ADDRESS = "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"
ACCOUNT = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

w3 = Web3(Web3.HTTPProvider(RPC))

# Load ABI from file relative to this module
BASE_DIR = Path(__file__).parent
ABI_FILE = BASE_DIR / "abi.json"
ARTIFACT_JSON = BASE_DIR.parent / "artifacts" / "contracts" / "MediChainFL.sol" / "MediChainFL.json"

if ABI_FILE.exists():
    abi_data = json.loads(ABI_FILE.read_text())
elif ARTIFACT_JSON.exists():
    abi_data = json.loads(ARTIFACT_JSON.read_text())
else:
    raise FileNotFoundError(f"ABI not found. Checked: {ABI_FILE} and {ARTIFACT_JSON}")

ABI = abi_data["abi"] if isinstance(abi_data, dict) else abi_data

contract = w3.eth.contract(address=PROXY_ADDRESS, abi=ABI)

def log_gradient_update(hash_value):
    nonce = w3.eth.get_transaction_count(ACCOUNT)
    tx = contract.functions.logUpdate(hash_value).build_transaction({
        "from": ACCOUNT,
        "nonce": nonce,
        "gas": 500000,
        "gasPrice": w3.to_wei("1", "gwei"),
    })

    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt
