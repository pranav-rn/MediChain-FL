from web3 import Web3
import json
from pathlib import Path
from typing import Optional

RPC = "http://127.0.0.1:8545"
PROXY_ADDRESS = "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"

# Hardhat default accounts - each hospital gets one
HOSPITAL_ACCOUNTS = {
    "hospital_A": {
        "address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        "private_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    },
    "hospital_B": {
        "address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
        "private_key": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
    },
    "hospital_C": {
        "address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
        "private_key": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
    },
    "hospital_D": {
        "address": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
        "private_key": "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"
    },
    "server": {
        "address": "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
        "private_key": "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a"
    }
}

# Default account for backward compatibility
ACCOUNT = HOSPITAL_ACCOUNTS["hospital_A"]["address"]
PRIVATE_KEY = HOSPITAL_ACCOUNTS["hospital_A"]["private_key"]

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

def log_gradient_update(hash_value, flagged=False, hospital_id: Optional[str] = None):
    """Log an update to the blockchain with flagged status from specific hospital"""
    # Use hospital-specific account if provided, otherwise use default
    if hospital_id and hospital_id in HOSPITAL_ACCOUNTS:
        account = HOSPITAL_ACCOUNTS[hospital_id]["address"]
        private_key = HOSPITAL_ACCOUNTS[hospital_id]["private_key"]
    else:
        account = ACCOUNT
        private_key = PRIVATE_KEY
    
    nonce = w3.eth.get_transaction_count(account)
    tx = contract.functions.logUpdate(hash_value, flagged).build_transaction({
        "from": account,
        "nonce": nonce,
        "gas": 500000,
        "gasPrice": w3.to_wei("1", "gwei"),
    })

    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt

def get_reputation(hospital_address):
    """Get reputation scores for a hospital"""
    successful, flagged, score = contract.functions.getReputation(hospital_address).call()
    return {
        "successful": successful,
        "flagged": flagged,
        "score": score
    }

def get_token_balance(hospital_address):
    """Get MCT token balance for a hospital"""
    balance = contract.functions.getTokenBalance(hospital_address).call()
    # Convert from wei to tokens (18 decimals)
    return balance / (10 ** 18)

def get_all_logs():
    """Get all update logs from the blockchain"""
    count = contract.functions.getLogsCount().call()
    logs = []
    for i in range(count):
        log = contract.functions.logs(i).call()
        logs.append({
            "index": i,
            "hospital": log[0],
            "gradientHash": log[1],
            "timestamp": log[2],
            "flagged": log[3]
        })
    return logs
