from web3 import Web3
import json
import time

# Connect to Hardhat
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
account = w3.eth.accounts[0]

# Load ABI
with open("../artifacts/contracts/ModelUpdateLogger.sol/ModelUpdateLogger.json") as f:
    artifact = json.load(f)

abi = artifact["abi"]
address = "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"  # Update if re-deployed
contract = w3.eth.contract(address=address, abi=abi)

def log_to_blockchain(node_id, model_hash):
    timestamp = int(time.time())

    tx = contract.functions.logModelUpdate(
        node_id,
        model_hash,
        timestamp
    ).transact({"from": account})

    receipt = w3.eth.wait_for_transaction_receipt(tx)
    print(f"[BLOCKCHAIN] Logged update from {node_id}: {model_hash}")
    return receipt
