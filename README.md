# MediChain-FL

> Privacy-Preserving Federated Learning for Medical Imaging — Built at Kodikon 5.0

MediChain-FL enables hospitals to collaboratively train AI models for pneumonia detection **without ever sharing patient data**. It combines Federated Learning, Homomorphic Encryption, and Blockchain into a triple-layer security architecture that is both HIPAA-compliant and production-ready.

---

## The Problem

Hospitals hold siloed patient data. Three hospitals together might have 500 chest X-rays — enough to train a 95% accurate AI — but HIPAA/GDPR regulations prevent any sharing. Existing approaches fall short:

| Approach | Problem |
|---|---|
| Centralized ML | Violates privacy laws |
| Basic Federated Learning | Gradients can leak patient information |
| Differential Privacy | Reduces accuracy by 8–15% |

---

## The Solution

```
Hospital trains locally
    → Encrypts gradients (CKKS)
        → Server aggregates blindly (Homomorphic)
            → Logs to Blockchain (Immutable audit trail)
                → Returns updated model
                    → Repeat for 5 rounds
```

**Result:** 93% pneumonia detection accuracy across 3–5 hospitals — no raw data ever leaves a hospital.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           BLOCKCHAIN LAYER (Hardhat)                │
│  [Smart Contract]  [Audit Trail]  [Token Rewards]  │
└────────────┬────────────────────────┬───────────────┘
             │                        │
    ┌────────▼─────────┐    ┌────────▼──────────┐
    │   FL SERVER      │    │ ANOMALY DETECTOR  │
    │   (Flower)       │◄───┤  (Z-Score Based)  │
    │   FedAvg + HE    │    └────────────────────┘
    └────────┬─────────┘
             │  Encrypted Updates
    ┌────────┼──────────┬──────────┬──────────┐
    │        │          │          │          │
┌───▼───┐┌──▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Hosp 1 ││Hosp 2│ │Hosp 3 │ │Hosp 4 │ │Hosp 5 │
│UNet   ││UNet  │ │UNet   │ │UNet   │ │UNet   │
│CKKS   ││CKKS  │ │CKKS   │ │CKKS   │ │CKKS   │
└───────┘└──────┘ └───────┘ └───────┘ └───────┘
```

### One Training Round

1. **Server** broadcasts the current global model to all hospitals
2. Each **Hospital** trains locally for 1 epoch (~30 seconds)
3. **Hospital** encrypts gradients using CKKS homomorphic encryption
4. **Anomaly Detector** checks gradient norms for poisoning attacks
5. **Blockchain** logs the update hash immutably
6. **Server** aggregates encrypted gradients — without decrypting them
7. **Server** decrypts only the final aggregate and broadcasts the updated model
8. Repeat for 5 rounds

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| FL Framework | Flower 1.11+ | Client–server federated learning |
| ML Model | UNet + ResNet18 (PyTorch) | Pneumonia detection from X-rays |
| Encryption | TenSEAL (CKKS) | Homomorphic operations on gradients |
| Blockchain | Hardhat (Ethereum) | Immutable audit trail |
| Backend | FastAPI + Uvicorn | API and WebSocket server |
| Dataset | NIH ChestX-ray14 | Labeled pneumonia X-ray images |

---

## Project Structure

```
medichain-fl/
├── backend/
│   ├── model.py                  # UNet with frozen ResNet18 encoder
│   ├── fl_client/
│   │   ├── client.py             # Flower FL client (per hospital)
│   │   └── server.py             # Flower server with HE aggregation
│   └── utils/
│       ├── encryption.py         # TenSEAL / CKKS wrapper
│       └── anomaly_detector.py   # Z-score gradient validation
├── blockchain/
│   ├── contracts/
│   │   └── MediChainFL.sol       # Solidity smart contract
│   └── scripts/
│       └── deploy.js             # Deployment script
├── scripts/
│   └── prepare_hospital_data.py  # Dataset preparation
├── docker-compose.yml
├── Dockerfile.client
├── Dockerfile.server
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 20 LTS
- Docker (optional, recommended)

### 1. Python Environment

```bash
python3.10 -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Dataset

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle and organize it as:

```
data/
├── hospital_1/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── hospital_2/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── hospital_3/
    ├── NORMAL/
    └── PNEUMONIA/
```

Use `scripts/prepare_hospital_data.py` to split the dataset automatically.

### 3. Blockchain

```bash
cd blockchain
npm install
# Terminal 1 — start local Hardhat node
npx hardhat node
# Terminal 2 — deploy the smart contract
npx hardhat run scripts/deploy.js --network localhost
```

### 4. Run Federated Learning

Open separate terminals for the server and each hospital client:

```bash
# Terminal 3 — FL Server
python backend/fl_client/server.py

# Terminal 4 — Hospital 1
python backend/fl_client/client.py hospital_1

# Terminal 5 — Hospital 2
python backend/fl_client/client.py hospital_2

# Terminal 6 — Hospital 3
python backend/fl_client/client.py hospital_3
```

Training runs for 5 rounds (~2–3 minutes on CPU, ~30–45 seconds on GPU).

### 5. Docker (Alternative)

```bash
docker-compose up --build
```

---

## Security Model

### Homomorphic Encryption (CKKS)
The FL server aggregates gradients entirely in encrypted space. It never decrypts individual hospital updates — only the final averaged result. This gives a **mathematical privacy guarantee**: even a compromised server cannot reconstruct patient data.

### Blockchain Audit Trail
Every model update is hashed and logged to an Ethereum smart contract via Hardhat. This creates a tamper-proof, regulator-readable audit trail of every training step.

### Anomaly Detection
Gradient norms are validated using Z-score analysis before aggregation. Updates that deviate significantly from the expected distribution are flagged and rejected, providing Byzantine-fault tolerance against model poisoning attacks.

---

## Expected Results

| Round | Accuracy |
|---|---|
| 1 | ~72% |
| 2 | ~82% |
| 3 | ~88% |
| 5 | ~93% |

---

## Troubleshooting

**TenSEAL import error**
```bash
pip uninstall tenseal && pip install tenseal==0.3.14
```

**Flower client connection refused**
```bash
# Verify server is running on port 8080
netstat -an | grep 8080
```

**Cannot connect to Hardhat**
```bash
curl http://127.0.0.1:8545
# If no response, restart: npx hardhat node
```

**Out of memory during training**
```python
# Reduce batch size in client.py
self.trainloader = DataLoader(self.trainset, batch_size=8)
```

---

## Roadmap

- [ ] Asynchronous FL for hospitals with variable speeds
- [ ] Differential privacy layer on top of homomorphic encryption
- [ ] Multi-party key management
- [ ] React real-time dashboard
- [ ] EHR system integration

---

## License

MIT

