const express = require("express");
const { ethers } = require("ethers");
const app = express();

app.use(express.json());

// Connect to Hardhat local blockchain
const provider = new ethers.JsonRpcProvider("http://127.0.0.1:8545");

// Load ABI of the Logic Contract (MediChainFL)
const contractABI = require("./artifacts/contracts/MediChainFL.sol/MediChainFL.json").abi;

// 👉 Replace with your PROXY ADDRESS from deploy_proxy.js output
const proxyAddress = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512";

// Global contract variable
let contract;

// Initialize contract with signer
async function initContract() {
    const signer = await provider.getSigner(0);
    contract = new ethers.Contract(proxyAddress, contractABI, signer);
    console.log("Contract initialized with proxy:", proxyAddress);
}

// Initialize on startup
initContract().catch(console.error);

/*
====================================================
 POST: Add File Metadata
====================================================
*/
app.post("/addFile", async (req, res) => {
    try {
        const { fileHash, owner, timestamp } = req.body;

        const tx = await contract.addFileMetadata(fileHash, owner, timestamp);
        await tx.wait();

        res.json({
            status: "success",
            transactionHash: tx.hash
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/*
====================================================
 GET: Get File Metadata
====================================================
*/
app.get("/file/:fileHash", async (req, res) => {
    try {
        const result = await contract.getFileMetadata(req.params.fileHash);
        res.json(result);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.listen(3000, () => console.log("Server running on port 3000"));
