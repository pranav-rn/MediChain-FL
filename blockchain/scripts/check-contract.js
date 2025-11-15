const { ethers } = require("hardhat");

async function main() {
  const address = "0x2279B7A0a67DB372996a5FaB50D91eAA73d2eBe6";
  
  // Check if contract exists
  const code = await ethers.provider.getCode(address);
  console.log("Contract code length:", code.length);
  console.log("Has code:", code !== "0x");
  
  if (code !== "0x") {
    // Try to interact with it
    const MediChainFL = await ethers.getContractFactory("MediChainFL");
    const contract = MediChainFL.attach(address);
    
    try {
      const owner = await contract.owner();
      console.log("Owner:", owner);
      
      const count = await contract.getLogsCount();
      console.log("Logs count:", count.toString());
      
      console.log("\n✅ Contract is accessible via ethers.js");
    } catch (err) {
      console.log("❌ Error calling contract:", err.message);
    }
  } else {
    console.log("❌ No contract at this address!");
  }
}

main().catch(console.error);