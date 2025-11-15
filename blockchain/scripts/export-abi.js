const fs = require('fs');
const path = require('path');

async function main() {
  const artifactPath = path.join(__dirname, '../artifacts/contracts/MediChainFL.sol/MediChainFL.json');
  const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
  
  const exportData = {
    contractAddress: "0x2279B7A0a67DB372996a5FaB50D91eAA73d2eBe6",
    abi: artifact.abi,
    rpcUrl: "http://172.17.144.1:8545",
    chainId: 31337
  };
  
  fs.writeFileSync(
    path.join(__dirname, '../contract-config.json'),
    JSON.stringify(exportData, null, 2)
  );
  
  console.log("✅ Contract config exported to contract-config.json");
  console.log("\nShare this file with your teammates!");
}

main().catch(console.error);