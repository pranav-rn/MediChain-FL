// test-contract.js
const { ethers, upgrades } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  
  console.log("🚀 Deploying contract...");
  const MediChainFL = await ethers.getContractFactory("MediChainFL");
  const proxy = await upgrades.deployProxy(MediChainFL, [deployer.address], { 
    initializer: "initialize" 
  });
  await proxy.waitForDeployment();
  
  const proxyAddress = await proxy.getAddress();
  console.log("✅ Proxy deployed to:", proxyAddress);
  
  console.log("\n📝 Calling logUpdate...");
  const tx = await proxy.logUpdate("test123");
  await tx.wait();
  console.log("✅ Transaction successful:", tx.hash);
  
  console.log("\n📊 Reading data...");
  const count = await proxy.getLogsCount();
  console.log("Total logs:", count.toString());
  
  const log = await proxy.logs(0);
  console.log("Log entry:", {
    hospital: log.hospital,
    gradientHash: log.gradientHash,
    timestamp: log.timestamp.toString()
  });
  
  console.log("\n🎉 All tests passed!");
}

main().catch((err) => { 
  console.error("❌ Error:", err.message); 
  process.exit(1); 
});