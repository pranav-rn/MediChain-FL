const { ethers, upgrades } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  const MediChainFL = await ethers.getContractFactory("MediChainFL");
  const proxy = await upgrades.deployProxy(MediChainFL, [deployer.address], { initializer: "initialize" });

  await proxy.waitForDeployment();
  const proxyAddress = await proxy.getAddress();
  console.log("Proxy deployed to (proxy address):", proxyAddress);
  console.log("Implementation address:", await upgrades.erc1967.getImplementationAddress(proxyAddress));
}

main().catch((err) => { console.error(err); process.exit(1); });
