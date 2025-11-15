const { ethers } = require("hardhat");

async function main() {
  const MediChainFL = await ethers.getContractFactory("MediChainFL");
  const iface = MediChainFL.interface;
  
  console.log("=== ALL FUNCTION SELECTORS ===\n");
  
  // Get all functions
  const functions = [
    "logUpdate(string)",
    "flagUpdate(uint256)",
    "getLogsCount()",
    "logs(uint256)",
    "owner()",
    "totalLogs()",
    "initialize(address)"
  ];
  
  functions.forEach(func => {
    try {
      const fragment = iface.getFunction(func);
      console.log(`${func}`);
      console.log(`Selector: ${fragment.selector}\n`);
    } catch (e) {
      console.log(`${func} - NOT FOUND\n`);
    }
  });
  
  console.log("\n=== POSTMAN REQUESTS ===\n");
  
  // 1. getLogsCount
  const getLogsCountData = iface.encodeFunctionData("getLogsCount", []);
  console.log("1. GET LOGS COUNT:");
  console.log(JSON.stringify({
    "jsonrpc": "2.0",
    "method": "eth_call",
    "params": [{
      "to": "0x2279B7A0a67DB372996a5FaB50D91eAA73d2eBe6",
      "data": getLogsCountData
    }, "latest"],
    "id": 2
  }, null, 2));
  
  // 2. totalLogs
  const totalLogsData = iface.encodeFunctionData("totalLogs", []);
  console.log("\n2. GET TOTAL LOGS:");
  console.log(JSON.stringify({
    "jsonrpc": "2.0",
    "method": "eth_call",
    "params": [{
      "to": "0x2279B7A0a67DB372996a5FaB50D91eAA73d2eBe6",
      "data": totalLogsData
    }, "latest"],
    "id": 3
  }, null, 2));
  
  // 3. owner
  const ownerData = iface.encodeFunctionData("owner", []);
  console.log("\n3. GET OWNER:");
  console.log(JSON.stringify({
    "jsonrpc": "2.0",
    "method": "eth_call",
    "params": [{
      "to": "0x2279B7A0a67DB372996a5FaB50D91eAA73d2eBe6",
      "data": ownerData
    }, "latest"],
    "id": 4
  }, null, 2));
  
  // 4. logs(0)
  const logsData = iface.encodeFunctionData("logs", [0]);
  console.log("\n4. GET LOG AT INDEX 0:");
  console.log(JSON.stringify({
    "jsonrpc": "2.0",
    "method": "eth_call",
    "params": [{
      "to": "0x2279B7A0a67DB372996a5FaB50D91eAA73d2eBe6",
      "data": logsData
    }, "latest"],
    "id": 5
  }, null, 2));
  
  // 5. flagUpdate(0)
  const flagUpdateData = iface.encodeFunctionData("flagUpdate", [0]);
  console.log("\n5. FLAG UPDATE AT INDEX 0:");
  console.log(JSON.stringify({
    "jsonrpc": "2.0",
    "method": "eth_sendTransaction",
    "params": [{
      "from": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
      "to": "0x2279B7A0a67DB372996a5FaB50D91eAA73d2eBe6",
      "data": flagUpdateData
    }],
    "id": 6
  }, null, 2));
}

main().catch(console.error);