const { ethers } = require("ethers");

const iface = new ethers.Interface([
  "function logUpdate(string memory gradientHash)"
]);

// Generate data for "GradientHash12345"
const data1 = iface.encodeFunctionData("logUpdate", ["GradientHash12345"]);
console.log("Data for 'GradientHash12345':");
console.log(data1);

// Generate data for "test123"
const data2 = iface.encodeFunctionData("logUpdate", ["test123"]);
console.log("\nData for 'test123':");
console.log(data2);

// Show the complete Postman request
console.log("\n=== COPY THIS ENTIRE REQUEST FOR POSTMAN ===\n");
console.log(JSON.stringify({
  "jsonrpc": "2.0",
  "method": "eth_sendTransaction",
  "params": [{
    "from": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
    "to": "0x2279B7A0a67DB372996a5FaB50D91eAA73d2eBe6",
    "data": data1
  }],
  "id": 1
}, null, 2));