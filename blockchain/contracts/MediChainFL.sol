// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";

contract MediChainFL is Initializable {
    struct UpdateRecord {
        address hospital;
        string gradientHash;      // Hash (SHA256/IPFS CID)
        uint256 timestamp;
        bool flagged;
    }

    // storage
    UpdateRecord[] public logs;
    address public owner;
    uint256 public totalLogs;

    // gap for future variable additions (storage layout safety)
    uint256[45] private __gap;

    event UpdateLogged(address indexed hospital, string gradientHash, uint256 timestamp);
    event UpdateFlagged(uint256 index, address hospital);

    // initializer replaces constructor
    function initialize(address _owner) public initializer {
        owner = _owner;
        totalLogs = 0;
    }

    function logUpdate(string memory gradientHash) public {
        logs.push(UpdateRecord(msg.sender, gradientHash, block.timestamp, false));
        totalLogs += 1;
        emit UpdateLogged(msg.sender, gradientHash, block.timestamp);
    }

    function flagUpdate(uint256 index) public {
        require(index < logs.length, "Invalid index");
        logs[index].flagged = true;
        emit UpdateFlagged(index, logs[index].hospital);
    }

    function getLogsCount() public view returns (uint256) {
        return logs.length;
    }
}
