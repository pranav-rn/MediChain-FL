// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/ERC20Upgradeable.sol";

contract MediChainFL is Initializable, ERC20Upgradeable {
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
    
    // Reputation System
    mapping(address => uint256) public successfulContributions;
    mapping(address => uint256) public flaggedContributions;
    
    // Token reward per successful contribution
    uint256 public constant REWARD_PER_CONTRIBUTION = 10 * 10**18; // 10 MCT tokens

    // gap for future variable additions (storage layout safety)
    uint256[42] private __gap;

    event UpdateLogged(address indexed hospital, string gradientHash, uint256 timestamp, bool flagged);
    event UpdateFlagged(uint256 index, address hospital);
    event ReputationUpdated(address indexed hospital, uint256 successfulCount, uint256 flaggedCount);
    event TokensRewarded(address indexed hospital, uint256 amount);

    // initializer replaces constructor
    function initialize(address _owner) public initializer {
        __ERC20_init("MediChain Contribution Token", "MCT");
        owner = _owner;
        totalLogs = 0;
    }

    function logUpdate(string memory gradientHash, bool _flagged) public {
        logs.push(UpdateRecord(msg.sender, gradientHash, block.timestamp, _flagged));
        totalLogs += 1;
        
        // Update reputation scores
        if (_flagged) {
            flaggedContributions[msg.sender]++;
            emit UpdateFlagged(logs.length - 1, msg.sender);
        } else {
            successfulContributions[msg.sender]++;
            // Reward tokens for good contribution
            _mint(msg.sender, REWARD_PER_CONTRIBUTION);
            emit TokensRewarded(msg.sender, REWARD_PER_CONTRIBUTION);
        }
        
        emit UpdateLogged(msg.sender, gradientHash, block.timestamp, _flagged);
        emit ReputationUpdated(msg.sender, successfulContributions[msg.sender], flaggedContributions[msg.sender]);
    }

    function flagUpdate(uint256 index) public {
        require(index < logs.length, "Invalid index");
        require(!logs[index].flagged, "Already flagged");
        
        logs[index].flagged = true;
        address hospital = logs[index].hospital;
        
        // Update reputation: decrease successful, increase flagged
        if (successfulContributions[hospital] > 0) {
            successfulContributions[hospital]--;
        }
        flaggedContributions[hospital]++;
        
        emit UpdateFlagged(index, hospital);
        emit ReputationUpdated(hospital, successfulContributions[hospital], flaggedContributions[hospital]);
    }

    function getLogsCount() public view returns (uint256) {
        return logs.length;
    }
    
    function getReputation(address hospital) public view returns (uint256 successful, uint256 flagged, uint256 score) {
        successful = successfulContributions[hospital];
        flagged = flaggedContributions[hospital];
        // Simple score: successful - flagged (can be negative in theory, but uint256 will underflow protect)
        if (successful >= flagged) {
            score = successful - flagged;
        } else {
            score = 0;
        }
        return (successful, flagged, score);
    }
    
    function getTokenBalance(address hospital) public view returns (uint256) {
        return balanceOf(hospital);
    }
}
