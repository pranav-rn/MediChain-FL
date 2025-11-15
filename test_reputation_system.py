#!/usr/bin/env python3
"""
Test script to demonstrate blockchain reputation and token system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "blockchain" / "python"))

from blockchain_client import (
    log_gradient_update, 
    get_reputation, 
    get_token_balance,
    get_all_logs,
    ACCOUNT
)

def main():
    print("=" * 70)
    print("BLOCKCHAIN REPUTATION & TOKEN SYSTEM TEST")
    print("=" * 70)
    
    # Test hospital address (using default account)
    hospital = ACCOUNT
    
    print(f"\n📍 Testing with Hospital Address: {hospital}")
    
    # Initial reputation check
    print("\n" + "=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    rep = get_reputation(hospital)
    tokens = get_token_balance(hospital)
    print(f"✅ Successful Contributions: {rep['successful']}")
    print(f"🚨 Flagged Contributions: {rep['flagged']}")
    print(f"⭐ Reputation Score: {rep['score']}")
    print(f"🪙 MCT Token Balance: {tokens:.2f} MCT")
    
    # Simulate good update
    print("\n" + "=" * 70)
    print("TEST 1: LOGGING GOOD UPDATE")
    print("=" * 70)
    print("📤 Submitting good gradient update...")
    receipt1 = log_gradient_update("good_gradient_hash_001", flagged=False)
    print(f"✅ Transaction confirmed in block #{receipt1['blockNumber']}")
    print(f"   Gas used: {receipt1['gasUsed']}")
    
    # Check reputation after good update
    rep = get_reputation(hospital)
    tokens = get_token_balance(hospital)
    print(f"\n📊 Updated Stats:")
    print(f"   ✅ Successful Contributions: {rep['successful']}")
    print(f"   🚨 Flagged Contributions: {rep['flagged']}")
    print(f"   ⭐ Reputation Score: {rep['score']}")
    print(f"   🪙 MCT Token Balance: {tokens:.2f} MCT (+10 MCT reward!)")
    
    # Simulate another good update
    print("\n" + "=" * 70)
    print("TEST 2: LOGGING ANOTHER GOOD UPDATE")
    print("=" * 70)
    print("📤 Submitting another good gradient update...")
    receipt2 = log_gradient_update("good_gradient_hash_002", flagged=False)
    print(f"✅ Transaction confirmed in block #{receipt2['blockNumber']}")
    
    rep = get_reputation(hospital)
    tokens = get_token_balance(hospital)
    print(f"\n📊 Updated Stats:")
    print(f"   ✅ Successful Contributions: {rep['successful']}")
    print(f"   🚨 Flagged Contributions: {rep['flagged']}")
    print(f"   ⭐ Reputation Score: {rep['score']}")
    print(f"   🪙 MCT Token Balance: {tokens:.2f} MCT")
    
    # Simulate malicious update
    print("\n" + "=" * 70)
    print("TEST 3: LOGGING MALICIOUS UPDATE")
    print("=" * 70)
    print("📤 Submitting malicious gradient update (FLAGGED)...")
    receipt3 = log_gradient_update("malicious_gradient_hash_666", flagged=True)
    print(f"🚨 Transaction confirmed in block #{receipt3['blockNumber']}")
    
    rep = get_reputation(hospital)
    tokens = get_token_balance(hospital)
    print(f"\n📊 Updated Stats:")
    print(f"   ✅ Successful Contributions: {rep['successful']}")
    print(f"   🚨 Flagged Contributions: {rep['flagged']} (⚠️ INCREASED!)")
    print(f"   ⭐ Reputation Score: {rep['score']}")
    print(f"   🪙 MCT Token Balance: {tokens:.2f} MCT (no reward for flagged)")
    
    # Display all logs
    print("\n" + "=" * 70)
    print("ALL BLOCKCHAIN LOGS")
    print("=" * 70)
    logs = get_all_logs()
    for log in logs:
        status = "🚨 FLAGGED" if log['flagged'] else "✅ VALID"
        print(f"{status} | Log #{log['index']}: {log['gradientHash'][:20]}...")
        print(f"         Hospital: {log['hospital'][:10]}...")
        print(f"         Timestamp: {log['timestamp']}")
        print()
    
    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"📊 Hospital {hospital[:10]}... has:")
    print(f"   ✅ {rep['successful']} successful contributions")
    print(f"   🚨 {rep['flagged']} flagged contributions")
    print(f"   ⭐ Reputation score: {rep['score']}")
    print(f"   🪙 Token balance: {tokens:.2f} MCT")
    print("\n💡 Key Insights:")
    print("   • Good updates earn 10 MCT tokens automatically")
    print("   • Flagged updates do NOT earn tokens")
    print("   • Reputation is transparent and immutable on blockchain")
    print("   • This creates a trustless incentive system!")
    print("=" * 70)

if __name__ == "__main__":
    main()
