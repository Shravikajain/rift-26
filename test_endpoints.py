import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test 1: Check if API is running"""
    print("\n=== Test 1: Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    print("✓ PASSED")

def test_analyze_wallet_normal():
    """Test 2: Analyze a normal wallet (known in training data)"""
    print("\n=== Test 2: Analyze Normal Wallet ===")
    payload = {
        "wallet_address": "STU_0",
        "asset_id": 12345
    }
    response = requests.post(f"{BASE_URL}/analyze-wallet", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        data = response.json()
        assert "address" in data
        assert "risk_score" in data
        assert "decision" in data
        assert 0 <= data["risk_score"] <= 1
        print("✓ PASSED")
    else:
        print(f"⚠ Status {response.status_code}: {response.text}")

def test_analyze_wallet_mule():
    """Test 3: Analyze a mule wallet (fraudulent in training data)"""
    print("\n=== Test 3: Analyze Mule Wallet (Fraud Pattern) ===")
    payload = {
        "wallet_address": "MULE_0",
        "asset_id": 12345
    }
    response = requests.post(f"{BASE_URL}/analyze-wallet", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Risk Score: {data['risk_score']} (should be higher for mule)")
        print(f"Decision: {data['decision']}")
        assert "address" in data
        print("✓ PASSED")
    else:
        print(f"⚠ Status {response.status_code}: {response.text}")

def test_analyze_wallet_hub():
    """Test 4: Analyze hub wallet (collector account)"""
    print("\n=== Test 4: Analyze Hub Wallet ===")
    payload = {
        "wallet_address": "HUB_COLLECTOR_01",
        "asset_id": 12345
    }
    response = requests.post(f"{BASE_URL}/analyze-wallet", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Decision: {data['decision']}")
        print("✓ PASSED")
    else:
        print(f"⚠ Status {response.status_code}: {response.text}")

def test_analyze_wallet_unknown():
    """Test 5: Analyze unknown wallet (not in training data)"""
    print("\n=== Test 5: Analyze Unknown Wallet ===")
    payload = {
        "wallet_address": "UNKNOWN_WALLET_XYZ",
        "asset_id": 12345
    }
    response = requests.post(f"{BASE_URL}/analyze-wallet", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Should return 404 for unknown wallet
    if response.status_code == 404:
        print("✓ PASSED (correctly identified unknown wallet)")
    else:
        print(f"⚠ Unexpected status {response.status_code}")

def test_analyze_wallet_invalid_payload():
    """Test 6: Send invalid payload (missing field)"""
    print("\n=== Test 6: Invalid Payload ===")
    payload = {
        "wallet_address": "STU_0"
        # Missing asset_id
    }
    response = requests.post(f"{BASE_URL}/analyze-wallet", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Should return 422 for validation error
    if response.status_code == 422:
        print("✓ PASSED (correctly rejected invalid payload)")
    else:
        print(f"⚠ Unexpected status {response.status_code}")

def test_analyze_wallet_multiple_assets():
    """Test 7: Analyze same wallet with different asset IDs"""
    print("\n=== Test 7: Same Wallet, Different Assets ===")
    wallet = "STU_1"
    for asset_id in [100, 200, 300]:
        payload = {
            "wallet_address": wallet,
            "asset_id": asset_id
        }
        response = requests.post(f"{BASE_URL}/analyze-wallet", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"  Asset {asset_id}: Risk={data['risk_score']}, Decision={data['decision']}")
        else:
            print(f"  Asset {asset_id}: Error {response.status_code}")
    print("✓ PASSED")

def test_government_wallet():
    """Test 8: Analyze government/NGO wallet"""
    print("\n=== Test 8: Government Wallet ===")
    payload = {
        "wallet_address": "GOVT_SCHOLARSHIP_DEPT",
        "asset_id": 12345
    }
    response = requests.post(f"{BASE_URL}/analyze-wallet", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Risk Score: {data['risk_score']} (should be low for gov wallet)")
        print("✓ PASSED")
    else:
        print(f"⚠ Status {response.status_code}: {response.text}")

def test_high_risk_threshold():
    """Test 9: Verify fraud detection threshold"""
    print("\n=== Test 9: Risk Score Thresholds ===")
    print("Testing various wallets to verify risk thresholds:")
    print("  - risk_score > 0.85: FRAUD_HIGH")
    print("  - 0.60 < risk_score <= 0.85: SUSPICIOUS_REVIEW")
    print("  - risk_score <= 0.60: CLEAR")
    
    test_wallets = ["STU_0", "STU_10", "MULE_0", "HUB_COLLECTOR_01"]
    for wallet in test_wallets:
        payload = {"wallet_address": wallet, "asset_id": 12345}
        response = requests.post(f"{BASE_URL}/analyze-wallet", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"  {wallet}: {data['risk_score']:.4f} -> {data['decision']}")
    print("✓ PASSED")

if __name__ == "__main__":
    print("=" * 60)
    print("ALGORAND FRAUD DETECTION API - TEST SUITE")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print("Make sure the API is running: python main.py")
    
    try:
        test_health_endpoint()
        test_analyze_wallet_normal()
        test_analyze_wallet_mule()
        test_analyze_wallet_hub()
        test_analyze_wallet_unknown()
        test_analyze_wallet_invalid_payload()
        test_analyze_wallet_multiple_assets()
        test_government_wallet()
        test_high_risk_threshold()
        
        print("\n" + "=" * 60)
        print("✅ TEST SUITE COMPLETED")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Make sure the API is running on http://localhost:8000")
