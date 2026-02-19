import torch
import pickle
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from algosdk.v2client import indexer
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

# --- 1. Model Definition (Must match training architecture) ---
class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=16):
        super(FraudGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- 2. FastAPI Setup & Resource Loading ---
app = FastAPI(title="Algorand AI Fraud Detection Service")

# Global state to hold model and mapping
state = {}

@app.on_event("startup")
def load_resources():
    try:
        # Load Algorand Indexer (Testnet example)
        state["algo_indexer"] = indexer.IndexerClient("", "https://testnet-idx.algonode.cloud", "")
        
        # Load LabelEncoder (The mapping of addresses to Graph IDs)
        try:
            with open("label_encoder.pkl", "rb") as f:
                state["encoder"] = pickle.load(f)
        except FileNotFoundError:
            print("WARNING: label_encoder.pkl not found. Features may not map correctly.")
            state["encoder"] = None
        
        # Load Model
        state["model"] = FraudGNN(in_channels=5)
        try:
            state["model"].load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
        except FileNotFoundError:
            print("WARNING: model.pt not found. Using untrained model.")
        state["model"].eval()
        print("AI Resources Loaded Successfully.")
    except Exception as e:
        print(f"ERROR loading resources: {e}")
        raise

# --- 3. Request Schemas ---
class FraudCheck(BaseModel):
    wallet_address: str
    asset_id: int

# --- 4. Logic Functions ---
def trigger_blockchain_freeze(wallet: str, asset_id: int):
    """
    Background Task: Calls the Algorand Smart Contract to freeze the asset
    if the risk score is high. Requires a 'Manager' or 'Freeze' account.
    """
    # implementation: Use algosdk to send a 'AssetFreeze' transaction
    print(f"!!! BLOCKCHAIN ACTION: Freezing Asset {asset_id} for wallet {wallet} !!!")

def get_wallet_transactions(wallet_address: str, asset_id: int):
    """
    Mock transaction fetcher for fraud pattern analysis.
    In production, this would query the Algorand blockchain.
    """
    import random
    import hashlib
    
    # Seed based on wallet address for consistent results
    random.seed(int(hashlib.md5(wallet_address.encode()).hexdigest(), 16) % (2**32))
    
    # Simulate transaction patterns based on wallet type
    if "MULE" in wallet_address:
        # Fraudulent pattern: many small transfers to hub
        txn_count = random.randint(15, 30)
        txns = [{
            "sender": wallet_address,
            "receiver": "HUB_COLLECTOR_01",
            "amount": random.randint(100, 500)
        } for _ in range(txn_count)]
    elif "HUB" in wallet_address:
        # Hub pattern: many inbound transactions
        txn_count = random.randint(20, 40)
        txns = [{
            "sender": f"MULE_{i % 10}",
            "receiver": wallet_address,
            "amount": random.randint(100, 500)
        } for i in range(txn_count)]
    elif "STU" in wallet_address:
        # Normal student: few transactions, larger amounts
        txn_count = random.randint(1, 5)
        txns = [{
            "sender": "GOVT_SCHOLARSHIP_DEPT",
            "receiver": wallet_address,
            "amount": random.randint(1000, 5000)
        } for _ in range(txn_count)]
    elif "GOVT" in wallet_address:
        # Government: many outbound to students
        txn_count = random.randint(40, 60)
        txns = [{
            "sender": wallet_address,
            "receiver": f"STU_{i % 50}",
            "amount": random.randint(1000, 5000)
        } for i in range(txn_count)]
    else:
        # Unknown: few transactions
        txn_count = random.randint(1, 3)
        txns = [{
            "sender": wallet_address,
            "receiver": "UNKNOWN_RECIPIENT",
            "amount": random.randint(100, 1000)
        } for _ in range(txn_count)]
    
    return txns

# --- 5. API Endpoints ---
@app.post("/analyze-wallet")
async def analyze_wallet(data: FraudCheck, background_tasks: BackgroundTasks):
    try:
        # A. Fetch Wallet Activity (Mock data for demo)
        txns = get_wallet_transactions(data.wallet_address, data.asset_id)

        # B. Check if wallet exists in our learned graph
        if state["encoder"] is None:
            raise HTTPException(status_code=503, detail="Encoder not loaded. Model not ready.")
        
        try:
            wallet_idx = state["encoder"].transform([data.wallet_address])[0]
        except ValueError:
            raise HTTPException(status_code=404, detail="Wallet address not found in historical graph data.")

        # C. Feature Extraction from transaction patterns
        # Features: [In-Degree, Out-Degree, Total_Amt, Structuring_Score, Account_Age]
        in_d = len([t for t in txns if t['receiver'] == data.wallet_address])
        out_d = len([t for t in txns if t['sender'] == data.wallet_address])
        total_amount = sum([t['amount'] for t in txns if t['sender'] == data.wallet_address])
        avg_amount = total_amount / max(out_d, 1) if out_d > 0 else 0
        structuring_score = 0.8 if out_d > 0 and avg_amount < 500 else 0.1  # Many small transfers
        account_age = 30
        
        # Create feature matrix with the wallet and a dummy neighbor node
        wallet_features = [in_d, out_d, avg_amount, structuring_score, account_age]
        dummy_features = [0, 0, 0, 0, 0]  # Dummy node representing the network
        features = torch.tensor([wallet_features, dummy_features], dtype=torch.float)
        
        # Create edge index connecting wallet (0) to dummy node (1) and vice versa
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        # D. Run AI Inference
        with torch.no_grad():
            output = state["model"](features, edge_index)
            probs = torch.exp(output)
            risk_score = float(probs[0][1]) # Probability of being a 'Mule' (use first node's fraud probability)

        # E. Determine Action
        decision = "CLEAR"
        if risk_score > 0.85:
            decision = "FRAUD_HIGH"
            background_tasks.add_task(trigger_blockchain_freeze, data.wallet_address, data.asset_id)
        elif risk_score > 0.60:
            decision = "SUSPICIOUS_REVIEW"

        return {
            "address": data.wallet_address,
            "risk_score": round(risk_score, 4),
            "decision": decision,
            "monitored_asset": data.asset_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "running", "network": "Algorand Testnet"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)