# Quick diagnostic: what actions is the agent taking?
from train_v4 import *
import numpy as np
import torch
import sys
sys.path.insert(0, '.')

# Load the model and test on a single observation

device = torch.device('cpu')
set_seed(42)

# Generate test bars
bars = make_bars_v3(n=300, seed=99)
cfg = EnvironmentConfig(
    episode_length=200, warmup_bars=60,
    initial_capital=100_000.0, random_start=False,
    reward=RewardConfig(),
)
env = SingleAssetEnvV4(bars, cfg, device)

# Create network and load best checkpoint
ncfg = NetworkConfig(
    architecture=Architecture.ATTENTION_LSTM,
    d_model=64, n_heads=4, n_layers=2, d_ff=128,
    lstm_hidden=64, lstm_layers=1, lstm_dropout=0.0,
    hidden_dims=(128, 64), dropout=0.1,
    use_noisy_net=True, use_dueling=True,
)
net = ActorCriticNetwork(ncfg, N_MARKET_FEATURES, N_PORTFOLIO_FEATURES,
                         N_SENTIMENT_FEATURES, N_TIME_FEATURES, ad=3, cont=True)

# Load checkpoint
ckpt = torch.load('/tmp/drl_v42_best.pt', map_location=device)
net.load_state_dict(ckpt['net'])
net.eval()

# Run a few steps and check actions
res = env.reset(seed=0)
actions_taken = []
positions_held = []
for step in range(50):
    mkt = res.obs_market_seq.unsqueeze(0)
    port = res.obs_portfolio_vec.unsqueeze(0)
    sent = res.obs_sentiment_vec.unsqueeze(0)
    tv = res.obs_time_vec.unsqueeze(0)

    ao, co, _ = net.get_action_and_value(mkt, port, sent, tv, det=True)
    action = ao.action.squeeze(0).numpy()
    actions_taken.append(action)

    res = env.step(action)
    has_pos = res.info.get('position', False)
    positions_held.append(has_pos)

actions = np.array(actions_taken)
print("Action analysis (deterministic mode):")
print(
    f"  Direction:  mean={actions[:,0].mean():.3f}, std={actions[:,0].std():.3f}")
print(
    f"  Size:       mean={actions[:,1].mean():.3f}, std={actions[:,1].std():.3f}")
print(
    f"  TP mult:    mean={actions[:,2].mean():.3f}, std={actions[:,2].std():.3f}")
print(
    f"  % in dead zone (±0.20): {(np.abs(actions[:,0]) < 0.20).mean()*100:.0f}%")
print(f"  % long: {(actions[:,0] > 0.20).mean()*100:.0f}%")
print(f"  % short: {(actions[:,0] < -0.20).mean()*100:.0f}%")
print(f"  Positions held: {sum(positions_held)}/50 steps")
print(f"  Capital: ${res.info.get('capital', 0):,.0f}")
