import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

PATH       = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
SEQ_LEN    = 32
BATCH_SIZE = 128
EPOCHS     = 50
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OCP_DEGREE = 5
LAMBDA_CC  = 0.3
LAMBDA_OCP = 0.2

POWER_LEVELS = [3300, 2700, 1650, 1000, 660, 330]
MODES        = {"C": 0, "D": 1}
FEATURE_COLS = ["P_AC", "C", "V", "Ta", "power_level", "mode"]
TARGET_COL   = "SOC"


def load_data(path):
    df = pd.read_csv(path, sep=";", header=0)
    df = df.drop(columns=[df.columns[-1]])
    df = df.rename(columns={"_C_2700": "Ta_C_2700", "_D_2700": "Ta_D_2700"})
    series_list = []
    for mode, mode_flag in MODES.items():
        for p in POWER_LEVELS:
            s = df[[f"SOC_{mode}_{p}", f"P_AC_{mode}_{p}",
                    f"C_{mode}_{p}", f"V_{mode}_{p}", f"Ta_{mode}_{p}"]].copy()
            s.columns = ["SOC", "P_AC", "C", "V", "Ta"]
            s["power_level"] = p
            s["mode"]        = mode_flag
            series_list.append(s.dropna().reset_index(drop=True))
    print(f"Series: {len(series_list)}  |  rows/series: {[len(s) for s in series_list]}")
    return series_list


def estimate_capacity(series_list):
    ratios = []
    for s in series_list:
        dSOC = s["SOC"].diff().dropna().values
        I    = s["C"].values[1:]
        mask = np.abs(dSOC) > 0.01
        if mask.sum() > 5:
            Q_est = -I[mask] / (dSOC[mask] / 100.0)
            ratios.extend(Q_est[(Q_est > 0) & (Q_est < 1e4)].tolist())
    Q = float(np.median(ratios))
    print(f"Estimated capacity Q = {Q:.2f} Ah")
    return Q


def fit_ocp(series_list):
    soc = np.concatenate([s["SOC"].values for s in series_list]) / 100.0
    v   = np.concatenate([s["V"].values   for s in series_list])
    coefs = np.polyfit(soc, v, OCP_DEGREE)
    r2    = r2_score(v, np.poly1d(coefs)(soc))
    print(f"OCP poly fit  R²={r2:.4f}  degree={OCP_DEGREE}")
    return coefs


def make_sequences(series_list):
    X_all, y_all, C_all, V_all, sp_all = [], [], [], [], []
    for s in series_list:
        feats  = s[FEATURE_COLS].values.astype(np.float32)
        target = s[TARGET_COL].values.astype(np.float32)
        curr   = s["C"].values.astype(np.float32)
        volt   = s["V"].values.astype(np.float32)
        for i in range(SEQ_LEN, len(s)):
            X_all.append(feats[i - SEQ_LEN:i])
            y_all.append(target[i])
            C_all.append(curr[i])
            V_all.append(volt[i])
            sp_all.append(target[i - 1])
    return (np.array(X_all), np.array(y_all),
            np.array(C_all), np.array(V_all), np.array(sp_all))


class BatteryGRU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.rnn  = nn.GRU(input_size, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def ocp_torch(soc_norm, coefs_t):
    result = torch.zeros_like(soc_norm)
    deg    = len(coefs_t) - 1
    for i, c in enumerate(coefs_t):
        result = result + c * soc_norm ** (deg - i)
    return result


def train(series_list, Q, ocp_coefs):
    X, y, C, V, sp = make_sequences(series_list)
    print(f"Total sequences: {len(X)}")

    x_sc = StandardScaler()
    y_sc = StandardScaler()
    X_s  = x_sc.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_s  = y_sc.fit_transform(y.reshape(-1, 1)).flatten()

    split  = int(0.8 * len(X_s))
    slices = lambda a: (a[:split], a[split:])
    X_tr,  X_val  = slices(X_s)
    y_tr,  y_val  = slices(y_s)
    C_tr,  C_val  = slices(C)
    V_tr,  V_val  = slices(V)
    sp_tr, sp_val = slices(sp)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32).to(DEVICE)
    X_tr, y_tr, C_tr, V_tr, sp_tr       = map(to_t, [X_tr, y_tr, C_tr, V_tr, sp_tr])
    X_val, y_val, C_val, V_val, sp_val  = map(to_t, [X_val, y_val, C_val, V_val, sp_val])

    ocp_t = torch.tensor(ocp_coefs, dtype=torch.float32).to(DEVICE)
    Q_t   = torch.tensor(Q,         dtype=torch.float32).to(DEVICE)

    y_mean = torch.tensor(y_sc.mean_[0],  dtype=torch.float32).to(DEVICE)
    y_std  = torch.tensor(y_sc.scale_[0], dtype=torch.float32).to(DEVICE)

    model   = BatteryGRU(X.shape[2]).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.SmoothL1Loss()
    history = {"train": [], "val": [], "cc": [], "ocp": []}
    best_val, best_ep = 1e9, 0

    print(f"\nTraining on {DEVICE}  |  {X_tr.shape[0]} train  {X_val.shape[0]} val\n")

    for epoch in range(EPOCHS):
        model.train()
        idx    = torch.randperm(X_tr.size(0))
        tr_tot = cc_tot = ocp_tot = 0.0

        for i in range(0, len(idx), BATCH_SIZE):
            b = idx[i:i + BATCH_SIZE]

            pred_s = model(X_tr[b])                          # scaled space
            pred_r = pred_s * y_std + y_mean                 # real SOC
            true_r = y_tr[b] * y_std + y_mean

            data_loss = loss_fn(pred_s, y_tr[b])

            # Coulomb counting: ΔSOC = -I / (Q*3600) per second
            soc_norm  = pred_r / 100.0
            sp_norm   = sp_tr[b] / 100.0
            cc_target = sp_norm - C_tr[b] / (Q_t * 3600.0)
            cc_loss   = torch.mean((soc_norm - cc_target) ** 2)

            # OCP residual: V ≈ OCP(SOC)
            ocp_pred  = ocp_torch(soc_norm.detach(), ocp_t)
            ocp_loss  = torch.mean((V_tr[b] - ocp_pred) ** 2)

            total = data_loss + LAMBDA_CC * cc_loss + LAMBDA_OCP * ocp_loss
            opt.zero_grad(); total.backward(); opt.step()

            n = len(b)
            tr_tot  += data_loss.item() * n
            cc_tot  += cc_loss.item()   * n
            ocp_tot += ocp_loss.item()  * n

        n_tr = len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        sched.step()
        history["train"].append(tr_tot / n_tr)
        history["val"].append(val_loss)
        history["cc"].append(cc_tot / n_tr)
        history["ocp"].append(ocp_tot / n_tr)

        if val_loss < best_val:
            best_val, best_ep = val_loss, epoch
            torch.save(model.state_dict(), "best_pinn_gru.pt")

        print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={tr_tot/n_tr:.5f}  val={val_loss:.5f}  "
              f"cc={cc_tot/n_tr:.5f}  ocp={ocp_tot/n_tr:.5f}", flush=True)

    print(f"\nBest val {best_val:.6f} at epoch {best_ep+1}")
    model.load_state_dict(torch.load("best_pinn_gru.pt", map_location=DEVICE))
    return model, x_sc, y_sc, X_val, y_val, history


def evaluate(model, y_sc, X_val, y_val):
    model.eval()
    with torch.no_grad():
        preds_s = model(X_val).cpu().numpy()
    preds = y_sc.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    trues = y_sc.inverse_transform(y_val.cpu().numpy().reshape(-1, 1)).flatten()
    r2   = r2_score(trues, preds)
    mae  = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    print(f"\n{'='*45}")
    print(f"R²   : {r2:.6f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"{'='*45}")
    return trues, preds


def plot(trues, preds, history, ocp_coefs, series_list):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0,0].plot(history["train"], label="Train")
    axes[0,0].plot(history["val"],   label="Val", ls="--")
    axes[0,0].set_title("Data Loss"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(history["cc"], color="darkorange")
    axes[0,1].set_title("Coulomb Counting Loss"); axes[0,1].grid(alpha=0.3)

    axes[0,2].plot(history["ocp"], color="green")
    axes[0,2].set_title("OCP Residual Loss"); axes[0,2].grid(alpha=0.3)

    r2  = r2_score(trues, preds)
    mae = mean_absolute_error(trues, preds)
    axes[1,0].scatter(trues, preds, alpha=0.3, s=8, color="steelblue")
    lims = [min(trues.min(), preds.min()), max(trues.max(), preds.max())]
    axes[1,0].plot(lims, lims, "r--", lw=1.5)
    axes[1,0].set_xlabel("True SOC"); axes[1,0].set_ylabel("Pred SOC")
    axes[1,0].set_title(f"Pred vs True  R²={r2:.4f}"); axes[1,0].grid(alpha=0.3)

    res = preds - trues
    axes[1,1].scatter(preds, res, alpha=0.3, s=8, color="darkorange")
    axes[1,1].axhline(0, color="red", lw=1.5, ls="--")
    axes[1,1].set_title(f"Residuals  MAE={mae:.4f}"); axes[1,1].grid(alpha=0.3)

    soc_r = np.linspace(0, 1, 200)
    ocp_fn = np.poly1d(ocp_coefs)
    soc_d  = np.concatenate([s["SOC"].values for s in series_list]) / 100.0
    v_d    = np.concatenate([s["V"].values   for s in series_list])
    axes[1,2].scatter(soc_d, v_d, alpha=0.1, s=4, color="steelblue", label="Data")
    axes[1,2].plot(soc_r, ocp_fn(soc_r), "r-", lw=2, label="OCP fit")
    axes[1,2].set_xlabel("SOC"); axes[1,2].set_ylabel("V")
    axes[1,2].set_title("OCP(SOC) Fit"); axes[1,2].legend(); axes[1,2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("soc_pinn_gru_results.png", dpi=150)
    plt.show()
    print("Saved: soc_pinn_gru_results.png")


series_list          = load_data(PATH)
Q                    = estimate_capacity(series_list)
ocp_coefs            = fit_ocp(series_list)
model, x_sc, y_sc, X_val, y_val, history = train(series_list, Q, ocp_coefs)
trues, preds         = evaluate(model, y_sc, X_val, y_val)
plot(trues, preds, history, ocp_coefs, series_list)

joblib.dump(x_sc, "x_scaler.pkl")
joblib.dump(y_sc, "y_scaler.pkl")
np.save("ocp_coefs.npy", ocp_coefs)
print("Saved: best_pinn_gru.pt | x_scaler.pkl | y_scaler.pkl | ocp_coefs.npy")