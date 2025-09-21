# app.py — Sports Betting Analyst + Gestionale (Conto Virtuale)
# -----------------------------------------------------------------------------
# Funzioni principali aggiunte:
# - Profilo giocatore in sidebar (username, creazione conto, saldo iniziale in €).
# - Persistenza locale per-utente (./data/<username>.json) con:
#     • profilo (saldo, creato_il, valuta)
#     • schedine: singole / multiple / sistemi (con sottoticket)
# - Salvataggio schedine direttamente dai tab "Multipla" e "Sistemi"
# - Verifica esiti (vinta/persa/void) con aggiornamento saldo e storico
# - Storico con filtri, esportazione CSV e reset opzionale
#
# Nota: persistenza su file locale funziona in ambienti dove il filesystem resta
#       disponibile fra run. Su hosting effimeri, usa Export/Import profilo.
# -----------------------------------------------------------------------------

import io
import os
import json
import math
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from collections import OrderedDict
from itertools import combinations

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
def do_rerun():
    # Streamlit >= 1.27
    try:
        st.rerun()
    except Exception:
        # Fallback per versioni più vecchie
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ============= CONFIG UI ======================================================
st.set_page_config(page_title="📊 Sports Betting Analyst — Gestionale", layout="wide")
st.title("📊 Sports Betting Analyst — Gestionale (conto virtuale)")
st.caption("Modello Poisson (Calcio) + gestione schedine e saldo virtuale (€)")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ============= UTILS STAGIONE & DOWNLOAD =====================================
def current_fd_code(today: Optional[datetime] = None) -> str:
    d = today or datetime.today()
    yy = d.year % 100
    return f"{yy:02d}{(yy+1)%100:02d}" if d.month >= 7 else f"{(yy-1)%100:02d}{yy:02d}"

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fd_csv(url_https: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url_https)
    except Exception:
        try:
            return pd.read_csv(url_https.replace("https://", "http://"))
        except Exception:
            r = requests.get(url_https, timeout=20, verify=False)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))

# ============= PERSISTENZA PROFILI ===========================================
def _profile_path(username: str) -> str:
    safe = "".join(c for c in username if c.isalnum() or c in ("_","-","."))
    return os.path.join(DATA_DIR, f"{safe.lower()}.json")

def load_profile(username: str) -> Dict:
    path = _profile_path(username)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # profilo nuovo
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "username": username,
        "currency": "EUR",
        "created_at": now,
        "balance": 0.0,
        "initial_balance": 0.0,
        "bets": []  # lista schedine
    }

def save_profile(profile: Dict) -> None:
    path = _profile_path(profile["username"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

def export_profile(profile: Dict) -> str:
    """Ritorna string JSON per download."""
    return json.dumps(profile, ensure_ascii=False, indent=2)

def import_profile_json(file) -> Dict:
    data = json.load(file)
    # sanity minima
    for k in ["username","balance","bets"]:
        if k not in data: raise ValueError("JSON profilo non valido.")
    return data

# ============= HELPER CALCOLO =================================================
@dataclass
class MatchupResult:
    prob_home: float
    prob_draw: float
    prob_away: float
    fair_odds_home: float
    fair_odds_draw: float
    fair_odds_away: float
    lam_home: float
    lam_away: float

def ev_from_prob_odds(p: float, odds: float, commission: float = 0.0) -> float:
    payoff = odds * (1.0 - commission) - 1.0
    return p * payoff - (1.0 - p)

def ev_with_push(p_win: float, p_push: float, odds: float, commission: float = 0.0) -> float:
    p_lose = max(0.0, 1.0 - p_win - p_push)
    payoff = odds * (1.0 - commission) - 1.0
    return p_win * payoff + p_push * 0.0 - p_lose * 1.0

def percent_fmt(x):
    try:
        return f"{100*float(x):.1f}%" if pd.notnull(x) and np.isfinite(float(x)) else "-"
    except:
        return "-"

# ============= GRAFICA ========================================================
def gauge_percent(title: str, value_pct: float, green_from: float = 66.0):
    v = max(0.0, min(100.0, float(value_pct)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix": "%"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "black"},
            "steps": [
                {"range": [0, 33], "color": "#f87171"},
                {"range": [33, green_from], "color": "#fbbf24"},
                {"range": [green_from, 100], "color": "#34d399"},
            ],
        },
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=10), height=220)
    return fig

def donut(labels, values, title="", height=260, showlegend=False):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.50)])
    fig.update_traces(textposition="inside", textinfo="percent+label", showlegend=False)
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        showlegend=showlegend,
        uniformtext_minsize=12,
        uniformtext_mode="hide",
    )
    return fig

# ============= CALCOLO CALCIO (Poisson) ======================================
def normalize_football_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
        "FTHG": "home_goals", "FTAG": "away_goals",
        "Home": "home", "Away": "away", "HG": "home_goals", "AG": "away_goals",
    }
    for src, dst in colmap.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    if "competition" not in df.columns:
        df["competition"] = ""
    if {"home_goals", "away_goals"}.issubset(df.columns):
        for c in ["home_goals", "away_goals"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    return df

def build_team_rates_football(df: pd.DataFrame, last_n_matches: int = 10) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    df = df.sort_values("date")
    frames = []
    for team in pd.unique(pd.concat([df["home"], df["away"]])):
        dft = df[(df["home"] == team) | (df["away"] == team)].tail(last_n_matches)
        frames.append(dft)
    dfl = pd.concat(frames).drop_duplicates()
    league_avg_home = dfl["home_goals"].mean()
    league_avg_away = dfl["away_goals"].mean()
    if not np.isfinite(league_avg_home) or league_avg_home <= 0: league_avg_home = 1.4
    if not np.isfinite(league_avg_away) or league_avg_away <= 0: league_avg_away = 1.1
    teams = pd.unique(pd.concat([dfl["home"], dfl["away"]]))
    att, deff = {}, {}
    for t in teams:
        played = ((dfl["home"] == t) | (dfl["away"] == t)).sum()
        gf = dfl.loc[dfl["home"] == t, "home_goals"].sum() + dfl.loc[dfl["away"] == t, "away_goals"].sum()
        ga = dfl.loc[dfl["home"] == t, "away_goals"].sum() + dfl.loc[dfl["away"] == t, "home_goals"].sum()
        att[t] = (gf / max(1, played)) / ((league_avg_home + league_avg_away) / 2.0)
        deff[t] = (ga / max(1, played)) / ((league_avg_home + league_avg_away) / 2.0)
        att[t] = 0.05 + 0.95 * att[t]
        deff[t] = 0.05 + 0.95 * deff[t]
    return att, deff, league_avg_home, league_avg_away

def score_matrix(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    H = np.arange(0, max_goals + 1); A = np.arange(0, max_goals + 1)
    ph = np.exp(-lam_home) * np.power(lam_home, H) / np.vectorize(math.factorial)(H)
    pa = np.exp(-lam_away) * np.power(lam_away, A) / np.vectorize(math.factorial)(A)
    M = np.outer(ph, pa)
    return M / M.sum()

def totals_distribution(M: np.ndarray) -> np.ndarray:
    max_goals = M.shape[0] - 1; Smax = 2 * max_goals
    pS = np.zeros(Smax + 1)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            pS[h + a] += M[h, a]
    return pS / pS.sum()

def diff_distribution(M: np.ndarray) -> Dict[int, float]:
    max_goals = M.shape[0] - 1
    p = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            d = h - a
            p[d] = p.get(d, 0.0) + M[h, a]
    s = sum(p.values())
    if s > 0:
        for k in list(p.keys()):
            p[k] /= s
    return p

def ou_probs(pS: np.ndarray, line: float) -> Tuple[float, float, float]:
    if float(line).is_integer():
        L = int(line); p_push = pS[L] if 0 <= L < len(pS) else 0.0
        p_over = pS[L+1:].sum(); p_under = pS[:L].sum()
    else:
        p_push = 0.0; th = math.floor(line)
        p_over = pS[th+1:].sum(); p_under = pS[:th+1].sum()
    return p_over, p_under, p_push

def btts_probs(M: np.ndarray) -> Tuple[float, float]:
    p_ng = M[0,:].sum() + M[:,0].sum() - M[0,0]
    return 1.0 - p_ng, p_ng

def handicap_home_probs(diff_p: Dict[int, float], handicap: float) -> Tuple[float, float, float]:
    eps = 1e-9; p_win = p_push = p_lose = 0.0
    integer_line = abs(handicap - round(handicap)) < 1e-9
    for d, p in diff_p.items():
        x = d - handicap
        if integer_line and abs(x) < eps: p_push += p
        elif x > 0: p_win += p
        else: p_lose += p
    tot = p_win + p_push + p_lose
    if tot > 0: p_win, p_push, p_lose = p_win/tot, p_push/tot, p_lose/tot
    return p_win, p_push, p_lose

def predict_1x2_poisson(df: pd.DataFrame, home: str, away: str, last_n_matches: int = 10) -> MatchupResult:
    att, deff, lavg_h, lavg_a = build_team_rates_football(df, last_n_matches)
    lam_home = lavg_h * att.get(home, 1.0) * deff.get(away, 1.0)
    lam_away = lavg_a * att.get(away, 1.0) * deff.get(home, 1.0)
    M = score_matrix(lam_home, lam_away, max_goals=10)
    p_home = np.tril(M, -1).sum(); p_draw = np.trace(M); p_away = np.triu(M, 1).sum()
    return MatchupResult(
        prob_home=p_home, prob_draw=p_draw, prob_away=p_away,
        fair_odds_home=(1.0 / max(1e-9, p_home)),
        fair_odds_draw=(1.0 / max(1e-9, p_draw)),
        fair_odds_away=(1.0 / max(1e-9, p_away)),
        lam_home=lam_home, lam_away=lam_away
    )

# ---------- MULTIPLA / SISTEMI ----------
def leg_probability(dfv: pd.DataFrame, home: str, away: str, last_n_matches: int,
                    market: str, param: Optional[float] = None) -> Tuple[float, str]:
    res = predict_1x2_poisson(dfv, home, away, last_n_matches=last_n_matches)
    M = score_matrix(res.lam_home, res.lam_away, max_goals=8)
    pS = totals_distribution(M)
    if market == "1X2-1":   return res.prob_home, f"{home} vincente"
    if market == "1X2-X":   return res.prob_draw, "Pareggio"
    if market == "1X2-2":   return res.prob_away, f"{away} vincente"
    if market == "DC-1X":   return res.prob_home + res.prob_draw, "Doppia 1X"
    if market == "DC-12":   return res.prob_home + res.prob_away, "Doppia 12"
    if market == "DC-X2":   return res.prob_draw + res.prob_away, "Doppia X2"
    if market == "BTTS-GG":
        pgg, _ = btts_probs(M); return pgg, "BTTS (GG)"
    if market == "BTTS-NG":
        _, png = btts_probs(M); return png, "BTTS (NG)"
    if market in ("OU-Over","OU-Under"):
        if param is None: param = 2.5
        pov, pun, _ = ou_probs(pS, float(param))
        return (pov, f"Over {param}") if market=="OU-Over" else (pun, f"Under {param}")
        # --- DNB (rimborso in caso di pareggio) ---
    if market == "DNB-1":
        # vince casa = win; pareggio = push
        return res.prob_home, "DNB Casa (rimborso se X)"
    if market == "DNB-2":
        # vince trasferta = win; pareggio = push
        return res.prob_away, "DNB Trasferta (rimborso se X)"
    raise ValueError("Market non supportato")

def fair_from_prob(p: float) -> float:
    return 1.0 / max(1e-12, p)

def acca_probability(legs: List[Dict]) -> float:
    p = 1.0
    for leg in legs:
        p *= float(leg["prob"])
    return p

SYSTEM_PRESETS = {
    "Trixie (3)": {2: 3, 3: 1},
    "Patent (3)": {1: 3, 2: 3, 3: 1},
    "Yankee (4)": {2: 6, 3: 4, 4: 1},
    "Canadian / Super Yankee (5)": {2:10, 3:10, 4:5, 5:1},
    "Heinz (6)": {2:15, 3:20, 4:15, 5:6, 6:1},
    "Super Heinz (7)": {2:21, 3:35, 4:35, 5:21, 6:7, 7:1},
    "Goliath (8)": {2:28, 3:56, 4:70, 5:56, 6:28, 7:8, 8:1},
}

def build_combos(legs: List[Dict], sizes: Dict[int,int]) -> List[Dict]:
    out = []
    n = len(legs)
    for k in sorted(sizes.keys()):
        if k>n: continue
        from itertools import combinations as _comb
        for combo in _comb(range(n), k):
            sel = [legs[i] for i in combo]
            p = 1.0; o = 1.0
            names = []
            for s in sel:
                p *= float(s["prob"]); o *= float(s["odds"]); names.append(s["label"])
            out.append({"type": f"{k}-fold","legs": sel,"combo_label": " + ".join(names),"prob": p,"odds": o})
    return out

def ev_value(p: float, odds: float, stake: float, commission: float = 0.0) -> float:
    payoff = odds * (1.0 - commission) - 1.0
    return stake * (p * payoff - (1.0 - p))

# --- Ultimi match / cruscotto helpers ----------------------------------------
def recent_matches(dfv: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    dft = dfv[(dfv["home"]==team) | (dfv["away"]==team)].sort_values("date", ascending=False).head(n).copy()
    rows = []
    for _, r in dft.iterrows():
        home, away = r["home"], r["away"]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        venue = "Casa" if home==team else "Trasferta"
        oppo  = away if home==team else home
        gf = hg if home==team else ag
        ga = ag if home==team else hg
        esito = "V" if gf>ga else ("N" if gf==ga else "P")
        rows.append({"Data": r["date"].date() if pd.notnull(r["date"]) else "",
                     "Sede": venue, "Avversario": oppo, "Risultato": f"{hg}-{ag}",
                     "Esito": esito, "GF": gf, "GS": ga})
    return pd.DataFrame(rows)

def wdl_pie(df_last: pd.DataFrame, height=260):
    v = int((df_last["Esito"]=="V").sum())
    n = int((df_last["Esito"]=="N").sum())
    p = int((df_last["Esito"]=="P").sum())
    return donut(["Vittorie","Pareggi","Sconfitte"], [v,n,p], "Ultime partite", height=height, showlegend=False)

def make_commentary(home: str, away: str, res: MatchupResult, pS: np.ndarray, M: np.ndarray) -> str:
    p1, px, p2 = res.prob_home, res.prob_draw, res.prob_away
    fav = max([(p1, f"{home} (1)"), (px, "Pareggio (X)"), (p2, f"{away} (2)")], key=lambda x: x[0])
    p_over25 = pS[3:].sum(); p_under25 = pS[:3].sum()
    p_btts = 1.0 - (M[0,:].sum() + M[:,0].sum() - M[0,0])
    max_goals = M.shape[0] - 1
    scores = sorted([(M[h,a], f"{h}-{a}") for h in range(max_goals+1) for a in range(max_goals+1)], reverse=True)[:3]

    if p_over25 >= 0.62: ou_tag = "partita da **Over 2.5**"
    elif p_under25 >= 0.62: ou_tag = "gara da **Under 2.5**"
    else: ou_tag = "totale gol **equilibrato** vicino alla 2.5"

    if p_btts >= 0.60: btts_tag = "alta probabilità di **BTTS: Sì**"
    elif p_btts <= 0.40: btts_tag = "propensione a **BTTS: No**"
    else: btts_tag = "BTTS **incerto**"

    if fav[1].endswith("(1)"):
        fav_line = f"Favorita: **{home}** (≈ {p1*100:.1f}%)."
        dnb_tip = f"**DNB {home}** consigliata (rimborso se X) dato il {p1*100:.1f}% vs pareggio {px*100:.1f}%."
    elif fav[1].endswith("(2)"):
        fav_line = f"Favorita: **{away}** (≈ {p2*100:.1f}%)."
        dnb_tip = f"**DNB {away}** consigliata (rimborso se X) dato il {p2*100:.1f}% vs pareggio {px*100:.1f}%."
    else:
        fav_line = f"Esito più probabile: **Pareggio** (≈ {px*100:.1f}%)."
        dnb_tip = "Match da **Doppia Chance**; su DNB nessuna delle due spicca chiaramente."

    gap = sorted([p1, px, p2], reverse=True)
    gap_val = gap[0] - gap[1]
    conf = "indicazione piuttosto **netta**" if gap_val>=0.15 else ("indicazione **moderata**" if gap_val>=0.07 else "situazione **molto equilibrata**")

    cs_str = ", ".join([f"{s} ({p*100:.1f}%)" for p, s in scores])
    md = []
    md.append("### 📝 Commento automatico")
    md.append(f"- {fav_line} ({conf}).")
    md.append(f"- Totali: {ou_tag} (Over ≈ **{p_over25*100:.1f}%**, Under ≈ **{p_under25*100:.1f}%**).")
    md.append(f"- Gol: {btts_tag} (BTTS Sì ≈ **{p_btts*100:.1f}%**).")
    md.append(f"- xG attesi: **{home} {res.lam_home:.2f}** vs **{away} {res.lam_away:.2f}** (totale **{(res.lam_home+res.lam_away):.2f}**).")
    md.append(f"- Correct score più probabili: {cs_str}.")
    md.append(f"- DNB (rimborso in caso di pareggio): {dnb_tip}")
    md.append("\n> Nota: modello Poisson su ultime N gare. All’inizio stagione la **varianza** è alta; aumenta la precisione col crescere del campione.")
    return "\n".join(md)

# ====================== RENDER CRUSCOTTO (Calcio) ============================
def render_match_dashboard(dfv: pd.DataFrame, home: str, away: str,
                           res: MatchupResult, pS: np.ndarray,
                           last_n_form: int = 5):
    st.subheader(f"🎛️ Cruscotto — {home} vs {away}")

    gcol1, gcol2, gcol3 = st.columns(3)
    with gcol1:
        st.plotly_chart(gauge_percent("Prob. 1 (Casa)", res.prob_home*100), use_container_width=True, key="gauge_home")
        st.caption(f"Fair: {res.fair_odds_home:.2f}")
    with gcol2:
        st.plotly_chart(gauge_percent("Prob. X", res.prob_draw*100, green_from=60), use_container_width=True, key="gauge_draw")
        st.caption(f"Fair: {res.fair_odds_draw:.2f}")
    with gcol3:
        st.plotly_chart(gauge_percent("Prob. 2 (Trasf.)", res.prob_away*100), use_container_width=True, key="gauge_away")
        st.caption(f"Fair: {res.fair_odds_away:.2f}")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.plotly_chart(donut(["1","X","2"], [res.prob_home, res.prob_draw, res.prob_away], "Distribuzione esiti (modello)"),
                        use_container_width=True, key="donut_1x2")
    with d2:
        pov25 = pS[3:].sum(); pun25 = pS[:3].sum()
        st.plotly_chart(donut(["Under 2.5","Over 2.5"], [pun25, pov25], "Under/Over 2.5"),
                        use_container_width=True, key="donut_ou25")
    with d3:
        st.metric("xG attese Casa", f"{res.lam_home:.2f}")
        st.metric("xG attese Trasferta", f"{res.lam_away:.2f}")
        st.metric("xG Totali", f"{(res.lam_home+res.lam_away):.2f}")

    st.divider()

    df_h = recent_matches(dfv, home, n=last_n_form)
    df_a = recent_matches(dfv, away, n=last_n_form)
    st.markdown(f"### Forma — ultimi {last_n_form} match")

    t1, t2 = st.columns(2)
    with t1: st.markdown(f"**{home}**")
    with t2: st.markdown(f"**{away}**")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        if not df_h.empty: st.plotly_chart(wdl_pie(df_h, height=260), use_container_width=True, key="wdl_home")
        else: st.caption("Nessuna partita recente trovata.")
    with r1c2:
        if not df_a.empty: st.plotly_chart(wdl_pie(df_a, height=260), use_container_width=True, key="wdl_away")
        else: st.caption("Nessuna partita recente trovata.")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        if not df_h.empty:
            gf_h, gs_h = df_h["GF"].sum(), df_h["GS"].sum()
            st.plotly_chart(donut(["Gol Fatti","Gol Subiti"], [gf_h, gs_h], "Gol (ultimi)", height=260, showlegend=False),
                            use_container_width=True, key="goals_home")
    with r2c2:
        if not df_a.empty:
            gf_a, gs_a = df_a["GF"].sum(), df_a["GS"].sum()
            st.plotly_chart(donut(["Gol Fatti","Gol Subiti"], [gf_a, gs_a], "Gol (ultimi)", height=260, showlegend=False),
                            use_container_width=True, key="goals_away")

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        if not df_h.empty: st.dataframe(df_h, use_container_width=True, height=200)
    with r3c2:
        if not df_a.empty: st.dataframe(df_a, use_container_width=True, height=200)

        # C) Distribuzione gol (modello) — totali + casa + trasferta
    # Totali (da pS)
    df_tot = pd.DataFrame({"Gol totali": np.arange(len(pS)), "Prob": pS})
    fig_tot = go.Figure([go.Bar(
        x=df_tot["Gol totali"], y=df_tot["Prob"]*100.0,
        text=[f"{p*100:.1f}%" for p in df_tot["Prob"]],
        textposition="outside", name="Prob %"
    )])
    fig_tot.update_layout(title="Gol totali (modello Poisson)", yaxis_title="%")

    # Marginali di squadra (Poisson univariata con λ stimato)
    max_k = 8
    k_vals = np.arange(0, max_k+1)

    pm_home = np.exp(-res.lam_home) * np.array([res.lam_home**k / math.factorial(k) for k in k_vals])
    pm_home = pm_home / pm_home.sum() if pm_home.sum()>0 else pm_home
    pm_away = np.exp(-res.lam_away) * np.array([res.lam_away**k / math.factorial(k) for k in k_vals])
    pm_away = pm_away / pm_away.sum() if pm_away.sum()>0 else pm_away

    fig_home = go.Figure([go.Bar(
        x=k_vals, y=pm_home*100.0,
        text=[f"{p*100:.1f}%" for p in pm_home],
        textposition="outside", name=f"{home}"
    )])
    fig_home.update_layout(title=f"{home} — distribuzione gol (modello)", yaxis_title="%")

    fig_away = go.Figure([go.Bar(
        x=k_vals, y=pm_away*100.0,
        text=[f"{p*100:.1f}%" for p in pm_away],
        textposition="outside", name=f"{away}"
    )])
    fig_away.update_layout(title=f"{away} — distribuzione gol (modello)", yaxis_title="%")

    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(fig_tot,  use_container_width=True, key="bar_totals_dash")
    with c2: st.plotly_chart(fig_home, use_container_width=True, key="bar_home_dash")
    with c3: st.plotly_chart(fig_away, use_container_width=True, key="bar_away_dash")

    st.divider()
    st.markdown(make_commentary(home, away, res, pS, score_matrix(res.lam_home, res.lam_away)))
    st.caption("⚠️ Modello Poisson semplice; usare come guida, non come verità assoluta.")

# ============= SIDEBAR: PROFILO & CONTO ======================================
with st.sidebar:
    st.header("👤 Profilo & conto (virtuale)")
    username = st.text_input("Nome utente", value=st.session_state.get("username",""))
    colp1, colp2 = st.columns([1,1])
    with colp1:
        load_btn = st.button("➜ Carica / Crea")
    with colp2:
        imp_file = st.file_uploader("Import profilo (.json)", type=["json"], label_visibility="collapsed")

    if load_btn and username.strip()=="":
        st.warning("Inserisci un nome utente.")
    if imp_file is not None:
        try:
            imported = import_profile_json(imp_file)
            st.session_state["profile"] = imported
            st.session_state["username"] = imported["username"]
            save_profile(imported)
            st.success(f"Profilo '{imported['username']}' importato.")
        except Exception as e:
            st.error(f"Import fallito: {e}")

    profile = st.session_state.get("profile")
    if (load_btn and username.strip()):
        profile = load_profile(username.strip())
        st.session_state["profile"] = profile
        st.session_state["username"] = username.strip()
        st.success(f"Profilo attivo: {profile['username']}")

    if profile is None:
        st.info("Crea o carica un profilo per attivare il conto virtuale e lo storico.")
        st.stop()

    st.markdown(f"**Utente:** `{profile['username']}`  •  **Valuta:** {profile.get('currency','EUR')}")
    st.metric("Saldo attuale", f"{profile['balance']:.2f} €")

    if profile["initial_balance"] == 0.0 and profile["balance"] == 0.0:
        init = st.number_input("Saldo iniziale (€)", min_value=0.0, value=100.0, step=10.0, help="Imposta il bankroll iniziale (una tantum).")
        if st.button("Imposta saldo iniziale"):
            profile["initial_balance"] = float(init)
            profile["balance"] = float(init)
            save_profile(profile)
            st.success(f"Saldo iniziale impostato a {init:.2f} €")

    dep_amt = st.number_input("Deposita (+)", min_value=0.0, value=0.0, step=10.0)
    wd_amt  = st.number_input("Preleva (−)", min_value=0.0, value=0.0, step=10.0)
    colp3, colp4, colp5 = st.columns(3)
    with colp3:
        if st.button("➕ Deposita"):
            if dep_amt>0:
                profile["balance"] += float(dep_amt)
                save_profile(profile); do_rerun()
    with colp4:
        if st.button("➖ Preleva"):
            if wd_amt>0 and profile["balance"]>=wd_amt:
                profile["balance"] -= float(wd_amt)
                save_profile(profile); do_rerun()
    with colp5:
        exp_json = export_profile(profile)
        st.download_button("⬇️ Esporta profilo", data=exp_json, file_name=f"{profile['username']}.json", mime="application/json")

# ====================== DATI CALCIO (download) ===============================
with st.sidebar:
    st.header("⚙️ Dati Calcio")
    last_n = st.slider("Finestra ultime gare per squadra", 5, 18, 10)
    popular_leagues = OrderedDict({
        "England — Premier League (E0)": "E0",
        "England — Championship (E1)": "E1",
        "England — League One (E2)": "E2",
        "England — League Two (E3)": "E3",
        "Scotland — Premiership (SC0)": "SC0",
        "Scotland — Championship (SC1)": "SC1",
        "Scotland — League One (SC2)": "SC2",
        "Scotland — League Two (SC3)": "SC3",
        "Italy — Serie A (I1)": "I1",
        "Italy — Serie B (I2)": "I2",
        "Spain — La Liga (SP1)": "SP1",
        "Spain — Segunda División (SP2)": "SP2",
        "Germany — Bundesliga (D1)": "D1",
        "Germany — 2. Bundesliga (D2)": "D2",
        "France — Ligue 1 (F1)": "F1",
        "France — Ligue 2 (F2)": "F2",
        "Netherlands — Eredivisie (N1)": "N1",
        "Belgium — Pro League (B1)": "B1",
        "Portugal — Primeira Liga (P1)": "P1",
        "Turkey — Süper Lig (T1)": "T1",
        "Greece — Super League (G1)": "G1",
    })
    league_label = st.selectbox("Lega", list(popular_leagues.keys()), index=0)
    custom_code = st.text_input("Codice lega personalizzato (E0, I1, SP2, ...)", value="").strip().upper()
    league_code = custom_code if custom_code else popular_leagues[league_label]
    suggest = current_fd_code()
    seasons = sorted(set(["2324", "2425", "2526", suggest]))
    default_idx = seasons.index(suggest) if suggest in seasons else len(seasons)-1
    season_code = st.selectbox("Stagione (codice 4 cifre)", seasons, index=default_idx)

# --- scarico
code = (season_code or "").strip()
if len(code) != 4 or not code.isdigit():
    st.error("Seleziona una stagione valida a 4 cifre."); st.stop()
url_https = f"https://www.football-data.co.uk/mmz4281/{code}/{league_code}.csv"
st.caption(f"📥 Scarico: {url_https}")
try:
    df_raw = fetch_fd_csv(url_https)
except Exception as e_last:
    st.error(f"Download CSV fallito: {e_last}"); st.stop()

dfv = normalize_football_columns(df_raw)
dfv["competition"] = league_label if not custom_code else f"{league_label} / {league_code}"

required = {"home","away","home_goals","away_goals","date"}
if not required.issubset(set(dfv.columns)):
    st.error(f"Colonne richieste assenti: {sorted(list(required - set(dfv.columns)))}"); st.stop()

# --- selezione match per analisi
teams = sorted(pd.unique(pd.concat([dfv["home"], dfv["away"]])).tolist())
c1, c2 = st.columns(2)
with c1: home = st.selectbox("Squadra Casa", teams, index=0)
with c2: away = st.selectbox("Squadra Trasferta", teams, index=min(1, len(teams)-1))

res = predict_1x2_poisson(dfv, home, away, last_n_matches=last_n)
M = score_matrix(res.lam_home, res.lam_away, max_goals=8)
pS = totals_distribution(M)
diff_p = diff_distribution(M)

# ====================== TABS PRINCIPALI ======================================
tabs = st.tabs([
    "Cruscotto", "1X2", "Doppia Chance & DNB", "Over/Under & BTTS",
    "Correct Score", "Handicap", "Multipla", "Sistemi", "💼 Schedine & Storico"
])

# ---- Tab 0
with tabs[0]:
    render_match_dashboard(dfv, home, away, res, pS, last_n_form=last_n)

# ---- Tab 1: 1X2
with tabs[1]:
    st.subheader("Probabilità stimate 1X2")
    cols = st.columns(3)
    cols[0].metric("1 (Casa)", f"{res.prob_home*100:.1f}%", help=f"Fair ≈ {res.fair_odds_home:.2f}")
    cols[1].metric("X (Pareggio)", f"{res.prob_draw*100:.1f}%", help=f"Fair ≈ {res.fair_odds_draw:.2f}")
    cols[2].metric("2 (Trasferta)", f"{res.fair_odds_away*0+res.prob_away*100:.1f}%", help=f"Fair ≈ {res.fair_odds_away:.2f}")

# ---- Tab 2: DC & DNB
with tabs[2]:
    st.subheader("Doppia Chance")
    p1x = res.prob_home + res.prob_draw
    p12 = res.prob_home + res.prob_away
    px2 = res.prob_draw + res.prob_away
    c1,c2,c3 = st.columns(3)
    c1.metric("1X", f"{p1x*100:.1f}%", help=f"Fair ≈ {1/max(1e-9,p1x):.2f}")
    c2.metric("12", f"{p12*100:.1f}%", help=f"Fair ≈ {1/max(1e-9,p12):.2f}")
    c3.metric("X2", f"{px2*100:.1f}%", help=f"Fair ≈ {1/max(1e-9,px2):.2f}")

    st.subheader("Draw No Bet (DNB) — solo statistica")
    st.info("Suggerimento statistico (senza quote): punta DNB sul lato con P(vittoria) maggiore; rimborso se pareggio.")
    st.write(
        "**Consiglio DNB:** " +
        (f"**{home}** (home) — P(win)≈{res.prob_home*100:.1f}%" if res.prob_home>res.prob_away
         else f"**{away}** (away) — P(win)≈{res.prob_away*100:.1f}%")
        + f" | P(draw)≈{res.prob_draw*100:.1f}%"
    )

# ---- Tab 3: O/U & BTTS
with tabs[3]:
    st.subheader("Over/Under & BTTS")
    default_lines=[0.5,1.5,2.5,3.5,4.5,5.5]
    ou_rows=[]
    for L in default_lines:
        pov,pun,ppu = ou_probs(pS, L)
        ou_rows.append({"Linea":L,"P(Over)":pov,"P(Under)":pun,"P(Push)":ppu,
                        "Fair Over":1/max(1e-9,pov),"Fair Under":1/max(1e-9,pun)})
    df_ou = pd.DataFrame(ou_rows)
    st.dataframe(df_ou.style.format(
        {"P(Over)":"{:.1%}","P(Under)":"{:.1%}","P(Push)":"{:.1%}","Fair Over":"{:.2f}","Fair Under":"{:.2f}"}
    ), use_container_width=True)

    st.subheader("BTTS")
    pgg,png = btts_probs(M)
    st.dataframe(pd.DataFrame({
        "Selezione":["GG","NG"], "Prob":[pgg,png]
    }).style.format({"Prob":"{:.1%}"}), use_container_width=True)

# ---- Tab 4: Correct Score
with tabs[4]:
    st.subheader("Correct Score — Top probabilità")
    max_goals = M.shape[0]-1
    records=[{"Score":f"{h}-{a}","Prob":M[h,a]} for h in range(max_goals+1) for a in range(max_goals+1)]
    dfcs = pd.DataFrame(records).sort_values("Prob", ascending=False).head(10)
    st.dataframe(dfcs.style.format({"Prob":"{:.1%}"}), use_container_width=True)

# ---- Tab 5: Handicap
with tabs[5]:
    st.subheader("Handicap Asiatico (Home)")
    h = st.selectbox("Linea", options=[-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0], index=4)
    pW,pP,pL = handicap_home_probs(diff_p, float(h))
    c1,c2,c3 = st.columns(3)
    c1.metric("Win %", f"{pW*100:.1f}%"); c2.metric("Push %", f"{pP*100:.1f}%"); c3.metric("Lose %", f"{pL*100:.1f}%")

# ---- Tab 6: Multipla (creazione + salvataggio)
with tabs[6]:
    st.subheader("🧮 Multipla — costruisci e salva")
    if "acca_legs" not in st.session_state:
        st.session_state.acca_legs = []

    colm1, colm2 = st.columns(2)
    with colm1:
        home_m = st.selectbox("Casa", teams, key="acca_home")
        away_m = st.selectbox("Trasferta", [t for t in teams if t != home_m], key="acca_away")
    with colm2:
        market = st.selectbox(
            "Mercato",
            ["1X2-1","1X2-X","1X2-2","DC-1X","DC-12","DC-X2","OU-Over","OU-Under","BTTS-GG","BTTS-NG","DNB-1","DNB-2"],
            key="acca_market"
        )
        line = None
        if market in ("OU-Over","OU-Under"):
            line = st.number_input("Linea O/U", min_value=0.5, max_value=8.5, value=2.5, step=0.5, key="acca_line")
        odds_in = st.number_input("Quota (decimale)", min_value=1.01, value=1.90, step=0.01, key="acca_odds")

    if st.button("➕ Aggiungi selezione"):
        p, desc = leg_probability(dfv, home_m, away_m, last_n, market, line)

        # Fair odds: per DNB usiamo p_win / (1 - p_push)
        if market in ("DNB-1", "DNB-2"):
            _res_dnb = predict_1x2_poisson(dfv, home_m, away_m, last_n_matches=last_n)
            p_push = _res_dnb.prob_draw
            fair = 1.0 / max(1e-12, (p / max(1e-12, (1.0 - p_push))))
        else:
            fair = fair_from_prob(p)

        st.session_state.acca_legs.append({
            "match": f"{home_m} vs {away_m}",
            "label": f"{home_m}-{away_m} | {desc}",
            "market": market,
            "line": line,
            "odds": float(odds_in),
            "prob": float(p),
            "fair": float(fair),
        })

    if st.session_state.acca_legs:
        df_acca = pd.DataFrame(st.session_state.acca_legs)
        st.dataframe(df_acca[["label","odds","prob","fair"]]
                     .rename(columns={"label":"Selezione","odds":"Quota","prob":"Prob.","fair":"Fair"})
                     .style.format({"Quota":"{:.2f}","Prob.":"{:.1%}","Fair":"{:.2f}"}),
                     use_container_width=True)

        cA, cB = st.columns(2)
        with cA:
            tot_stake = st.number_input("Stake totale (€)", min_value=1.0, value=10.0, step=1.0, key="acca_stake")
        with cB:
            save_btn = st.button("💾 Salva come MULTIPLA (scarica dal saldo)")

        acca_p = acca_probability(st.session_state.acca_legs)
        acca_odds = float(np.prod([x["odds"] for x in st.session_state.acca_legs]))
        potential_win = float(tot_stake * acca_odds)
        st.metric("Vincita potenziale", f"{potential_win:.2f} €", help="Stake × Quota combinata")
        st.metric("Prob. multipla", f"{acca_p*100:.1f}%")
        st.metric("Quota combinata", f"{acca_odds:.2f}")

        if save_btn:
            if profile["balance"] < tot_stake:
                st.error("Saldo insufficiente.")
            else:
                bet_id = str(uuid.uuid4())
                bet = {
                    "id": bet_id,
                    "type": "multiple",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "stake": float(tot_stake),
                    "combined_odds": float(acca_odds),
                    "potential_return": float(tot_stake * acca_odds),
                    "status": "open",
                    "legs": st.session_state.acca_legs,
                    "notes": ""
                }
                profile["bets"].append(bet)
                profile["balance"] -= float(tot_stake)
                save_profile(profile)
                st.session_state.acca_legs = []
                st.success(f"Multipla salvata (ID: {bet_id}). Stake scalato dal saldo.")
                do_rerun()
    else:
        st.info("Aggiungi almeno una selezione.")

# ---- Tab 7: Sistemi (creazione + salvataggio)
with tabs[7]:
    st.subheader("🧩 Sistemi ridotti — crea e salva")
    if "sys_legs" not in st.session_state:
        st.session_state.sys_legs = []

    colx1, colx2 = st.columns(2)
    with colx1:
        home_s = st.selectbox("Casa", teams, key="sys_home")
        away_s = st.selectbox("Trasferta", [t for t in teams if t != home_s], key="sys_away")
    with colx2:
        market_s = st.selectbox(
            "Mercato",
            ["1X2-1","1X2-X","1X2-2","DC-1X","DC-12","DC-X2","OU-Over","OU-Under","BTTS-GG","BTTS-NG","DNB-1","DNB-2"],
            key="sys_market"
        )
        line_s = None
        if market_s in ("OU-Over","OU-Under"):
            line_s = st.number_input("Linea O/U (sistemi)", min_value=0.5, max_value=8.5, value=2.5, step=0.5, key="sys_line")
        odds_s = st.number_input("Quota (decimale) (sistemi)", min_value=1.01, value=1.90, step=0.01, key="sys_odds")

    # Usa un'etichetta diversa per evitare conflitti con il bottone del Tab 6
    if st.button("➕ Aggiungi selezione al sistema", key="btn_add_sys"):
        p, desc = leg_probability(dfv, home_s, away_s, last_n, market_s, line_s)

        # Fair odds: per DNB usiamo p_win / (1 - p_push)
        if market_s in ("DNB-1", "DNB-2"):
            _res_dnb = predict_1x2_poisson(dfv, home_s, away_s, last_n_matches=last_n)
            p_push = _res_dnb.prob_draw
            fair = 1.0 / max(1e-12, (p / max(1e-12, (1.0 - p_push))))
        else:
            fair = fair_from_prob(p)

        st.session_state.sys_legs.append({
            "match": f"{home_s} vs {away_s}",
            "label": f"{home_s}-{away_s} | {desc}",
            "market": market_s,
            "line": line_s,
            "odds": float(odds_s),
            "prob": float(p),
            "fair": float(fair),
        })

    if st.session_state.sys_legs:
        nlegs = len(st.session_state.sys_legs)
        st.info(f"Selezioni nel sistema: **{nlegs}** (min 3, max 8)")
        df_sys_sel = pd.DataFrame(st.session_state.sys_legs)
        st.dataframe(df_sys_sel[["label","odds","prob","fair"]]
                     .rename(columns={"label":"Selezione","odds":"Quota","prob":"Prob.","fair":"Fair"})
                     .style.format({"Quota":"{:.2f}","Prob.":"{:.1%}","Fair":"{:.2f}"}),
                     use_container_width=True)

        preset_name = st.selectbox("Preset sistema", list(SYSTEM_PRESETS.keys())+["Personalizzato"], key="preset_sys")
        if preset_name != "Personalizzato":
            sizes = SYSTEM_PRESETS[preset_name]
            required_n = int(preset_name.split("(")[-1].replace(")",""))
            if nlegs != required_n:
                st.warning(f"Questo preset richiede **{required_n}** selezioni. Attuali: **{nlegs}**.")
        else:
            colk1, colk2 = st.columns(2)
            min_k = colk1.slider("Da n-fold", 2, min(6, nlegs), 2, key="sys_min_k")
            max_k = colk2.slider("A n-fold", min_k, min(8, nlegs), min(4, nlegs), key="sys_max_k")
            sizes = {k: None for k in range(min_k, max_k+1)}

        tickets = build_combos(st.session_state.sys_legs, sizes)
        nt = len(tickets)
        if nt == 0:
            st.warning("Nessun ticket generato con i parametri attuali.")
        else:
            total_stake_sys = st.number_input("Stake totale sistema (€)", min_value=1.0, value=10.0, step=1.0, key="sys_total_stake")
            stake_per = total_stake_sys / nt
            rows = []
            for t in tickets:
                rows.append({"Ticket": t["type"], "Selezioni": t["combo_label"], "Quota": t["odds"], "Prob.": t["prob"], "Stake": stake_per})
            df_tk = pd.DataFrame(rows)
            st.dataframe(df_tk.style.format({"Quota":"{:.2f}","Prob.":"{:.1%}","Stake":"{:.2f}"}), use_container_width=True)
            # Vincita potenziale se TUTTI i subticket vincono
            potential_sys_return = float(sum(stake_per * float(t["odds"]) for t in tickets))

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Ticket generati", f"{nt}")
            with m2:
                st.metric("Vincita potenziale (tutti vincenti)", f"{potential_sys_return:.2f} €")

            if st.button("💾 Salva SISTEMA (scarica stake dal saldo)"):
                if profile["balance"] < total_stake_sys:
                    st.error("Saldo insufficiente.")
                else:
                    bet_id = str(uuid.uuid4())
                    # salvo sistema come bet con sottoticket
                    subtickets = []
                    for t in tickets:
                        subtickets.append({
                            "id": str(uuid.uuid4()),
                            "type": t["type"],
                            "legs": t["legs"],
                            "combo_label": t["combo_label"],
                            "odds": float(t["odds"]),
                            "prob": float(t["prob"]),
                            "stake": float(stake_per),
                            "status": "open",
                            "return_amount": 0.0
                        })
                    bet = {
                        "id": bet_id,
                        "type": "system",
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "stake": float(total_stake_sys),
                        "status": "open",
                        "subtickets": subtickets,
                        "notes": ""
                    }
                    profile["bets"].append(bet)
                    profile["balance"] -= float(total_stake_sys)
                    save_profile(profile)
                    st.session_state.sys_legs = []
                    st.success(f"Sistema salvato (ID: {bet_id}). Stake scalato dal saldo.")
                    do_rerun()
    else:
        st.info("Aggiungi almeno 3 selezioni.")

# ---- Tab 8: Storico & Verifica esiti
with tabs[8]:
    st.subheader("💼 Schedine & storico")
    bets = profile.get("bets", [])
    if not bets:
        st.info("Nessuna schedina salvata finora.")
    else:
        # Mappa ID bet -> indice reale nell’array profile["bets"]
        bet_idx_map = {b["id"]: i for i, b in enumerate(bets)}

        # Filtro
        status_filter = st.multiselect("Stato", ["open","won","lost","void","partial"],
                                       default=["open","won","lost","void","partial"])
        type_filter = st.multiselect("Tipo", ["single","multiple","system"],
                                     default=["multiple","system","single"])
        filtered = [b for b in bets if b["status"] in status_filter and b["type"] in type_filter]

        # Render
        for b in sorted(filtered, key=lambda x: x.get("created_at",""), reverse=True):
            real_i = bet_idx_map[b["id"]]
            real_b = profile["bets"][real_i]

            with st.expander(f"[{real_b['type'].upper()}] ID: {real_b['id']} • creato: {real_b.get('created_at','')} • stato: {real_b['status']}"):
                if real_b["type"] == "multiple":
                    st.write(
                        f"Stake: **{real_b['stake']:.2f} €**  |  "
                        f"Quota: **{real_b['combined_odds']:.2f}**  |  "
                        f"Potenziale: **{real_b['potential_return']:.2f} €**"
                    )
                    legs_df = pd.DataFrame(real_b["legs"])
                    st.dataframe(
                        legs_df[["label","odds","prob","fair"]]
                            .rename(columns={"label":"Selezione","odds":"Quota","prob":"Prob.","fair":"Fair"})
                            .style.format({"Quota":"{:.2f}","Prob.":"{:.1%}","Fair":"{:.2f}"}),
                        use_container_width=True
                    )

                    if real_b["status"] == "open":
                        colmb1, colmb2, colmb3 = st.columns(3)
                        with colmb1:
                            won = st.button("✅ Esito: VINTA", key=f"mult_win_{real_b['id']}")
                        with colmb2:
                            lost = st.button("❌ Esito: PERSA", key=f"mult_lost_{real_b['id']}")
                        with colmb3:
                            void = st.button("↩️ VOID / Rimborso", key=f"mult_void_{real_b['id']}")

                        if won or lost or void:
                            if won:
                                ret = float(real_b["stake"] * real_b["combined_odds"])
                                profile["balance"] += ret
                                real_b["status"] = "won"
                                real_b["return_amount"] = ret
                            elif lost:
                                real_b["status"] = "lost"
                                real_b["return_amount"] = 0.0
                            else:  # void
                                profile["balance"] += float(real_b["stake"])
                                real_b["status"] = "void"
                                real_b["return_amount"] = float(real_b["stake"])

                            # Riassegna nell’array e persisti
                            profile["bets"][real_i] = real_b
                            st.session_state["profile"] = profile
                            save_profile(profile)
                            do_rerun()
                    else:
                        st.write(f"Ritorno: **{real_b.get('return_amount',0.0):.2f} €**")

                elif real_b["type"] == "system":
                    st.write(f"Stake totale: **{real_b['stake']:.2f} €**  |  Sottoticket: **{len(real_b['subtickets'])}**")

                    # Tabella sottoticket
                    rows = []
                    for t in real_b["subtickets"]:
                        rows.append({
                            "ID sub": t["id"][:8],
                            "Tipo": t["type"],
                            "Quota": t["odds"],
                            "Stake": t["stake"],
                            "Prob": t["prob"],
                            "Stato": t["status"],
                            "Ritorno": t.get("return_amount", 0.0),
                            "Selezioni": t["combo_label"]
                        })
                    df_sub = pd.DataFrame(rows)
                    st.dataframe(
                        df_sub.style.format({"Quota":"{:.2f}","Stake":"{:.2f}","Prob":"{:.1%}","Ritorno":"{:.2f}"}),
                        use_container_width=True
                    )

                    # Pannello esito per ogni sub (aggiorna l'oggetto reale)
                    for j, t in enumerate(real_b["subtickets"]):
                        if t["status"] == "open":
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                if st.button(f"✅ VINTA ({t['id'][:8]})", key=f"sys_win_{t['id']}"):
                                    t["status"] = "won"
                                    t["return_amount"] = float(t["stake"] * t["odds"])
                                    real_b["subtickets"][j] = t
                                    profile["bets"][real_i] = real_b
                                    st.session_state["profile"] = profile
                                    save_profile(profile)
                                    do_rerun()
                            with c2:
                                if st.button(f"❌ PERSA ({t['id'][:8]})", key=f"sys_lost_{t['id']}"):
                                    t["status"] = "lost"
                                    t["return_amount"] = 0.0
                                    real_b["subtickets"][j] = t
                                    profile["bets"][real_i] = real_b
                                    st.session_state["profile"] = profile
                                    save_profile(profile)
                                    do_rerun()
                            with c3:
                                if st.button(f"↩️ VOID ({t['id'][:8]})", key=f"sys_void_{t['id']}"):
                                    t["status"] = "void"
                                    t["return_amount"] = float(t["stake"])
                                    real_b["subtickets"][j] = t
                                    profile["bets"][real_i] = real_b
                                    st.session_state["profile"] = profile
                                    save_profile(profile)
                                    do_rerun()

                    # Chiusura sistema: somma ritorni dei sub chiusi
                    if real_b["status"] == "open":
                        all_closed = all(t["status"] != "open" for t in real_b["subtickets"])
                        partial = any(t["status"] != "open" for t in real_b["subtickets"])
                        if all_closed:
                            total_ret = float(sum(t.get("return_amount", 0.0) for t in real_b["subtickets"]))
                            profile["balance"] += total_ret
                            real_b["status"] = "won" if total_ret > real_b["stake"] else (
                                "void" if math.isclose(total_ret, real_b["stake"], rel_tol=1e-6) else "lost"
                            )
                            real_b["return_amount"] = total_ret
                            profile["bets"][real_i] = real_b
                            st.session_state["profile"] = profile
                            save_profile(profile)
                            st.success(f"Sistema chiuso. Ritorno totale: {total_ret:.2f} €")
                            do_rerun()
                        elif partial:
                            real_b["status"] = "partial"
                            profile["bets"][real_i] = real_b
                            st.session_state["profile"] = profile
                            save_profile(profile)
                            st.info("Sistema in stato PARZIALE: chiudi tutti i sottoticket per accreditare il ritorno totale.")
                    else:
                        st.write(f"Ritorno accreditato: **{real_b.get('return_amount',0.0):.2f} €** (stato: {real_b['status']})")

                else:  # single (non usata qui ma lasciata per completezza)
                    st.write("Schedina singola.")
                    if real_b["status"] == "open":
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if st.button("✅ VINTA", key=f"sg_win_{real_b['id']}"):
                                ret = float(real_b["stake"] * real_b.get("odds", 1.0))
                                profile["balance"] += ret
                                real_b["status"] = "won"
                                real_b["return_amount"] = ret
                                profile["bets"][real_i] = real_b
                                st.session_state["profile"] = profile
                                save_profile(profile)
                                do_rerun()
                        with c2:
                            if st.button("❌ PERSA", key=f"sg_lost_{real_b['id']}"):
                                real_b["status"] = "lost"
                                real_b["return_amount"] = 0.0
                                profile["bets"][real_i] = real_b
                                st.session_state["profile"] = profile
                                save_profile(profile)
                                do_rerun()
                        with c3:
                            if st.button("↩️ VOID", key=f"sg_void_{real_b['id']}"):
                                profile["balance"] += float(real_b["stake"])
                                real_b["status"] = "void"
                                real_b["return_amount"] = float(real_b["stake"])
                                profile["bets"][real_i] = real_b
                                st.session_state["profile"] = profile
                                save_profile(profile)
                                do_rerun()

        st.divider()

        # Esportazione storico CSV
        if bets:
            out = []
            for b in bets:
                if b["type"] == "multiple":
                    out.append({
                        "id": b["id"], "type": b["type"], "created_at": b.get("created_at",""),
                        "stake": b["stake"], "status": b["status"], "return": b.get("return_amount",0.0),
                        "combined_odds": b.get("combined_odds", np.nan),
                        "legs": " | ".join([l["label"] for l in b["legs"]])
                    })
                elif b["type"] == "system":
                    out.append({
                        "id": b["id"], "type": b["type"], "created_at": b.get("created_at",""),
                        "stake": b["stake"], "status": b["status"], "return": b.get("return_amount",0.0),
                        "combined_odds": np.nan,
                        "legs": f"{len(b.get('subtickets', []))} subtickets"
                    })
                else:
                    out.append({
                        "id": b["id"], "type": b["type"], "created_at": b.get("created_at",""),
                        "stake": b["stake"], "status": b["status"], "return": b.get("return_amount",0.0),
                        "combined_odds": b.get("odds", np.nan),
                        "legs": b.get("label", "")
                    })
            df_hist = pd.DataFrame(out)
            st.dataframe(df_hist, use_container_width=True)
            csv = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Esporta storico (CSV)", data=csv,
                               file_name=f"{profile['username']}_storico.csv", mime="text/csv")

        # Reset completo (cautela)
        with st.expander("⚠️ Reset profilo"):
            st.warning("Azzera saldo e cancella tutte le schedine del profilo corrente (irreversibile).")
            if st.button("Svuota profilo (saldo=0, schedine vuote)"):
                profile["balance"] = 0.0
                profile["initial_balance"] = 0.0
                profile["bets"] = []
                st.session_state["profile"] = profile
                save_profile(profile)
                st.success("Profilo azzerato.")
                do_rerun()
                
# ============= NOTE FINALI ====================================================
with st.expander("ℹ️ Note"):
    st.markdown("""
- Conto **virtuale** con valuta fissa EUR. Ogni profilo è salvato in `./data/username.json`.
- Le schedine **Multiple** accreditano l'intero ritorno soltanto alla chiusura (vinta/persa/void).
- I **Sistemi** sono composti da sottoticket: il ritorno totale è la somma dei sottoticket a esito chiuso.
- Il modello è **statistico** e usa un Poisson semplice sulle ultime N gare: non incorpora infortuni, calendario, ecc.
- In avvio stagione l'incertezza è maggiore (**varianza** più alta); conviene preferire mercati protetti (DNB, DC).
""")
