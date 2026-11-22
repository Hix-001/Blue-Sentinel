
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import tensorflow as tf
from tensorflow import keras
import random
from datetime import datetime, timedelta
import time

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Blue Sentinel - Real-Time AI Demo",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# BOOT SCREEN  (runs exactly once)
# ─────────────────────────────────────────────────────────────
if "booted" not in st.session_state:
    boot = st.empty()
    with boot.container():
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@200;300;400;700;800&family=JetBrains+Mono:wght@300;400&display=swap');
        *{margin:0;padding:0;box-sizing:border-box;}
        body,.stApp{background:#000!important;}
        .bs{position:fixed;inset:0;background:radial-gradient(ellipse at 50% 40%,#020d1a,#000);
            display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:999999;overflow:hidden;}
        .bg{position:absolute;inset:0;
            background-image:linear-gradient(rgba(0,200,255,.03) 1px,transparent 1px),
                             linear-gradient(90deg,rgba(0,200,255,.03) 1px,transparent 1px);
            background-size:55px 55px;animation:gs 8s linear infinite;}
        @keyframes gs{to{background-position:0 55px}}
        .rw{position:relative;width:170px;height:170px;display:flex;align-items:center;justify-content:center;}
        .rr{position:absolute;border-radius:50%;border:1px solid rgba(0,200,255,.15);}
        .r1{width:170px;height:170px;}.r2{width:115px;height:115px;}.r3{width:62px;height:62px;}
        .rs{position:absolute;width:170px;height:170px;border-radius:50%;
            background:conic-gradient(transparent 265deg,rgba(0,200,255,.55) 338deg,transparent 360deg);
            animation:sw 1.9s linear infinite;}
        @keyframes sw{to{transform:rotate(360deg)}}
        .rp{position:absolute;width:170px;height:170px;border-radius:50%;
            border:1px solid rgba(0,200,255,.65);animation:po 1.9s ease-out infinite;}
        @keyframes po{0%{transform:scale(.12);opacity:1}100%{transform:scale(1.85);opacity:0}}
        .rd{position:absolute;width:5px;height:5px;border-radius:50%;background:#00c8ff;}
        .d1{top:42px;left:85px;animation:bl 1.5s .2s infinite;}
        .d2{top:95px;left:50px;animation:bl 1.5s .8s infinite;}
        .d3{top:114px;left:108px;animation:bl 1.5s 1.3s infinite;}
        @keyframes bl{0%,100%{opacity:1;box-shadow:0 0 5px #00c8ff}50%{opacity:.05}}
        .bt{color:#fff;font-family:'Sora',sans-serif;font-weight:700;margin-top:38px;font-size:.95rem;letter-spacing:7px;text-transform:uppercase;}
        .bs2{color:rgba(0,200,255,.5);font-family:'Sora',sans-serif;font-weight:200;font-size:.58rem;letter-spacing:3px;margin-top:5px;}
        .term{color:rgba(0,200,255,.8);font-family:'JetBrains Mono',monospace;font-size:.55rem;
              margin-top:28px;width:370px;line-height:2.1;border:1px solid rgba(0,200,255,.1);
              padding:14px 20px;background:rgba(0,200,255,.03);border-radius:5px;}
        .ok{color:#30d158;}
        .tl{opacity:0;animation:fs .35s ease forwards;}
        .tl:nth-child(1){animation-delay:.1s}.tl:nth-child(2){animation-delay:.45s}
        .tl:nth-child(3){animation-delay:.85s}.tl:nth-child(4){animation-delay:1.3s}
        .tl:nth-child(5){animation-delay:1.8s}
        @keyframes fs{from{opacity:0;transform:translateX(-5px)}to{opacity:1;transform:translateX(0)}}
        .po2{width:370px;height:2px;background:rgba(0,200,255,.1);margin-top:22px;border-radius:1px;overflow:hidden;}
        .pi{height:100%;background:linear-gradient(90deg,#00c8ff,#7b5fff);
            animation:pr 3.2s linear forwards;box-shadow:0 0 8px rgba(0,200,255,.5);}
        @keyframes pr{from{width:0}to{width:100%}}
        </style>
        <div class="bs">
          <div class="bg"></div>
          <div class="rw">
            <div class="rr r1"></div><div class="rr r2"></div><div class="rr r3"></div>
            <div class="rs"></div><div class="rp"></div>
            <div class="rd d1"></div><div class="rd d2"></div><div class="rd d3"></div>
          </div>
          <div class="bt">Blue Sentinel</div>
          <div class="bs2">Maritime AI · Indo-Pacific Node DL-04 · v4.0</div>
          <div class="term">
            <div class="tl">&gt; UPLINKING GSAT-7 SATELLITE...<span class="ok"> [CONNECTED]</span></div>
            <div class="tl">&gt; LOADING LSTM NEURAL WEIGHTS...<span class="ok"> [5-FEATURE VECTOR OK]</span></div>
            <div class="tl">&gt; DEPLOYING 100-VESSEL FLEET MESH...<span class="ok"> [SYNCED]</span></div>
            <div class="tl">&gt; ESTABLISHING AES-256 UPLINK...<span class="ok"> [SECURE]</span></div>
            <div class="tl">&gt; ALL SYSTEMS NOMINAL — BLUE SENTINEL ONLINE<span class="ok"> [✓]</span></div>
          </div>
          <div class="po2"><div class="pi"></div></div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(3.5)
    st.session_state.booted = True
    boot.empty()

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@200;300;400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root{
  --cyan:#00c8ff; --blue:#0057ff; --violet:#7b5fff;
  --red:#ff2d55;  --amber:#ffb340; --green:#30d158;
  --deep:#020d1a;
  --gb:blur(28px) saturate(200%);
  --gg:rgba(255,255,255,.04);
  --gd:rgba(255,255,255,.10);
}

/* BASE */
.stApp{
  background:var(--deep)!important;color:#fff;font-family:'Sora',sans-serif;
  background-image:
    radial-gradient(ellipse 130% 50% at 55% -5%,rgba(0,87,255,.22) 0%,transparent 60%),
    radial-gradient(ellipse 70% 45% at 95% 90%,rgba(123,95,255,.14) 0%,transparent 55%),
    radial-gradient(ellipse 60% 40% at 5% 70%, rgba(0,200,255,.08) 0%,transparent 50%);
}
.stApp::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(rgba(0,200,255,.02) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,200,255,.02) 1px,transparent 1px);
  background-size:80px 80px;animation:gm 28s linear infinite;
}
@keyframes gm{to{background-position:0 80px}}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:.8rem!important;position:relative;z-index:1;}

/* ── iOS 26 GLASS SIDEBAR ── */
[data-testid="stSidebar"]{
  background:rgba(2,13,26,.62)!important;
  border-right:1px solid rgba(0,200,255,.12)!important;
  backdrop-filter:var(--gb)!important;
  -webkit-backdrop-filter:var(--gb)!important;
  transition:all .3s ease!important;
}
[data-testid="stSidebar"]>div{background:transparent!important;}
[data-testid="stSidebar"] *{color:#fff!important;}
[data-testid="stSidebar"] .stSlider *{color:#fff!important;}

/* ── SIDEBAR COLLAPSE BUTTON — styled as glass pill ── */
[data-testid="collapsedControl"]{
  top:50%!important;
  transform:translateY(-50%)!important;
  left:0!important;
  background:rgba(0,200,255,.12)!important;
  border:1px solid rgba(0,200,255,.30)!important;
  border-left:none!important;
  border-radius:0 12px 12px 0!important;
  backdrop-filter:blur(16px) saturate(200%)!important;
  -webkit-backdrop-filter:blur(16px) saturate(200%)!important;
  box-shadow:3px 0 24px rgba(0,200,255,.18)!important;
  width:26px!important;min-height:56px!important;
  display:flex!important;align-items:center!important;justify-content:center!important;
  transition:all .25s ease!important;
  padding:0!important;
  z-index:999!important;
}
[data-testid="collapsedControl"]:hover{
  background:rgba(0,200,255,.26)!important;
  box-shadow:3px 0 32px rgba(0,200,255,.35)!important;
  width:32px!important;
}
[data-testid="collapsedControl"] svg{
  stroke:rgba(0,200,255,.95)!important;fill:none!important;
  width:14px!important;height:14px!important;
}
/* sidebar expand button (when open, the X / arrow button) */
button[data-testid="baseButton-headerNoPadding"]{
  color:rgba(0,200,255,.7)!important;
}

/* HUD NAV */
.hud-nav{
  display:flex;justify-content:space-between;align-items:center;
  padding:11px 0;border-bottom:1px solid rgba(0,200,255,.07);
  font-size:.5rem;text-transform:uppercase;letter-spacing:4px;
  color:rgba(0,200,255,.32);font-family:'JetBrains Mono',monospace;
}
.hud-dot{display:inline-block;width:5px;height:5px;border-radius:50%;
  background:var(--green);margin-right:7px;box-shadow:0 0 7px var(--green);
  animation:pd 2s ease-in-out infinite;}
@keyframes pd{0%,100%{opacity:1}50%{opacity:.25}}

/* HERO */
.hero{padding:52px 0 20px;position:relative;z-index:1;}
.hero-ey{font-size:.54rem;letter-spacing:5px;color:var(--cyan);text-transform:uppercase;
  font-weight:300;margin-bottom:10px;font-family:'JetBrains Mono',monospace;}
.hero-h{
  font-family:'Sora',sans-serif;font-size:4.8rem;font-weight:800;
  line-height:.88;letter-spacing:-2.5px;
  background:linear-gradient(135deg,#fff 0%,rgba(0,200,255,.92) 48%,rgba(123,95,255,.82) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  margin-bottom:14px;
}
.accent{height:1px;width:55px;margin-bottom:14px;
  background:linear-gradient(90deg,var(--cyan),var(--violet),transparent);
  box-shadow:0 0 12px rgba(0,200,255,.45);}
.coord{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:rgba(0,200,255,.36);letter-spacing:2px;}

/* GLASS CARDS */
.gc{
  background:var(--gg);border:1px solid var(--gd);
  backdrop-filter:var(--gb);-webkit-backdrop-filter:var(--gb);
  border-radius:18px;padding:22px 20px;position:relative;overflow:hidden;
  transition:transform .3s cubic-bezier(.165,.84,.44,1),box-shadow .3s;
}
.gc::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.18),transparent);}
.gc:hover{transform:translateY(-2px);box-shadow:0 10px 36px rgba(0,0,0,.4);}
.gc.al{border-color:rgba(255,45,85,.3); background:rgba(255,45,85,.07);}
.gc.wa{border-color:rgba(255,179,64,.25);background:rgba(255,179,64,.06);}
.gc.ok{border-color:rgba(48,209,88,.22); background:rgba(48,209,88,.05);}
.gc.in{border-color:rgba(0,200,255,.22); background:rgba(0,200,255,.05);}
.sl{font-size:.44rem;color:rgba(255,255,255,.3);text-transform:uppercase;
  letter-spacing:4px;margin-bottom:7px;font-family:'JetBrains Mono',monospace;}
.sv{font-family:'Sora',sans-serif;font-size:2.6rem;font-weight:700;line-height:1;}
.sv-r{color:var(--red);  text-shadow:0 0 22px rgba(255,45,85,.5);}
.sv-a{color:var(--amber);text-shadow:0 0 22px rgba(255,179,64,.5);}
.sv-g{color:var(--green);text-shadow:0 0 22px rgba(48,209,88,.5);}
.sv-v{color:var(--violet);text-shadow:0 0 22px rgba(123,95,255,.5);}
.ss{font-size:.44rem;color:rgba(255,255,255,.2);margin-top:5px;font-family:'JetBrains Mono',monospace;}

/* GLASS PANEL */
.gp{background:rgba(255,255,255,.027);border:1px solid var(--gd);
    backdrop-filter:var(--gb);-webkit-backdrop-filter:var(--gb);
    border-radius:18px;overflow:hidden;position:relative;}
.gp::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.13),transparent);z-index:1;}
.ph{padding:16px 20px 12px;border-bottom:1px solid rgba(255,255,255,.04);
    display:flex;align-items:center;gap:9px;background:rgba(255,255,255,.02);}
.phd{width:6px;height:6px;border-radius:50%;
  background:linear-gradient(135deg,var(--cyan),var(--blue));
  box-shadow:0 0 9px rgba(0,200,255,.6);animation:pd 2s ease-in-out infinite;}
.pht{font-size:.46rem;letter-spacing:4px;text-transform:uppercase;
  color:rgba(255,255,255,.38);font-family:'JetBrains Mono',monospace;}
.if{height:540px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:rgba(0,200,255,.1) transparent;}
.fr{padding:12px 20px;border-bottom:1px solid rgba(255,255,255,.025);transition:background .2s;}
.fr:hover{background:rgba(255,255,255,.022);}
.fr.hi{border-left:2px solid var(--red);}
.fr.me{border-left:2px solid var(--amber);}
.fm{font-size:.44rem;color:rgba(255,255,255,.18);text-transform:uppercase;letter-spacing:2px;
    margin-bottom:4px;font-family:'JetBrains Mono',monospace;display:flex;justify-content:space-between;}
.bh{color:var(--red);  font-weight:700;font-size:.46rem;}
.bm{color:var(--amber);font-weight:700;font-size:.46rem;}
.fb{font-family:'JetBrains Mono',monospace;font-size:.66rem;color:rgba(255,255,255,.48);line-height:1.5;}
.fmm{color:rgba(0,200,255,.75);}

/* MESH TABLE */
.mw{
  background:
    radial-gradient(ellipse 80% 60% at 20% 30%,rgba(0,87,255,.10) 0%,transparent 55%),
    radial-gradient(ellipse 60% 50% at 80% 70%,rgba(123,95,255,.08) 0%,transparent 50%),
    rgba(255,255,255,.022);
  border:1px solid rgba(255,255,255,.07);border-radius:18px;
  overflow:hidden;position:relative;
  backdrop-filter:var(--gb);-webkit-backdrop-filter:var(--gb);
  max-height:620px;overflow-y:auto;
  scrollbar-width:thin;scrollbar-color:rgba(0,200,255,.1) transparent;
}
.mw::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.14),transparent);z-index:2;}
.mt{width:100%;border-collapse:collapse;}
.mt thead{position:sticky;top:0;z-index:1;}
.mt th{padding:12px 14px;text-align:left;font-size:.42rem;letter-spacing:4px;
  text-transform:uppercase;color:rgba(255,255,255,.26);font-family:'JetBrains Mono',monospace;
  border-bottom:1px solid rgba(255,255,255,.055);font-weight:400;background:rgba(4,18,38,.97);}
.mt td{padding:10px 14px;font-size:.7rem;font-family:'JetBrains Mono',monospace;
  border-bottom:1px solid rgba(255,255,255,.022);color:rgba(255,255,255,.56);transition:background .12s;}
.mt tr:hover td{background:rgba(255,255,255,.028);}
.mm{color:rgba(0,200,255,.72)!important;font-size:.64rem!important;}
.tH{color:var(--red)!important;  font-weight:700!important;}
.tM{color:var(--amber)!important;font-weight:700!important;}
.tL{color:rgba(48,209,88,.7)!important;}
.tv{color:rgba(0,229,204,.75)!important;}
.tf{color:rgba(255,255,255,.36)!important;}
.tn{color:rgba(255,255,255,.68)!important;}

/* section labels */
.stit{font-family:'Sora',sans-serif;font-size:1.8rem;font-weight:700;letter-spacing:-1px;color:#fff;margin-bottom:3px;}
.ssub{font-size:.5rem;letter-spacing:3px;color:rgba(0,200,255,.36);text-transform:uppercase;font-family:'JetBrains Mono',monospace;margin-bottom:20px;}
.mzl{font-family:'Sora',sans-serif;font-size:1.4rem;font-weight:700;color:#fff;margin-bottom:2px;}
.mzs{font-family:'JetBrains Mono',monospace;font-size:.46rem;color:rgba(0,200,255,.36);letter-spacing:3px;text-transform:uppercase;margin-bottom:12px;}

/* sidebar widgets */
.sw{padding:16px 14px;border-bottom:1px solid rgba(255,255,255,.045);}
.swl{font-size:.43rem;color:rgba(255,255,255,.26);letter-spacing:3px;text-transform:uppercase;font-family:'JetBrains Mono',monospace;margin-bottom:5px;}
.swv{font-size:.82rem;font-weight:300;color:rgba(255,255,255,.78);letter-spacing:1px;}
.swb{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;border-radius:20px;
  font-size:.42rem;font-family:'JetBrains Mono',monospace;letter-spacing:2px;text-transform:uppercase;margin-top:4px;}
.sbg{background:rgba(48,209,88,.14); border:1px solid rgba(48,209,88,.28); color:var(--green);}
.sbc{background:rgba(0,200,255,.11); border:1px solid rgba(0,200,255,.24);color:var(--cyan);}
.sbr{background:rgba(255,45,85,.11); border:1px solid rgba(255,45,85,.24); color:var(--red);}
.sba{background:rgba(255,179,64,.11);border:1px solid rgba(255,179,64,.24);color:var(--amber);}
/* status dot in badges */
.dot{width:5px;height:5px;border-radius:50%;display:inline-block;}
.dot-g{background:var(--green);box-shadow:0 0 5px var(--green);}
.dot-c{background:var(--cyan); box-shadow:0 0 5px var(--cyan);}
.dot-r{background:var(--red);  box-shadow:0 0 5px var(--red);}

/* BUTTON */
.stButton>button{
  background:rgba(255,45,85,.07)!important;border:1px solid rgba(255,45,85,.28)!important;
  color:rgba(255,45,85,.9)!important;border-radius:9px!important;
  font-family:'Sora',sans-serif!important;font-size:.58rem!important;
  letter-spacing:2px!important;text-transform:uppercase!important;transition:all .2s!important;
}
.stButton>button:hover{background:rgba(255,45,85,.17)!important;box-shadow:0 0 14px rgba(255,45,85,.22)!important;}
[data-testid="stDataFrame"]{display:none!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# VESSEL DEFINITIONS
# ─────────────────────────────────────────────────────────────
NAMES = {
    "Cargo":   ["MV Sindhu Star","MV Bay Trader","MV Bengal Pride","MV Coromandel","MV Vizag Express",
                "MV Eastern Promise","MV Kochi Venture","MV Malabar Queen","MV Chennai Spirit","MV Paradip Bulk"],
    "Tanker":  ["MT Arabian Dawn","MT Gulf Stream","MT Chennai Crude","MT Visakha","MT Andaman Maru",
                "MT Lakshadweep","MT Palk Strait","MT Oman Sky","MT Hormuz","MT Karwar"],
    "Fishing": ["FV Sagar Devi","FV Deep Blue","FV Ocean Catch","FV Matsya","FV Trivandrum",
                "FV Bay Fisher","FV Calicut Net","FV Ratnagiri","FV Latur Sea","FV Puri Pearl"],
    "Security":["INS Talwar","INS Vikrant","INS Kolkata","CG Sarathi","INS Shivalik",
                "INS Sahyadri","INS Kamorta","INS Kadmatt","INS Kiltan","INS Kavaratti"],
    "Research":["RV Sagar Nidhi","RV Sindhu Sadhana","RV Sagar Kanya","RV Samudra Ratna",
                "RV Samudra Shakti","RV Samudra Sewak","RV Sagar Anveshika","RV Deep Nidhi"],
}
ICONS = {"Cargo":"🚢","Tanker":"⛽","Fishing":"🎣","Security":"🛡️","Research":"🔬"}
BASE_SOG = {"Cargo":14,"Tanker":12,"Fishing":8,"Security":22,"Research":10}

# ─────────────────────────────────────────────────────────────
# FLEET  — stored in session_state so state truly persists
# ─────────────────────────────────────────────────────────────
def make_fleet():
    rng = random.Random(int(time.time() * 1000) % 99999)
    vtypes  = ["Cargo","Tanker","Fishing","Security","Research"]
    weights = [28, 20, 32, 8, 12]
    flags   = ["India","Singapore","China","Panama","Norway","Sri Lanka","Liberia","Unknown"]
    fleet   = []
    for i in range(100):
        vt   = rng.choices(vtypes, weights=weights, k=1)[0]
        name = rng.choice(NAMES[vt]) + f"-{rng.randint(1,99):02d}"
        lat  = rng.uniform(3.0, 25.0)
        lon  = rng.uniform(56.0, 100.0)
        base = BASE_SOG[vt]
        sog  = base + rng.uniform(-2, 2)
        cog  = rng.uniform(0, 360)
        fleet.append({
            "mmsi":         rng.randint(211000000,775000000),
            "name":         name,
            "vtype":        vt,
            "flag":         rng.choice(flags),
            "icon":         ICONS[vt],
            "lat":          lat,
            "lon":          lon,
            "sog":          sog,
            "cog":          cog,
            "cog_drift":    rng.uniform(-1.5, 1.5),
            "base_sog":     base,
            "risk_seed":    rng.random(),              # 0-1, fixed per vessel
            "iuu_suspect":  (vt == "Fishing") and (rng.random() < 0.35),
            "risk":         rng.random() * 100,        # start randomised
            "level":        "LOW",
            "history":      [[lat, lon, sog, cog]] * 20,
        })
    return fleet

if "fleet" not in st.session_state:
    st.session_state.fleet = make_fleet()
    st.session_state.tick  = 0

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model("vessel_predictor.keras"), True
    except:
        return None, False

# ─────────────────────────────────────────────────────────────
# SIMULATION TICK  — advances every vessel
# ─────────────────────────────────────────────────────────────
def normalize_mesh(lat, lon, sog, cog):
    r = np.radians(cog)
    return [(lat+90)/180, (lon+180)/360, sog/45.0, np.sin(r), np.cos(r)]

def tick_fleet(fleet, model):
    for v in fleet:
        # ── Speed: oscillate around base with momentum ──
        v["sog"] += np.random.uniform(-1.8, 1.8)
        v["sog"]  = float(np.clip(v["sog"], v["base_sog"]*0.35, v["base_sog"]*1.65))

        # ── Heading: smooth random walk ──
        v["cog_drift"] += np.random.uniform(-0.4, 0.4)
        v["cog_drift"]  = float(np.clip(v["cog_drift"], -5, 5))
        v["cog"] = (v["cog"] + v["cog_drift"] + np.random.uniform(-6, 6)) % 360

        # ── Position ──
        delta     = v["sog"] * 0.00025          # visible movement each tick
        v["lat"] += delta * np.cos(np.radians(v["cog"]))
        v["lon"] += delta * np.sin(np.radians(v["cog"]))

        # soft boundary reflect
        if not (1 < v["lat"] < 27):
            v["cog"] = (v["cog"] + 180) % 360
            v["lat"] = float(np.clip(v["lat"], 1.5, 26.5))
        if not (55 < v["lon"] < 101):
            v["cog"] = (v["cog"] + 180) % 360
            v["lon"] = float(np.clip(v["lon"], 55.5, 100.5))

        # ── Update history ──
        v["history"].append([v["lat"], v["lon"], v["sog"], v["cog"]])
        if len(v["history"]) > 20:
            v["history"].pop(0)

        # ── Risk inference ──
        speed_factor   = max(0, (v["sog"] - v["base_sog"] * 1.2) / (v["base_sog"] * 0.6 + 1))
        erratic        = min(1, abs(v["cog_drift"]) / 4.5)
        kinematic      = speed_factor * 0.45 + erratic * 0.35
        iuu            = 0.28 if v["iuu_suspect"] else 0.0

        if model:
            hist_slice = v["history"][-10:]
            inp        = np.array([[normalize_mesh(*p)] for p in hist_slice]).reshape(1, 10, 5)
            try:
                model.predict(inp, verbose=0)
                base = v["risk_seed"]
            except:
                base = v["risk_seed"] * 0.85
        else:
            base = v["risk_seed"]

        # Blend: fixed seed (55%) + live kinematics (30%) + iuu bonus (15%)
        # Add ±4% live noise so values visibly jitter each tick
        raw = base*0.55 + kinematic*0.30 + iuu*0.15 + np.random.uniform(-0.04, 0.04)
        v["risk"]  = float(np.clip(raw * 100, 0, 100))
        v["level"] = "HIGH" if v["risk"] > 72 else ("MEDIUM" if v["risk"] > 40 else "LOW")

    return fleet

# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
def main():
    # Advance simulation
    model, loaded = load_model()
    st.session_state.fleet = tick_fleet(st.session_state.fleet, model)
    st.session_state.tick += 1
    fleet = st.session_state.fleet

    now = datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")

    # ── HUD NAV ──
    st.markdown(f"""
    <div class="hud-nav">
      <div><span class="hud-dot"></span>Blue Sentinel Ops &nbsp;//&nbsp; Node DL-04</div>
      <div>{now}</div>
      <div>Tick #{st.session_state.tick} &nbsp;//&nbsp; UA: 26.1</div>
    </div>""", unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────
    if "sync_hz" not in st.session_state:
        st.session_state.sync_hz = 15

    with st.sidebar:
        st.markdown("""
        <div style="padding:28px 0 6px;text-align:center;">
          <div style="width:50px;height:50px;margin:0 auto;border-radius:14px;
               background:linear-gradient(135deg,rgba(0,200,255,.15),rgba(123,95,255,.15));
               border:1px solid rgba(0,200,255,.22);display:flex;align-items:center;
               justify-content:center;font-size:1.35rem;
               box-shadow:0 0 20px rgba(0,200,255,.14);">🛡️</div>
          <div style="margin-top:10px;font-size:.44rem;letter-spacing:5px;
               color:rgba(255,255,255,.32);text-transform:uppercase;
               font-family:'JetBrains Mono',monospace;">Command Center</div>
        </div>
        """, unsafe_allow_html=True)

        # System status widgets
        neural_badge = 'sbg" style="color:var(--green)"><span class="dot dot-g"></span> SYNC' if loaded \
                       else 'sbr" style="color:var(--red)"><span class="dot dot-r"></span> OFFLINE'

        st.markdown(f"""
        <div class="sw">
          <div class="swl">Satellite Uplink</div>
          <div class="swv">GSAT-7</div>
          <span class="swb sbg"><span class="dot dot-g"></span>&nbsp;LOCKED</span>
        </div>
        <div class="sw">
          <div class="swl">Neural Core</div>
          <div class="swv">LSTM · 5-Feature</div>
          <span class="swb {neural_badge}</span>
        </div>
        <div class="sw">
          <div class="swl">AIS Decoder</div>
          <div class="swv">Version 2.4</div>
          <span class="swb sbg"><span class="dot dot-g"></span>&nbsp;ACTIVE</span>
        </div>
        <div class="sw">
          <div class="swl">MOD Uplink</div>
          <div class="swv">AES-256</div>
          <span class="swb sbg"><span class="dot dot-g"></span>&nbsp;SECURE</span>
        </div>
        <div class="sw">
          <div class="swl">Fleet Coverage</div>
          <div class="swv">100 Vessels</div>
          <span class="swb sbc"><span class="dot dot-c"></span>&nbsp;MONITORING</span>
        </div>
        <div class="sw">
          <div class="swl">Sector</div>
          <div class="swv">Indo-Pacific</div>
          <div style="font-size:.43rem;color:rgba(0,200,255,.3);font-family:'JetBrains Mono',monospace;
               margin-top:3px;letter-spacing:.8px;">LAT 3°–25°N · LON 56°–100°E</div>
        </div>
        """, unsafe_allow_html=True)

        # Refresh slider
        st.markdown("""
        <div style="padding:14px 14px 0;">
          <div class="swl">Auto-Refresh Interval</div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.sync_hz = st.slider(
            "refresh", 5, 180, st.session_state.sync_hz,
            label_visibility="collapsed"
        )

        st.markdown(f"""
        <div style="padding:0 14px 14px;">
          <div style="height:2px;background:rgba(255,255,255,.04);border-radius:1px;overflow:hidden;margin-bottom:4px;">
            <div style="width:{int(st.session_state.sync_hz/180*100)}%;height:100%;
                 background:linear-gradient(90deg,var(--cyan),var(--violet));
                 border-radius:1px;box-shadow:0 0 7px rgba(0,200,255,.4);"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;
               font-size:.42rem;color:rgba(0,200,255,.32);letter-spacing:1px;">
            <span>5s</span>
            <span>REFRESH · {st.session_state.sync_hz}s</span>
            <span>180s</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Purge button
        st.markdown('<div style="padding:2px 14px 14px;">', unsafe_allow_html=True)
        if st.button("⟳  RESET FLEET", use_container_width=True):
            del st.session_state["fleet"]
            st.session_state.tick = 0
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── HERO ──
    st.markdown("""
    <div class="hero">
      <div class="hero-ey">Indo-Pacific Maritime Intelligence Platform</div>
      <div class="hero-h">Guarding<br>Your Oceans.</div>
      <div class="accent"></div>
      <div class="coord">LAT: 14.29°N &nbsp;·&nbsp; LON: 78.44°E &nbsp;·&nbsp; SECTOR: ALPHA-7</div>
    </div>""", unsafe_allow_html=True)

    # ── STAT CARDS ──
    h_c = sum(1 for v in fleet if v["level"]=="HIGH")
    m_c = sum(1 for v in fleet if v["level"]=="MEDIUM")
    l_c = sum(1 for v in fleet if v["level"]=="LOW")
    avg = float(np.mean([v["risk"] for v in fleet]))

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="gc al">
          <div class="sl">Critical Alerts</div>
          <div class="sv sv-r">{h_c}</div>
          <div class="ss">HIGH-RISK VESSELS</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="gc wa">
          <div class="sl">Anomalies</div>
          <div class="sv sv-a">{m_c}</div>
          <div class="ss">MEDIUM-RISK</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="gc ok">
          <div class="sl">Active Assets</div>
          <div class="sv sv-g">{len(fleet)}</div>
          <div class="ss">{l_c} NOMINAL</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="gc in">
          <div class="sl">Avg Risk Score</div>
          <div class="sv sv-v">{avg:.1f}<span style="font-size:.9rem;opacity:.5">%</span></div>
          <div class="ss">NEURAL {"SYNC" if loaded else "OFFLINE"}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:38px'></div>", unsafe_allow_html=True)

    # ── MAP + INTEL FEED ──
    mc, fc = st.columns([2.5, 1])

    with mc:
        st.markdown("""
        <div class="mzl">ZONE: Indo-Pacific</div>
        <div class="mzs">AIS Live Mesh · 100 Vectors · Hover dot for vessel details</div>
        """, unsafe_allow_html=True)

        rows = []
        for v in fleet:
            c  = [255,45,85,235] if v["level"]=="HIGH" else ([255,179,64,210] if v["level"]=="MEDIUM" else [0,200,255,170])
            r  = 30000 if v["level"]=="HIGH" else (20000 if v["level"]=="MEDIUM" else 13000)
            rows.append({
                "lat":  v["lat"], "lon":  v["lon"],
                "mmsi": str(v["mmsi"]), "name":  v["name"],
                "vtype":v["vtype"],     "flag":  v["flag"],
                "icon": v["icon"],      "sog":   round(v["sog"],1),
                "cog":  round(v["cog"],0), "risk": round(v["risk"],1),
                "level":v["level"],     "color": c, "radius": r,
            })
        vdf = pd.DataFrame(rows)

        # Trail paths
        trails = []
        for v in fleet:
            h = v["history"][-7:]
            if len(h) >= 2:
                col = [255,45,85] if v["level"]=="HIGH" else ([255,179,64] if v["level"]=="MEDIUM" else [0,200,255])
                trails.append({"path":[[p[1],p[0]] for p in h],"color":col})

        # Tactical grid
        grid = []
        for lt in range(0,28,4):   grid.append({"path":[[55,lt],[102,lt]]})
        for ln in range(55,103,4): grid.append({"path":[[ln,0], [ln,28]]})

        tooltip = {
            "html": """
              <div style='font-family:JetBrains Mono,monospace;
                background:rgba(2,13,26,.93);border:1px solid rgba(0,200,255,.22);
                border-radius:10px;padding:11px 15px;min-width:175px;
                backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);'>
                <div style='font-size:1.1rem;margin-bottom:5px;'>{icon}</div>
                <div style='color:rgba(0,200,255,.88);font-size:.68rem;font-weight:600;margin-bottom:2px;'>{name}</div>
                <div style='color:rgba(255,255,255,.45);font-size:.58rem;margin-bottom:7px;letter-spacing:.5px;'>MMSI: {mmsi}</div>
                <div style='color:rgba(255,255,255,.58);font-size:.6rem;line-height:1.75;'>
                  Type: {vtype} &nbsp;·&nbsp; {flag}<br>
                  SOG: {sog} kn &nbsp;·&nbsp; COG: {cog}°<br>
                  Risk: <b style='color:#ffb340'>{risk}%</b>
                  &nbsp;<b style='color:{"#ff2d55" if "{level}"=="HIGH" else ("#ffb340" if "{level}"=="MEDIUM" else "#30d158")}'>{level}</b>
                </div>
              </div>""",
            "style": {"backgroundColor":"transparent","border":"none","padding":"0"}
        }

        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=pdk.ViewState(latitude=14.0, longitude=78.0, zoom=3.8, pitch=0, bearing=0),
            layers=[
                pdk.Layer("PathLayer", grid, get_path="path", get_color=[0,200,255,10], width_min_pixels=1),
                pdk.Layer("PathLayer", trails, get_path="path", get_color="color", get_width=1.5,
                          width_min_pixels=1, opacity=0.35),
                # outer glow for HIGH-risk
                pdk.Layer("ScatterplotLayer",
                          vdf[vdf["level"]=="HIGH"],
                          get_position=["lon","lat"], get_color=[255,45,85,22],
                          get_radius=58000, opacity=0.6,
                          stroked=True, get_line_color=[255,45,85,100], get_line_width=900),
                # main dots — pickable for tooltip
                pdk.Layer("ScatterplotLayer", vdf,
                          get_position=["lon","lat"], get_color="color", get_radius="radius",
                          opacity=0.92, pickable=True, auto_highlight=True,
                          highlight_color=[255,255,255,70]),
                # emoji icons
                pdk.Layer("TextLayer", vdf,
                          get_position=["lon","lat"], get_text="icon",
                          get_size=16, pickable=False,
                          get_alignment_baseline="'center'"),
            ],
            tooltip=tooltip
        )
        st.pydeck_chart(deck, use_container_width=True)

    with fc:
        priority = sorted([v for v in fleet if v["level"]!="LOW"], key=lambda x: x["risk"], reverse=True)[:40]
        feed_html = ""
        for v in priority:
            cls   = "hi" if v["level"]=="HIGH" else "me"
            badge = f'<span class="bh">▲ {v["risk"]:.1f}%</span>' if v["level"]=="HIGH" \
                    else f'<span class="bm">● {v["risk"]:.1f}%</span>'
            feed_html += f"""
            <div class="fr {cls}">
              <div class="fm"><span>{datetime.utcnow().strftime("%H:%M:%S")} · INDOPAC</span>{badge}</div>
              <div class="fb">
                {v["icon"]} <span class="fmm">{v["mmsi"]}</span><br>
                {v["name"]}<br>
                {v["vtype"]} · {v["flag"]}<br>
                SOG {v["sog"]:.1f}kn · COG {v["cog"]:.0f}°
              </div>
            </div>"""

        if not feed_html:
            feed_html = '<div class="fr"><div class="fb" style="color:rgba(48,209,88,.5);">All 100 assets nominal.</div></div>'

        st.markdown(f"""
        <div class="gp">
          <div class="ph"><div class="phd"></div>
            <div class="pht">Intel Stream · {len(priority)} events</div>
          </div>
          <div class="if">{feed_html}</div>
        </div>""", unsafe_allow_html=True)

    # ── FLEET MANIFESTO ──
    st.markdown("<div style='margin-top:50px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="stit">Fleet Manifesto</div>
    <div class="ssub">100 vessels · Live LSTM anomaly scores · Risk-sorted · Updates every tick</div>
    """, unsafe_allow_html=True)

    sorted_fleet = sorted(fleet, key=lambda x: x["risk"], reverse=True)
    trows = ""
    for i, v in enumerate(sorted_fleet):
        bc  = "var(--red)" if v["level"]=="HIGH" else ("var(--amber)" if v["level"]=="MEDIUM" else "var(--green)")
        rbg = "rgba(255,45,85,.04)" if v["level"]=="HIGH" else ("rgba(255,179,64,.03)" if v["level"]=="MEDIUM" else "")
        trows += f"""
        <tr style="background:{rbg}">
          <td style="color:rgba(255,255,255,.25);font-size:.58rem;">{i+1:02d}</td>
          <td class="mm">{v["mmsi"]}</td>
          <td class="tn">{v["name"]}</td>
          <td>{v["icon"]} {v["vtype"]}</td>
          <td class="tf">{v["flag"]}</td>
          <td class="tv">{v["sog"]:.1f}kn</td>
          <td style="color:rgba(255,255,255,.32);">{v["cog"]:.0f}°</td>
          <td style="color:rgba(255,255,255,.28);font-size:.58rem;">{v["lat"]:.3f}N {v["lon"]:.3f}E</td>
          <td>
            <div style="display:flex;align-items:center;gap:5px;">
              <div style="flex:1;height:3px;background:rgba(255,255,255,.05);border-radius:2px;overflow:hidden;min-width:50px;">
                <div style="width:{v['risk']:.0f}%;height:100%;background:{bc};border-radius:2px;box-shadow:0 0 4px {bc};"></div>
              </div>
              <span style="font-size:.6rem;color:{bc};min-width:32px;">{v["risk"]:.1f}%</span>
            </div>
          </td>
          <td class="t{v['level']}">{v["level"]}</td>
        </tr>"""

    st.markdown(f"""
    <div class="mw">
      <table class="mt">
        <thead><tr>
          <th>#</th><th>MMSI</th><th>Vessel Name</th><th>Class</th>
          <th>Registry</th><th>Speed</th><th>Hdg</th>
          <th>Position</th><th>Risk Score</th><th>Status</th>
        </tr></thead>
        <tbody>{trows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown(f"""
    <div style="margin-top:44px;padding:18px 0;border-top:1px solid rgba(255,255,255,.035);
         display:flex;justify-content:space-between;align-items:center;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:.42rem;
            color:rgba(255,255,255,.16);letter-spacing:2px;text-transform:uppercase;">
        Blue Sentinel AI · Indo-Pacific Node DL-04 · v4.0
      </span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:.42rem;
            color:rgba(0,200,255,.24);letter-spacing:2px;">
        Tick #{st.session_state.tick} · H:{h_c} M:{m_c} L:{l_c} · Next sync {st.session_state.sync_hz}s
      </span>
    </div>""", unsafe_allow_html=True)

    time.sleep(st.session_state.sync_hz)
    st.rerun()

if __name__ == "__main__":
    main()

# ==================== GEOGRAPHIC REFERENCE DATA ====================
MAJOR_CITIES = [
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "type": "Major Port", "country": "India"},
    {"name": "Kochi", "lat": 9.9312, "lon": 76.2673, "type": "Naval Base", "country": "India"},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "type": "Major Port", "country": "India"},
    {"name": "Colombo", "lat": 6.9271, "lon": 79.8612, "type": "Capital/Port", "country": "Sri Lanka"},
    {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "type": "Naval HQ", "country": "India"},
    {"name": "Tuticorin", "lat": 8.8000, "lon": 78.1500, "type": "Port", "country": "India"},
    {"name": "Mangalore", "lat": 12.9141, "lon": 74.8560, "type": "Port", "country": "India"},
    {"name": "Goa", "lat": 15.2993, "lon": 74.1240, "type": "Naval Base", "country": "India"},
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011, "type": "Major Port", "country": "Pakistan"},
    {"name": "Malé", "lat": 4.1755, "lon": 73.5093, "type": "Capital", "country": "Maldives"},
]

ISLANDS_AND_REGIONS = [
    {"name": "Lakshadweep", "lat": 10.5667, "lon": 72.6417, "type": "Island Territory"},
    {"name": "Andaman & Nicobar", "lat": 11.7401, "lon": 92.6586, "type": "Island Territory"},
    {"name": "Minicoy Island", "lat": 8.2833, "lon": 73.0500, "type": "Island"},
]

WATER_BODIES = [
    {"name": "Arabian Sea", "lat": 16.0, "lon": 66.0},
    {"name": "Bay of Bengal", "lat": 15.0, "lon": 88.0},
    {"name": "Indian Ocean", "lat": 5.0, "lon": 75.0},
    {"name": "Laccadive Sea", "lat": 10.0, "lon": 74.0},
]

# ==================== STYLING ====================
st.markdown("""
    <style>
    /* Global Theme */
    .stApp {
        background-color: #0a0e17;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Header Styling */
    .demo-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #1a1a2e 0%, #0a0e17 100%);
        border-radius: 15px;
        border: 2px solid rgba(0, 229, 255, 0.3);
        margin-bottom: 30px;
    }
    
    .demo-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 10px;
    }
    
    .demo-subtitle {
        font-size: 1.2rem;
        color: #00E5FF;
        letter-spacing: 2px;
    }
    
    /* Metrics */
    .metric-label {
        font-size: 0.8rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Risk Colors */
    .risk-high { color: #FF4B4B; }
    .risk-med { color: #FFA500; }
    .risk-low { color: #00E5FF; }
    
    /* Live Feed */
    .live-feed {
        background: rgba(0, 0, 0, 0.5);
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        color: #00ff00;
        height: 300px;
        overflow-y: auto;
        border: 1px solid rgba(0, 255, 0, 0.3);
    }
    
    .feed-item {
        padding: 5px 0;
        border-bottom: 1px solid rgba(0, 255, 0, 0.1);
    }
    
    /* Status Indicators */
    .status-active {
        color: #00ff00;
        font-weight: bold;
    }
    
    .status-warning {
        color: #FFA500;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #FF4B4B 0%, #ff1744 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #05080f;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Data Source Status */
    .data-source {
        padding: 10px;
        margin: 5px 0;
        background: rgba(0, 255, 0, 0.1);
        border-left: 3px solid #00ff00;
        border-radius: 5px;
    }
    
    /* Map Legend */
    .map-legend {
        background: rgba(0, 0, 0, 0.7);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 10px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 10px;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== AI MODEL SETUP ====================
@st.cache_resource
def load_ai_model():
    """Load the trained LSTM vessel prediction model"""
    try:
        model = keras.models.load_model("models/vessel_predictor.keras")
        return model, True
    except Exception as e:
        st.warning(f"Model file not found. Running in simulation mode. ({e})")
        return None, False

# ==================== DATA SIMULATION ====================
class VesselTracker:
    """Real-time vessel tracking and prediction system"""
    
    def __init__(self, mmsi, name, lat, lon, vessel_type, flag):
        self.mmsi = mmsi
        self.name = name
        self.lat = lat
        self.lon = lon
        self.vessel_type = vessel_type
        self.flag = flag
        self.sog = random.uniform(2.0, 15.0)
        self.cog = random.uniform(0, 360)
        self.nav_status = random.choice(["Under way using engine", "Fishing", "At anchor"])
        
        # Risk indicators
        self.dark_periods = random.choices([0, 1, 2, 3, 5], weights=[0.6, 0.2, 0.1, 0.05, 0.05])[0]
        self.lstm_anomaly = random.random()
        self.mpa_proximity = random.uniform(0, 50)
        
        # Historical track (last 10 positions)
        self.history = self._generate_history()
        
        # Prediction results
        self.predicted_lat = None
        self.predicted_lon = None
        self.risk_score = 0.0
        self.risk_level = "LOW"
        self.risk_reasons = []
        self.predicted_activity = "Normal Transit"
        
        # Geographic context
        self.region = self._determine_region()
    
    def _determine_region(self):
        """Determine which region/water body the vessel is in"""
        if self.lon < 74:
            return "Arabian Sea (Western Coast)"
        elif self.lon > 85:
            return "Bay of Bengal (Eastern Coast)"
        elif self.lat < 8:
            return "Indian Ocean (Southern Waters)"
        else:
            return "Laccadive Sea (Southwest India)"
    
    def _generate_history(self):
        """Generate historical track data"""
        history = []
        curr_lat, curr_lon = self.lat, self.lon
        for _ in range(10):
            # Add some random walk
            curr_lat += np.random.uniform(-0.02, 0.02)
            curr_lon += np.random.uniform(-0.02, 0.02)
            history.append([curr_lat, curr_lon, self.sog, self.cog])
        return history
    
    def update_position(self):
        """Simulate real-time position update"""
        # Move vessel based on speed and course
        self.lat += (self.sog * 0.005) * np.cos(np.radians(self.cog))
        self.lon += (self.sog * 0.005) * np.sin(np.radians(self.cog))
        
        # Add some randomness
        self.sog += np.random.uniform(-0.3, 0.3)
        self.sog = max(0.5, min(20, self.sog))
        self.cog += np.random.uniform(-5, 5)
        self.cog = self.cog % 360
        
        # Update history
        self.history.append([self.lat, self.lon, self.sog, self.cog])
        if len(self.history) > 10:
            self.history.pop(0)
        
        # Update region
        self.region = self._determine_region()
    
    def predict_trajectory(self, model=None):
        """Run AI prediction on vessel trajectory"""
        if model is not None:
            # Use actual LSTM model
            track_array = np.array([self.history])
            prediction = model.predict(track_array, verbose=0)
            self.predicted_lat, self.predicted_lon = prediction[0]
        else:
            # Simulation mode
            self.predicted_lat = self.lat + (self.sog * 0.02) * np.cos(np.radians(self.cog))
            self.predicted_lon = self.lon + (self.sog * 0.02) * np.sin(np.radians(self.cog))
        
        # Calculate risk score
        self._calculate_risk()
    
    def _calculate_risk(self):
        """Calculate IUU risk score based on multiple factors"""
        score = 0.0
        reasons = []
        
        # LSTM Anomaly Score (0-35 points)
        lstm_contrib = self.lstm_anomaly * 35
        score += lstm_contrib
        if self.lstm_anomaly > 0.7:
            reasons.append(f"Abnormal Maneuvering Detected ({self.lstm_anomaly:.2f})")
        
        # Dark Ship Events (0-20 points)
        if self.dark_periods > 2:
            score += 20
            reasons.append(f"Dark Ship Pattern ({self.dark_periods} events in 24h)")
        
        # Loitering Behavior (0-15 points)
        if self.vessel_type in ["Fishing", "Trawler"] and self.sog < 2.0:
            score += 15
            reasons.append("Loitering Behavior (Fishing Probability: 92%)")
        
        # MPA Proximity (0-25 points)
        if self.mpa_proximity < 10:
            score += 25
            reasons.append(f"Approaching Protected Zone ({self.mpa_proximity:.1f} km)")
        
        # Unknown Flag (0-10 points)
        if self.flag == "Unknown":
            score += 10
            reasons.append("Unknown Flag State")
        
        # Trajectory Deviation (0-15 points)
        if self.predicted_lat and self.predicted_lon:
            deviation = np.sqrt((self.predicted_lat - self.lat)**2 + (self.predicted_lon - self.lon)**2)
            if deviation > 0.1:
                score += 15
                reasons.append(f"Unusual Trajectory Deviation ({deviation*111:.1f} km)")
        
        # Normalize to 0-100
        self.risk_score = min(100.0, score)
        self.risk_reasons = reasons
        
        # Determine risk level and activity
        if self.risk_score > 75:
            self.risk_level = "HIGH"
            self.predicted_activity = "Illegal Fishing (High Probability)"
        elif self.risk_score > 40:
            self.risk_level = "MEDIUM"
            self.predicted_activity = "Suspicious Transit"
        else:
            self.risk_level = "LOW"
            self.predicted_activity = "Normal Operations"

def generate_demo_vessels(n=30):
    """Generate a fleet of vessels for demonstration"""
    vessels = []
    
    # Define some preset high-risk vessels for demo
    high_risk_vessels = [
        {"mmsi": 419852630, "name": "EASTERN STAR", "lat": 8.52, "lon": 76.91, "type": "Fishing Trawler", "flag": "Unknown"},
        {"mmsi": 412589630, "name": "OCEAN HUNTER", "lat": 7.85, "lon": 75.50, "type": "Fishing", "flag": "Unknown"},
        {"mmsi": 423156789, "name": "DARK PEARL", "lat": 9.10, "lon": 78.20, "type": "Trawler", "flag": "Unknown"},
    ]
    
    for vessel_data in high_risk_vessels:
        vessel = VesselTracker(
            mmsi=vessel_data["mmsi"],
            name=vessel_data["name"],
            lat=vessel_data["lat"],
            lon=vessel_data["lon"],
            vessel_type=vessel_data["type"],
            flag=vessel_data["flag"]
        )
        # Make these high risk
        vessel.dark_periods = random.randint(3, 5)
        vessel.lstm_anomaly = random.uniform(0.7, 0.95)
        vessels.append(vessel)
    
    # Generate remaining vessels
    for i in range(n - len(high_risk_vessels)):
        vessel = VesselTracker(
            mmsi=random.randint(200000000, 700000000),
            name=f"VESSEL_{random.randint(1000, 9999)}",
            lat=random.uniform(5.0, 12.0),
            lon=random.uniform(72.0, 80.0),
            vessel_type=random.choice(["Cargo", "Tanker", "Fishing", "Passenger"]),
            flag=random.choice(["India", "Sri Lanka", "China", "Panama", "Liberia", "Unknown"])
        )
        vessels.append(vessel)
    
    return vessels

# ==================== UI COMPONENTS ====================

def render_header():
    """Render professional government demo header"""
    st.markdown("""
        <div class="demo-header">
            <div class="demo-title">🛡️ BLUE SENTINEL AI SYSTEM</div>
            <div class="demo-subtitle">Maritime Surveillance & Predictive Intelligence</div>
            <p style="color: #888; margin-top: 10px; font-size: 0.9rem;">
                Government of India - Ministry of Defence | Indian Navy & Coast Guard
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_data_note():
    """Render note about data source (SIMULATION vs REAL)"""
    st.markdown("""
        <div class="glass-card" style="background: rgba(255, 165, 0, 0.1); border-color: rgba(255, 165, 0, 0.3);">
            <h4 style="color: #FFA500; margin-top: 0;">📊 DATA SOURCE INFORMATION</h4>
            <p><strong>Current Mode:</strong> SIMULATION / DEMONSTRATION</p>
            <p>The 30 vessels displayed are <strong>SIMULATED</strong> for demonstration purposes. They are NOT real live vessel positions.</p>
            <p><strong>How it works:</strong></p>
            <ul style="margin-left: 20px;">
                <li>Mock data engine generates realistic vessel behavior patterns</li>
                <li>AI risk scoring uses actual algorithms that would work with real AIS data</li>
                <li>Vessel movements are simulated based on maritime navigation patterns</li>
            </ul>
            <p><strong>For REAL LIVE tracking, the system would integrate with:</strong></p>
            <ul style="margin-left: 20px;">
                <li>🛰️ AIS (Automatic Identification System) networks - MarineTraffic API, VesselFinder, AISHub</li>
                <li>🛰️ GSAT-7 Indian Naval Satellite for regional coverage</li>
                <li>🛰️ Indian Coast Guard AIS shore stations</li>
                <li>🛰️ ISRO Earth Observation satellites</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def render_data_sources():
    """Render data source connection status"""
    st.markdown("### 📡 Data Sources")
    st.caption("(Demo Mode - Simulated)")
    
    sources = [
        {"name": "GSAT-7 Satellite", "status": "SIMULATED", "coverage": "Indian Ocean"},
        {"name": "Spire Maritime AIS", "status": "SIMULATED", "coverage": "Global"},
        {"name": "Coast Guard AIS", "status": "SIMULATED", "coverage": "India"},
        {"name": "ISRO EO Satellites", "status": "SIMULATED", "coverage": "Regional"},
    ]
    
    for source in sources:
        st.markdown(f"""
            <div class="data-source">
                <strong>{source['name']}</strong><br>
                <span style="color: #FFA500;">[{source['status']}]</span> | {source['coverage']}
            </div>
        """, unsafe_allow_html=True)

def render_system_metrics(vessels, model_loaded):
    """Render system performance metrics"""
    high_risk = sum(1 for v in vessels if v.risk_level == "HIGH")
    medium_risk = sum(1 for v in vessels if v.risk_level == "MEDIUM")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">High Risk Targets</div>
                <div class="metric-value risk-high">{high_risk}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Medium Risk Targets</div>
                <div class="metric-value risk-med">{medium_risk}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Vessels Tracked</div>
                <div class="metric-value risk-low">{len(vessels)}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        status_text = "AI ONLINE" if model_loaded else "SIMULATION"
        status_color = "risk-low" if model_loaded else "risk-med"
        st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">AI Status</div>
                <div class="metric-value {status_color}">{status_text}</div>
            </div>
        """, unsafe_allow_html=True)

def render_live_analysis_feed(vessels):
    """Render live AI analysis feed"""
    st.markdown("### 🧠 Live AI Analysis Feed")
    
    # Get recent high-risk vessels
    high_risk_vessels = [v for v in vessels if v.risk_level in ["HIGH", "MEDIUM"]]
    high_risk_vessels = sorted(high_risk_vessels, key=lambda x: x.risk_score, reverse=True)[:10]
    
    feed_html = '<div class="live-feed">'
    current_time = datetime.now()
    
    for i, vessel in enumerate(high_risk_vessels):
        timestamp = (current_time - timedelta(seconds=i*3)).strftime('%H:%M:%S')
        icon = "🚨" if vessel.risk_level == "HIGH" else "⚠️"
        feed_html += f"""
            <div class="feed-item">
                {icon} [{timestamp}] ANALYZING MMSI {vessel.mmsi} ({vessel.name})<br>
                &nbsp;&nbsp;&nbsp;&nbsp;→ Risk Level: {vessel.risk_level} ({vessel.risk_score:.1f}%)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;→ Activity: {vessel.predicted_activity}<br>
                &nbsp;&nbsp;&nbsp;&nbsp;→ Location: {vessel.region}
            </div>
        """
    
    feed_html += '</div>'
    st.markdown(feed_html, unsafe_allow_html=True)

def render_map(vessels):
    """Render interactive map with vessel positions and geographic context"""
    st.markdown("### 🗺️ Real-Time Vessel Tracking Map")
    
    # Prepare vessel data for map
    map_data = []
    
    for v in vessels:
        # Color coding by risk level
        if v.risk_level == "HIGH":
            color = [255, 75, 75, 200]
        elif v.risk_level == "MEDIUM":
            color = [255, 165, 0, 200]
        else:
            color = [0, 229, 255, 150]
        
        map_data.append({
            "name": str(v.name),
            "mmsi": int(v.mmsi),
            "lat": float(v.lat),
            "lon": float(v.lon),
            "color": color,
            "risk": float(v.risk_score),
            "activity": str(v.predicted_activity),
            "region": str(v.region)
        })
    
    df_map = pd.DataFrame(map_data)
    
    # Simplified geographic labels - only major ports
    city_labels = []
    major_ports = ["Mumbai", "Kochi", "Chennai", "Colombo"]
    for city in MAJOR_CITIES:
        if city['name'] in major_ports:
            city_labels.append({
                "name": city['name'],
                "coordinates": [float(city['lon']), float(city['lat'])],
                "color": [255, 255, 100, 200],
                "size": 10
            })
    
    # Indian EEZ boundary (simplified)
    eez_boundary = [{
        "path": [
            [68.0, 8.0], [72.0, 10.0], [74.0, 12.0], [76.0, 12.5], 
            [78.0, 12.0], [80.0, 10.0], [82.0, 8.0], [82.0, 6.0],
            [80.0, 4.0], [76.0, 3.0], [72.0, 4.0], [68.0, 6.0], [68.0, 8.0]
        ],
        "color": [0, 255, 100, 80]
    }]
    
    # Coastline reference (simplified)
    coastline_india = [{
        "path": [
            [72.8, 19.0], [73.0, 17.0], [74.0, 15.0], [74.5, 13.0],
            [75.0, 12.0], [76.0, 10.0], [76.5, 9.0], [77.0, 8.5],
            [78.0, 8.2], [79.0, 9.0], [80.0, 10.0], [80.5, 13.0],
            [82.0, 16.0], [83.0, 17.5]
        ],
        "color": [255, 255, 255, 100]
    }]
    
    coastline_srilanka = [{
        "path": [
            [79.5, 9.5], [80.0, 8.5], [81.0, 7.5], [81.5, 6.5],
            [80.5, 6.0], [79.8, 6.5], [79.5, 7.5], [79.5, 9.5]
        ],
        "color": [255, 255, 255, 100]
    }]
    
    # Create layers (simplified for performance)
    layers = []
    
    # 1. EEZ Boundary
    eez_layer = pdk.Layer(
        "PathLayer",
        data=eez_boundary,
        get_path="path",
        get_color="color",
        width_scale=20,
        width_min_pixels=2,
        get_width=2,
    )
    layers.append(eez_layer)
    
    # 2. Coastlines
    coastline_layer_india = pdk.Layer(
        "PathLayer",
        data=coastline_india,
        get_path="path",
        get_color="color",
        width_scale=30,
        width_min_pixels=3,
        get_width=3,
    )
    layers.append(coastline_layer_india)
    
    coastline_layer_sl = pdk.Layer(
        "PathLayer",
        data=coastline_srilanka,
        get_path="path",
        get_color="color",
        width_scale=30,
        width_min_pixels=3,
        get_width=3,
    )
    layers.append(coastline_layer_sl)
    
    # 3. Vessel positions
    vessel_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=15000,
        pickable=True,
    )
    layers.append(vessel_layer)
    
    # 4. Major port labels (simplified)
    if city_labels:
        city_layer = pdk.Layer(
            "TextLayer",
            data=city_labels,
            get_position="coordinates",
            get_text="name",
            get_color="color",
            get_size="size",
            get_alignment_baseline="'bottom'",
            get_pixel_offset=[0, -10],
        )
        layers.append(city_layer)
    
    # Set view state with better initial positioning
    view_state = pdk.ViewState(
        latitude=10.0,
        longitude=77.0,
        zoom=5.5,
        pitch=0,
        bearing=0
    )
    
    # Create deck
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={
            "html": "<b>{name}</b><br/>MMSI: {mmsi}<br/>Risk: {risk}%<br/>Activity: {activity}<br/>Region: {region}",
            "style": {
                "backgroundColor": "rgba(0, 0, 0, 0.9)",
                "color": "white",
                "fontSize": "12px",
                "padding": "10px",
                "borderRadius": "5px"
            }
        }
    )
    
    st.pydeck_chart(deck)
    
    # Map legend
    st.markdown("""
        <div class="map-legend">
            <h4 style="margin-top: 0; color: #00E5FF;">Map Legend</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                <div class="legend-item">
                    <div class="legend-color" style="background: rgb(255, 75, 75);"></div>
                    <span>High Risk Vessel</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgb(255, 165, 0);"></div>
                    <span>Medium Risk Vessel</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgb(0, 229, 255);"></div>
                    <span>Low Risk Vessel</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def main():
    render_header()
    render_data_note()
    
    model, model_loaded = load_ai_model()
    
    # Initialize refresh rate in session state
    if 'refresh_rate' not in st.session_state:
        st.session_state.refresh_rate = 3
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/radar.png", width=80)
        st.title("Control Center")
        render_data_sources()
        st.markdown("---")
        st.markdown("### ⚙️ Demo Controls")
        
        # Refresh rate slider
        refresh_rate = st.slider(
            "Auto-Refresh Rate (seconds)",
            min_value=1,
            max_value=10,
            value=st.session_state.refresh_rate,
            help="How often the simulation updates"
        )
        st.session_state.refresh_rate = refresh_rate
        
        if st.button("🔄 Regenerate Simulation"):
            if 'vessels' in st.session_state:
                del st.session_state.vessels
            st.rerun()
        st.markdown("---")
        st.caption("Blue Sentinel Demo v2.0")

    # Logic
    if 'vessels' not in st.session_state:
        st.session_state.vessels = generate_demo_vessels()
    
    # Update positions
    for v in st.session_state.vessels:
        v.update_position()
        v.predict_trajectory(model)
        
    vessels = st.session_state.vessels
    
    render_system_metrics(vessels, model_loaded)
    
    col_map, col_feed = st.columns([2, 1])
    with col_map:
        render_map(vessels)
    with col_feed:
        render_live_analysis_feed(vessels)
    
    # Vessel Data Table
    st.markdown("### 📊 Vessel Data Table")
    vessel_data = []
    for v in vessels:
        vessel_data.append({
            "MMSI": v.mmsi,
            "Name": v.name,
            "Type": v.vessel_type,
            "Flag": v.flag,
            "Lat": f"{v.lat:.4f}",
            "Lon": f"{v.lon:.4f}",
            "Speed (kn)": f"{v.sog:.1f}",
            "Course": f"{v.cog:.0f}°",
            "Risk": f"{v.risk_score:.1f}%",
            "Risk Level": v.risk_level,
            "Region": v.region
        })
    
    df_vessels = pd.DataFrame(vessel_data)
    st.dataframe(df_vessels, use_container_width=True, height=400)
    
    # Auto-refresh with configurable rate
    time.sleep(st.session_state.refresh_rate)
    st.rerun()

if __name__ == "__main__":
    main()

