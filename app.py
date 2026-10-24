"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone - Project 1
Run:  streamlit run app.py
"""
import json
import re
import urllib.parse
from datetime import datetime

import streamlit as st
from PIL import Image

try:
    import google.generativeai as genai
except ImportError:  # app still works in demo mode
    genai = None

st.set_page_config(page_title="Construction Intelligence", page_icon="🏗️", layout="wide")

# ───────────────────────────── Styling ─────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500;600&display=swap');
:root{--ink:#1d2327;--slab:#eceae4;--steel:#4a5560;--hi-vis:#f2b705;--ok:#2e7d4f;--warn:#c77700;--bad:#b3261e;}
html,body,[class*="css"]{font-family:'Barlow',sans-serif;}
h1,h2,h3,.hero-title{font-family:'Barlow Condensed',sans-serif!important;letter-spacing:.2px;}
.hero{background:var(--ink);color:#fff;padding:1.4rem 1.6rem;border-radius:6px;border-left:10px solid var(--hi-vis);margin-bottom:1rem;}
.hero-title{font-size:2.2rem;font-weight:700;margin:0;line-height:1.1;}
.hero-sub{color:#c9ced3;margin:.3rem 0 0;}
.metric{background:var(--slab);border-radius:6px;padding:.9rem 1rem;border-bottom:4px solid var(--steel);}
.metric .v{font-family:'Barlow Condensed';font-size:2rem;font-weight:700;color:var(--ink);line-height:1.1;}
.metric .l{color:var(--steel);font-size:.85rem;}
.pill{display:inline-block;padding:.15rem .7rem;border-radius:99px;font-weight:600;font-size:.82rem;color:#fff;}
.pill.low,.pill.ok{background:var(--ok);} .pill.medium,.pill.warn{background:var(--warn);}
.pill.high,.pill.bad{background:var(--bad);} .pill.critical{background:#6d0f0a;}
.alert{padding:.8rem 1rem;border-radius:6px;margin:.4rem 0;background:#fff7dc;border-left:6px solid var(--hi-vis);color:var(--ink);}
.alert.bad{background:#fdecea;border-left-color:var(--bad);}
.demo-note{font-size:.85rem;color:var(--steel);}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────── Helpers ─────────────────────────────
def metric(label, value):
    st.markdown(f'<div class="metric"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def pill(text, kind=None):
    kind = (kind or text).lower()
    return f'<span class="pill {kind}">{text}</span>'


def inr(amount: float) -> str:
    """Format a rupee amount as Crores / Lakhs."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:,.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount / 1e5:,.2f} Lakhs"
    return f"₹{amount:,.0f}"


def parse_json(text: str) -> dict:
    """Extract a JSON object from an LLM reply (handles ```json fences)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model reply did not contain JSON.")
    return json.loads(text[start : end + 1])


def gemini_model(system_instruction=None):
    genai.configure(api_key=st.session_state.api_key)
    kwargs = {"system_instruction": system_instruction} if system_instruction else {}
    return genai.GenerativeModel(st.session_state.model_name, **kwargs)


def live_mode() -> bool:
    return bool(st.session_state.get("api_key")) and genai is not None


def analyze_image(prompt: str, image: Image.Image, demo_result: dict) -> tuple[dict, bool]:
    """Returns (result, used_live_model). Falls back to demo data on any failure."""
    if not live_mode():
        return demo_result, False
    try:
        reply = gemini_model().generate_content([prompt, image])
        return parse_json(reply.text), True
    except Exception as exc:  # network, quota, bad JSON, invalid key
        st.warning(f"Live analysis failed ({type(exc).__name__}: {exc}). Showing demonstration output instead.")
        return demo_result, False


def load_image(upload) -> Image.Image:
    img = Image.open(upload)
    img = img.convert("RGB")
    img.thumbnail((1280, 1280))
    return img


# ───────────────────────────── Prompts & demo data ─────────────────────────────
PPE_PROMPT = """You are a certified construction safety officer auditing a site photo against OSHA 1926 and
IS 456 / IS 3764 site-safety practice. Check each visible worker for: helmet, high-vis vest, steel-toe boots,
safety harness (when working at height). Respond with ONLY valid JSON in this schema:
{"safety_score": int 0-100,
 "overall_status": "Compliant" | "Needs Improvement" | "Non-Compliant",
 "workers_detected": int,
 "ppe_detection": {"helmet": bool, "high_vis_vest": bool, "steel_toe_boots": bool, "safety_harness": bool},
 "hazards": [{"hazard": str, "severity": "Low"|"Medium"|"High"|"Critical", "remedy": str}],
 "summary": str}
If something cannot be seen, mark it false and say so in the summary."""

CRACK_PROMPT = """You are a structural engineer diagnosing a concrete / masonry surface photo per IS 456 and
IS 13935. Respond with ONLY valid JSON in this schema:
{"defect_type": "Hairline" | "Settlement" | "Structural Shear" | "Thermal Expansion" | "Spalling",
 "estimated_width_mm": float,
 "severity_score": int 0-100,
 "urgency": "Monitor" | "Repair within 3 months" | "Repair within 2 weeks" | "Immediate - restrict access",
 "root_causes": [str],
 "recommended_repairs": [str],
 "summary": str}
Typical guide: <0.3 mm hairline; 0.3-1 mm repair-worthy; >1 mm or diagonal/stepped suggests structural concern."""

DEMO_PPE = {
    "safety_score": 62, "overall_status": "Needs Improvement", "workers_detected": 3,
    "ppe_detection": {"helmet": True, "high_vis_vest": True, "steel_toe_boots": False, "safety_harness": False},
    "hazards": [
        {"hazard": "Worker on elevated platform without harness", "severity": "Critical",
         "remedy": "Stop work at height; issue full-body harness anchored to a rated point."},
        {"hazard": "Open-toe or non-safety footwear observed", "severity": "High",
         "remedy": "Enforce steel-toe boot policy at site entry checkpoint."},
        {"hazard": "Loose material stacked near edge", "severity": "Medium",
         "remedy": "Install toe boards and relocate materials 2 m from the edge."},
        {"hazard": "Untidy walkway with exposed rebar ends", "severity": "Low",
         "remedy": "Fit rebar caps and clear the access route."},
    ],
    "summary": "DEMO OUTPUT - head and torso protection present; footwear and fall protection gaps found.",
}
DEMO_CRACK = {
    "defect_type": "Settlement", "estimated_width_mm": 1.8, "severity_score": 68,
    "urgency": "Repair within 2 weeks",
    "root_causes": ["Differential foundation settlement", "Poor soil compaction under footing", "Water ingress softening sub-grade"],
    "recommended_repairs": ["Install crack monitors (tell-tales) for 4 weeks", "Epoxy injection for stabilised cracks",
                            "Structural re-grouting / underpinning if movement continues", "Fix drainage near the foundation"],
    "summary": "DEMO OUTPUT - stepped diagonal crack pattern consistent with settlement; verify with a site inspection.",
}

# ───────────────────────────── Cost model ─────────────────────────────
STRUCTURES = {  # base construction cost INR per sqft, steel kg/sqft, cement bags/sqft
    "Residential (RCC Frame)": (2200, 4.0, 0.45),
    "Commercial Complex": (2800, 4.8, 0.5),
    "Industrial Shed (PEB)": (1700, 6.0, 0.3),
    "Luxury Villa": (3800, 4.5, 0.5),
    "Load-bearing Masonry": (1600, 1.5, 0.55),
}
TIERS = {  # (cost multiplier, market value INR per sqft)
    "Tier 1 (Mumbai, Delhi NCR, Bengaluru)": (1.25, 12000),
    "Tier 2 (Pune, Jaipur, Lucknow, Kochi)": (1.0, 6500),
    "Tier 3 (Smaller towns & districts)": (0.8, 3200),
}


def estimate(structure, sqft, tier, pct):
    rate, steel, cement = STRUCTURES[structure]
    mult, market = TIERS[tier]
    budget = rate * mult * sqft * 1.12  # +12% contingency & approvals
    spent = budget * pct / 100
    land_building = market * sqft
    # Asset value grows with completion: land share (35%) plus built share by completion
    asset = land_building * (0.35 + 0.65 * pct / 100)
    return dict(budget=budget, spent=spent, asset=asset, full_value=land_building,
                steel_t=steel * sqft / 1000, cement_bags=cement * sqft)


MILESTONES = [(0, "Site prep"), (10, "Foundation"), (30, "Structure"), (55, "Masonry & services"),
              (75, "Finishing"), (95, "Handover")]

WEATHER = {
    "Normal": ("ok", ["Standard toolbox talk at 8:00 AM.", "Routine inspection of scaffolds and lifting gear.",
                      "Maintain wet-curing of concrete for at least 7 days (IS 456)."]),
    "High Wind": ("bad", ["Suspend crane and hoist operations when gusts exceed 36 km/h; stow jibs.",
                          "Secure loose sheeting, formwork and scaffold netting.",
                          "Halt work on exposed edges and tall scaffolds.", "Keep personnel out of the crane swing zone."]),
    "Heavy Rain": ("bad", ["Electrical hazard: isolate temporary power, check RCCB/ELCB, keep cables off wet ground.",
                           "Do not pour concrete; cover fresh work with tarpaulin.", "Pump out excavations; check shoring for collapse risk.",
                           "Anti-slip measures on ramps and walkways."]),
    "Extreme Heat": ("warn", ["Shift heavy work to early morning and late afternoon; mandatory hydration breaks.",
                              "Concrete curing: pour in cooler hours, use retarders, wet-cure and shade slabs (avoid plastic shrinkage cracks).",
                              "Monitor workers for heat-stress symptoms; provide shaded rest areas.", "Check hot-surface contact on steel and formwork."]),
}

CHAT_SYSTEM = ("You are a senior Indian construction engineer assisting on-site teams. Give concise, practical answers "
               "referencing IS codes (IS 456, IS 800, IS 875, NBC 2016) and OSHA where relevant. Use INR for costs. "
               "Say so when a question needs a licensed engineer's on-site verification.")


def demo_chat_reply(q: str) -> str:
    q = q.lower()
    if "m20" in q or "mix" in q:
        return "M20 nominal mix is 1:1.5:3 (cement:sand:aggregate), roughly 8 bags of cement per m³ with w/c ≈ 0.5. Cure for at least 7 days."
    if "curing" in q or "cure" in q:
        return "Start curing 12-24 h after placing. Keep concrete continuously wet for 7 days (10 days for blended cements or hot weather), per IS 456."
    if "crack" in q:
        return "Cracks under 0.3 mm are usually cosmetic; above that, find the cause first. Monitor, then seal or epoxy-inject. Diagonal or widening cracks need an engineer."
    if "steel" in q or "rebar" in q:
        return "Typical RCC buildings use 3.5-4.5 kg of steel per sqft built-up. Use Fe500D with cover per IS 456 exposure class."
    if "safety" in q or "ppe" in q:
        return "Minimum PPE: helmet, high-vis vest, safety boots, gloves; harness above 1.8 m. Run a toolbox talk daily."
    return "Demo mode: add a Gemini API key in the sidebar for full answers. Try asking about curing, M20 mix, cracks, steel or PPE."


# ───────────────────────────── Session state ─────────────────────────────
st.session_state.setdefault("api_key", "")
st.session_state.setdefault("model_name", "gemini-3.5-flash-lite")
st.session_state.setdefault("chat", [])

# ───────────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.header("🏗️ Control room")
    key = st.text_input("Google Gemini API key", type="password", value=st.session_state.api_key,
                        help="Never stored on disk. Leave empty for demonstration mode.")
    st.session_state.api_key = key.strip()
    st.session_state.model_name = st.text_input("Model", value=st.session_state.model_name)
    if live_mode():
        st.success("Live mode: Gemini multimodal analysis")
    else:
        st.info("Demonstration mode: sample outputs, no API calls")
        if genai is None:
            st.caption("google-generativeai is not installed.")
    st.divider()
    st.subheader("Architecture")
    st.markdown(
        "- **Vision:** image + JSON-contract prompt → Gemini\n- **Parser:** fence-safe JSON extraction\n"
        "- **Cost engine:** rate × tier × area\n- **Weather rules:** local rule base\n- **Assistant:** multi-turn Gemini chat\n"
        "- **Storage:** none (session only)"
    )
    st.subheader("Modules")
    st.markdown("1. Safety & PPE audit\n2. Crack diagnostics\n3. Planning & valuation\n4. Site assistant")

st.markdown(
    '<div class="hero"><p class="hero-title">Multimodal Construction Intelligence Platform</p>'
    '<p class="hero-sub">Safety audits, crack diagnostics, cost and valuation, and weather-aware site protocols.</p></div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["🦺 Safety & PPE", "🧱 Crack diagnostics", "📐 Planning & valuation", "🌦️ Site assistant"])

# ───────────────────────────── Tab 1: PPE ─────────────────────────────
with tab1:
    st.subheader("Site safety and PPE audit")
    up = st.file_uploader("Upload a site photo", type=["jpg", "jpeg", "png"], key="ppe_up")
    if up:
        try:
            img = load_image(up)
        except Exception as exc:
            st.error(f"Could not read that image: {exc}")
            img = None
        if img:
            c1, c2 = st.columns([1, 1.2])
            c1.image(img, use_container_width=True)
            if c2.button("Run safety audit", type="primary", key="ppe_btn"):
                with st.spinner("Auditing site photo..."):
                    st.session_state.ppe, st.session_state.ppe_live = analyze_image(PPE_PROMPT, img, DEMO_PPE)
    elif "ppe" not in st.session_state:
        st.caption("Upload a JPG or PNG to begin. Demonstration mode returns sample output.")

    r = st.session_state.get("ppe")
    if r:
        try:
            if not st.session_state.get("ppe_live"):
                st.markdown('<p class="demo-note">Demonstration output</p>', unsafe_allow_html=True)
            score = int(r.get("safety_score", 0))
            status = r.get("overall_status", "Unknown")
            kind = "ok" if score >= 80 else "warn" if score >= 55 else "bad"
            m1, m2, m3 = st.columns(3)
            with m1: metric("Safety score", f"{score}/100")
            with m2: metric("Workers detected", r.get("workers_detected", "–"))
            with m3: st.markdown(f'<div class="metric"><div class="v">{pill(status, kind)}</div><div class="l">Overall status</div></div>', unsafe_allow_html=True)
            st.progress(max(0, min(score, 100)) / 100)
            st.markdown("**PPE checklist**")
            labels = {"helmet": "Helmet", "high_vis_vest": "High-vis vest", "steel_toe_boots": "Steel-toe boots", "safety_harness": "Safety harness"}
            cols = st.columns(4)
            for col, (k, name) in zip(cols, labels.items()):
                ok = bool(r.get("ppe_detection", {}).get(k))
                col.markdown(f"{name}<br>{pill('Present' if ok else 'Missing', 'ok' if ok else 'bad')}", unsafe_allow_html=True)
            st.markdown("**Identified hazards**")
            for h in r.get("hazards", []):
                sev = h.get("severity", "Low")
                st.markdown(f"{pill(sev)} &nbsp;**{h.get('hazard','')}**<br><span class='demo-note'>Remedy: {h.get('remedy','')}</span>", unsafe_allow_html=True)
                st.write("")
            st.info(r.get("summary", ""))
            st.download_button("Download JSON report", json.dumps(r, indent=2), "ppe_safety_report.json", "application/json")
        except Exception as exc:
            st.error(f"Unexpected report format: {exc}")
            st.json(r)

# ───────────────────────────── Tab 2: Cracks ─────────────────────────────
with tab2:
    st.subheader("Crack diagnostics and structural integrity")
    up = st.file_uploader("Upload a concrete or wall surface photo", type=["jpg", "jpeg", "png"], key="crack_up")
    if up:
        try:
            img = load_image(up)
        except Exception as exc:
            st.error(f"Could not read that image: {exc}")
            img = None
        if img:
            c1, c2 = st.columns([1, 1.2])
            c1.image(img, use_container_width=True)
            if c2.button("Diagnose defect", type="primary", key="crack_btn"):
                with st.spinner("Analysing surface..."):
                    st.session_state.crack, st.session_state.crack_live = analyze_image(CRACK_PROMPT, img, DEMO_CRACK)
    elif "crack" not in st.session_state:
        st.caption("Upload a close-up of the cracked surface, ideally with a scale reference such as a coin or ruler.")

    r = st.session_state.get("crack")
    if r:
        try:
            if not st.session_state.get("crack_live"):
                st.markdown('<p class="demo-note">Demonstration output</p>', unsafe_allow_html=True)
            sev = int(r.get("severity_score", 0))
            kind = "ok" if sev < 35 else "warn" if sev < 65 else "bad"
            m1, m2, m3 = st.columns(3)
            with m1: metric("Defect type", r.get("defect_type", "–"))
            with m2: metric("Estimated width", f"{r.get('estimated_width_mm', '–')} mm")
            with m3: metric("Severity score", f"{sev}/100")
            st.progress(max(0, min(sev, 100)) / 100)
            st.markdown(f"**Urgency:** {pill(r.get('urgency', '–'), kind)}", unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                st.markdown("**Probable root causes**")
                for x in r.get("root_causes", []): st.markdown(f"- {x}")
            with b:
                st.markdown("**Recommended repairs**")
                for x in r.get("recommended_repairs", []): st.markdown(f"- {x}")
            st.info(r.get("summary", ""))
            st.caption("AI screening only. Structural decisions need inspection by a licensed engineer.")
            st.download_button("Download JSON report", json.dumps(r, indent=2), "crack_diagnostic_report.json", "application/json")
        except Exception as exc:
            st.error(f"Unexpected report format: {exc}")
            st.json(r)

# ───────────────────────────── Tab 3: Planning ─────────────────────────────
with tab3:
    st.subheader("Project planning and real estate valuation")
    c1, c2 = st.columns(2)
    structure = c1.selectbox("Structure type", list(STRUCTURES))
    tier = c2.selectbox("Location tier", list(TIERS))
    sqft = c1.number_input("Built-up area (sqft)", min_value=100, max_value=1_000_000, value=2000, step=100)
    pct = c2.slider("Completion (%)", 0, 100, 40)
    e = estimate(structure, sqft, tier, pct)

    m1, m2, m3 = st.columns(3)
    with m1: metric("Total project budget", inr(e["budget"]))
    with m2: metric("Capital expenditure incurred", inr(e["spent"]))
    with m3: metric("Current asset valuation", inr(e["asset"]))
    m4, m5, m6 = st.columns(3)
    with m4: metric("Steel requirement", f"{e['steel_t']:.1f} tonnes")
    with m5: metric("Cement requirement", f"{e['cement_bags']:,.0f} bags")
    with m6: metric("Valuation on completion", inr(e["full_value"]))

    stage = [n for p, n in MILESTONES if pct >= p][-1]
    st.markdown(f"**Milestone: {stage}**")
    st.progress(pct / 100)
    st.caption(" → ".join(f"{n} ({p}%)" for p, n in MILESTONES))
    st.caption("Planning estimates include 12% contingency; valuation uses indicative market rates. Verify with local quotes and a registered valuer.")

    st.markdown("**Conceptual 3D rendering**")
    style = st.selectbox("Rendering style", ["Photorealistic architectural render", "Modern minimalist 3D visualization", "Sketch-to-3D concept"])
    default_prompt = (f"{style} of a {structure.split(' (')[0].lower()} building, {int(sqft)} sqft, "
                      f"Indian {tier.split(' (')[0]} city, golden hour, detailed facade, landscaping")
    prompt = st.text_area("Image prompt", default_prompt)
    if st.button("Generate concept render"):
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width=1024&height=640&nologo=true&seed={datetime.now().microsecond}"
        st.session_state.render_url = url
    if st.session_state.get("render_url"):
        try:
            st.image(st.session_state.render_url, caption="Generated via Pollinations AI (conceptual only)", use_container_width=True)
        except Exception as exc:
            st.error(f"Render service unavailable: {exc}")

# ───────────────────────────── Tab 4: Assistant ─────────────────────────────
with tab4:
    st.subheader("Weather-aware daily advisory")
    w = st.radio("Today's conditions", list(WEATHER), horizontal=True)
    kind, items = WEATHER[w]
    for it in items:
        st.markdown(f'<div class="alert {"bad" if kind == "bad" else ""}">{it}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Site engineering assistant")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if q := st.chat_input("Ask about curing, mix design, cracks, PPE, costs..."):
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                if live_mode():
                    history = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                               for m in st.session_state.chat[:-1]]
                    ctx = f"[Today's weather on site: {w}] {q}"
                    chat = gemini_model(CHAT_SYSTEM).start_chat(history=history)
                    ans = chat.send_message(ctx).text
                else:
                    ans = demo_chat_reply(q)
            except Exception as exc:
                ans = f"Assistant error ({type(exc).__name__}). Check your API key and model name.\n\nFallback: {demo_chat_reply(q)}"
            st.markdown(ans)
        st.session_state.chat.append({"role": "assistant", "content": ans})
    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()