import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from io import StringIO

# -----------------------------------------------------------
# CONFIG & GLOBAL CSS (CENTER ALL MAIN CONTENT)
# -----------------------------------------------------------
st.set_page_config(page_title="SPKC - Profile Matching", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------
# TITLE & HEADER
# -----------------------------------------------------------
st.title("Aplikasi - Sistem Pendukung Keputusan Cerdas (SPKC)")
st.subheader("Profile Matching")
st.write("")
st.write("Aplikasi ini diajukan untuk memenuhi tugas SPKC")
st.write("")
st.write("Aplikasi ini menghitung profile matching (gap analysis) dan meranking kandidat berdasarkan kesesuaian profil ideal.")

cols = st.columns(100)
with cols[17]:
    st.image(r"https://pertanian.uma.ac.id/wp-content/uploads/2019/12/UGM.png", width=1280)

st.write("Oleh:")
st.subheader("Nezar Abdilah Prakasa (563414)")
st.write("Program Studi Magister Kecerdasan Artifisial")
st.write("Fakultas Matematika dan Ilmu Pengetahuan Alam")
st.write("Universitas Gadjah Mada")
st.write("2025")

# -----------------------------------------------------------
# INPUT SECTION
# -----------------------------------------------------------
st.markdown("---")
st.subheader("Input Data")
use_dummy = st.checkbox("Gunakan data dummy", value=False)


uploaded_file = None
paste_data = None



if not use_dummy:
    # st.subheader("Upload File atau Paste Data")
    uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

    paste_data = st.text_area(
        "Atau paste tabel dari Excel (Ctrl+C → Ctrl+V):",
        placeholder="Candidate\tTechnical\tExperience\tCommunication\tLeadership\tProblem Solving\nAlice\t85\t75\t80\t82\t78"
    )

# Template
use_template = st.checkbox("Tampilkan & download sample template", value=False)
if use_template:
    template = pd.DataFrame({
        "Candidate":["Alice","Bob","Charlie","Dewi","Eka"],
        "Technical":[85,78,92,88,81],
        "Experience":[80,74,89,85,77],
        "Communication":[78,82,75,80,79],
        "Leadership":[82,79,90,83,78],
        "Problem Solving":[84,80,88,86,82]
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        template.to_excel(writer, index=False, sheet_name="template")
    buf.seek(0)
    st.download_button("Download template (xlsx)", buf, file_name="template_profile_matching.xlsx")

# -----------------------------------------------------------
# DATA HANDLING
# -----------------------------------------------------------

if use_dummy:
    st.success("Menggunakan Dummy Data")
    df_scores = pd.DataFrame({
        "Candidate":["Alice","Bob","Charlie","Dewi","Eka"],
        "Technical":[85,78,92,88,81],
        "Experience":[80,74,89,85,77],
        "Communication":[78,82,75,80,79],
        "Leadership":[82,79,90,83,78],
        "Problem Solving":[84,80,88,86,82]
    })

else:
    # PRIORITAS: PASTE → FILE UPLOAD
    if paste_data and paste_data.strip() != "":
        try:
            df_scores = pd.read_csv(StringIO(paste_data), sep=None, engine="python")
            st.success("Data berhasil dibaca dari paste!")
        except Exception as e:
            st.error(f"Paste error: {e}")
            st.stop()

    elif uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_scores = pd.read_csv(uploaded_file)
            else:
                df_scores = pd.read_excel(uploaded_file)
            st.success("File berhasil dibaca!")
        except:
            st.error("Format file tidak bisa dibaca. Pastikan format benar.")
            st.stop()
    else:
        st.warning("Upload file atau paste data terlebih dahulu.")
        st.stop()

# Identify column groups
cols = df_scores.columns.tolist()
candidate_col = cols[0]
criteria_cols = cols[1:]

# -----------------------------------------------------------
# GAP-WEIGHT TABLE
# -----------------------------------------------------------
default_map = {
    0: 5.0, 1: 4.5, -1: 4.0, 2: 3.5, -2: 3.0,
    3: 2.5, -3: 2.0, 4: 1.5, -4: 1.0
}

st.markdown("---")
st.subheader("Gap → Weight Conversion")

conv_df = pd.DataFrame(sorted(default_map.items()), columns=["Gap","Weight"])
conv_edit = st.data_editor(conv_df)
conv_dict = dict(zip(conv_edit["Gap"], conv_edit["Weight"]))

# -----------------------------------------------------------
# SHOW DATA
# -----------------------------------------------------------
st.header("Data Kandidat")
st.dataframe(df_scores)

# -----------------------------------------------------------
# CRITERIA METADATA
# -----------------------------------------------------------
st.markdown("---")
st.subheader("Atur Bobot & Tipe Kriteria")

criteria_meta = []
for c in criteria_cols:
    w = st.number_input(f"Bobot {c} (0–100)", 0.0, 100.0, 10.0, key=f"w_{c}")
    t = st.selectbox(f"Tipe {c}", ["Benefit","Cost"], key=f"t_{c}")
    criteria_meta.append({"name":c, "weight":w, "type":t})

# -----------------------------------------------------------
# IDEAL PROFILE
# -----------------------------------------------------------
st.markdown("---")
st.subheader("Profil Ideal")

ideal = {}
for c in criteria_cols:
    maxv = float(df_scores[c].max())
    minv = float(df_scores[c].min())
    is_cost = any(m['name']==c and m['type']=="Cost" for m in criteria_meta)
    default_val = minv if is_cost else maxv
    ideal[c] = st.number_input(f"Ideal {c}", value=default_val)

# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------
def map_gap_to_weight(gap, conv):
    g = int(round(gap))
    if g in conv:
        return conv[g]
    keys = sorted(conv.keys())
    nearest = min(keys, key=lambda x: abs(x - g))
    return conv[nearest]

def compute_profile_matching(df, cand_col, meta, ideal_dict, conv):
    df = df.copy()
    gaps = {}
    weights = {}
    for m in meta:
        c = m["name"]
        if m["type"] == "Benefit":
            gap = df[c] - ideal_dict[c]
        else:
            gap = ideal_dict[c] - df[c]
        gaps[c] = gap
        weights[c] = gap.apply(lambda g: map_gap_to_weight(g, conv))
    return pd.concat([df[[cand_col]],
                      pd.DataFrame({f"GAP_{k}":gaps[k] for k in gaps}),
                      pd.DataFrame({f"W_{k}":weights[k] for k in weights})],
                     axis=1)

# -----------------------------------------------------------
# CALCULATION
# -----------------------------------------------------------
results_basic = compute_profile_matching(df_scores, candidate_col, criteria_meta, ideal, conv_dict)

total_weight = sum(m["weight"] for m in criteria_meta)
for m in criteria_meta:
    m["norm_weight"] = m["weight"] / total_weight if total_weight > 0 else 0

nw = {m["name"]:m["norm_weight"] for m in criteria_meta}

st.subheader("Core & Secondary Factor")

core = st.multiselect("Pilih Core Factor", criteria_cols, default=criteria_cols[:len(criteria_cols)//2])
sec = [c for c in criteria_cols if c not in core]

def weighted_factor(row, group):
    num = sum(row[f"W_{c}"] * nw[c] for c in group)
    den = sum(nw[c] for c in group)
    return num/den if den > 0 else 0

results = results_basic.copy()
results["CF"] = results.apply(lambda r: weighted_factor(r, core), axis=1)
results["SF"] = results.apply(lambda r: weighted_factor(r, sec), axis=1)

pct_cf = st.slider("Persentase CF", 0, 100, 60)
pct_sf = 100 - pct_cf

results["Final Score"] = results["CF"]*(pct_cf/100) + results["SF"]*(pct_sf/100)
results["Rank"] = results["Final Score"].rank(ascending=False, method="min").astype(int)

# -----------------------------------------------------------
# OUTPUT TABLE
# -----------------------------------------------------------
st.subheader("Hasil Akhir")
st.dataframe(results.sort_values("Rank"))

# -----------------------------------------------------------
# SPIDER CHART
# -----------------------------------------------------------
st.subheader("Spiderweb Radar Chart")

criteria_list = criteria_cols
ideal_weight = conv_dict[0]

fig = go.Figure()

for _, row in results.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[row[f"W_{c}"] for c in criteria_list],
        theta=criteria_list,
        fill="toself",
        name=row[candidate_col],
        opacity=0.55
    ))

fig.add_trace(go.Scatterpolar(
    r=[ideal_weight]*len(criteria_list),
    theta=criteria_list,
    fill="toself",
    name="Ideal Profile",
    opacity=0.25
))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    showlegend=True,
    width=850,
    height=650
)

st.plotly_chart(fig)

# -----------------------------------------------------------
# DOWNLOAD RESULT
# -----------------------------------------------------------
buf2 = io.BytesIO()
with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
    results.to_excel(writer, index=False, sheet_name="results")
buf2.seek(0)

st.download_button("Download hasil (xlsx)", buf2, file_name="profile_matching_result.xlsx")



