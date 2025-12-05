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
    /* CENTER THE MAIN CONTAINER */
    .block-container {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }

    /* CENTER ALL TEXT ELEMENTS */
    h1, h2, h3, h4, h5, h6, p, div, span {
        text-align: center !important;
    }

    /* CENTER ALL IMAGES */
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
st.title("Model Sistem Pendukung Keputusan Cerdas untuk Menentukan Kandidat Terbaik dalam Promosi Jabatan")
st.subheader("Profile Matching")
st.write("Aplikasi ini diajukan untuk memenuhi tugas SPKC")

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("https://ugm.ac.id/wp-content/uploads/2022/11/LOGO-UGM-BAKU-tnp-back-grou-300x300.jpg")

with col3:
    st.write(' ')

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
        except Exception as e:
            st.error(f"Format file tidak bisa dibaca. Pastikan format benar. ({e})")
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
# IDEAL PROFILE
# -----------------------------------------------------------
st.markdown("---")
st.subheader("Profil Ideal")

ideal = {}
for c in criteria_cols:
    maxv = float(df_scores[c].max())
    # default = max (karena semua benefit)
    default_val = maxv
    ideal[c] = st.number_input(f"Ideal {c}", value=default_val, key=f"ideal_{c}")

# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------
def map_gap_to_weight(gap, conv):
    # round to nearest integer gap then map; if beyond keys, find nearest key
    g = int(round(gap))
    if g in conv:
        return conv[g]
    keys = sorted(conv.keys())
    nearest = min(keys, key=lambda x: abs(x - g))
    return conv[nearest]

def compute_profile_matching(df, cand_col, criteria_list, ideal_dict, conv):
    df = df.copy()
    gaps = {}
    weights = {}
    for c in criteria_list:
        # All treated as Benefit (candidate - ideal)
        gap = df[c] - ideal_dict[c]
        gaps[c] = gap
        weights[c] = gap.apply(lambda g: map_gap_to_weight(g, conv))
    return pd.concat([df[[cand_col]],
                      pd.DataFrame({f"GAP_{k}":gaps[k] for k in gaps}),
                      pd.DataFrame({f"W_{k}":weights[k] for k in weights})],
                     axis=1)

# -----------------------------------------------------------
# CALCULATION
# -----------------------------------------------------------
results_basic = compute_profile_matching(df_scores, candidate_col, criteria_cols, ideal, conv_dict)

st.markdown("---")
st.subheader("Core & Secondary Factor")

core = st.multiselect("Pilih Core Factor", criteria_cols, default=criteria_cols[:len(criteria_cols)//2])
sec = [c for c in criteria_cols if c not in core]

def simple_avg_factor(row, group):
    # simple average of W_{c} across group (equal weighting per variable)
    if len(group) == 0:
        return 0
    s = sum(row[f"W_{c}"] for c in group)
    return s / len(group)

results = results_basic.copy()
results["CF"] = results.apply(lambda r: simple_avg_factor(r, core), axis=1)
results["SF"] = results.apply(lambda r: simple_avg_factor(r, sec), axis=1)

pct_cf = st.slider("Persentase CF (%)", 0, 100, 60)
st.write("Secondary Factor otomatis terhitung 1-CF")
pct_sf = 100 - pct_cf

results["Final Score"] = results["CF"]*(pct_cf/100) + results["SF"]*(pct_sf/100)
results["Rank"] = results["Final Score"].rank(ascending=False, method="min").astype(int)

# -----------------------------------------------------------
# OUTPUT TABLE
# -----------------------------------------------------------
st.subheader("Hasil Akhir")
st.dataframe(results.sort_values("Rank").reset_index(drop=True))

# -----------------------------------------------------------
# SPIDER CHART
# -----------------------------------------------------------
st.subheader("Spiderweb Radar Chart")

criteria_list = criteria_cols
ideal_weight = conv_dict.get(0, list(conv_dict.values())[0])

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
