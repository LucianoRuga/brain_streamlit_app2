import ssl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import community as community_louvain
from nilearn import datasets, surface
import streamlit as st

# ============================================================
# 0. FIX SSL PER NILEARN (se serve)
# ============================================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ============================================================
# 1. FUNZIONI UTILI
# ============================================================

@st.cache_resource
def load_fsaverage():
    """Scarica e carica le superfici fsaverage (una sola volta)."""
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    l_verts, l_faces = surface.load_surf_data(fsaverage.pial_left)
    r_verts, r_faces = surface.load_surf_data(fsaverage.pial_right)
    return l_verts, l_faces, r_verts, r_faces


def compute_connectome_figure(atlas, edges_ids, edges_names,
                              shrink=0.75, brain_opacity=0.6):
    """Costruisce la figura Plotly 3D del connectome."""

    # --- 1. Prepara dati di base ---
    atlas = atlas.reset_index(drop=True)
    n_nodes = len(atlas)

    # Matrice di adiacenza
    adj_matrix = np.zeros((n_nodes, n_nodes))
    id_to_index = {row.roi_id: idx for idx, row in atlas.iterrows()}

    for _, row in edges_ids.iterrows():
        i = id_to_index.get(row["source_id"])
        j = id_to_index.get(row["target_id"])
        if i is not None and j is not None:
            adj_matrix[i, j] = row["weight"]
            adj_matrix[j, i] = row["weight"]

    # Louvain
    G = nx.from_numpy_array(adj_matrix)
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    atlas["community"] = atlas.index.map(partition)
    n_comm = len(atlas["community"].unique())

    # --- 2. Carica superfici cervello ---
    l_verts, l_faces, r_verts, r_faces = load_fsaverage()

    # --- 3. Normalizzazione + recenter + shrink ---
    brain_min = np.min(np.vstack([l_verts, r_verts]), axis=0)
    brain_max = np.max(np.vstack([l_verts, r_verts]), axis=0)

    nodes_min = atlas[["x", "y", "z"]].min().values
    nodes_max = atlas[["x", "y", "z"]].max().values

    scale = (brain_max - brain_min) / (nodes_max - nodes_min)
    atlas[["x", "y", "z"]] = (atlas[["x", "y", "z"]] - nodes_min) * scale + brain_min

    mesh_center = (brain_min + brain_max) / 2
    nodes_center = atlas[["x", "y", "z"]].mean().values
    atlas[["x", "y", "z"]] = atlas[["x", "y", "z"]] - nodes_center + mesh_center

    atlas[["x", "y", "z"]] = mesh_center + (atlas[["x", "y", "z"]] - mesh_center) * shrink

    coords = atlas[["x", "y", "z"]].values
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # --- 4. Mesh cervello ---
    brain_lh = go.Mesh3d(
        x=l_verts[:, 0], y=l_verts[:, 1], z=l_verts[:, 2],
        i=l_faces[:, 0], j=l_faces[:, 1], k=l_faces[:, 2],
        color='lightgray',
        opacity=brain_opacity,
        hoverinfo='none',
        name='LH'
    )

    brain_rh = go.Mesh3d(
        x=r_verts[:, 0], y=r_verts[:, 1], z=r_verts[:, 2],
        i=r_faces[:, 0], j=r_faces[:, 1], k=r_faces[:, 2],
        color='lightgray',
        opacity=brain_opacity,
        hoverinfo='none',
        name='RH'
    )

    # --- 5. Archi con nomi (se edges_names disponibile) ---
    edge_x, edge_y, edge_z = [], [], []
    edge_text = []

    if edges_names is not None:
        # assumo stessa lunghezza e ordine di edges_ids
        for idx, row in edges_names.iterrows():
            src_name = row["source_name"]
            trg_name = row["target_name"]

            src_id = edges_ids.iloc[idx]["source_id"]
            trg_id = edges_ids.iloc[idx]["target_id"]

            i = id_to_index[src_id]
            j = id_to_index[trg_id]

            edge_x += [x[i], x[j], None]
            edge_y += [y[i], y[j], None]
            edge_z += [z[i], z[j], None]

            edge_text.append(f"{src_name} → {trg_name}")
    else:
        # fallback: niente nomi
        for _, row in edges_ids.iterrows():
            i = id_to_index[row["source_id"]]
            j = id_to_index[row["target_id"]]
            edge_x += [x[i], x[j], None]
            edge_y += [y[i], y[j], None]
            edge_z += [z[i], z[j], None]
            edge_text.append(f"{row['source_id']} → {row['target_id']}")

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='cyan', width=1.5),
        hoverinfo='text',
        text=edge_text,
        name='Connessioni'
    )

    # --- 6. Nodi per community ---
    fig = go.Figure(data=[brain_lh, brain_rh, edge_trace])

    colors = px.colors.qualitative.Dark24
    unique_comm = sorted(atlas["community"].unique())

    for i_comm, c in enumerate(unique_comm):
        comm_df = atlas[atlas["community"] == c]
        fig.add_trace(go.Scatter3d(
            x=comm_df["x"],
            y=comm_df["y"],
            z=comm_df["z"],
            mode="markers",
            marker=dict(size=7, color=colors[i_comm % len(colors)], opacity=0.85),
            text=comm_df["roi_name"],
            hoverinfo="text",
            name=f"Community {c}"
        ))

    # --- 7. Layout ---
    fig.update_layout(
        title=f"Connectome — {n_comm} Community (Louvain)",
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )

    return fig


# ============================================================
# 2. INTERFACCIA STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Brain Connectome 3D",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>🧠 Brain Connectome 3D Viewer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color: gray;'>Carica i tuoi file CSV (atlas + connessioni) e visualizza il connectome in 3D.</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Parametri & File")

    atlas_file = st.file_uploader(
        "Atlas (CSV con colonne: roi_id, roi_name, x, y, z)",
        type=["csv"],
        help="Il file atlas_164.csv o equivalente."
    )

    edges_ids_file = st.file_uploader(
        "Edge list (ID) – CSV con colonne: source_id, target_id, weight",
        type=["csv"],
        help="Es. cond_music_sub-01_edgelist_ids.csv"
    )

    edges_names_file = st.file_uploader(
        "Edge list (Nomi) – opzionale",
        type=["csv"],
        help="Es. cond_music_sub-01_edgelist_names.csv"
    )

    st.markdown("---")
    shrink = st.slider(
        "Shrink dei nodi verso il centro del cervello",
        min_value=0.5, max_value=1.0, value=0.75, step=0.01,
        help="Valori più bassi = nodi più interni."
    )
    brain_opacity = st.slider(
        "Trasparenza cervello",
        min_value=0.1, max_value=1.0, value=0.6, step=0.05,
        help="0.1 = molto trasparente, 1.0 = opaco."
    )

    run_button = st.button("🚀 Genera Connectome 3D")

# Area principale
if run_button:
    if (atlas_file is None) or (edges_ids_file is None):
        st.error("Per favore carica almeno **Atlas** e **Edge list (ID)**.")
    else:
        with st.spinner("Elaborazione in corso..."):
            atlas_df = pd.read_csv(atlas_file)
            edges_ids_df = pd.read_csv(edges_ids_file)
            edges_names_df = pd.read_csv(edges_names_file) if edges_names_file is not None else None

            fig = compute_connectome_figure(
                atlas=atlas_df,
                edges_ids=edges_ids_df,
                edges_names=edges_names_df,
                shrink=shrink,
                brain_opacity=brain_opacity
            )

        st.success("Connectome generato!")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 Info sui dati"):
            st.write("**Numero di ROI:**", len(atlas_df))
            st.write("**Numero di connessioni (righe edge list):**", len(edges_ids_df))
else:
    st.info("⬅️ Carica i file nella sidebar, imposta i parametri e clicca **“Genera Connectome 3D”**.")
