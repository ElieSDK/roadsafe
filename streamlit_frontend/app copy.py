
# import os
# from pathlib import Path
# from datetime import datetime
# import io

# import streamlit as st
# from PIL import Image
# from streamlit_js_eval import get_geolocation
# from streamlit_folium import st_folium
# import folium

# # ---------- Page config ----------
# st.set_page_config(
#     page_title="Photo + Location Uploader",
#     page_icon="📍",
#     layout="centered",
# )

# # ---------- Title & Intro ----------
# st.markdown(
#     """
#     <style>
#     .big-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.25rem;}
#     .subtle {opacity: 0.7; margin-top: 0; margin-bottom: 1rem;}
#     .card {
#         background: #111;
#         border: 1px solid #2a2a2a;
#         border-radius: 18px;
#         padding: 18px;
#     }
#     </style>
#     <div class="big-title">📷 Upload a photo — auto-capture your location</div>
#     <p class="subtle">When you upload an image, we'll ask the browser for your current GPS coordinates (with your permission).</p>
#     """,
#     unsafe_allow_html=True,
# )

# uploads_dir = Path("uploads")
# uploads_dir.mkdir(exist_ok=True)

# # ---------- File Uploader ----------
# with st.container():
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     file = st.file_uploader("Drop an image here (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
#     st.markdown("</div>", unsafe_allow_html=True)

# loc = None
# if file is not None:
#     # Load image and preview
#     image = Image.open(file).convert("RGB")
#     st.image(image, caption="Preview", use_column_width=True)

#     # Ask browser for geolocation (will prompt the user)
#     with st.spinner("Requesting your location from the browser..."):
#         loc = get_geolocation()

#     # Show location results
#     if loc and all(k in loc for k in ["coords"]):
#         coords = loc["coords"]
#         lat = coords.get("latitude", None)
#         lon = coords.get("longitude", None)
#         acc = coords.get("accuracy", None)

#         if lat is not None and lon is not None:
#             st.success(f"📍 Location captured: **{lat:.6f}, {lon:.6f}** (± ~{int(acc) if acc else '—'} m)")

#             # Map preview
#             m = folium.Map(location=[lat, lon], zoom_start=15)
#             folium.Marker([lat, lon], popup="You are here").add_to(m)
#             st_folium(m, width=700, height=420)

#             # Save the image & metadata
#             ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
#             fname = f"{ts}_{lat:.6f}_{lon:.6f}.jpg"
#             out_path = uploads_dir / fname
#             image.save(out_path, format="JPEG", quality=92)

#             meta = {
#                 "timestamp_utc": ts,
#                 "latitude": lat,
#                 "longitude": lon,
#                 "accuracy_m": acc,
#                 "filename": fname,
#             }
#             # Write a simple per-upload JSON sidecar
#             with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
#                 import json as _json
#                 _json.dump(meta, f, ensure_ascii=False, indent=2)

#             st.info("Saved locally to ./uploads/ with a JSON sidecar file.")
#         else:
#             st.warning("Couldn't read latitude/longitude from the browser response.")
#     else:
#         st.warning("Location access was denied or unavailable. You can allow location in your browser and try again.")

# # ---------- Footer ----------
# with st.expander("ℹ️ How it works / privacy"):
#     st.write(
#         "- Your location is requested **by your browser** using HTML5 Geolocation.\n"
#         "- If you deny the permission, we won't receive any coordinates.\n"
#         "- Uploaded photos and their metadata are saved locally in the `uploads/` folder."
#     )
