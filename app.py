import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
from datetime import datetime

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "weights/best_final.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logo processing thresholds
IOU_THRESHOLD_LOGO_INSIDE = 0.5  # IoU threshold for detecting logos inside detected pallets (avoid double counting)
DISTANCE_THRESHOLD_NEARBY = 50   # Distance threshold for considering logos near pallets (pixels)

# ==========================
# UTILITY FUNCTIONS
# ==========================
def iou(box1, box2):
    """Compute Intersection over Union (IoU) between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="iGPS Pallet Detector", layout="centered")
st.markdown("""
<style>
.main {background-color: #f4f4f8;}
h1 {text-align: center; color: #2c3e50; font-weight: 700; margin-bottom: 0.5rem;}
.subheader {text-align: center; font-size: 1rem; color: #555; margin-bottom: 2rem;}
.stButton>button {background-color: #2c3e50; color: white; border-radius: 6px; padding: 0.6rem 1.2rem; font-weight: 500;}
.stDownloadButton>button {background-color: #1f4e79; color: white;}
.confidence-slider {padding-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📦 iGPS Pallet Detector</h1>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Upload an image to detect <strong>iGPS pallets</strong></div>", unsafe_allow_html=True)

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# ==========================
# CONFIDENCE SLIDER
# ==========================
st.markdown("<div class='confidence-slider'>", unsafe_allow_html=True)
confidence = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# INFERENCE
# ==========================
if uploaded_file:
    with st.spinner("🔍 Running YOLO inference..."):
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run YOLO inference
        results = model(image, conf=confidence, device=DEVICE)
        result = results[0]
        annotated_image = result.plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # ==========================
        # PROCESS DETECTIONS
        # ==========================
        igps_boxes = []
        non_igps_boxes = []
        logos = []
        detections = []

        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf_score, cls = box
            cls_name = model.names[int(cls)]
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            detections.append({
                "class": cls_name,
                "confidence": round(conf_score, 2),
                "bbox": bbox
            })

            if cls_name == "igps":
                igps_boxes.append(bbox)
            elif cls_name == "non-igps":
                non_igps_boxes.append(bbox)
            elif cls_name == "igps_logo":
                logos.append(bbox)

        # ==========================
        # ENHANCED LOGO PROCESSING WITH 3 LOGICS
        # ==========================
        
        def is_logo_inside_pallet(logo_box, pallet_box, threshold=0.5):
            """Check if logo is significantly inside a pallet bounding box"""
            return iou(logo_box, pallet_box) > threshold
        
        def is_logo_near_pallet(logo_box, pallet_box, distance_threshold=DISTANCE_THRESHOLD_NEARBY):
            """Check if logo is near a pallet (within distance threshold)"""
            logo_center_x = (logo_box[0] + logo_box[2]) / 2
            logo_center_y = (logo_box[1] + logo_box[3]) / 2
            pallet_center_x = (pallet_box[0] + pallet_box[2]) / 2
            pallet_center_y = (pallet_box[1] + pallet_box[3]) / 2
            
            distance = ((logo_center_x - pallet_center_x)**2 + (logo_center_y - pallet_center_y)**2)**0.5
            return distance <= distance_threshold
        
        # Start with detected iGPS pallets
        igps_count = len(igps_boxes)
        additional_igps_from_logos = []
        
        # Process each logo with the 3 implemented logics
        for logo in logos:
            # LOGIC 1: Don't double count - check if logo is inside any detected iGPS pallet
            logo_inside_detected_igps = False
            for igps_pallet in igps_boxes:
                if is_logo_inside_pallet(logo, igps_pallet, IOU_THRESHOLD_LOGO_INSIDE):
                    logo_inside_detected_igps = True
                    break
            
            if logo_inside_detected_igps:
                # Skip this logo as it's already counted as part of detected iGPS pallet
                continue
            
            # LOGIC 2: Check if logo is near any iGPS pallet (don't ignore nearby detections)
            logo_near_igps = False
            for igps_pallet in igps_boxes:
                if is_logo_near_pallet(logo, igps_pallet):
                    # Even if iGPS pallet detection missed this logo's pallet, 
                    # but there's a nearby iGPS pallet, still count the logo
                    logo_near_igps = True
                    break
            
            # LOGIC 3: Check if logo is in a stack of non-iGPS pallets
            logo_in_non_igps_area = False
            for non_igps_pallet in non_igps_boxes:
                if is_logo_inside_pallet(logo, non_igps_pallet, 0.3):  # Lower threshold for non-iGPS area
                    logo_in_non_igps_area = True
                    break
            
            # Count the logo as additional iGPS pallet based on the logics
            should_count_logo = False
            
            if logo_near_igps:
                # LOGIC 2: Logo near iGPS pallet - count it (missed iGPS detection)
                should_count_logo = True
            elif logo_in_non_igps_area:
                # LOGIC 3: Logo detected in non-iGPS stack - count as iGPS pallet
                should_count_logo = True
            elif len(igps_boxes) == 0 and len(non_igps_boxes) > 0:
                # LOGIC 3: No iGPS detected but logo present in image with non-iGPS pallets
                should_count_logo = True
            elif len(igps_boxes) == 0 and len(non_igps_boxes) == 0:
                # Standalone logo with no other pallets detected
                should_count_logo = True
            
            if should_count_logo:
                additional_igps_from_logos.append(logo)
                igps_count += 1

        # Add the additional iGPS boxes identified from logos
        igps_boxes.extend(additional_igps_from_logos)
        
        total_pallets = igps_count + len(non_igps_boxes)

        # ==========================
        # DISPLAY RESULTS
        # ==========================
        if detections:
            st.image(annotated_image_rgb, caption="🖼️ Detected Pallets", use_container_width=True)
            st.markdown(f"### 📦 Total Pallets Detected: {total_pallets}")

            # Enhanced info about logo-based detections
            original_igps_count = len([d for d in detections if d["class"] == "igps"])
            logos_processed = len(logos)
            additional_from_logos = len(additional_igps_from_logos)
            
            if additional_from_logos > 0:
                st.success(f"✅ {additional_from_logos} additional iGPS pallet(s) identified through logo detection!")
                
            if logos_processed > additional_from_logos:
                ignored_logos = logos_processed - additional_from_logos
                st.info(f"ℹ️ {ignored_logos} logo(s) ignored (already within detected iGPS pallets - no double counting)")

            st.markdown("### 📊 Number of Pallets Detected per Class")
            st.write(f"**igps:** {igps_count} (Original: {original_igps_count}, From logos: {additional_from_logos})")
            st.write(f"**non-igps:** {len(non_igps_boxes)}")
            if logos_processed > 0:
                st.write(f"**logos detected:** {logos_processed}")

            st.markdown("### 🧾 Detection Details")
            st.dataframe(detections, use_container_width=True)

            # JSON output with enhanced logo information
            json_data = json.dumps({
                "filename": uploaded_file.name,
                "timestamp": datetime.now().isoformat(),
                "total_pallets": total_pallets,
                "igps_count": igps_count,
                "igps_breakdown": {
                    "original_detections": len([d for d in detections if d["class"] == "igps"]),
                    "additional_from_logos": len(additional_igps_from_logos),
                    "logos_processed": len(logos)
                },
                "non_igps_count": len(non_igps_boxes),
                "detections": detections,
                "logo_processing_summary": {
                    "total_logos_detected": len(logos),
                    "logos_counted_as_igps": len(additional_igps_from_logos),
                    "logos_ignored_double_count": len(logos) - len(additional_igps_from_logos)
                }
            }, indent=4)

            json_filename = f"results_{os.path.splitext(uploaded_file.name)[0]}.json"
            with st.expander("📄 View JSON Result"):
                st.code(json_data, language="json")
            st.download_button(
                label="📥 Download Detection JSON",
                data=json_data,
                file_name=json_filename,
                mime="application/json",
                use_container_width=True
            )
        else:
            st.warning("🚫 No pallets or logos detected in this image.")
else:
    st.info("📤 Upload a marketplace image to begin.")
