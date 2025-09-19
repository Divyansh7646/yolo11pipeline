import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
from datetime import datetime

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="iGPS Pallet Detector", layout="centered")

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "weights/best_final.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logo processing thresholds
IOU_THRESHOLD_LOGO_INSIDE = 0.5
DISTANCE_THRESHOLD_NEARBY = 50

# ==========================
# UTILITY FUNCTIONS
# ==========================
def iou(box1, box2):
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

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at {MODEL_PATH}. Please check the file path.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

# ==========================
# HEADER
# ==========================
st.title("📦 iGPS Pallet Detector")
st.write("Upload an image to detect **iGPS pallets**")

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ==========================
# CONFIDENCE SLIDER
# ==========================
confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# ==========================
# INFERENCE
# ==========================
if uploaded_file:
    with st.spinner("🔍 Running YOLO inference..."):
        try:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("❌ Failed to decode uploaded image. Please try a different image.")
                st.stop()

            # Run YOLO inference
            results = model(image, conf=confidence, device=DEVICE)
            result = results[0]

            # Annotate image safely
            try:
                annotated_image = result.plot()
                if annotated_image is not None:
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                else:
                    annotated_image_rgb = None
            except Exception as e:
                st.warning(f"⚠️ Could not generate annotated image: {str(e)}")
                annotated_image_rgb = None

            # ==========================
            # PROCESS DETECTIONS
            # ==========================
            igps_boxes = []
            non_igps_boxes = []
            logos = []
            detections = []

            # Check if boxes exist and have data
            if hasattr(result.boxes, 'data') and result.boxes.data is not None and len(result.boxes.data) > 0:
                for box in result.boxes.data.tolist():
                    x1, y1, x2, y2, conf_score, cls = box
                    cls_name = model.names[int(cls)]
                    bbox = [int(x1), int(y1), int(x2), int(y2)]

                    detections.append({
                        "class": cls_name,
                        "confidence": round(float(conf_score), 2),
                        "bbox": bbox
                    })

                    if cls_name == "igps":
                        igps_boxes.append(bbox)
                    elif cls_name == "non-igps":
                        non_igps_boxes.append(bbox)
                    elif cls_name == "igps_logo":
                        logos.append(bbox)

            # ==========================
            # LOGO PROCESSING
            # ==========================
            def is_logo_inside_pallet(logo_box, pallet_box, threshold=0.5):
                return iou(logo_box, pallet_box) > threshold

            def is_logo_near_pallet(logo_box, pallet_box, distance_threshold=DISTANCE_THRESHOLD_NEARBY):
                logo_center_x = (logo_box[0] + logo_box[2]) / 2
                logo_center_y = (logo_box[1] + logo_box[3]) / 2
                pallet_center_x = (pallet_box[0] + pallet_box[2]) / 2
                pallet_center_y = (pallet_box[1] + pallet_box[3]) / 2
                distance = ((logo_center_x - pallet_center_x)**2 + (logo_center_y - pallet_center_y)**2)**0.5
                return distance <= distance_threshold

            igps_count = len(igps_boxes)
            additional_igps_from_logos = []

            for logo in logos:
                logo_inside_detected_igps = any(is_logo_inside_pallet(logo, igps) for igps in igps_boxes)
                if logo_inside_detected_igps:
                    continue

                logo_near_igps = any(is_logo_near_pallet(logo, igps) for igps in igps_boxes)
                logo_in_non_igps_area = any(is_logo_inside_pallet(logo, non, 0.3) for non in non_igps_boxes)

                should_count_logo = False
                if logo_near_igps or logo_in_non_igps_area:
                    should_count_logo = True
                elif len(igps_boxes) == 0:
                    should_count_logo = True

                if should_count_logo:
                    additional_igps_from_logos.append(logo)
                    igps_count += 1

            igps_boxes.extend(additional_igps_from_logos)
            total_pallets = igps_count + len(non_igps_boxes)

            # ==========================
            # DISPLAY RESULTS
            # ==========================
            if detections:
                # Display annotated image with proper parameter
                if annotated_image_rgb is not None:
                    st.image(annotated_image_rgb, caption="🖼️ Detected Pallets", use_column_width=True)
                else:
                    # Display original image if annotation fails
                    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(original_rgb, caption="🖼️ Original Image (annotation failed)", use_column_width=True)
                    st.warning("🚫 Could not generate annotated image for display.")

                st.subheader(f"📦 Total Pallets Detected: {total_pallets}")

                original_igps_count = len([d for d in detections if d["class"] == "igps"])
                logos_processed = len(logos)
                additional_from_logos = len(additional_igps_from_logos)

                if additional_from_logos > 0:
                    st.success(f"✅ {additional_from_logos} additional iGPS pallet(s) identified through logo detection!")

                if logos_processed > additional_from_logos:
                    ignored_logos = logos_processed - additional_from_logos
                    st.info(f"ℹ️ {ignored_logos} logo(s) ignored (already within detected iGPS pallets)")

                st.markdown("### 📊 Number of Pallets Detected per Class")
                st.write(f"**igps:** {igps_count} (Original: {original_igps_count}, From logos: {additional_from_logos})")
                st.write(f"**non-igps:** {len(non_igps_boxes)}")
                if logos_processed > 0:
                    st.write(f"**logos detected:** {logos_processed}")

                st.markdown("### 🧾 Detection Details")
                st.dataframe(detections, use_container_width=True)

                json_data = json.dumps({
                    "filename": uploaded_file.name,
                    "timestamp": datetime.now().isoformat(),
                    "total_pallets": total_pallets,
                    "igps_count": igps_count,
                    "igps_breakdown": {
                        "original_detections": original_igps_count,
                        "additional_from_logos": additional_from_logos,
                        "logos_processed": logos_processed
                    },
                    "non_igps_count": len(non_igps_boxes),
                    "detections": detections,
                    "logo_processing_summary": {
                        "total_logos_detected": logos_processed,
                        "logos_counted_as_igps": additional_from_logos,
                        "logos_ignored_double_count": logos_processed - additional_from_logos
                    }
                }, indent=4)

                json_filename = f"results_{os.path.splitext(uploaded_file.name)[0]}.json"
                with st.expander("📄 View JSON Result"):
                    st.code(json_data, language="json")
                st.download_button(
                    label="📥 Download Detection JSON",
                    data=json_data,
                    file_name=json_filename,
                    mime="application/json"
                )
            else:
                st.warning("🚫 No pallets or logos detected in this image.")
                
        except Exception as e:
            st.error(f"❌ An error occurred during inference: {str(e)}")
            st.error("Please check your model file and try again.")
            
else:
    st.info("📤 Upload a marketplace image to begin.")