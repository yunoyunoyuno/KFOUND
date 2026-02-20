#!/bin/bash
# ============================================================================
# KFOUND: Full Training & Evaluation Pipeline
# ============================================================================
# Usage:
#   ./run_pipeline.sh [OPTIONS]
#
# Examples:
#   # Full pipeline with defaults (KAN g9 d4, DINO backbone, with MaskCut)
#   ./run_pipeline.sh
#
#   # Skip decoder training (use pretrained KFOUND decoder weights)
#   ./run_pipeline.sh --skip-decoder-training
#
#   # Skip MaskCut (Stage 3) — recommended based on ablation study
#   ./run_pipeline.sh --skip-stage3
#
#   # Use DetCo backbone instead of DINO
#   ./run_pipeline.sh --backbone detco
#
#   # Quick run: skip decoder training + skip MaskCut
#   ./run_pipeline.sh --skip-decoder-training --skip-stage3
#
#   # Full pipeline with specific seed
#   ./run_pipeline.sh --seed 42
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Default Parameters
# ============================================================================
SEED=0
DECODER="Kan"
GRID=9
DEGREE=4
SMOOTHING="dcrf"
BACKBONE="dino"
THRESHOLD=0.004
MASKCUT=1

SKIP_DECODER=false
SKIP_STAGE3=false
SKIP_EVAL=false

# Paths (relative to repo root)
DECODER_WEIGHT="./weights/decoder_weights_kfound_g9d4.pt"
DATA_DIR="./data"
KFOUND_CONFIG="./KFOUND/configs/kfound_DUTS-TR.yaml"

# ============================================================================
# Parse Arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)           SEED="$2"; shift 2 ;;
        --decoder)        DECODER="$2"; shift 2 ;;
        --grid)           GRID="$2"; shift 2 ;;
        --degree)         DEGREE="$2"; shift 2 ;;
        --smoothing)      SMOOTHING="$2"; shift 2 ;;
        --backbone)       BACKBONE="$2"; shift 2 ;;
        --threshold)      THRESHOLD="$2"; shift 2 ;;
        --skip-decoder-training)  SKIP_DECODER=true; shift ;;
        --skip-stage3)    SKIP_STAGE3=true; MASKCUT=0; shift ;;
        --skip-evaluation) SKIP_EVAL=true; shift ;;
        --decoder-weight) DECODER_WEIGHT="$2"; shift 2 ;;
        --help|-h)
            head -28 "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Derived paths
if [ "$DECODER" == "Kan" ]; then
    VARIANT="KanG${GRID}D${DEGREE}_seed${SEED}"
    DECODER_ARG="KAN"
else
    VARIANT="Conv_seed${SEED}"
    DECODER_ARG="Conv"
fi

MASK_DIR="${DATA_DIR}/coco20k_kfound_${SMOOTHING}/${VARIANT}"
LABEL_DIR="${DATA_DIR}/coco20k_kfound_${SMOOTHING}/${VARIANT}"
ANNO_DIR="./pred_annotations/${VARIANT}"

if [ "$MASKCUT" -eq 1 ]; then
    MK_SUFFIX="mk"
else
    MK_SUFFIX="no_mk"
fi

if [ "$BACKBONE" == "dino" ]; then
    OUTPUT_MODEL_DIR="./output_faster_rcnn/${VARIANT}"
else
    OUTPUT_MODEL_DIR="./output_detco/${VARIANT}"
fi

# ============================================================================
# Print Configuration
# ============================================================================
echo "=============================================="
echo "  KFOUND Pipeline Configuration"
echo "=============================================="
echo "  Seed:           ${SEED}"
echo "  Decoder:        ${DECODER} (grid=${GRID}, degree=${DEGREE})"
echo "  Smoothing:      ${SMOOTHING}"
echo "  Backbone:       ${BACKBONE}"
echo "  Threshold:      ${THRESHOLD}"
echo "  MaskCut:        $([ "$MASKCUT" -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
echo "  Skip decoder:   ${SKIP_DECODER}"
echo "  Decoder weight: ${DECODER_WEIGHT}"
echo "  Output model:   ${OUTPUT_MODEL_DIR}"
echo "=============================================="
echo ""

# ============================================================================
# Stage 0: Train KFOUND Decoder (optional)
# ============================================================================
if [ "$SKIP_DECODER" = false ]; then
    echo ""
    echo "############################################################"
    echo "# Stage 0: Train KFOUND Decoder on DUTS-TR"
    echo "############################################################"
    
    cd KFOUND
    python3 main_kfound_train.py \
        --decoder "${DECODER}" \
        --g "${GRID}" \
        --d "${DEGREE}" \
        --s "${SMOOTHING}" \
        --seed "${SEED}" \
        --log-dir ./outputs \
        --dataset-dir "../${DATA_DIR}/"
    cd ..
    
    # Find the best decoder weights (latest checkpoint)
    DECODER_OUTPUT_DIR="KFOUND/outputs/${SMOOTHING}/${DECODER}/KFOUND-DUTS-TR-${GRID}-d${DEGREE}"
    LATEST_WEIGHT=$(ls -t "${DECODER_OUTPUT_DIR}"/decoder_weights_niter*.pt 2>/dev/null | head -1)
    
    if [ -n "$LATEST_WEIGHT" ]; then
        DECODER_WEIGHT="$LATEST_WEIGHT"
        echo "Using trained decoder weights: ${DECODER_WEIGHT}"
    else
        echo "WARNING: No decoder weights found in ${DECODER_OUTPUT_DIR}"
        echo "Falling back to: ${DECODER_WEIGHT}"
    fi
else
    echo ""
    echo ">>> Skipping Stage 0 (decoder training)"
    echo ">>> Using pretrained weights: ${DECODER_WEIGHT}"
fi

# ============================================================================
# Stage 1: Background Extraction (generate masks on COCO20k)
# ============================================================================
echo ""
echo "############################################################"
echo "# Stage 1: Background Extraction"
echo "############################################################"

for MODE in train val; do
    echo ""
    echo "--- Stage 1: ${MODE} set ---"
    python3 stage1_background_extraction.py \
        --mode "${MODE}" \
        --decoder "${DECODER_ARG}" \
        --seed "${SEED}" \
        --g "${GRID}" \
        --d "${DEGREE}" \
        --s "${SMOOTHING}" \
        --weight "${DECODER_WEIGHT}" \
        --output_dir "${MASK_DIR}"
done

# ============================================================================
# Stage 2-3: Create Pseudo Bounding Boxes
# ============================================================================
echo ""
echo "############################################################"
echo "# Stage 2-3: Create Pseudo Bounding Boxes"
if [ "$MASKCUT" -eq 1 ]; then
    echo "# (with MaskCut complementary generation)"
else
    echo "# (without MaskCut — recommended)"
fi
echo "############################################################"

for MODE in train val; do
    echo ""
    echo "--- Stage 2-3: ${MODE} set ---"
    python3 stage2_3_create_pseudo_bboxes.py \
        --lower-threshold "${THRESHOLD}" \
        --upper-threshold 1.01 \
        --mode "${MODE}" \
        --use_maskcut "${MASKCUT}" \
        --seed "${SEED}" \
        --labels_output_dir "${LABEL_DIR}/" \
        --verbose \
        --mask_folder "${MASK_DIR}/"
done

# ============================================================================
# Pre-Stage 4: Convert YOLO labels to COCO JSON format
# ============================================================================
echo ""
echo "############################################################"
echo "# Pre-Stage 4: Annotation Conversion (YOLO → COCO JSON)"
echo "############################################################"

for MODE in train val; do
    echo ""
    echo "--- Pre-Stage 4: ${MODE} set ---"
    python3 pre_stage4.py \
        --mode "${MODE}" \
        --type "${MK_SUFFIX}" \
        --t "${THRESHOLD}" \
        --label_basename "${LABEL_DIR}/" \
        --output_path "${ANNO_DIR}/"
done

# ============================================================================
# Stage 4: Train Faster R-CNN Class-Agnostic Detector
# ============================================================================
echo ""
echo "############################################################"
echo "# Stage 4: Train Faster R-CNN (${BACKBONE} backbone)"
echo "############################################################"

if [ "$BACKBONE" == "dino" ]; then
    python3 stage4_train_class_agnostic_detector.py \
        --seed "${SEED}" \
        --annotation_train "${ANNO_DIR}/train/pseudo_bboxes_${THRESHOLD}_${MK_SUFFIX}.json" \
        --annotation_val "${ANNO_DIR}/val/pseudo_bboxes_${THRESHOLD}_${MK_SUFFIX}.json" \
        --output_dir "${OUTPUT_MODEL_DIR}"
elif [ "$BACKBONE" == "detco" ]; then
    python3 stage4_train_detco_detector.py \
        --seed "${SEED}" \
        --annotation_train "${ANNO_DIR}/train/pseudo_bboxes_${THRESHOLD}_${MK_SUFFIX}.json" \
        --annotation_val "${ANNO_DIR}/val/pseudo_bboxes_${THRESHOLD}_${MK_SUFFIX}.json" \
        --output_dir "${OUTPUT_MODEL_DIR}"
else
    echo "ERROR: Unknown backbone: ${BACKBONE} (use 'dino' or 'detco')"
    exit 1
fi

# Verify training output
MODEL_FINAL="${OUTPUT_MODEL_DIR}/model_final.pth"
if [ ! -f "$MODEL_FINAL" ]; then
    echo "ERROR: Training failed! Model file not found: ${MODEL_FINAL}"
    exit 1
fi
echo "Training complete! Model saved: ${MODEL_FINAL}"

# ============================================================================
# Stage 5: Evaluation (optional)
# ============================================================================
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "############################################################"
    echo "# Stage 5: Evaluation"
    echo "############################################################"

    echo ""
    echo "--- Evaluating on COCO20k (train) ---"
    python3 test_detector_coco.py \
        --model_weights "${MODEL_FINAL}" \
        --split train 2>&1 | tee "${OUTPUT_MODEL_DIR}/eval_coco_train.txt"

    echo ""
    echo "--- Evaluating on COCO20k (val) ---"
    python3 test_detector_coco.py \
        --model_weights "${MODEL_FINAL}" \
        --split val 2>&1 | tee "${OUTPUT_MODEL_DIR}/eval_coco_val.txt"

    echo ""
    echo "--- Evaluating on VOC 2007 ---"
    python3 test_detector_voc.py \
        --model_weights "${MODEL_FINAL}" \
        --year 2007 2>&1 | tee "${OUTPUT_MODEL_DIR}/eval_voc07.txt"

    echo ""
    echo "--- Evaluating on VOC 2012 ---"
    python3 test_detector_voc.py \
        --model_weights "${MODEL_FINAL}" \
        --year 2012 2>&1 | tee "${OUTPUT_MODEL_DIR}/eval_voc12.txt"
else
    echo ""
    echo ">>> Skipping evaluation"
fi

# ============================================================================
echo ""
echo "=============================================="
echo "  KFOUND Pipeline Complete!"
echo "=============================================="
echo "  Model: ${MODEL_FINAL}"
echo "=============================================="
