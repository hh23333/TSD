OUTPUT_DIR='logs/occ_duke/tsd_3'

# # python3 train_tsd.py \
# # --config_file configs/OCC_Duke/vit_base.yml \
# # INPUT.MASK_PREPROCESS eight \
# # MODEL.NUM_PARTS 8 \
# # SOLVER.BASE_LR 0.004 \
# # MODEL.DEVICE_ID "('0')" \
# # MODEL.WITH_FORE_HEAD True \
# # MODEL.WITH_DIVERSE_LOSS True \
# # MODEL.PRETRAIN_PATH logs/pretrain_model/jx_vit_base_p16_224-80ecf9dd.pth \
# # OUTPUT_DIR $OUTPUT_DIR

# python3 test_tsd.py \
# --config_file configs/OCC_Duke/vit_base.yml \
# TEST.WEIGHT $OUTPUT_DIR/transformer_90.pth \
# INPUT.MASK_PREPROCESS eight \
# MODEL.NUM_PARTS 8 \
# MODEL.DEVICE_ID "('0')" \
# MODEL.WITH_FORE_HEAD True \
# DATASETS.NAMES new_occ_duke \
# TEST.NEW_METRIC True \
# OUTPUT_DIR $OUTPUT_DIR

python3 train_tsd.py \
--config_file configs/OCC_Duke/vit_base.yml \
INPUT.MASK_PREPROCESS three \
MODEL.NUM_PARTS 3 \
SOLVER.BASE_LR 0.004 \
MODEL.DEVICE_ID "('1')" \
MODEL.WITH_FORE_HEAD True \
MODEL.WITH_DIVERSE_LOSS True \
MODEL.PRETRAIN_PATH logs/pretrain_model/jx_vit_base_p16_224-80ecf9dd.pth \
OUTPUT_DIR $OUTPUT_DIR

python3 test_tsd.py \
--config_file configs/OCC_Duke/vit_base.yml \
TEST.WEIGHT $OUTPUT_DIR/transformer_120.pth \
INPUT.MASK_PREPROCESS three \
MODEL.NUM_PARTS 3 \
MODEL.DEVICE_ID "('1')" \
MODEL.WITH_FORE_HEAD True \
DATASETS.NAMES new_occ_duke \
TEST.NEW_METRIC True \
OUTPUT_DIR $OUTPUT_DIR