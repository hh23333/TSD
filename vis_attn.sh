python3 -m pdb vis_attn.py \
--config_file configs/OCC_Duke/vit_base.yml \
TEST.WEIGHT /home/omnisky/hh/TransReID/logs/occ_duke_part/vit_base_eight_part6_proc6_tran5_5part/transformer_120.pth \
INPUT.MASK_PREPROCESS five \
MODEL.NUM_PARTS 5 \
MODEL.DEVICE_ID "('0')" \
MODEL.WITH_FORE_HEAD True \
MODEL.WITH_DIVERSE_LOSS True \
DATASETS.NAMES new_occ_duke \
OUTPUT_DIR logs/occ_reid_part/vit_base_eight_part6_proc6_tran5_5part