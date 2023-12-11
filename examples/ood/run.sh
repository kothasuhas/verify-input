# Update the path to alpha-beta-CROWN to match your setup
PYTHONPATH=$PYTHONPATH:../../../abcrown/complete_verifier/ python oc.py \
    --config ood.yaml \
    --onnx_path ood.onnx \
    --vnnlib_path ood.vnnlib \
    --apply_output_constraints_to /input /input.3 /18 \
    --optimize_disjuncts_separately \
    --tighten_input_bounds \
    --init_iteration 1000 \
    --alpha_lr_decay 0.99 \
    --lr_init_alpha 0.4 \
    --early_stop_patience 1000 \
    --directly_optimize /18 \
    --oc_lr 0.01
