python prepare.sh

# Update the path to alpha-beta-CROWN to match your setup
PYTHONPATH=$PYTHONPATH:../../../abcrown/complete_verifier/ python oc.py \
    --config double_integrator.yaml \
    --vnnlib_path double_integrator.vnnlib \
    --apply_output_constraints_to BoundLinear \
    --tighten_input_bounds \
    --oc_lr 0.05 \
    --init_iteration 100 \
    --alpha_lr_decay 1.0 \
    --lr_init_alpha 0.1 \
    --early_stop_patience 1000 \
    --best_of_oc_and_no_oc
