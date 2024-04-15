rm -rf results
mkdir results

rm files/double_integrator_*.onnx

python files/prepare.py

if [[ -z "${ABCROWN_PATH}" ]]; then
	ABCROWN_PATH=../../../alpha-beta-CROWN
fi

echo "Using alpha-beta-CROWN installation at '$ABCROWN_PATH'. Change path by setting the ABCROWN_PATH environment variable"

# Update the path to alpha-beta-CROWN to match your setup
PYTHONPATH=$PYTHONPATH:../:$ABCROWN_PATH/complete_verifier/ python files/oc.py \
    --config files/double_integrator.yaml \
    --vnnlib_path files/double_integrator.vnnlib \
    --apply_output_constraints_to BoundInput BoundLinear \
    --tighten_input_bounds \
    --oc_lr 0.05 \
    --init_iteration 100 \
    --alpha_lr_decay 1.0 \
    --lr_init_alpha 0.1 \
    --early_stop_patience 1000 \
    --best_of_oc_and_no_oc
