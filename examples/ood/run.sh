rm -rf results
mkdir results

python files/prepare.py

if [[ -z "${ABCROWN_PATH}" ]]; then
	ABCROWN_PATH=../../../alpha-beta-CROWN
fi

echo "Using alpha-beta-CROWN installation at '$ABCROWN_PATH'. Change path by setting the ABCROWN_PATH environment variable"

# Update the path to alpha-beta-CROWN to match your setup
PYTHONPATH=$PYTHONPATH:../:$ABCROWN_PATH/complete_verifier/ python files/oc.py --config files/ood.yaml
