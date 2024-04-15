rm -rf results
mkdir results

if [[ -z "${ABCROWN_PATH}" ]]; then
	ABCROWN_PATH=../../../alpha-beta-CROWN
fi


echo "Using alpha-beta-CROWN installation at '$ABCROWN_PATH'. Change path by setting the ABCROWN_PATH environment variable"

CONFIG_FILE=$ABCROWN_PATH/complete_verifier/exp_configs/vnncomp23/yolo-xiangruzh.yaml
if [ ! -f $CONFIG_FILE ]; then
        echo "Expected to find the .yaml config file for the yolo benchmark at '$CONFIG_FILE'. Make sure to set the correct ABCROWN_PATH environment variable!"
        exit 1
fi

echo "#################################"
echo "# Warnings"
echo "# 1) Make sure to have a GPU with at least 24GB memory. Otherwise, this benchmark will fail!"
echo "# 2) This example will take approximately 20 hours to run! We recommend to run it inside a tmux session."
echo "# 3) This example will modify the external file $CONFIG_FILE. Do not manually edit it until this example terminates."
echo "#################################"
sleep 30

export VNNCOMP_PYTHON_PATH=$(dirname $(which python))

cp files/yolo-xiangruzh.yaml.invprop $CONFIG_FILE
./files/run_all_categories.sh v1 $ABCROWN_PATH/vnncomp_scripts . results/invprop.csv /dev/null "yolo" all

cp files/yolo-xiangruzh.yaml.orig $CONFIG_FILE
./files/run_all_categories.sh v1 $ABCROWN_PATH/vnncomp_scripts . results/orig.csv /dev/null "yolo" all

python files/plot.py
