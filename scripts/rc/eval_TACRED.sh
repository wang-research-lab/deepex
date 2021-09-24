echo "Run relation classification experiments"
echo "***************************************"
cd scripts/rc/
python3 post_process.py --task=TACRED
python3 evaluation.py --task=TACRED
cd ../..
echo "***************************************"
