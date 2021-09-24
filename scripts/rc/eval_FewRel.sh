echo "Run relation classification experiments"
echo "***************************************"
cd scripts/rc/
python3 post_process.py --task=FewRel
python3 evaluation.py --task=FewRel
cd ../..
echo "***************************************"
