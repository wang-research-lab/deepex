rm -rf ./data/FewRel
mkdir -p ./data/FewRel/
cd ./data/FewRel/
if [ ! -d "./pid2name.json" ];then
    wget https://raw.githubusercontent.com/thunlp/FewRel/master/data/pid2name.json
fi
if [ ! -d "./val_wiki.json" ];then
    wget https://raw.githubusercontent.com/thunlp/FewRel/master/data/val_wiki.json
fi
cd ../..
cd scripts/rc/
echo "Preparing FewRel dataset..."
python dataset_preparation.py --task=FewRel
cd ../..
echo "Done."
