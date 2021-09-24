# Please first download TACRED dataset from [This Link](https://catalog.ldc.upenn.edu/LDC2018T24). The downloaded file should be named as `tacred_LDC2018T24.tgz`.
rm -rf tacred
tar zxvf tacred_LDC2018T24.tgz
mkdir -p ./data/TACRED
mv ./tacred/data/json/test.json ./data/TACRED/test.json
cd scripts/rc/
echo "Preparing TACRED dataset..."
python dataset_preparation.py --task=TACRED
cd ../..
echo "Done."
