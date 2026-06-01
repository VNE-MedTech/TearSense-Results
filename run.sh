rm -rf outputs/ external_assessor/ __pycache__/

cp -r ../OutputLibraryTearSense/outputs ./

source ~/.zshrc
conda activate tear_assessor

python run.py
