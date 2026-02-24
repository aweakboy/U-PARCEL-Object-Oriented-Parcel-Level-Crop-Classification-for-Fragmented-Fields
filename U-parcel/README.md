# U-PARCEL Core Code


## Structure
- `src/` : model backbones (UTAE + ParcelPooling), losses, metrics, datasets
- `train_*.py` : training scripts
- `test_*.py` / `eval.py` : evaluation scripts
- `pre_*.py` : data preprocessing utilities
- `class_weights_all.pt` : class weights used by the parcel loss (optional)

## Setup
```bash
pip install -r requirements.txt
```

## Run
Please edit the dataset path variables inside the `train_*.py` / `test_*.py` / `pre_*.py` scripts (search for `DATA_ROOT`), then run for example:
```bash
python train_parcel.py
```
