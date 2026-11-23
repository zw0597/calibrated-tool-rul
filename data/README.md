Place the raw data here (see the main README, Section 2):

    data/uniwear/phm2010_bundle_high_resolution.csv
    data/uniwear/nuaa_orthogonal_bundle_high_resolution.csv
    data/mill.mat
    data/scidata/FeatureAndMetadata_Milling.csv   (scidata_study.py)
    data/cmapss/train_FD001.txt                   (cmapss_study.py)

then run `python prep.py` and `python prep_mill.py` from the repository root to create `data/win.pkl`.
The raw and windowed data are not redistributed in this repository.
