## Data

### Preprocess data
- Run the following command to pre-process `MDS` data in `raw_data/MDS` to save the results to `dataset/MDS` 
  ```
  python convert_mds.py --input_dir raw_data/MDS --output_dir dataset/MDS --overwrite
  ```

## Data storage structure
- Raw data should have the following structure
    ```
    slowMRI/
        - data/
            - raw_data/
                - fastMRI/
                    - multicoil_test/
                        - file_brain_AXT2_200_2000110.h5
                        - file_brain_AXT2_200_2000124.h5
                        - file_brain_AXT2_200_2000129.h5
                        - ...
                    - mutlicoil_val/
                        - file_brain_AXFLAIR_200_6002462.h5
                        - file_brain_AXFLAIR_200_6002471.h5
                        - file_brain_AXFLAIR_200_6002477.h5
                        - ...
                - HCP/
                    - 103818/
                        - unprocessed/
                            - 3T/
                                - T1w_MPR1/
                                    - 103818_3T_AFI.nii.gz
                                    - 103818_3T_BIAS_32CH.nii.gz
                                    - 103818_3T_BIAS_BC.nii.gz
                                    - ...
                                - T1w_MPR2/
                                    - 103818_3T_AFI.nii.gz
                                    - 103818_3T_BIAS_32CH.nii.gz
                                    - 103818_3T_BIAS_BC.nii.gz
                                    - ...
                                - T2w_SCP1/
                                    - 103818_3T_AFI.nii.gz
                                    - 103818_3T_BIAS_32CH.nii.gz
                                    - 103818_3T_BIAS_BC.nii.gz
                                    - ...
                    - 105923/
                    - 111312/
                    - ...
                - MDS
                    - SR_002_NHSRI_V0.mat
                    - SR_002_NHSRI_V1.mat
                    - SR_005_BRIC1_V0.mat
                    - ...
    ```
- the processed dataset should have the following structure
    ```
    slowMRI/
        - data/
            - dataset/
                - MDS
                    - hr_samples.h5
                    - lr_samples.h5
                    - info.json
                - HCP
                    - hr_samples.h5
                    - info.json
                - fastMRI
                    - hr_samples.h5
                    - info.json
    ```