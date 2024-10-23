# Positioning_Refinement
## Software Name: Geometric Correction Software for Satellite Geographic Products Using Joint Multi-Temporal and Multi-Source Base Maps

This software performs geometric correction of large-scale geographic products (DSM, DOM) based on multi-source and multi-temporal base maps. It supports data sources such as Google Earth, ArcGIS, and SRTM (8-bit, WGS84 UTM coordinate system, and ellipsoidal height). The software allows for the input of building masks for the area to achieve better geometric correction accuracy (uint8 TIFF format) and implements a block-wise overall adjustment method.

This software is intended for academic research purposes only and has obtained copyright registration (Registration No.: 2023SR0580603). For any issues encountered during use, please contact: luoqy26@mail2.sysu.edu.cn

## Keywords
1. All;
2. Feature_Matching;
3. Geometric_Correction;
4. Precision_Evaluation.

Please enter the following information in order: 1. Keyword; 2. Folder containing the data; 3. Folder where match_pairs.exe is located; 4. Number of blocks (square units); 5. Whether the building mask for the area has been generated (1 for yes, 0 for no); 6. Type of building mask (DOM or BASEMAP); 7. Whether to use the base map building mask for elevation confidence weighting (1 for yes, 0 for no); 8. Whether to perform parameter tuning (1 for yes, 0 for no); and 9. Type of parameter tuning (RANSAC_Iter or Basemap_Num; this parameter is not needed if tuning is not required).

If using the multi-temporal base map control point averaging algorithms (2017: global averaging; 2024: multi-temporal averaging), please enter the following information in order: 1. Keyword; 2. Folder containing the data; 3. Folder where match_pairs.exe is located; 4. Number of blocks (square units); and 5. Method name (2017 or 2024).



## Note

Data should be stored in the "assets" folder at the same level as match_pairs.exe.

match_pairs.exe is based on the SuperPoint and SuperGlue algorithms. Specific files and sample regional data (Vaihingen area) can be downloaded via the following Quark Cloud link:

「VAIHINGEN_AREA.zip」

Link：https://pan.quark.cn/s/c1181c2c58fc

Extraction Code：bm1Y

「match_pairs_assets_empty.zip」 (You can extract VAIHINGEN_AREA.zip and place it in the "assets" folder)

Link：https://pan.quark.cn/s/e2f93eb9bbc2

Extraction Code：NmYy

## Command line run example:
D:\DSM_Rectification_by_Feature_Matching\DSM_Rectification_by_Feature_Matching.exe Geometric_Correction E:\Geometric_Correction_exe\match_pairs\assets\VAIHINGEN_AREA E:\Geometric_Correction_exe\match_pairs 9 1 BASEMAP 1 0
