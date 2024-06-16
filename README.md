# Positioning_Refinement
软件名称：联合多时相多源底图的卫星地理产品几何校正软件
本计算机软件实现基于多源多时相底图的大范围地理产品（DSM、DOM）的几何校正，支持Google Earth、ArcGIS、SRTM等数据源底图（8bit、WGS84 UTM坐标系、大地高），支持输入该区域建筑物掩膜达到更佳的几何校正精度（uint8的tif），采用分块整体平差方法实现
该软件仅用于学术科研用途，已获软件著作权，登记号：2023SR0580603，如在使用中遇到问题，请邮件联系luoqy26@mail2.sysu.edu.cn

Keywords:
1. All;
2. Feature_Matching;
3. Geometric_Correction;
4. Precision_Evaluation.

请依次输入1.Keyword;2.数据所在文件夹;3.match_pairs.exe所在文件夹;4.分块数量(平方数);5、是否已生成该区域的建筑物掩膜(1/ 0);6、建筑物掩膜类型(DOM/ BASEMAP);7、是否使用底图建筑物掩膜来进行高程置信度赋权(1/ 0);8、是否进行调参(1/ 0);9、调参类型(RANSAC_Iter/ Basemap_Num，若不需要调参则不需要输入该参数)。请注意数据需要存放于match_pairs.exe同级的assets文件夹

match_pairs.exe基于superpoint和superglue算法，具体文件和示例区域数据（Vaihingen区域）可通过夸克网盘链接下载：

命令行运行示例：
D:\DSM_Rectification_by_Feature_Matching\DSM_Rectification_by_Feature_Matching.exe Geometric_Correction E:\Geometric_Correction_exe\match_pairs\assets\VAIHINGEN_AREA E:\Geometric_Correction_exe\match_pairs 9 1 BASEMAP 1 0
