# Mowing detection

## 20220912：适配到非洲草地割草事件检测。


===========================================================

&copy;
Copyright 2021 Marcel Schwieder and Max Wesemeyer

This algorithm was developed to estimate mowing events on grasslands from dense vegetation index time series derived from Sentinel-2 and Landsat data.
It was developed and tested for grasslands in Germany (see maps here: https://ows.geo.hu-berlin.de/webviewer/mowing-detection/).
While the thresholds used to identify mowing events are derived from the time series itself, some parameters might be adjusted for your specific 
use case. Details regarding the indivídual processing steps are described in the Schwieder et al. 2021 (open accesss).

![mowing detection scheme](scheme.jpg)

# Output
The algorithm is pixel based. The output is a raster stack with 17 bands that contain:

- B1: Sum of mowing events
- B2: Maximum data gap in original time series
- B3: absolute clear sky observations (CSO)
- B4: CSO/potential observations (*100)
- B5 - B11: DOY of detected mowing events
- B12: Mean VI value of the defined grassland season
- B13: Median VI value of the defined grassland season
- B14: VI standard deviation of the defined grassland season
- B15: Sum of differences between interpolated and original values (*100)
- B16: Sum of differences between interpolated and original values * data availability (*100)
- B17: Processing error [0,1]

## Usage 

Choose a vegetation index of your choice (tested with EVI and NDVI). 
It is recommended to not use the above and below noise filters, as they might filter out potential mowing events.


The following parameters might be changed in the mowingDetection.py UDF (search for the function: detectMow_S2_new):
- GLstart and GLend (defines the approximate length of grassland season in which you expect the main mowing activity; make sure too include a buffer)
- PSstart and PSend (defines the approximate length of the main vegetation season; i.e., time of the year in which you expect at least one peak)
- GFstd and posEval (sensitivity of thresholds; i.e., width of gaussian function and number of positive evaluations)

## References

- Schwieder, M., Wesemeyer, M., Frantz, D., Pfoch, K., Erasmi, S., Pickert, J., Nendel, C., Hostert, P.. (2021): **Mapping grassland mowing events across Germany based on combined Sentinel-2 and Landsat 8 time series**. *Remote Sensing of Environment X(XX)*, XXX-XXX; [XXX10.3390/rs5126481](https://doi.org/10.3390/XX)
