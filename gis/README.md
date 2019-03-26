国土地理院の[公開データ](http://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-v2_3.html)を基に、[GeoPandas](http://geopandas.org) の dissolve と simplify を用いて、市区町村の境界情報を含む [shapefile](https://github.com/takashi-matsushita/lab/tree/master/gis/map) を作成した. 

* map/prefecture.shp : 都道府県境界
* map/town.shp : 市区町村境界

政府統計情報のコピー
* 05k29-2.xls : [都道府県，男女別人口及び人口性比－総人口，日本人人口（平成29年10月１日現在）](http://www.stat.go.jp/data/jinsui/2017np/)

* c01.csv : [男女別人口－全国，都道府県（大正９年～平成27年）](https://www.e-stat.go.jp)
