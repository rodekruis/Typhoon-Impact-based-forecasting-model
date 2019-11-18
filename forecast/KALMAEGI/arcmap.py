import arcpy

arcpy.env.overwriteOutput = True

mxd = arcpy.mapping.MapDocument("basemap.mxd")
arcpy.MakeFeatureLayer_management("201927W_gust_2019101600.shp", "gust")
arcpy.SaveToLayerFile_management("gust","201927W_gust_2019101600.lyr","RELATIVE")
df = arcpy.mapping.ListDataFrames(mxd)[0]
addLayer = arcpy.mapping.Layer("201927W_gust_2019101600.lyr")
arcpy.mapping.AddLayer(df, addLayer, "AUTO_ARRANGE")

arcpy.MakeFeatureLayer_management("201927W_pasttrack_2019101600.shp", "past track")
arcpy.SaveToLayerFile_management("past track","201927W_pasttrack_2019101600.lyr","RELATIVE")
addLayer = arcpy.mapping.Layer("201927W_pasttrack_2019101600.lyr")
arcpy.mapping.AddLayer(df, addLayer, "AUTO_ARRANGE")

arcpy.MakeFeatureLayer_management("201927W_forecasttrack_2019101600.shp", "forecast track")
arcpy.SaveToLayerFile_management("forecast track","201927W_forecasttrack_2019101600.lyr","RELATIVE")
addLayer = arcpy.mapping.Layer("201927W_forecasttrack_2019101600.lyr")
arcpy.mapping.AddLayer(df, addLayer, "AUTO_ARRANGE")

arcpy.MakeFeatureLayer_management("201927W_track_2019101600.shp", "track")
arcpy.SaveToLayerFile_management("track","201927W_track_2019101600.lyr","RELATIVE")
addLayer = arcpy.mapping.Layer("201927W_track_2019101600.lyr")
arcpy.mapping.AddLayer(df, addLayer, "AUTO_ARRANGE")

mxd.saveACopy("201927W_gust_2019101600.mxd")