import pandas as pd
import feedparser
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def check_active_typhoon():
    """
    Get events from GDACS
    for Tropical Cyclone Advisory Domain(TCAD) 4°N 114°E, 28°N 114°E, 28°N 145°E and 4°N 145°N.
    """
    TCDA=np.array([[145, 28], [145, 4], [114, 5], [114, 28],[145, 28]])  # TCDA area
    #PAR=np.array([[145, 35], [145, 5], [115, 5], [115, 35],[145, 35]])  # PAR area
    Pacific_basin=['wp','nwp','NWP','west pacific','north west pacific','northwest pacific']   
    feed = feedparser.parse('feed://gdacs.org/xml/rss.xml')
    current_events = pd.DataFrame(feed.entries)
    polygon = Polygon(TCDA) # create polygon
    event_tc = current_events.query("gdacs_eventtype == '{}'".format('TC'))
    Activetyphoon=[]
    for ind,row in event_tc.iterrows():
        p_cor=np.array(row['where']['coordinates'])
        point =Point(p_cor[0],p_cor[1])
        #print(point.within(polygon)) # check if a point is in the polygon 
        if point.within(polygon):
            eventid=row['gdacs_eventid']
            Activetyphoon.append(row['gdacs_eventname'][:row['gdacs_eventname'].rfind('-')])
            #print(row['gdacs_eventname'][:row['gdacs_eventname'].rfind('-')])      
    return Activetyphoon