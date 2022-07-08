import pandas as pd
import feedparser
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def retrieve_all_gdacs_events():
    """
    Reads in the RSS feed from GDACS and returns all current events in a pandas data frame
    """
    feed = feedparser.parse('feed://gdacs.org/xml/rss.xml')
    events_out = pd.DataFrame(feed.entries)
    return events_out


def get_specific_events(gdacs_events_in, event_code_in):
    """
    Filters the GDACS events by type of event. Available event codes:
    EQ: earthquake
    TC: tropical cyclone
    DR: drought
    FL: flood
    VO: volcano
    Requires a pandas data frame as input. Returns a pandas data frame
    """
    return gdacs_events_in.query("gdacs_eventtype == '{}'".format(event_code_in))

def get_current_TC_events():
    current_events = retrieve_all_gdacs_events()
    tc_events = get_specific_events(current_events, 'TC')
    return tc_events


def check_active_typhoon():
    """
    Get events from GDACS
    for Tropical Cyclone Advisory Domain(TCAD) 4°N 114°E, 28°N 114°E, 28°N 145°E and 4°N 145°N.
    """
    TCDA=np.array([[145, 28], [145, 4], [114, 5], [114, 28],[145, 28]])  # TCDA area
    #PAR=np.array([[145, 35], [145, 5], [115, 5], [115, 35],[145, 35]])  # PAR area
    Pacific_basin=['wp','nwp','NWP','west pacific','north west pacific','northwest pacific']     
    polygon = Polygon(TCDA) # create polygon
    event_tc = get_current_TC_events()
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