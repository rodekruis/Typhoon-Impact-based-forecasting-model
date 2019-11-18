##Project Name : Triggr model for Forecast-based Financing in the Philippines
Description :
Organize process of running priority models.
The following steps have been implemented for the pilot Phase automation

Step 1: Check for active typhoon events in the Philippines area. This is done first by checking  for disaster events  listed on GDACS website https://www.gdacs.org/ , from those active events select typhoon events.

Step 2: From those typhoon events check if they are located in the Philippines responsibility (PAR) area defined by the following binding box PAR=[[145, 35], [145, 5], [115, 5], [115, 35],[145, 35]]  which is approximately the area shown in the figure below.
   
Step 3: Extract name and other relevant information for the typhoon event located in PAR
Step 4: Download Typhoon forecast data from UCL and Rainfall forecast Data from NOAA
Step 5:  Run the trigger model based on input data for the new typhoon 
Step 6: Send an automated email with information on forecasted impact of the new typhoon for users identified by the local FbF team. For the pilot phase this information will be sent via a temporary email account created for the trigger model. The tentative list for automated email recipients has five email addresses. 
Step 7: Repeat the above steps 1 to 6 every 6 hours – until landfall.




