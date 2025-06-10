# Hindcasts

1. Make a file in this directory called `constants.py`. Add a variable 
   called `save_dir` that is a string and points to where you want to 
   save the hindcasts.
2. From this directory, execute:
   ```
   python 01_download_hindcasts.py
   ```
   This will download all the xml ECMWF hindcasts from rda.ucar.edu. 
   Note this takes a very long time (~1 day) and a lot of disk space.
3. Execute:
   ```
   python 02_convert_to_csv.py 
   ```
   This uses the file `data/wind_data/input/typhoon_events.cs` (from 
   the fiducial dataset) to select typhoons to pull from the xml files,
   and convert them to csv.
4. Run the main pipeline, here are some example parameters: 
   ```shell
   run-typhoon-model \ 
      --no-azure \
      --use-hindcast \
      --remote_directory=20061127120000 \
      --typhoonname=DURIAN \
      --local-directory=${SAVE_DIR}/csv/
   ```
   For this example, the typhoon DURIAN is used, which made landfall on
   30 Nov 2006, therefore we run on 27 Nov 2006 12:00, to get the forecast
   72 hour ahead. For the local directory, we use the same one as specified
   in `save_dir` in the constants file, plus `csv` because that is where
   the csv summaries are located.