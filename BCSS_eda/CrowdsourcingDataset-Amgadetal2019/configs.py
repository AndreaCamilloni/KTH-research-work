APIURL = 'https://demo.kitware.com/histomicstk/api/v1/'
# apiKey = None  # interactive login
apiKey = 'n0Kp1ez8YOnOiWNoACryzeBlIzbUDW3iOD2DmPLI'

source_folder_id = '5bbdeba3e629140048d017bb'

SAVEPATH = 'CrowdsourcingDataset-Amgadetal2019/'
ROIBOUNDSPATH = 'CrowdsourcingDataset-Amgadetal2019/meta/roiBounds.csv'

# Set either MPP or MAG.
# If both are None, base (scan) magnification is used.

# Microns-per-pixel -- best use this
# MPP of 0.25 is "standardized" at 40x using original Aperio scanners
# MPP = None
MPP = 0.20

# If you prefer to use whatever magnification is reported
MAG = None
# MAG = 40.0

# What things to download? -- comment out whet you dont want
PIPELINE = (
    'images',
    'masks',
    #'annotations',
)

# if you only want to download data for specific slides
#SLIDES_TO_KEEP = ['TCGA-A7-A26F', 'TCGA-OL-A66I', 'TCGA-EW-A1OV', 'TCGA-A2-A0YE', 'TCGA-C8-A12V', 'TCGA-BH-A1F6', 'TCGA-AR-A0TS', 'TCGA-A2-A0D0', 'TCGA-BH-A0B9', 'TCGA-BH-A0BL', 'TCGA-A7-A4SD', 'TCGA-D8-A1JL', 'TCGA-D8-A142', 'TCGA-E2-A150', 'TCGA-OL-A97C', 'TCGA-AR-A2LH', 'TCGA-AN-A0AT', 'TCGA-A7-A0DA', 'TCGA-BH-A1EW', 'TCGA-E2-A1AZ', 'TCGA-BH-A0AV', 'TCGA-EW-A1P7', 'TCGA-A2-A0D2', 'TCGA-S3-AA15', 'TCGA-BH-A1FC', 'TCGA-D8-A1XQ', 'TCGA-AO-A0J2', 'TCGA-AR-A1AY', 'TCGA-BH-A0BG', 'TCGA-AR-A0TU', 'TCGA-BH-A0RX', 'TCGA-C8-A1HJ', 'TCGA-GM-A3XL', 'TCGA-AR-A1AR', 'TCGA-C8-A3M7', 'TCGA-E2-A14R', 'TCGA-E2-A1B6', 'TCGA-A7-A26F', 'TCGA-OL-A66I', 'TCGA-EW-A1OV', 'TCGA-A2-A0YE', 'TCGA-C8-A12V', 'TCGA-BH-A1F6', 'TCGA-AR-A0TS', 'TCGA-A2-A0D0', 'TCGA-BH-A0B9', 'TCGA-BH-A0BL', 'TCGA-A7-A4SD', 'TCGA-D8-A1JL', 'TCGA-D8-A142', 'TCGA-E2-A150', 'TCGA-OL-A97C', 'TCGA-AR-A2LH', 'TCGA-AN-A0AT', 'TCGA-A7-A0DA', 'TCGA-BH-A1EW', 'TCGA-E2-A1AZ', 'TCGA-BH-A0AV', 'TCGA-EW-A1P7', 'TCGA-A2-A0D2', 'TCGA-S3-AA15', 'TCGA-BH-A1FC', 'TCGA-D8-A1XQ', 'TCGA-AO-A0J2', 'TCGA-AR-A1AY', 'TCGA-BH-A0BG', 'TCGA-AR-A0TU', 'TCGA-BH-A0RX', 'TCGA-C8-A1HJ', 'TCGA-GM-A3XL', 'TCGA-AR-A1AR', 'TCGA-C8-A3M7', 'TCGA-E2-A14R', 'TCGA-E2-A1B6']
SLIDES_TO_KEEP = None#['TCGA-OL-A66P', 'TCGA-OL-A6VO', 'TCGA-S3-AA10']
