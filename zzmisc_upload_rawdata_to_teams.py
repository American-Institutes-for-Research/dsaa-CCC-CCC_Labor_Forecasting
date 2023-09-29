import pandas as pd
# from office365.runtime.auth.authentication_context import AuthenticationContext
# from office365.sharepoint.client_context import ClientContext
# import os
# creds = pd.read_excel("C:/AnacondaProjects/sp creds.xlsx", header =None)
# siteurl = sharepoint_url = 'https://msair.sharepoint.com/sites/CCC-LaborMarketForecast/Shared%20Documents/Forms/AllItems.aspx?FolderCTID=0x012000A26C6B97EB558F48A0695584C7D90852&id=%2Fsites%2FCCC%2DLaborMarketForecast%2FShared%20Documents%2FGeneral%2FEMSI%20data%2F08212023%20update&viewid=68eb643d%2Dac7c%2D4f5d%2Daaaa%2Debd53be6cb07'
# localpath = 'working/2023 data pull chunk.csv'
# remotepath ='Shared%20Documents/Forms/AllItems.aspx?FolderCTID=0x012000A26C6B97EB558F48A0695584C7D90852&id=%2Fsites%2FCCC-LaborMarketForecast%2FShared%20Documents%2FGeneral%2FEMSI%20data%2F08212023%20update&viewid=68eb643d-ac7c-4f5d-aaaa-ebd53be6cb07'
# ctx_auth = AuthenticationContext(siteurl) # should also be the siteurl
# ctx_auth.acquire_token_for_user(creds.loc[0,0], creds.loc[1,0])
# ctx = ClientContext(siteurl, ctx_auth)

for idx,df in enumerate(pd.read_csv('data/2023_update/AIR Datapull Expanded.csv', chunksize=10000)):
    print(idx)
    if idx >= 550:
        df.to_csv('working/data chunks/2023 data pull chunk #'+ str(idx)+'.csv')
