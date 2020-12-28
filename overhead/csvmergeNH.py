import pandas as pd 

# path10 = pd.read_csv('s2bt3Cam10_AltCord.csv')
path11 = pd.read_csv('s2bt3Cam11_AltCord.csv')
path12 = pd.read_csv('s2bt3Cam12_AltCord.csv')
path14 = pd.read_csv('s2bt3Cam14_AltCord.csv')


merger1 = pd.concat([path12,path11])

FinalP = pd.concat([merger1,path14],sort=False)

FinalP.to_csv('test3.csv', index=True)