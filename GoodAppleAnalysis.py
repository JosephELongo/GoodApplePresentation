import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#Load the data into a dataframe
df = pd.read_csv("GoodAppleUneditedData.csv")

#Rename columns so they match the file more accurately
#When reading the csv, pandas includes a column that is entirely empty and then adds on an extra column for some reason
#To get around that, label the columns as Empty and Empty 2 and then drop them
columnNames = ["Date", "Month", "Channel 1 Views", "Channel 1 Interactions", "Channel 2 Views", "Channel 2 Interactions",
                "Site Visitors", "Empty", "Week", "Term 1", "Term 2", "Term 3"]

df.columns = columnNames

#Drop the empty columns and the extra row added
df.drop(labels = ["Empty"], axis = 1, inplace = True)

df.drop(0, inplace = True)

#Create two new dataframe's: one for all the data corresponding to daily and another for all the weekly data
#Also, drop the month column for the daily dataframe, it seems unnecessary for now
#Lastly, drop any rows that are empty

dfDaily = df.loc[:, ["Date", "Channel 1 Views", "Channel 1 Interactions", "Channel 2 Views", "Channel 2 Interactions", "Site Visitors"]]
dfDaily.dropna(inplace = True)

dfWeekly = df.loc[:, ["Week", "Term 1", "Term 2", "Term 3"]]
dfWeekly.dropna(inplace = True)

#Unfortunately, pandas reads the csv file as containing strings instead of ints
#Therefore, each column that should only have ints needs to be converted ie every column except Date and Week
#However, because the numbers contain commas, the commas must also be replaced with an empty character

dailyColumnsExcludingDate = ["Channel 1 Views", "Channel 1 Interactions", "Channel 2 Views", "Channel 2 Interactions", "Site Visitors"]
for i in dailyColumnsExcludingDate:
    dfDaily[i] = dfDaily[i].str.replace(",", "").astype(int)

weeklyColumnsExcludingDate = ["Term 1", "Term 2", "Term 3"]
for j in weeklyColumnsExcludingDate:
    dfWeekly[j] = dfWeekly[j].str.replace(",", "").astype(int)

#Now, calculate the correlation coefficents for a variety of variables
correlationsDaily = {}
#Channel 1 Views and Interactions
correlationsDaily["C1 Views and Interactions"] = dfDaily["Channel 1 Views"].corr(dfDaily["Channel 1 Interactions"])
#Channel 2 Views and Interactions
correlationsDaily["C2 Views and Interactions"] = dfDaily["Channel 2 Views"].corr(dfDaily["Channel 2 Interactions"])
#Channel 1 Views and Site Visitors
correlationsDaily["C1 Views and Visitors"] = dfDaily["Channel 1 Views"].corr(dfDaily["Site Visitors"])
#Channel 1 Interactions and Site Visitors
correlationsDaily["C1 Interactions and Visitors"] = dfDaily["Channel 1 Interactions"].corr(dfDaily["Site Visitors"])
#Channel 2 Views and Site Visitors
correlationsDaily["C2 Views and Visitors"] = dfDaily["Channel 2 Views"].corr(dfDaily["Site Visitors"])
#Channel 2 Interactions and Site Visitors
correlationsDaily["C2 Interactions and Visitors"] = dfDaily["Channel 2 Interactions"].corr(dfDaily["Site Visitors"])

#The comment below prints out all of the above correlations
#print(correlationsDaily)

#Repeat this process for weekly search data
correlationsWeekly = {}
#Term 1 and 2
correlationsWeekly["Term 1 and 2"] = dfWeekly["Term 1"].corr(dfWeekly["Term 2"])
#Term 1 and 3
correlationsWeekly["Term 1 and 3"] = dfWeekly["Term 1"].corr(dfWeekly["Term 3"])
#Term 2 and 3
correlationsWeekly["Term 2 and 3"] = dfWeekly["Term 2"].corr(dfWeekly["Term 3"])

#The comment below prints out all of the above correlations
#print(correlationsWeekly)

#Now, we need to convert the daily data into a weekly format so we can compare it to the search terms
#I do not know if the week data represents the previous week or the next week; I am going to assume it's the previous week
#Because of the limited daily data, the first week of full data is January 12th, 2014 (I am assuming that data represents the days from January 6th - January 12th)
#Therefore, we pull all the weekly data from row 316 on to get the data from 1/12/2014 - 7/19/2015
#However, since we have incomplete daily data, I am going to drop the last three weeks (there are 395 rows, to drop the last 3 I index up to 392)
#In the end, our data goes from 1/12/2014 - 6/28/2015

#The copy() here is to prevent the error message "A value is trying to be set on a copy of a slice from a DataFrame"
#Instead of only slicing dfWeekly, I am now copying the slice which seems to stop the error and give the intended result
dailyAndWeeklyData = dfWeekly.loc[316:392, :].copy()

for i in dailyColumnsExcludingDate:
    weeklyVals = []
    for j in range(5, len(dfDaily[i])-2, 7):
        #Because I dropped row 0 before, indexing starts at 1
        #The first point of weekly data is January 5th, 2014
        #Therefore, I start indexing at the 6th point of data and drop the first week of data
        #Then, since the last week of data ends at June 28th, I drop the the last two data points by stopping at the length of the column - 2
        weeklyVals.append(dfDaily[i][j:j+7].sum())
    
    dailyAndWeeklyData[i] = weeklyVals

#Now, with all of our data on the same time scale, we can calculate correlations between all of our values
finalCorrelations = {}

columnsExcludingWeek = ["Term 1", "Term 2", "Term 3", "Channel 1 Views", "Channel 1 Interactions", "Channel 2 Views", "Channel 2 Interactions", "Site Visitors"]
for i in columnsExcludingWeek:
    for j in columnsExcludingWeek:
        finalCorrelations[i + " and " + j] = dailyAndWeeklyData[i].corr(dailyAndWeeklyData[j])

#The commented out code below prints the correlations in a relatively easy to read form
#for i in finalCorrelations:
#    print(i + ": " + str(finalCorrelations[i]))

#Now, let's run a linear regression with X = Channel 2 Views, Term 2, and Channel 1 Views based on the highest correlations and y = Site Visitors
X = dailyAndWeeklyData[["Channel 2 Views", "Term 2", "Channel 1 Views"]]
y = dailyAndWeeklyData["Site Visitors"]
reg = LinearRegression().fit(X, y)

#The commented out prints below show the coefficients of the regression, the MAE of the regression, and the average value of Site Visitors
#print(reg.coef_)
#print(mean_absolute_error(dailyAndWeeklyData["Site Visitors"], reg.predict(dailyAndWeeklyData[["Channel 2 Views", "Term 2", "Channel 1 Views"]])))
#print(y.mean())