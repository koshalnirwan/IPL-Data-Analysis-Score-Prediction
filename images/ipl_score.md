

```python
import pandas as pd
import pickle
```


```python
df = pd.read_csv('ipl2017.csv')
```


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mid</th>
      <th>date</th>
      <th>venue</th>
      <th>bat_team</th>
      <th>bowl_team</th>
      <th>batsman</th>
      <th>bowler</th>
      <th>runs</th>
      <th>wickets</th>
      <th>overs</th>
      <th>runs_last_5</th>
      <th>wickets_last_5</th>
      <th>striker</th>
      <th>non-striker</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2008-04-18</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>SC Ganguly</td>
      <td>P Kumar</td>
      <td>1</td>
      <td>0</td>
      <td>0.1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2008-04-18</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>BB McCullum</td>
      <td>P Kumar</td>
      <td>1</td>
      <td>0</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2008-04-18</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>BB McCullum</td>
      <td>P Kumar</td>
      <td>2</td>
      <td>0</td>
      <td>0.2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2008-04-18</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>BB McCullum</td>
      <td>P Kumar</td>
      <td>2</td>
      <td>0</td>
      <td>0.3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2008-04-18</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>BB McCullum</td>
      <td>P Kumar</td>
      <td>2</td>
      <td>0</td>
      <td>0.4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>2008-04-18</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>BB McCullum</td>
      <td>P Kumar</td>
      <td>2</td>
      <td>0</td>
      <td>0.5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>2008-04-18</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>BB McCullum</td>
      <td>P Kumar</td>
      <td>3</td>
      <td>0</td>
      <td>0.6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>2008-04-18</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>BB McCullum</td>
      <td>Z Khan</td>
      <td>3</td>
      <td>0</td>
      <td>1.1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>2008-04-18</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>BB McCullum</td>
      <td>Z Khan</td>
      <td>7</td>
      <td>0</td>
      <td>1.2</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2008-04-18</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>BB McCullum</td>
      <td>Z Khan</td>
      <td>11</td>
      <td>0</td>
      <td>1.3</td>
      <td>11</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>222</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = pd.read_csv('deliveries.csv')
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>inning</th>
      <th>batting_team</th>
      <th>bowling_team</th>
      <th>over</th>
      <th>ball</th>
      <th>batsman</th>
      <th>non_striker</th>
      <th>bowler</th>
      <th>is_super_over</th>
      <th>...</th>
      <th>bye_runs</th>
      <th>legbye_runs</th>
      <th>noball_runs</th>
      <th>penalty_runs</th>
      <th>batsman_runs</th>
      <th>extra_runs</th>
      <th>total_runs</th>
      <th>player_dismissed</th>
      <th>dismissal_kind</th>
      <th>fielder</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>Sunrisers Hyderabad</td>
      <td>Royal Challengers Bangalore</td>
      <td>1</td>
      <td>1</td>
      <td>DA Warner</td>
      <td>S Dhawan</td>
      <td>TS Mills</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Sunrisers Hyderabad</td>
      <td>Royal Challengers Bangalore</td>
      <td>1</td>
      <td>2</td>
      <td>DA Warner</td>
      <td>S Dhawan</td>
      <td>TS Mills</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Sunrisers Hyderabad</td>
      <td>Royal Challengers Bangalore</td>
      <td>1</td>
      <td>3</td>
      <td>DA Warner</td>
      <td>S Dhawan</td>
      <td>TS Mills</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Sunrisers Hyderabad</td>
      <td>Royal Challengers Bangalore</td>
      <td>1</td>
      <td>4</td>
      <td>DA Warner</td>
      <td>S Dhawan</td>
      <td>TS Mills</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>Sunrisers Hyderabad</td>
      <td>Royal Challengers Bangalore</td>
      <td>1</td>
      <td>5</td>
      <td>DA Warner</td>
      <td>S Dhawan</td>
      <td>TS Mills</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



## Data Cleaning


```python
df = df.drop(['date','bowler','batsman','striker','non-striker'],axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mid</th>
      <th>venue</th>
      <th>bat_team</th>
      <th>bowl_team</th>
      <th>runs</th>
      <th>wickets</th>
      <th>overs</th>
      <th>runs_last_5</th>
      <th>wickets_last_5</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>1</td>
      <td>0</td>
      <td>0.1</td>
      <td>1</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>1</td>
      <td>0</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>2</td>
      <td>0</td>
      <td>0.2</td>
      <td>2</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>2</td>
      <td>0</td>
      <td>0.3</td>
      <td>2</td>
      <td>0</td>
      <td>222</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>M Chinnaswamy Stadium</td>
      <td>Kolkata Knight Riders</td>
      <td>Royal Challengers Bangalore</td>
      <td>2</td>
      <td>0</td>
      <td>0.4</td>
      <td>2</td>
      <td>0</td>
      <td>222</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.bat_team.unique()
```




    array(['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
           'Mumbai Indians', 'Deccan Chargers', 'Kings XI Punjab',
           'Royal Challengers Bangalore', 'Delhi Daredevils',
           'Kochi Tuskers Kerala', 'Pune Warriors', 'Sunrisers Hyderabad',
           'Rising Pune Supergiants', 'Gujarat Lions',
           'Rising Pune Supergiant'], dtype=object)




```python
current_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals','Mumbai Indians',
                'Kings XI Punjab','Royal Challengers Bangalore', 'Delhi Daredevils','Sunrisers Hyderabad']
```


```python
df = df[(df['bat_team'].isin(current_teams))&(df['bowl_team'].isin(current_teams))]
```


```python
df.bat_team.unique()
```




    array(['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
           'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
           'Delhi Daredevils', 'Sunrisers Hyderabad'], dtype=object)




```python
df.bowl_team.unique()
```




    array(['Royal Challengers Bangalore', 'Kings XI Punjab',
           'Delhi Daredevils', 'Rajasthan Royals', 'Mumbai Indians',
           'Chennai Super Kings', 'Kolkata Knight Riders',
           'Sunrisers Hyderabad'], dtype=object)




```python
df = df.replace('Punjab Cricket Association IS Bindra Stadium, Mohali','Punjab Cricket Association Stadium, Mohali')
df.venue.unique()
```




    array(['M Chinnaswamy Stadium',
           'Punjab Cricket Association Stadium, Mohali', 'Feroz Shah Kotla',
           'Wankhede Stadium', 'Sawai Mansingh Stadium',
           'MA Chidambaram Stadium, Chepauk', 'Eden Gardens',
           'Dr DY Patil Sports Academy', 'Newlands', "St George's Park",
           'Kingsmead', 'SuperSport Park', 'Buffalo Park',
           'New Wanderers Stadium', 'De Beers Diamond Oval',
           'OUTsurance Oval', 'Brabourne Stadium',
           'Sardar Patel Stadium, Motera',
           'Himachal Pradesh Cricket Association Stadium',
           'Subrata Roy Sahara Stadium',
           'Rajiv Gandhi International Stadium, Uppal',
           'Shaheed Veer Narayan Singh International Stadium',
           'JSCA International Stadium Complex', 'Sheikh Zayed Stadium',
           'Sharjah Cricket Stadium', 'Dubai International Cricket Stadium',
           'Barabati Stadium', 'Maharashtra Cricket Association Stadium',
           'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
           'Holkar Cricket Stadium'], dtype=object)




```python
# Considering only those stadiums where more than 4 matches are played
df2 = pd.DataFrame(df.groupby('venue')['mid'].unique())
df2['matches'] = [len(i) for i in df2.mid]
df2.drop('mid',axis=1,inplace=True)
df2 = df2[df2['matches']>5].reset_index()
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>venue</th>
      <th>matches</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brabourne Stadium</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dubai International Cricket Stadium</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Eden Gardens</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Feroz Shah Kotla</td>
      <td>46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Himachal Pradesh Cricket Association Stadium</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>JSCA International Stadium Complex</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Kingsmead</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M Chinnaswamy Stadium</td>
      <td>49</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MA Chidambaram Stadium, Chepauk</td>
      <td>40</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Punjab Cricket Association Stadium, Mohali</td>
      <td>38</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Rajiv Gandhi International Stadium, Uppal</td>
      <td>24</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sardar Patel Stadium, Motera</td>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sawai Mansingh Stadium</td>
      <td>27</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sharjah Cricket Stadium</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sheikh Zayed Stadium</td>
      <td>7</td>
    </tr>
    <tr>
      <th>15</th>
      <td>St George's Park</td>
      <td>6</td>
    </tr>
    <tr>
      <th>16</th>
      <td>SuperSport Park</td>
      <td>7</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Wankhede Stadium</td>
      <td>46</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df[df['venue'].isin(df2.venue)].drop('mid',axis=1)
```


```python
# starting from 6th over for every match as we require minimum of last five overs runs and wickets
df = df[df['overs']>=5.0]
```

##  Handeling Categorical Variables


```python
df_encode = pd.get_dummies(data=df,columns=['venue','bat_team','bowl_team'])
```


```python
df_encode.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>runs</th>
      <th>wickets</th>
      <th>overs</th>
      <th>runs_last_5</th>
      <th>wickets_last_5</th>
      <th>total</th>
      <th>venue_Brabourne Stadium</th>
      <th>venue_Dubai International Cricket Stadium</th>
      <th>venue_Eden Gardens</th>
      <th>venue_Feroz Shah Kotla</th>
      <th>...</th>
      <th>bat_team_Royal Challengers Bangalore</th>
      <th>bat_team_Sunrisers Hyderabad</th>
      <th>bowl_team_Chennai Super Kings</th>
      <th>bowl_team_Delhi Daredevils</th>
      <th>bowl_team_Kings XI Punjab</th>
      <th>bowl_team_Kolkata Knight Riders</th>
      <th>bowl_team_Mumbai Indians</th>
      <th>bowl_team_Rajasthan Royals</th>
      <th>bowl_team_Royal Challengers Bangalore</th>
      <th>bowl_team_Sunrisers Hyderabad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>61</td>
      <td>0</td>
      <td>5.1</td>
      <td>59</td>
      <td>0</td>
      <td>222</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>61</td>
      <td>1</td>
      <td>5.2</td>
      <td>59</td>
      <td>1</td>
      <td>222</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>61</td>
      <td>1</td>
      <td>5.3</td>
      <td>59</td>
      <td>1</td>
      <td>222</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>61</td>
      <td>1</td>
      <td>5.4</td>
      <td>59</td>
      <td>1</td>
      <td>222</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>61</td>
      <td>1</td>
      <td>5.5</td>
      <td>58</td>
      <td>1</td>
      <td>222</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
df_encode = df_encode[['venue_Brabourne Stadium',
   'venue_Dubai International Cricket Stadium', 'venue_Eden Gardens',
   'venue_Feroz Shah Kotla','venue_Himachal Pradesh Cricket Association Stadium',
   'venue_JSCA International Stadium Complex', 'venue_Kingsmead',
   'venue_M Chinnaswamy Stadium', 'venue_MA Chidambaram Stadium, Chepauk',
   'venue_Punjab Cricket Association Stadium, Mohali','venue_Rajiv Gandhi International Stadium, Uppal',
   'venue_Sardar Patel Stadium, Motera', 'venue_Sawai Mansingh Stadium',      
   'venue_Sharjah Cricket Stadium', 'venue_Sheikh Zayed Stadium',
   "venue_St George's Park", 'venue_SuperSport Park','venue_Wankhede Stadium',
   'bat_team_Chennai Super Kings','bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
   'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians','bat_team_Rajasthan Royals',
   'bat_team_Royal Challengers Bangalore','bat_team_Sunrisers Hyderabad',
   'bowl_team_Chennai Super Kings','bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
   'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians','bowl_team_Rajasthan Royals',
   'bowl_team_Royal Challengers Bangalore','bowl_team_Sunrisers Hyderabad',
   'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','total']]
```


```python
y = df_encode.total
X = df_encode.drop('total',axis=1)
```


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
```


```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
```


```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```


```python
model = RandomForestRegressor(n_estimators=5)
model.fit(X_train,y_train)
```




    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=None, max_features='auto', max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=5, n_jobs=None, oob_score=False,
                          random_state=None, verbose=0, warm_start=False)




```python
from sklearn.ensemble import AdaBoostRegressor
adb_regressor = AdaBoostRegressor(base_estimator=model, n_estimators=100)
adb_regressor.fit(X_train, y_train)
```




    AdaBoostRegressor(base_estimator=RandomForestRegressor(bootstrap=True,
                                                           ccp_alpha=0.0,
                                                           criterion='mse',
                                                           max_depth=None,
                                                           max_features='auto',
                                                           max_leaf_nodes=None,
                                                           max_samples=None,
                                                           min_impurity_decrease=0.0,
                                                           min_impurity_split=None,
                                                           min_samples_leaf=1,
                                                           min_samples_split=2,
                                                           min_weight_fraction_leaf=0.0,
                                                           n_estimators=5,
                                                           n_jobs=None,
                                                           oob_score=False,
                                                           random_state=None,
                                                           verbose=0,
                                                           warm_start=False),
                      learning_rate=1.0, loss='linear', n_estimators=100,
                      random_state=None)




```python
model.score(X_test,y_test)
```




    0.943052690951396




```python
prediction = model.predict
```


```python
df_2 = pickle.load(open('field.pickle','rb'))
df_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fielder</th>
      <th>Catches</th>
      <th>Run Outs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A Ashish Reddy</td>
      <td>8.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A Chandila</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A Chopra</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A Flintoff</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Kumble</td>
      <td>9.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A Mishra</td>
      <td>16.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A Mithun</td>
      <td>7.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A Mukund</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A Mukund (sub)</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A Nehra</td>
      <td>17.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A Singh</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A Symonds</td>
      <td>20.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>AA Bilakhia</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>AA Chavan</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>AA Jhunjhunwala</td>
      <td>10.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>AB Agarkar</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>AB Dinda</td>
      <td>7.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>AB de Villiers</td>
      <td>81.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>AB de Villiers (sub)</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>AC Blizzard</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>AC Gilchrist</td>
      <td>51.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>AC Thomas</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AC Voges</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>AD Mascarenhas</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AD Mathews</td>
      <td>19.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>AD Russell</td>
      <td>7.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>AF Milne</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>AG Murtaza</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>AG Paunikar</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>AJ Finch</td>
      <td>21.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>419</th>
      <td>UT Yadav</td>
      <td>23.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>420</th>
      <td>Umar Gul</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>421</th>
      <td>V Kohli</td>
      <td>60.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>422</th>
      <td>V Sehwag</td>
      <td>34.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>423</th>
      <td>V Shankar</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>424</th>
      <td>V Shankar (sub)</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>425</th>
      <td>VR Aaron</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>426</th>
      <td>VRV Singh</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>427</th>
      <td>VS Malik</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>428</th>
      <td>VVS Laxman</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>429</th>
      <td>VY Mahesh</td>
      <td>6.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>430</th>
      <td>W Jaffer</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>431</th>
      <td>WA Mota</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>432</th>
      <td>WD Parnell</td>
      <td>5.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>433</th>
      <td>WD Parnell (sub)</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>434</th>
      <td>WP Saha</td>
      <td>51.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>435</th>
      <td>WP Saha (sub)</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>436</th>
      <td>WPUJC Vaas</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>437</th>
      <td>Washington Sundar</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>438</th>
      <td>Y Gnaneswara Rao</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>439</th>
      <td>Y Nagar</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>440</th>
      <td>Y Nagar (sub)</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>Y Venugopal Rao</td>
      <td>13.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>442</th>
      <td>YK Pathan</td>
      <td>37.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>443</th>
      <td>YS Chahal</td>
      <td>13.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>444</th>
      <td>YV Takawale</td>
      <td>13.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>445</th>
      <td>Yashpal Singh</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>446</th>
      <td>Younis Khan</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>447</th>
      <td>Yuvraj Singh</td>
      <td>28.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>448</th>
      <td>Z Khan</td>
      <td>20.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>449 rows × 3 columns</p>
</div>




```python
dd = pd.read_csv('deliveries.csv')
dd = dd.replace('Rising Pune Supergiants','Rising Pune Supergiant')
```


```python
pickle_out = open('fetch_data.pickle','wb')
pickle.dump(dd,pickle_out)
pickle_out.close()
```
