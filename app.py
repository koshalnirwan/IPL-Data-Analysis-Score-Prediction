import pickle
import pandas as pd
import numpy as np
import plotly as py
import streamlit as st
import plotly.express as px
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

@st.cache(allow_output_mutation=True)
def fetch_data():
    df_2 = pd.read_csv('deliveries.csv')
    df_2 = df_2.replace('Rising Pune Supergiants','Rising Pune Supergiant')    
    return df_2

@st.cache(allow_output_mutation=True)
def score_predictor():
    df_encode = pickle.load(open('data.pickle','rb'))
    
    y = df_encode.total
    X = df_encode.drop('total',axis=1)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X_train,y_train)
    model.score(X_test,y_test)
    
    return df_encode,model

df = fetch_data()
data,model = score_predictor()


st.sidebar.header('CHOOSE THE TOPICS')
radio = st.sidebar.radio('',['Introduction','Raw Dataset','Score Predictor','Team Stats','Batting Stats','Bowling Stats','Fielding Stats','Interesting Insights'])

########### FUNCTIONS #########################################################
def pie_chart(data,name,value,color,pull,hole,title,fcolor='green'):
    
    fig = px.pie(data,names=name,values=value,hole=hole,height=600,width=600)
    
    fig.layout.paper_bgcolor='rgba(0,0,0,0)'
    fig.layout.plot_bgcolor='rgba(0,0,0,0)'
    fig.update_layout(title_font_color='blue',title_font_family="Courier New",
                      title_font_size=20,legend_title_font_color='green',
                    title={
                    'text': title,
                    'y':1.0,
                    'x':0.55,
                    'xanchor': 'center',
                    'yanchor': 'top'})
    fig.update_traces(
                    pull=pull,
                    marker=dict(colors=color),
                    textfont_size=12,
                    textfont_color=fcolor,
                    textinfo='label+value+percent',
                    hoverinfo='label+value+percent',
                    insidetextorientation='horizontal'           #['horizontal', 'radial', 'tangential', 'auto']
                    )
    fig.update_layout(showlegend=False,annotations=[dict(text='IPL DATA',font_size=20,x=0.5,y=0.5,showarrow=False,font_color='red')])            
        
    st.plotly_chart(fig)
        
def bar_chart(data,x,y,color,title):
    
    fig = px.bar(data,x=x,y=y,height=500,width=700)
    fig.layout.paper_bgcolor='rgba(0,0,0,0)'
    fig.layout.plot_bgcolor='rgba(0,0,0,0)'
    fig.update_layout(title_font_color='blue',title_font_family="Courier New",title_font_size=20,
            title={
            'text': title,
            'y':1.0,
            'x':0.55,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig.update_traces(marker=dict(color=color))
    st.plotly_chart(fig)
    
def bar_chart_h(data,x,y,color,title):
    
    fig = px.bar(data,x=y,y=x,height=500,width=700,orientation='h')
    fig.layout.paper_bgcolor='rgba(0,0,0,0)'
    fig.layout.plot_bgcolor='rgba(0,0,0,0)'
    fig.update_layout(title_font_color='blue',title_font_family="Courier New",title_font_size=20,
            title={
            'text': title,
            'y':1.0,
            'x':0.55,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig.update_traces(marker=dict(color=color))
    st.plotly_chart(fig)
            
def scatter_chart(data,x,y,color,size,title):
    
    fig = px.scatter(data,x=x,y=y,color=color,size=size,height=500,width=700)
    fig.update_layout(title_font_color='blue',title_font_family="Courier New",showlegend=False,title_font_size=20,
    title={
    'text': title,
    'y':1.0,
    'x':0.55,
    'xanchor': 'center',
    'yanchor': 'top'})                                             
    st.plotly_chart(fig)
    
def funnel_chart(data,x,y,color,width,title):
    
    fig = px.funnel(data,x=x,y=y,height=600,width=width)
    fig.layout.paper_bgcolor='rgba(0,0,0,0)'
    fig.layout.plot_bgcolor='rgba(0,0,0,0)'
    fig.update_layout(title_font_color='blue',title_font_family="Courier New",title_font_size=20,
    title={
    'text': title,
    'y':1.0,
    'x':0.55,
    'xanchor': 'center',
    'yanchor': 'top'})
    fig.update_traces(textinfo='value+percent total',textposition='inside',textfont_size=12,marker=dict(color=color))
    st.plotly_chart(fig)

team_color=['blue','red','#ee9b14','purple','#73d3f7','#f4f719','pink','#f0693d','#65c8b7']
team_bar = ['blue','red','#ee9b14','purple','#73d3f7','#f4f719','pink','#f0693d','#65c8b7','slateblue','#ff4500','#00bfff','brown']
team_bar_20 = ['blue','red','#ee9b14','purple','#73d3f7','#f4f719','pink','#f0693d','#65c8b7','slateblue','#ff4500','#00bfff','brown','#ee9b14',
               'red','purple','pink','#00bfff','slateblue','#65c8b7']
team_bar_h = ['brown','#00bfff','#ff4500','slateblue','#65c8b7','#f0693d','pink','#f4f719','#73d3f7','purple','#ee9b14','red','blue']            
pull=[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
  
if radio == 'Raw Dataset':
    st.header('Few Rows of data used for Analysis')
    st.table(df.head())
    
    st.header('Few Rows of data used for Score Prediction')
    st.table(data.head())
 
elif radio == 'Introduction':
    st.title('Analysing IPL Data (2008 - 2017)')
    st.subheader('')
    st.markdown("""
    <img src="https://www.insidesport.co/wp-content/uploads/2019/12/Banner.jpg" width="600" height="250">"""
                ,unsafe_allow_html=True)
    st.subheader(' ')
    st.markdown("""<h2><b>1. DATA VISUALIZATION</h2> 
                        <ul>
                        <li>Raw Dataset is taken, analysed and brought it in the form of stats</li>
                        <li>Stats for Teams, Batting stats, Bowling stats, Fielding stats and some interesting IPL stats </li>
                        <li>This analysed data is used for visualization</li>
                        <li>Each section contains the analysed data or derived data from raw dataset</li>
                        </ul>
                    <h2><b>2. SCORE PREDICTION</h2>
                        <ul>
                        <li>For score Prediction the dataset is different from what I have used in Visualization</li>
                        <li>User needs to enter the required information to Predict score from a particular point </li>
                        <li>Random Forest algorithm is used for building this Score Prediction Model</li>
                        </ul>
                    """, unsafe_allow_html=True)
    
elif radio == 'Score Predictor':
    st.markdown("""
    <img src="https://image-cdn.essentiallysports.com/wp-content/uploads/20200320194356/ipl-2020-players-auction-what-to-expect-1600x900.jpg" width="700" height="150">"""
                ,unsafe_allow_html=True)
    st.markdown(""" 
                <h2 style="color:slateblue;text-align:center;"><b><i>IPL Score Predictor
                </h2>""",unsafe_allow_html=True)
    
    stadium = [i[6:] for i in data.columns[:18]]
    bat_team = [i[9:] for i in data.columns[18:26]]
    bowl_team = [i[10:] for i in data.columns[26:34]]
    bowl_team.reverse()
    st.subheader('Select Ground')
    set_1 = st.selectbox('',stadium)
    temp = list()
    if set_1 == 'Brabourne Stadium':
        temp = temp + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif set_1 == 'Dubai International Cricket Stadium':
        temp = temp + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif set_1 == 'Eden Gardens':
        temp = temp + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]    
    elif set_1 == 'Feroz Shah Kotla':
        temp = temp + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif set_1 == 'Himachal Pradesh Cricket Association Stadium':
        temp = temp + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]    
    elif set_1 == 'JSCA International Stadium Complex':
        temp = temp + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]    
    elif set_1 == 'Kingsmead':
        temp = temp + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]    
    elif set_1 == 'M Chinnaswamy Stadium':
        temp = temp + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]    
    elif set_1 == 'MA Chidambaram Stadium, Chepauk':
        temp = temp + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
    elif set_1 == 'Punjab Cricket Association Stadium, Mohali':
        temp = temp + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0] 
    elif set_1 == 'Rajiv Gandhi International Stadium, Uppal':
        temp = temp + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0] 
    elif set_1 == 'Sardar Patel Stadium, Motera':
        temp = temp + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0] 
    elif set_1 == 'Sawai Mansingh Stadium':
        temp = temp + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0] 
    elif set_1 == 'Sharjah Cricket Stadium':
        temp = temp + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0] 
    elif set_1 == 'Sheikh Zayed Stadium':
        temp = temp + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0] 
    elif set_1 == "St George's Park":
        temp = temp + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0] 
    elif set_1 == 'SuperSport Park':
        temp = temp + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0] 
    elif set_1 == 'Wankhede Stadium':
        temp = temp + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1] 
    
    st.subheader('Select Batting Team')    
    set_2 = st.selectbox(' ',bat_team)
    if set_2 == 'Chennai Super Kings':
        temp = temp + [1,0,0,0,0,0,0,0]
    elif set_2 == 'Delhi Daredevils':
        temp = temp + [0,1,0,0,0,0,0,0]
    elif set_2 == 'Kings XI Punjab':
        temp = temp + [0,0,1,0,0,0,0,0]
    elif set_2 == 'Kolkata Knight Riders':
        temp = temp + [0,0,0,1,0,0,0,0]
    elif set_2 == 'Mumbai Indians':
        temp = temp + [0,0,0,0,1,0,0,0]
    elif set_2 == 'Rajasthan Royals':
        temp = temp + [0,0,0,0,0,1,0,0]
    elif set_2 == 'Royal Challengers Bangalore':
        temp = temp + [0,0,0,0,0,0,1,0]
    elif set_2 == 'Sunrisers Hyderabad':
        temp = temp + [0,0,0,0,0,0,0,1]
    
    st.subheader('Select Bowling Team')    
    set_3 = st.selectbox('  ',bowl_team)
    if set_3 == 'Chennai Super Kings':
        temp = temp + [1,0,0,0,0,0,0,0]
    elif set_3 == 'Delhi Daredevils':
        temp = temp + [0,1,0,0,0,0,0,0]
    elif set_3 == 'Kings XI Punjab':
        temp = temp + [0,0,1,0,0,0,0,0]
    elif set_3 == 'Kolkata Knight Riders':
        temp = temp + [0,0,0,1,0,0,0,0]
    elif set_3 == 'Mumbai Indians':
        temp = temp + [0,0,0,0,1,0,0,0]
    elif set_3 == 'Rajasthan Royals':
        temp = temp + [0,0,0,0,0,1,0,0]
    elif set_3 == 'Royal Challengers Bangalore':
        temp = temp + [0,0,0,0,0,0,1,0]
    elif set_3 == 'Sunrisers Hyderabad':
        temp = temp + [0,0,0,0,0,0,0,1]
    
    st.subheader('Runs Scored')    
    runs = st.number_input('   ',min_value=1,max_value=200,value=50)
    st.subheader('Wickets Down')
    wickets = st.selectbox('',[0,1,2,3,4,5,6,7,8,9])
         
    balls = []
    for i in [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
        for j in [0.1,0.2,0.3,0.4,0.5,0.6]:
            balls.append(i+j) 
    st.subheader('Overs (5.0 - 19.5)')
    overs = st.selectbox('    ',balls) 
    
    st.subheader('Runs Scored in Last 5 Overs')
    runs_last_5 = st.number_input('     ',min_value=1,max_value=120,value=30)
    
    st.subheader('Wickets Fell in Last 5 Overs')
    wickets_last_5 = st.selectbox('      ',[0,1,2,3,4,5,6,7,8,9])
    
    st.subheader('')
    button = st.button('PREDICT SCORE')
    
    temp = temp + [runs,wickets,overs,runs_last_5,wickets_last_5]
    pred_data = np.array([temp])
    prediction = int(model.predict(pred_data))
    lower = prediction - 5
    upper = prediction + 5
    if button:
        if wickets < wickets_last_5:
            st.error('Total Wickets are less than wickets fell in last 5 overs')
        elif set_2 == set_3:
            st.error('Batting and Bowling Teams Cannot be Same')
        else:
            st.success(f'Score will be between {lower} to {upper} Runs'.format(lower,upper))
     
elif radio == 'Team Stats':
    st.markdown(""" 
                <h2 style="color:black;text-align:center;"><b><i>All The Teams Related Stats
                </h2></div>""",unsafe_allow_html=True) 
    m_total = pickle.load(open('team_stats.pickle','rb'))    
        
    ts = st.selectbox('',['ALL Teams','Mumbai Indians','Royal Challengers','Kings XI Punjab','Kolkata Knight Riders',
                          'Delhi Daredevils','Chennai Super Kings','Rajasthan Royals','Sunrisers Hyderabad','Deccan Chargers'])
    if ts == 'ALL Teams':
        tsm = st.selectbox('Select Teams With Most - ',['Matches Played','Runs Scored','Dot balls','Fours','Sixes','Wickets Fell','Derived Data'])
        st.subheader('')
        if tsm == 'Matches Played':
            funnel_chart(m_total.sort_values('matches',ascending=False),'matches','Teams',team_bar,600,'Matches Played by Each Team')
            bar_chart(m_total.sort_values('matches'),'Teams','matches',team_bar_h,'Matches Played by Each Team')
            st.subheader('Data Derived from Raw Data')
            st.table(m_total[['Teams','matches']])
        elif tsm == 'Runs Scored':
            pie_chart(m_total.head(9),'Teams','total_runs',team_color,pull,0.6,'Runs Scored by Top 9 Teams')
            bar_chart_h(m_total.sort_values('total_runs'),'Teams','total_runs',team_bar_h,'Runs Scored by Teams')
            st.subheader('Data Derived from Raw Data')
            st.table(m_total[['Teams','total_runs']])
        elif tsm == 'Dot balls':
            pie_chart(m_total.head(9),'Teams','dots',team_color,pull,0.4,'Dot balls Played by Top 9 Teams')
            bar_chart(m_total,'Teams','dots',team_bar,'Dot balls Played by Teams')
            st.subheader('Data Derived from Raw Data')
            st.table(m_total[['Teams','dots']])
        elif tsm == 'Fours':
            funnel_chart(m_total.sort_values('fours',ascending=False),'fours','Teams',team_bar,700,'Fours Hit by Teams')
            bar_chart(m_total,'Teams','fours',team_bar,'Fours Hit by Teams')
            st.subheader('Data Derived from Raw Data')
            st.table(m_total[['Teams','fours']])
        elif tsm == 'Sixes':
            pie_chart(m_total.head(9),'Teams','six',team_color,pull,0.6,'Sixes Hit by Top 9 Teams')
            bar_chart(m_total.sort_values('six'),'Teams','six','#65c8b7','Sixes Hit by Teams')
            st.subheader('Data Derived from Raw Data')
            st.table(m_total[['Teams','six']])
        elif tsm == 'Wickets Fell':
            funnel_chart(m_total.sort_values('wickets_fell'),'wickets_fell','Teams',team_bar_h,600,'Wickets Fell for each Team')
            bar_chart(m_total.sort_values('wickets_fell'),'Teams','wickets_fell','#65c8b7','Wickets Fell for each Team')                     
            st.subheader('Data Derived from Raw Data')
            st.table(m_total[['Teams','wickets_fell']])
        else:
            st.subheader('Data Derived from Raw Data')
            st.table(m_total)
    
    def team_data(data,team):
        team_stats = data[data['Teams']==team][['dots','singles','doubles','threes','fours','six','wickets_fell']]
        team_stats = team_stats.T.reset_index()
        team_stats = team_stats.rename(columns={i:'Balls hit for' if i == 'index' else 'total balls' for i in team_stats})
        return team_stats
    
    def team_data_more(data):
        data['doubles'] = data['doubles']*2
        data['threes'] = data['threes']*3
        data['fours'] = data['fours']*4
        data['six'] = data['six']*6
        return data
        
    if ts == 'Mumbai Indians':
        mi = team_data(m_total,'Mumbai Indians')
        pie_chart(mi,'Balls hit for','total balls',team_color[:6],pull[:6],0.6,'How Mumbai Indians Played The Balls','black')
        mi_m = team_data_more(m_total)
        mi_m1 = team_data(mi_m,'Mumbai Indians')
        mi_m1 = mi_m1.rename(columns={'total balls':'runs scored'}).drop([0,6])
        funnel_chart(mi_m1.sort_values('runs scored',ascending=False),'runs scored','Balls hit for',team_color[:5],600,'How Mumbai Indians Scored Runs with Bat')
        pie_chart(mi_m1,'Balls hit for','runs scored',team_color[3:9],pull[:6],0.4,'How Mumbai Indians Scored Runs with Bat','black')    
    elif ts == 'Royal Challengers':
        rcb = team_data(m_total,'Royal Challengers Bangalore')
        pie_chart(rcb,'Balls hit for','total balls',team_color[2:8],pull[:6],0.6,'How Royal Challengers Played The Balls','black')
        rcb_m = team_data_more(m_total)
        rcb_m1 = team_data(rcb_m,'Royal Challengers Bangalore')
        rcb_m1 = rcb_m1.rename(columns={'total balls':'runs scored'}).drop([0,6])
        pie_chart(rcb_m1,'Balls hit for','runs scored',team_color[4:],pull[:6],0.7,'How Royal Challengers Scored Runs with Bat','black')
    elif ts == 'Kings XI Punjab':
        pun = team_data(m_total,'Kings XI Punjab')
        pie_chart(pun,'Balls hit for','total balls',team_bar[6:12],pull[:6],0.6,'How Kings XI Punjab Played The Balls','black')
        pun_m = team_data_more(m_total)
        pun_m1 = team_data(pun_m,'Kings XI Punjab')
        pun_m1 = pun_m1.rename(columns={'total balls':'runs scored'}).drop([0,6])
        pie_chart(pun_m1,'Balls hit for','runs scored',team_color[0:6],pull[:6],0.8,'How Kings XI Punjab Scored Runs with Bat','black')
    elif ts == 'Kolkata Knight Riders':
        kkr = team_data(m_total,'Kolkata Knight Riders')
        pie_chart(kkr,'Balls hit for','total balls',team_bar[8:],pull[:6],0.6,'How Kolkata Knight Riders Played The Balls','black')
        kkr_m = team_data_more(m_total)
        kkr_m1 = team_data(kkr_m,'Kolkata Knight Riders')
        kkr_m1 = kkr_m1.rename(columns={'total balls':'runs scored'}).drop([0,6])
        funnel_chart(kkr_m1.sort_values('runs scored',ascending=False),'runs scored','Balls hit for',team_color[4:],600,'How Kolkata Knight Riders Scored Runs with Bat')
        pie_chart(kkr_m1,'Balls hit for','runs scored',team_bar_h[3:9],pull[:6],0.3,'Kolkata Knight Riders Scored Runs with Bat','black')
    elif ts == 'Delhi Daredevils':
        dd = team_data(m_total,'Delhi Daredevils')
        pie_chart(dd,'Balls hit for','total balls',team_color[:6],pull[:6],0.6,'How Delhi Daredevils Played The Balls','black')
        dd_m = team_data_more(m_total)
        dd_m1 = team_data(dd_m,'Delhi Daredevils')
        dd_m1 = dd_m1.rename(columns={'total balls':'runs scored'}).drop([0,6])
        pie_chart(dd_m1,'Balls hit for','runs scored',team_bar[5:11],pull[:6],0.5,'How Delhi Daredevils Scored Runs with Bat','black')
    elif ts == 'Chennai Super Kings':
        csk = team_data(m_total,'Chennai Super Kings')
        pie_chart(csk,'Balls hit for','total balls',team_color[2:8],pull[:6],0.6,'How Chennai Super Kings Played The Balls','black')
        csk_m = team_data_more(m_total)
        csk_m1 = team_data(csk_m,'Chennai Super Kings')
        csk_m1 = csk_m1.rename(columns={'total balls':'runs scored'}).drop([0,6])
        pie_chart(csk_m1,'Balls hit for','runs scored',team_bar_h[0:6],pull[:6],0.3,'How Chennai Super Kings Scored Runs with Bat','black')
    elif ts == 'Rajasthan Royals':
        rr = team_data(m_total,'Rajasthan Royals')
        pie_chart(rr,'Balls hit for','total balls',team_color[:6],pull[:6],0.6,'How Rajasthan Royals Played The Balls','black')
        rr_m = team_data_more(m_total)
        rr_m1 = team_data(rr_m,'Rajasthan Royals')
        rr_m1 = rr_m1.rename(columns={'total balls':'runs scored'}).drop([0,6])
        pie_chart(rr_m1,'Balls hit for','runs scored',team_color[3:9],pull[:6],0.6,'How Rajasthan Royals Scored Runs with Bat','black')
    elif ts == 'Sunrisers Hyderabad':
        srh = team_data(m_total,'Sunrisers Hyderabad')
        pie_chart(srh,'Balls hit for','total balls',team_bar[4:10],pull[:6],0.6,'How Sunrisers Hyderabad Played The Balls','black')
        srh_m = team_data_more(m_total)
        srh_m1 = team_data(srh_m,'Sunrisers Hyderabad')
        srh_m1 = srh_m1.rename(columns={'total balls':'runs scored'}).drop([0,6])
        pie_chart(srh_m1,'Balls hit for','runs scored',team_color[1:7],pull[:6],0.3,'How Sunrisers Hyderabad Scored Runs with Bat','black')
    elif ts == 'Deccan Chargers':
        dc = team_data(m_total,'Deccan Chargers')
        pie_chart(dc,'Balls hit for','total balls',team_color[:6],pull[:6],0.6,'How Deccan Chargers Played The Balls','black')
        dc_m = team_data_more(m_total)
        dc_m1 = team_data(dc_m,'Deccan Chargers')
        dc_m1 = dc_m1.rename(columns={'total balls':'runs scored'}).drop([0,6])
        pie_chart(dc_m1,'Balls hit for','runs scored',team_bar_h[1:7],pull[:6],0.6,'How Deccan Chargers Scored Runs with Bat','black')
        funnel_chart(dc_m1.sort_values('runs scored',ascending=False),'runs scored','Balls hit for',team_color[:5],600,'How Deccan Chargers Scored Runs with Bat')    
        
elif radio == 'Batting Stats': 
       st.markdown(""" 
                    <h2 style="color:black;text-align:center;"><b><i>All The Batting Stats
                    </h2></div>""",unsafe_allow_html=True)
       bat = pickle.load(open('bat.pickle','rb'))
       
       bat = bat[bat['Innings Played']>10].set_index('batsman')
       select = st.selectbox('',['Individual Player Stats','Compare Players','Query Based Stats','Derived Data'])
       if select == 'Individual Player Stats':
           ips = st.selectbox('Select Player',list(bat.index)) 
           bat_1 = bat.loc[ips]['Out':'Sixes']
           pie_chart(bat_1,bat_1.index,ips,team_bar[0:6],pull[:6],0.6,f'How {ips} Faced Balls'.format(ips),'black')
           st.subheader(f'{ips}   Batting Stats')
           st.table(bat.loc[ips])                     
       elif select == 'Compare Players':
           player = st.multiselect('',list(bat.index),default=['MS Dhoni','V Kohli','CH Gayle'])
           st.table(bat.loc[player])
       elif select == 'Query Based Stats':        
           top = st.selectbox('Players With Most -',['Innings Played','Out','Fours','Sixes','Runs','Strike Rate','Average'])                           
           num = st.slider('Select Number',min_value=5,max_value=20,value=10)
            
           if top == 'Innings Played':
               bar_chart(bat.sort_values(top,ascending=False).head(num),bat.sort_values(top,ascending=False).index[:num],
                              'Innings Played',team_bar_20[:num],f'Top {num} Batsmen With Most {top}'.format(num,top))
               st.subheader('DATA')
               st.table(bat.sort_values(top,ascending=False).head(num))
           elif top == 'Out':
               bar_chart(bat.sort_values(top,ascending=False).head(num),bat.sort_values(top,ascending=False).index[:num],
                              'Out',team_bar_20[:num],f'Top {num} Batsmen With Most {top}'.format(num,top))
               st.subheader('DATA')
               st.table(bat.sort_values(top,ascending=False).head(num))
           elif top == 'Fours':
               bar_chart(bat.sort_values(top,ascending=False).head(num),bat.sort_values(top,ascending=False).index[:num],
                              'Fours',team_bar_20[:num],f'Top {num} Batsmen With Most {top}'.format(num,top))
               st.subheader('DATA')
               st.table(bat.sort_values(top,ascending=False).head(num))
           elif top == 'Sixes':
               bar_chart(bat.sort_values(top,ascending=False).head(num),bat.sort_values(top,ascending=False).index[:num],
                              'Sixes',team_bar_20[:num],f'Top {num} Batsmen With Most {top}'.format(num,top))
               st.subheader('DATA')
               st.table(bat.sort_values(top,ascending=False).head(num))
           elif top == 'Runs':
               bar_chart(bat.sort_values(top,ascending=False).head(num),bat.sort_values(top,ascending=False).index[:num],
                              'Runs',team_bar_20[:num],f'Top {num} Batsmen With Most {top}'.format(num,top))
               st.subheader('DATA')
               st.table(bat.sort_values(top,ascending=False).head(num))
           elif top == 'Strike Rate':
               bar_chart(bat.sort_values(top,ascending=False).head(num),bat.sort_values(top,ascending=False).index[:num],
                              'Strike Rate',team_bar_20[:num],f'Top {num} Batsmen With Most {top}'.format(num,top))
               st.subheader('DATA')
               st.table(bat.sort_values(top,ascending=False).head(num))
           elif top == 'Average':
               bar_chart(bat.sort_values(top,ascending=False).head(num),bat.sort_values(top,ascending=False).index[:num],
                              'Average',team_bar_20[:num],f'Top {num} Batsmen With Most {top}'.format(num,top))
               st.subheader('DATA')
               st.table(bat.sort_values(top,ascending=False).head(num))
       else:
           st.table(bat)
            
elif radio == 'Bowling Stats':
    st.markdown(""" 
                <h2 style="color:black;text-align:center;"><b><i>All The Bowling Stats
                </h2></div>""",unsafe_allow_html=True)
    bowl = pickle.load(open('bowl.pickle','rb'))
    
    bowl = bowl[bowl['Overs']>=40].set_index('Bowler')
    select = st.selectbox('',['Query Based Stats','Bowlers Stats','Derived Data'])                           
    if select == 'Bowlers Stats':
       player = st.multiselect('Select Bowlers',list(bowl.index),default=['SL Malinga','Z Khan','A Mishra'])
       st.table(bowl.loc[player])
    elif select == 'Query Based Stats':        
       top = st.selectbox('Players With Most -',['Balls Bowled','Dots','Wickets','Four Against','Six Against','Runs','Strike Rate','Average','Economy'])                           
       num = st.slider('Select Number',min_value=5,max_value=20,value=10)
       
       if top == 'Balls Bowled':
           bar_chart(bowl.sort_values(top,ascending=False).head(num),bowl.sort_values(top,ascending=False).index[:num],
                          'Balls Bowled',team_bar_20[:num],f'Top {num} Bowlers With Most {top}'.format(num,top))
           st.subheader('DATA')
           st.table(bowl.sort_values(top,ascending=False).head(num))
       elif top == 'Dots':
           bar_chart(bowl.sort_values(top,ascending=False).head(num),bowl.sort_values(top,ascending=False).index[:num],
                          'Dots',team_bar_20[:num],f'Top {num} Bowlers With Most {top}'.format(num,top))
           st.subheader('DATA')
           st.table(bowl.sort_values(top,ascending=False).head(num))
       elif top == 'Wickets':
           bar_chart(bowl.sort_values(top,ascending=False).head(num),bowl.sort_values(top,ascending=False).index[:num],
                          'Wickets',team_bar_20[:num],f'Top {num} Bowlers With Most {top}'.format(num,top))
           st.subheader('DATA')
           st.table(bowl.sort_values(top,ascending=False).head(num))
       elif top == 'Four Against':
           bar_chart(bowl.sort_values(top,ascending=False).head(num),bowl.sort_values(top,ascending=False).index[:num],
                          'Four Against',team_bar_20[:num],f'Top {num} Bowlers With Most {top}'.format(num,top))
           st.subheader('DATA')
           st.table(bowl.sort_values(top,ascending=False).head(num))
       elif top == 'Six Against':
           bar_chart(bowl.sort_values(top,ascending=False).head(num),bowl.sort_values(top,ascending=False).index[:num],
                          'Six Against',team_bar_20[:num],f'Top {num} Bowlers With Most {top}'.format(num,top))
           st.subheader('DATA')
           st.table(bowl.sort_values(top,ascending=False).head(num))
       elif top == 'Runs':
           bar_chart(bowl.sort_values(top,ascending=False).head(num),bowl.sort_values(top,ascending=False).index[:num],
                          'Runs',team_bar_20[:num],f'Top {num} Bowlers With Most {top}'.format(num,top))
           st.subheader('DATA')
           st.table(bowl.sort_values(top,ascending=False).head(num))
       elif top == 'Strike Rate':
           bar_chart(bowl.sort_values(top,ascending=False).tail(num),bowl.sort_values(top,ascending=False).index[:num],
                          'Strike Rate',team_bar_20[:num],f'Top {num} Bowlers With Least {top}'.format(num,top))
           st.subheader('DATA')
           st.table(bowl.sort_values(top,ascending=False).tail(num))
       elif top == 'Average':
           bar_chart(bowl.sort_values(top,ascending=False).tail(num),bowl.sort_values(top,ascending=False).index[:num],
                          'Average',team_bar_20[:num],f'Top {num} Bowlers With Least {top}'.format(num,top))
           st.subheader('DATA')
           st.table(bowl.sort_values(top,ascending=False).tail(num))
       elif top == 'Economy':
           bar_chart(bowl.sort_values(top,ascending=False).tail(num),bowl.sort_values(top,ascending=False).index[:num],
                          'Economy',team_bar_20[:num],f'Top {num} Bowlers With Least {top}'.format(num,top))
           st.subheader('DATA')
           st.table(bowl.sort_values(top,ascending=False).tail(num))
    else:
       st.table(bowl)

elif radio == 'Fielding Stats':
    st.markdown(""" 
                <h2 style="color:black;text-align:center;"><b><i>All The Fielding Stats
                </h2></div>""",unsafe_allow_html=True)
    field = pickle.load(open('field.pickle','rb'))

    field = field[field['Catches']>=5].set_index('fielder')
    select = st.selectbox('',['Query Based Stats','Fielders Stats','Derived Data'])                           
    if select == 'Fielders Stats':
       player = st.multiselect('Select Players',list(field.index),default=['AB de Villiers','MS Dhoni','KA Pollard'])
       st.table(field.loc[player])
    elif select == 'Query Based Stats':        
       top = st.selectbox('Players With Most -',['Catches','Run Outs'])                           
       num = st.slider('Select Number',min_value=5,max_value=20,value=10)
       
       if top == 'Catches':
           scatter_chart(field.sort_values(top,ascending=False).head(num),field.sort_values(top,ascending=False).index[:num],
                         'Catches',team_bar_20[:num],'Catches',f'Top {num} Fielders With Most {top}'.format(num,top))
           bar_chart(field.sort_values(top,ascending=False).head(num),field.sort_values(top,ascending=False).index[:num],
                          'Catches',team_bar_20[:num],f'Top {num} Fielders With Most {top}'.format(num,top))
           
           st.subheader('DATA')
           st.table(field.sort_values(top,ascending=False).head(num))
       elif top == 'Run Outs':
           scatter_chart(field.sort_values(top,ascending=False).head(num),field.sort_values(top,ascending=False).index[:num],
                         'Run Outs',team_bar_20[:num],'Run Outs',f'Top {num} Fielders With Most {top}'.format(num,top))
           bar_chart(field.sort_values(top,ascending=False).head(num),field.sort_values(top,ascending=False).index[:num],
                          'Run Outs',team_bar_20[:num],f'Top {num} Fielders With Most {top}'.format(num,top))           
           st.subheader('DATA')
           st.table(field.sort_values(top,ascending=False).head(num))
    else:
       st.table(field.sort_values('Catches',ascending=False))
    
elif radio == 'Interesting Insights':
    st.markdown(""" 
                <h2 style="color:black;text-align:center;"><b><i>Interesting IPL Stats
                </h2></div>""",unsafe_allow_html=True)
    select = st.selectbox('',['Number of Wickets Fell On Each Ball','Number of Wickets Fell In Each Over','Super Overs Played',
                              'How Wickets Fell for Each Team','How Each Team Have Taken Wickets','How Each Bowler Have Taken Wickets'])
    def dismissal_type(kind,column):
            wicket_type = df.groupby(column)['dismissal_kind'].value_counts()
            wicket_type = pd.DataFrame(wicket_type).rename(columns={'dismissal_kind':kind}).reset_index()
            wicket_type = wicket_type[wicket_type['dismissal_kind']==kind].drop('dismissal_kind',axis=1)
            return wicket_type
        
    if select == 'Number of Wickets Fell On Each Ball':
        ball_w = df.copy()
        ball_w['player_dismissed'] = df.player_dismissed.fillna(0)
        ball_w['player_dismissed'] = [0 if i==0 else 1 for i in ball_w.player_dismissed]
        ball_w2 = ball_w.groupby('ball')['player_dismissed'].value_counts()
        ball_w2 = pd.DataFrame(ball_w2).rename(columns={'player_dismissed':'Wickets'}).reset_index().rename(columns={'ball':'ball_no'})
        ball_w2 = ball_w2[ball_w2['player_dismissed']==1].drop('player_dismissed',axis=1)
        
        pie_chart(ball_w2,ball_w2['ball_no'],'Wickets',team_bar[0:9],pull[:6],0.6,'Number of Wickets Fell On Each Ball of an Over','black')
        bar_chart(ball_w2,'ball_no','Wickets',team_bar[4:13],'Number of Wickets Fell On Each Ball of Over')
        st.subheader('Derived Data')
        st.table(ball_w2.set_index('ball_no'))

    elif select == 'Number of Wickets Fell In Each Over':
        over_w = df.copy()
        over_w['player_dismissed'] = df.player_dismissed.fillna(0)
        over_w['player_dismissed'] = [0 if i==0 else 1 for i in over_w.player_dismissed]       
        over_w2 = over_w.groupby('over')['player_dismissed'].value_counts()
        over_w2 = pd.DataFrame(over_w2).rename(columns={'player_dismissed':'Wickets'}).reset_index().rename(columns={'over':'over_no'})
        over_w2 = over_w2[over_w2['player_dismissed']==1].drop('player_dismissed',axis=1)
        bar_chart(over_w2,'over_no','Wickets',team_bar_20,'Number of Wickets Fell In Each Over')
        pie_chart(over_w2,over_w2['over_no'],'Wickets',team_bar_20,pull[:6],0.6,'Number of Wickets Fell In Each Over','black')
        scatter_chart(over_w2,'over_no','Wickets',team_bar_20,'Wickets','Number of Wickets Fell In Each Over')
        st.subheader('Derived Data')
        st.table(over_w2.set_index('over_no'))

    elif select == 'Super Overs Played':
        super_over = df.groupby(['batting_team'])['is_super_over'].value_counts()
        super_over = pd.DataFrame(super_over).rename(columns={'is_super_over':'Super Overs'}).reset_index()
        super_over = super_over[super_over['is_super_over']==1].drop('is_super_over',axis=1)
        super_over['Super Overs'] = round(super_over['Super Overs'] / 8).astype('int')
        bar_chart(super_over,'batting_team','Super Overs',team_bar_20,'Number of Wickets Fell In Each Over')
        st.subheader('Derived Data')
        st.table(super_over.set_index('batting_team'))

    elif select == 'How Wickets Fell for Each Team':
        
        b_catch = dismissal_type('caught','batting_team')
        b_bowled = dismissal_type('bowled','batting_team')
        b_run_out = dismissal_type('run out','batting_team')
        b_lbw = dismissal_type('lbw','batting_team')
        b_stump = dismissal_type('stumped','batting_team')
        b_cnb = dismissal_type('caught and bowled','batting_team')
        b_hitwicket = dismissal_type('hit wicket','batting_team')
        
        batting_team_list = [b_catch,b_bowled,b_lbw,b_stump,b_cnb,b_hitwicket,b_run_out]
        b_kind = reduce(lambda left,right: pd.merge(left,right,on=['batting_team'],how='outer'),batting_team_list).fillna(0)
        
        b_kind['hit wicket'] = b_kind['hit wicket'].fillna(0).astype('int')
        b_kind['total'] = b_kind['caught']+b_kind['bowled']+b_kind['run out']+b_kind['lbw']+b_kind['stumped'] + \
                          b_kind['caught and bowled']+b_kind['hit wicket']
        bt_list = list(b_kind.batting_team)
        bt_list.append('Derived Data')
        select_2 = st.selectbox('Select Team',bt_list)
        if select_2 == 'Derived Data':
            st.table(b_kind.set_index('batting_team'))
        else:
            b_kind2 = b_kind.drop('total',axis=1).set_index('batting_team').loc[select_2].T.reset_index().rename(columns={'index':'Type'})
            pie_chart(b_kind2,b_kind2.Type,select_2,team_bar_20[9:16],pull[:6],0.6,f'How Wickets Fell for {select_2}'.format(select_2),'black')
            bar_chart(b_kind2,b_kind2.Type,select_2,team_bar_20[4:11],f'How Wickets Fell for {select_2}'.format(select_2))
        st.subheader('Derived Data')
        st.table(b_kind2.set_index('Type'))
        
    elif select == 'How Each Team Have Taken Wickets':
        
        f_catch = dismissal_type('caught','bowling_team')
        f_bowled = dismissal_type('bowled','bowling_team')
        f_run_out = dismissal_type('run out','bowling_team')
        f_lbw = dismissal_type('lbw','bowling_team')
        f_stump = dismissal_type('stumped','bowling_team')
        f_cnb = dismissal_type('caught and bowled','bowling_team')
        f_hitwicket = dismissal_type('hit wicket','bowling_team')
        
        field_team_list = [f_catch,f_bowled,f_lbw,f_stump,f_cnb,f_hitwicket,f_run_out]
        f_kind = reduce(lambda left,right: pd.merge(left,right,on=['bowling_team'],how='outer'),field_team_list).fillna(0)
        
        f_kind['hit wicket'] = f_kind['hit wicket'].fillna(0).astype('int')
        f_kind['total'] = f_kind['caught']+f_kind['bowled']+f_kind['run out']+f_kind['lbw']+f_kind['stumped'] + \
                          f_kind['caught and bowled']+f_kind['hit wicket']
        f_list = list(f_kind.bowling_team)
        f_list.append('Derived Data')
        select_3 = st.selectbox('Select Team',f_list)
        if select_3 == 'Derived Data':
            st.table(f_kind.set_index('bowling_team'))       
        else:
            f_kind2 = f_kind.drop('total',axis=1).set_index('bowling_team').loc[select_3].T.reset_index().rename(columns={'index':'Type'})
            bar_chart(f_kind2,f_kind2.Type,select_3,team_bar_20[1:8],f'How {select_3} Have Taken Wickets'.format(select_3))
            pie_chart(f_kind2,f_kind2.Type,select_3,team_bar_20[4:11],pull[:6],0.6,f'How {select_3} Have Taken Wickets'.format(select_3),'black')            
        st.subheader('Derived Data')
        st.table(f_kind2.set_index('Type'))
    
    elif select == 'How Each Bowler Have Taken Wickets':
       
        bowler_catch = dismissal_type('caught','bowler')
        bowler_bowled = dismissal_type('bowled','bowler')
        bowler_lbw = dismissal_type('lbw','bowler')
        bowler_stump = dismissal_type('stumped','bowler')
        bowler_cnb = dismissal_type('caught and bowled','bowler')
        bowler_hitwicket = dismissal_type('hit wicket','bowler')
        
        bowler_data_list = [bowler_catch,bowler_bowled,bowler_lbw,bowler_stump,bowler_cnb,bowler_hitwicket]
        bowler_kind = reduce(lambda left,right: pd.merge(left,right,on=['bowler'],how='outer'),bowler_data_list).fillna(0)
        
        bowler_kind['total'] = bowler_kind['caught']+bowler_kind['bowled']+bowler_kind['lbw']\
                               +bowler_kind['stumped']+bowler_kind['caught and bowled']+bowler_kind['hit wicket']
        bowler_kind = bowler_kind[bowler_kind['total']>15].sort_values('total',ascending=False)
        bo_list = list(bowler_kind.bowler)
        bo_list.append('Derived Data')
        select_4 = st.selectbox('Select Bowler',bo_list)
        if select_4 == 'Derived Data':
            st.table(bowler_kind.set_index('bowler'))
        else:
            bowler_kind2 = bowler_kind.drop('total',axis=1).set_index('bowler').loc[select_4].T.reset_index().rename(columns={'index':'Type'})
            pie_chart(bowler_kind2,bowler_kind2.Type,select_4,team_bar_20[1:7],pull[:6],0.6,f'How {select_4} Have Taken Wickets'.format(select_4),'black')  
            bar_chart(bowler_kind2,bowler_kind2.Type,select_4,team_bar_20[6:12],f'How {select_4} Have Taken Wickets'.format(select_4))
        st.subheader('Derived Data')
        st.table(bowler_kind2.set_index('Type'))

st.sidebar.success('author : @Koshal')






