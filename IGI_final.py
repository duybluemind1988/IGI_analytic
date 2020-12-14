import streamlit as st
import time
import os
from datetime import datetime, timedelta
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

#import statistics as st

import warnings
warnings.filterwarnings("ignore")

import glob
from fbprophet import Prophet

#-----------------Design_layout main side-----------------#
st.markdown('<style>h1{color: green;}</style>', unsafe_allow_html=True)
st.title('Incoming material quality control')

st.subheader('Created by: DNN')
st.header("Information")

text2= st.text_input("1. Please input folder name for data analysis (MUST)",'//Vn01w2k16v18/data/Copyroom/Test_software/Data/Membrane 3000S/')
path=text2 +'/'
#path='//Vn01w2k16v18/data/Copyroom/Test_software/Data/Membrane 3000S/'
#st.write('path input: '+path)

st.write('Accept any public folders like copy room or IGIR-SIS, currently not allow private folder')
text3=st.text_input("2. Please input folder name for saving (option)",'//Vn01w2k16v18/data/Copyroom/Test_software/Data/Save')
path3=text3+'/'

#-----------------Design_layout left side-----------------#
st.sidebar.title("Help")
#st.sidebar.info("2. Number of Sample: all sample need to be measured each inspection")
st.sidebar.info("1. Folder name: Folder contains all excel file related to specific product")
st.sidebar.info("2. Name file: product name or item config for saving. Ex: Membrane_33AA079")

st.sidebar.title("About author")
st.sidebar.info("if you have questionns, please contact to DNN@sonion.com")
st.sidebar.title("Software")
st.sidebar.info("This web app was written by Python program language (offline mode). Please use\
                this app inside Sonion only")
#---------------sort file by created_time-------------------------# 

files = list(filter(os.path.isfile, glob.glob(path + "*")))
files.sort(key=lambda x: os.path.getctime(x)) # sort by create date, not modify date
file_df=pd.DataFrame(files,columns=['file_name'])
file_df['Created_time']=file_df.file_name.apply(lambda x: time.ctime(os.path.getctime(x)) )
file_df['Last_modifie']=file_df.file_name.apply(lambda x: time.ctime(os.path.getmtime(x)) )
file_df.Created_time=pd.to_datetime(file_df.Created_time)
cutting_time='2016-07'
file_df=file_df[file_df.Created_time >cutting_time]
all_files=file_df.file_name.tolist()
material_name=all_files[-1]# Problem ?
material_name=material_name.split('/')[-1].split('.')[0]
material_name
st.text('number of files: '+str(len(all_files)))
#st.text('all files len: '+str(len(all_files)))

### add all dataframe to list of dataframe:
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def create_sheet_dict(all_files):
    sheet_all={}
    n=0
    count_file=1
    for filename in all_files:
      #print(filename)
      #print('count_file: ',str(count_file) + '/' + str(len(all_files)))
      b='Processing_file: '+ str(count_file) + '/' + str(len(all_files))
      #sys.stdout.write('\r'+b)  
      #st.text(b)
      count_file+=1
      try:
        df = pd.read_excel(filename, index_col = None, header = 0)
        #df.to_excel('file'+str(n)+'.xlsx')
      except:
        print('File cannot be read: '+filename)
        st.text('File cannot be read: '+filename)

      # find begin and end col 
      try:
        # find postion row with Pos/items
        a=df[df.columns[0]].str.contains('Pos.')
        item_row=df[a==True].index[0]
        values_col=df.iloc[item_row,:] # specific row that contain dimension name
        #print(values_col)
        values_col.reset_index(drop=True,inplace=True)
        begin_col=1 # bat dau tu cot thu 1 tro di tu vi tri Pos.\Item
        try:
          end_col=values_col[values_col.isnull() == True].index[0]
        except: # mot so truong hop cot cuoi cung khong phai Nan value thi phai lay cot cuoi cua val_col +1
          end_col=values_col[-1:].index[0]+1
        #print('begin_col',begin_col)
        #print('end_col',end_col)
      except:
        continue

      dim_dict={} # add all value, USL, LSL, UCL... in sheet
      n+=1
      if n==23:n=0
      for name in df.columns[begin_col:end_col]:
        new_df=pd.DataFrame()

        # find begin value collect (check if value is numeric or not):
        try:
            value_begin_row=df[df[df.columns[0]].apply(lambda x: isinstance(x, int))].index[0]
        except:
            value_begin_row= item_row+11
        #Problem:  index 0 is out of bounds for axis 0 with size 0
        new_df['Value']=df[name][value_begin_row:] #19: 5 or 10... value after this row
        # Problem
        new_df=new_df.reset_index(drop=True)
        #print(new_df)
        #print(new_df.applymap(np.isreal))
        try:
          non_nummeric_row=new_df[new_df.applymap(np.isreal).values==False].index[0] # App o cuoi dong value 
        except:
          non_nummeric_row=new_df[new_df.isnull().values==True].index[0] # Nan o cuoi dong value
          #continue
        new_df=new_df[:non_nummeric_row]
        new_df['Value']=new_df['Value'].astype('float32')

        #print(df)
        # Check Date: (find row and column contains date)
        row_date=''
        for i in range(7,10):
          try:
            row_date=df[df[df.columns[i]].apply(lambda x: isinstance(x, datetime))].index[0]
            column_date=i
          except:
            continue
        if row_date=='':
            continue
        #print(row_date,column_date)
        new_df['Date']=df[df.columns[column_date]][row_date]
        # Problem: local variable 'column_date' referenced before assignment ? (solved by check row_date)
        #print(new_df)
        #new_df['Date']=new_df['Date'].apply(lambda x: x.strftime("%Y %m %d %H"))
        new_df['Date']=pd.to_datetime(new_df['Date'])
        new_df['Date']=new_df['Date'].apply(lambda x : x+timedelta(hours=n)) # and hour from 1 to 23 to avoid duplicate
        
        # Add id(lot) vao dataframe:
        try:
            new_df['ID_No']=df['Unnamed: 9'][2]
        except:
            new_df['ID_No']='' 
        # chuyen vi tri date ra phia ngoai cung, value vao phia trong
        cols = new_df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        new_df=new_df[cols]

        # Add USL, LSL, UCL, LCL, Mean:
        new_df['USL']=df[name][item_row+2] # max
        new_df['LSL']=df[name][item_row+3] # min
        new_df['Nominal']=df[name][item_row+8] # min
        new_df[new_df.columns[2:]]=new_df[new_df.columns[2:]].astype('float32')
        dim_name=df[name][item_row]
        dim_dict[dim_name]=new_df
      sheet_all[filename]=dim_dict
    return sheet_all
    
sheet_all=create_sheet_dict(all_files) # Run function above
#st.text('len(sheet_all): '+str(len(sheet_all)))

#--------------Base file and concat-----------------#

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def create_base_file_and_concat(sheet_all):
    
    base_file=sheet_all[list(sheet_all.keys())[-1]].copy() # lay file cuoi cung
    first_dim=list(base_file.keys())[0]
    num_sub_sample=len(base_file[first_dim])
    # concat all process in each weeks based on keys value:
    for file_name in list(sheet_all.keys())[:-1]: # ko tính base week nên từ 1: 
      other_file=sheet_all[file_name]
      for dim_name in other_file.keys(): # dict all process
        for dim_name_base in base_file.keys(): # dict all process
            if dim_name_base==dim_name: 
              base_file[dim_name]=\
              pd.concat([base_file[dim_name],other_file[dim_name]])
              base_file[dim_name]=base_file[dim_name].sort_values(by='Date').reset_index(drop=True).dropna() # drop NaN value in row
    # find max len:            
    max_len=len(base_file[list(base_file.keys())[0]])
    for name in base_file.keys():
        if len(base_file[name]) > max_len: # Invalid comparison between dtype=datetime64[ns] and int
            max_len=len(base_file[name])
    # drop non numeric dim:
    print('before drop non numeric dim: ',base_file.keys())
    dim_drop=[]
    for name in base_file.keys():
      print(name,len(base_file[name]))
      if len(base_file[name]) < 0.1*max_len:
        print('drop dim: ',name)
        dim_drop.append(name)
    for dim in dim_drop:
      base_file.pop(dim,None)
      #base_file=base_file.pop(dim,None)
    print('after drop non numeric dim: ',base_file.keys())
    return base_file,num_sub_sample

base_file,num_sub_sample=create_base_file_and_concat(sheet_all)
st.text('num_sub_sample: '+ str(num_sub_sample))
#st.dataframe(base_file[list(base_file.keys())[0]])

#----------------Calculate UCL, LCL, Cp, Cpk, Ppk base on stdev 25 lates subgroup sample----#
constants={
    2:1.128,3:1.693,4:2.059,5:2.326,6:2.534,7:2.704,8:2.847, 9: 2.970,10: 3.078,
    11: 3.173,12: 3.258,13: 3.336,14: 3.407,15: 3.472,16: 3.532,17:3.588,18:3.640,
    19:3.689,20:3.735,
}

def process_performance(df,num_sub_sample):
  #print('dim: ',name)
  n=num_sub_sample
  num_sample=n*25 # chuyển từ 25 dữ liệu thành 50 dữ liệu tránh trường hợp các value trùng nhau
  
  df_temp=df[-num_sample:].reset_index(drop=True)
  #print(len(df))
  #print(len(df_temp))
  usl=df_temp.USL[0]
  lsl=df_temp.LSL[0]
  #m=df_temp.Value.mean()
  m=df_temp.Value.mean() # mean 25 latest sample

  #Ppk
  sigma=np.std(df_temp.Value)
  Pp = float(usl - lsl) / (6*sigma)
  Ppu = float(usl - m) / (3*sigma)
  Ppl = float(m - lsl) / (3*sigma)
  Ppk = np.min([Ppu, Ppl])
  

  #UCL, LCL, Mean
  k=3
  df['UCL']=df_temp['Value'].mean() + sigma*k
  df['LCL']=df_temp['Value'].mean() - sigma*k
  df['Mean']=df_temp['Value'].mean()

  #Cpk
  
  temp=df_temp.groupby('Date').agg({'Value':['min','max']})
  temp['Range']=temp['Value','max']-temp['Value','min']
  Range=temp['Range'].mean()

  if n <= 20:
    sigma_within = Range/constants[n] # Problem in this line ?
  else:
    sigma_within = Range/constants[20]

  Cp = float(usl - lsl) / (6*sigma_within) #ProblemL float division by zero
  Cpu = float(usl - m) / (3*sigma_within)
  Cpl = float(m - lsl) / (3*sigma_within)
  Cpk = np.min([Cpu, Cpl])
  #print('Pp:{:.2f} , Ppk: {:.2f}'.format(Pp,Ppk))
  #print('Cp:{:.2f} , Cpk:{:.2f}'.format(Cp,Cpk))
  if np.isnan(usl):
    Cpk=Cpl
    Ppk=Ppl
  elif np.isnan(lsl):
    Cpk=Cpu
    Ppk=Ppu
  else:
    Cpk = np.min([Cpu, Cpl])
    Ppk = np.min([Ppu, Ppl])
  Cp=round(Cp,2)
  Cpk=round(Cpk,2)
  Pp=round(Pp,2)
  Ppk=round(Ppk,2)
  return Cp,Cpk,Pp,Ppk


@st.cache(allow_output_mutation=True)
def create_process_indicator(base_file): 
    process_indicator_dict={}
    process_indicator_df=pd.DataFrame(columns=['Dim','Cp','Pp','Cpk','Ppk'])
    #process_indicator_df.columns=['Dim','Cp','Cpk','Pb','Ppk']
    i=0
    for dim_name in list(base_file.keys()):
      print(dim_name)
      df=base_file[dim_name]
      #try: # object column cannot be calculated process indicator (OK, Not OK...) How to remove object colum in the beginning ?
      Cp,Cpk,Pp,Ppk=process_performance(df,num_sub_sample) 
      #except: continue
      process_indicator_dict[dim_name]=[Cp,Pp,Cpk,Ppk]
      process_indicator_df.loc[i]=dim_name,Cp, Pp, Cpk, Ppk
      i+=1

    process_indicator_df=process_indicator_df.sort_values(by='Ppk').reset_index(drop=True)

    #conver process indicator dict to list name:
    name_list=[]
    for name in process_indicator_dict.keys():
      dim_name='dim: ' + name + ' Cp: ' + str(process_indicator_dict[name][0]) + ' Pp: '+ \
      str(process_indicator_dict[name][1]) +' Cpk: ' \
      + str(process_indicator_dict[name][2]) +' Ppk: '+ str(process_indicator_dict[name][3]) 
      name_list.append(dim_name)
    return process_indicator_df,name_list

process_indicator_df,name_list=create_process_indicator(base_file)

st.subheader("Process indicator: Cp, Pb, Cpk, Ppk")
limit=st.number_input("Please input lower limit of Cpk and Ppk",value=1.33)
limit=float(limit)
#limit = 1.33  # sigma: 4, Yield: 99.99%   
def hightlight_price(row):
    ret = ["" for _ in row.index]
    if row.Cpk < limit or row.Ppk < limit:
      ret[row.index.get_loc("Dim")] = "background-color: yellow"
    if row.Cpk < limit:    
      ret[row.index.get_loc("Cpk")] = "background-color: yellow"
    if row.Ppk < limit:  
      ret[row.index.get_loc("Ppk")] = "background-color: yellow"
    return ret

@st.cache(allow_output_mutation=True)
def highlight_process_indicator(process_indicator_df): 
    return process_indicator_df.style.apply(hightlight_price, axis=1)
process_indicator_df=highlight_process_indicator(process_indicator_df)

st.text('Highlight yellow for dim below lower limit:')
st.dataframe(process_indicator_df)
#st.dataframe(base_file[list(base_file.keys())[0]])
#st.dataframe(name_list)

#----------------------------------------
st.subheader('(Optional:) Please input start date and end date for data analysis')
st.write('Skip this step or leave it blank if you need full time range')
start_date= st.text_input("Please input start Date (Ex: 2018, 2018/07, 2018/07/31)")
end_date= st.text_input("Please input end Date (Ex:  2019, 2019/08, 2018/08/20)")

#print(base_file)

#----------LINE CHART--------------------------------------------#
st.header("Control chart (group value each day)")
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def line_chart(base_file,name_list):
    df_dict=base_file
    fig = make_subplots(          # Dim name
        rows=len(df_dict), cols=1,
        #shared_xaxes=True, # share same axis
        #vertical_spacing=0.05, # adjust spacing between charts
        #column_widths=[0.8, 0.2],
        #subplot_titles=(list(df_dict.keys()))
        subplot_titles=(name_list)
    )
    i=1
    for name in list(df_dict.keys()): #also group
      #print(name)
      df=df_dict[name]
      df=df.sort_values(by=['Date'])
      non_object_column=list(df.select_dtypes(exclude=['object']).columns)
      for a in df.columns[1:]:
        if a in non_object_column :
          df[a] = df[a].round(decimals=3) # problem here
      df=df.reset_index(drop=True)
      df_group=df.groupby('Date').aggregate({'ID_No': 'max','Value': 'mean','USL': 'mean',
      'LSL': 'mean','UCL': 'mean','LCL': 'mean','Mean': 'mean'})
      #df=df.set_index('Date')
      if start_date != '' and end_date != '':
        df_group=df_group[start_date:end_date] 
      #Control chart 1 
      fig.append_trace(go.Scatter(
                              x=df_group.index, y=df_group['Value'],mode='lines+markers',name='Mean ',
                              line=dict( color='#4280F5'),text=df_group['ID_No']
                              ),row=i, col=1)
      #USL, LSL
      fig.append_trace(go.Scatter(x=df_group.index, y=df_group['USL'],name='USL ', line=dict( color='#FF5733'),mode='lines'),row=i, col=1)
      fig.append_trace(go.Scatter(x=df_group.index, y=df_group['LSL'],name='LSL ',line=dict( color='#FF5733'),mode='lines'),row=i, col=1)
      #fig.append_trace(go.Scatter(x=df['Datef'], y=df['Nominal'],name='Nominal '+name,line=dict( color='#FF5733')),row=i, col=1)
      # UCL, LCL
      fig.append_trace(go.Scatter(x=df_group.index, y=df_group['UCL'],name='UCL ', line=dict( color='#33C2FF'),mode='lines'),row=i, col=1)
      fig.append_trace(go.Scatter(x=df_group.index, y=df_group['LCL'],name='LCL ', line=dict( color='#33C2FF'),mode='lines'),row=i, col=1)
      fig.append_trace(go.Scatter(x=df_group.index, y=df_group['Mean'],name='Mean ', line=dict( color='#33C2FF'),mode='lines'),row=i, col=1)
      i=i+1


    fig.update_layout(height=200*len(df_dict), width=1200, title_text=material_name)
    #fig update each process (contain a lot of dim inside)
    return fig
fig_new=line_chart(base_file,name_list)    
st.plotly_chart(fig_new)
#------------------Save Line chart plotly-----------------------------------------#

if st.checkbox('Save Line chart'):
#if st.button("Save Line chart"):
    fig_new.write_html(path3+material_name+'_line.html')
    st.write('Path file: ',path3+material_name+'_line.html')
#---------------------------Box chart--------------------------------------#
st.header("Box chart")
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def box_chart(base_file,name_list):
    
    df_dict=base_file
    fig = make_subplots(          # Dim name
        rows=len(df_dict), cols=1,
        #shared_xaxes=True, # share same axis
        #vertical_spacing=0.05, # adjust spacing between charts
        #column_widths=[0.8, 0.2],
        #subplot_titles=(list(df_dict.keys()))
        subplot_titles=(name_list)
    )
    i=1
    for name in list(df_dict.keys()): #also group
      df=df_dict[name]
      df=df.sort_values(by=['Date'])
      non_object_column=list(df.select_dtypes(exclude=['object']).columns)
      for a in df.columns[1:]:
        if a in non_object_column :
          df[a] = df[a].round(decimals=3)
      df=df.reset_index(drop=True)
      df=df.set_index('Date')
      if start_date != '' and end_date != '':
        df=df[start_date:end_date]
      #Control chart 1 
      fig.append_trace(go.Box(
                              x=df.index, y=df['Value'],name='value '+name,
                              line=dict( color='#4280F5'),text=df['ID_No'],boxpoints='all'
                              ),row=i, col=1)
      #USL, LSL
      fig.append_trace(go.Scatter(x=df.index, y=df['USL'],name='USL '+name, line=dict( color='#FF5733'),mode='lines'),row=i, col=1)
      fig.append_trace(go.Scatter(x=df.index, y=df['LSL'],name='LSL '+name,line=dict( color='#FF5733'),mode='lines'),row=i, col=1)
      fig.append_trace(go.Scatter(x=df.index, y=df['Nominal'],name='Nominal '+name,line=dict( color='#FF5733')),row=i, col=1)
      i=i+1


    fig.update_layout(height=200*len(df_dict), width=1200, title_text=material_name)
    return fig

fig_new_2=box_chart(base_file,name_list)    
st.plotly_chart(fig_new_2)
#----------------------------Box chart plotly Save-----------------------
if st.checkbox('Save Box chart'):
#if st.button("Save Line chart"):
    fig_new_2.write_html(path3+material_name+'_box.html')
    st.write('Path file: ',path3+material_name+'_box.html')



#---------------------------DataFrame----------------------------
st.header("Data Frame")
if st.checkbox('Show dataframe'):
#if st.button("Show dataframe"):
    # selectbox
    option = st.selectbox(
    'Which dimension to show ?',list(base_file.keys()))
    st.dataframe(base_file[option])

if st.checkbox('Save dataframe'):
    for name in base_file.keys():
        base_file[name].to_csv(path3+name+'.csv')
        st.write('Path file: ',path3+name+'.csv')

#-----------------Fore cast--------------------------#
from fbprophet import Prophet
@st.cache(suppress_st_warning=True)
def predict_prophet(base_file):
    DFdict_final=base_file
    DF_predict_final_={} # only contail predict
    #DF_predict_all_final={} # contain predict and past value
    for name in DFdict_final.keys(): #also group
      print(name)
      df_temp_2=DFdict_final[name] # gia tri mean thong thuong
      df_temp_2=df_temp_2.groupby('Date').mean()
      df_predict=df_temp_2.copy()
      #df_predict=df_predict.resample('1M').ffill() # gia tri mean da resample
      df_predict=df_predict.resample('M').mean().dropna()
      df_fb=df_predict.reset_index()#
      df_fb=df_fb[[df_fb.columns[0],df_fb.columns[1]]] # old : column 0,1. New: column 0, 2
      df_fb.columns=['ds','y']

      # Prophet will by default fit weekly and yearly seasonalities,
      model = Prophet( #yearly_seasonality=True
                    ) #instantiate Prophet with only yearly seasonality as our data is monthly 
      model.fit(df_fb) #fit the model with your dataframe

      # predict for five months in the furure and MS - month start is the frequency
      future = model.make_future_dataframe(periods = predict_future, freq = 'M')  
      # now lets make the forecasts
      forecast = model.predict(future)
      new_forecast=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
      new_forecast['USL']=df_temp_2['USL'][0]
      new_forecast['LSL']=df_temp_2['LSL'][0]
      #new_forecast['Nominal']=df_temp_2['Mean'][0]
      new_forecast['UCL']=df_temp_2['UCL'][0]
      new_forecast['LCL']=df_temp_2['LCL'][0]
      new_forecast['mean']=df_temp_2['Mean'][0]
      DF_predict_final_[name]=new_forecast 
    return DF_predict_final_

#------------------------------------Prophet Plotly-------------------------------#
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def line_chart_prophet(base_file,DF_predict_final_,name_list):
    DFdict_final=base_file
    DF_predict_final=DF_predict_final_
    i=1
    #Layout
    fig = make_subplots(
        rows=len(DFdict_final), cols=1,
        #shared_xaxes=True,
        #vertical_spacing=0.05,
        #column_widths=[0.8, 0.2],
        subplot_titles=(name_list)
    )

    for name in DFdict_final.keys(): #also group
      df=DFdict_final[name].copy()
      df=df.sort_values(by=['Date'])
      non_object_column=list(df.select_dtypes(exclude=['object']).columns)
      for a in df.columns[1:]:
        if a in non_object_column :
          df[a] = df[a].round(decimals=3)
      df=df.groupby('Date').mean()
      #Control chart 1 
      fig.append_trace(go.Scatter(
                              x=df.index, y=df['Value'],
                              mode='lines+markers',
                              name='mean ' + name,line=dict( color='#4280F5')
                              ),row=i, col=1
                    )
      #Predict:
      df_predict=DF_predict_final[name].copy()
      fig.append_trace(go.Scatter(
                          x=df_predict['ds'], y=df_predict['yhat'],
                          mode='lines+markers',
                          name='mean predict ' + name,line=dict( color='#060891')
                          ),row=i, col=1
                )

      #USL, LSL
      fig.append_trace(go.Scatter(x=df_predict['ds'], y=df_predict['USL'],name='USL '+name, line=dict( color='#FF5733'),mode='lines'),row=i, col=1)
      fig.append_trace(go.Scatter(x=df_predict['ds'], y=df_predict['LSL'],name='LSL '+name,line=dict( color='#FF5733'),mode='lines'),row=i, col=1)
      #fig.append_trace(go.Scatter(x=df_predict['ds'], y=df_predict['Nominal'],name='Nominal '+name,line=dict( color='#FF5733')),row=i, col=1)
      # UCL, LCL
      #fig.append_trace(go.Scatter(x=df_predict['ds'], y=df_predict['UCL'],name='UCL '+name, line=dict( color='#33C2FF')),row=i, col=1)
      #fig.append_trace(go.Scatter(x=df_predict['ds'], y=df_predict['LCL'],name='LCL '+name, line=dict( color='#33C2FF')),row=i, col=1)
      #fig.append_trace(go.Scatter(x=df_predict['ds'], y=df_predict['mean'],name='mean '+name, line=dict( color='#33C2FF')),row=i, col=1)
      # Fill between:
      fig.append_trace(go.Scatter(x=df_predict['ds'], y=df_predict['yhat_upper'],name='yhat_upper '+name, line=dict( color='#9FA0F8'),fill=None,mode='lines'),row=i, col=1)
      fig.append_trace(go.Scatter(x=df_predict['ds'], y=df_predict['yhat_lower'],name='yhat_lower '+name, line=dict( color='#9FA0F8'), fill='tonexty',mode='lines'),row=i, col=1)


      #Final layout:
      #fig.update_layout(height=400, width=1400, title_text='Line chart'+name)
      i=i+1

    fig.update_layout(height=200*len(DFdict_final), width=1200, title_text=material_name)
    return fig

st.header("Prediction (Just for reference only)")

if st.checkbox('Please tick to this box if you need prediction'):
    num_predict=st.number_input("Please input number of Months for prediction",value=3)
    num_predict=int(num_predict)
    predict_future=num_predict # months

    DF_predict_final_=predict_prophet(base_file)
    fig_new_3=line_chart_prophet(base_file,DF_predict_final_,name_list)
    st.plotly_chart(fig_new_3)
    
    if st.checkbox('Save predict Line chart'):
        fig_new_2.write_html(path3+material_name+'_predict_line.html')
        st.write('Path file: ',path3+material_name+'_predict_line.html')

#------------------Save prediction chart plotly-----------------------------------------#

