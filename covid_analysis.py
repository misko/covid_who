import pandas as pd
import sys
from datetime import timedelta
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d,uniform_filter1d
from scipy.signal import find_peaks
from adjustText import adjust_text
import requests
import io

def process_district(name, df,prefix,top_peaks,category):
	#new_cases=df['New_cases'].values
	new_cases=df[categories[category]['col']].values
	#restrict for countries with some minimum number of cases
	start_idx=0
	while start_idx<len(new_cases) and new_cases[start_idx]<categories[category]['start_threshold']: # 1000 
		start_idx+=1
	if start_idx>=len(new_cases):
		return
	date_time_obj = datetime.datetime.strptime(df.iloc[start_idx]['Date_reported'].split("T")[0], '%Y-%m-%d')
	
	end_idx=start_idx
	smooth=3
	while end_idx<len(new_cases):
		if new_cases[end_idx]<=categories[category]['end_threshold']:
			if smooth==0:
				break
			else:
				smooth-=1
		else:
			smooth=min(3,smooth+1)	
		end_idx+=1
		
	new_cases=new_cases[start_idx:end_idx]
	if len(new_cases)%2==1:
		new_cases=new_cases[:-1]
	if len(new_cases)<categories[category]['min_days']:
		return

	#different smoothing options, trying to remove high freq (low period) noise
	smooth_fn = lambda x : uniform_filter1d(x,10)
	#smooth_fn = lambda x : gaussian_filter1d(x,5)

	new_cases_smooth = smooth_fn(new_cases)
	sp = np.fft.rfft(new_cases-new_cases_smooth)
	power=np.abs(sp)**2
	angle=np.angle(sp)
	periods = 1.0/np.fft.rfftfreq(new_cases.shape[-1])

	sp_inv = sp.copy()
	slack=0.5
	for idx in range(1,len(periods)):
		period=periods[idx]
		if (period<(3.5+slack) and period>(3.5-slack)) or (period<(7+slack) and period>(7-slack)):
			pass
		else:
			sp_inv[idx]=0
	sp_inv_i = np.fft.irfft(sp_inv)
	try:
		print( "ERR", name,np.abs(((new_cases-new_cases_smooth)-sp_inv_i)).mean())
	except:
		pass
			
		shift=-phase*(period/(2*np.pi))
		if idx==0:
			max_date = date_time_obj+timedelta(days=shift )
			weekday=max_date.strftime('%A')
			axs[1+idx].axvline(x=shift,color='black')

	peaks = [ { 'power':power[peak],'phase':angle[peak],'period':periods[peak],'shift':-angle[peak]*(periods[peak]/(2*np.pi)) } for peak in power.argsort()[:][::-1] if power[peak]>1e4]
	#print(timedelta(days=peaks[0]['shift']))
	#print(date_time_obj , " :::", (date_time_obj + timedelta(days=peaks[0]['shift'])))
	weekday = (date_time_obj + timedelta(days=peaks[0]['shift'])).strftime('%A')
	top_peaks.append((name,weekday,peaks[:2]))
	print(name,weekday)

	#plot power fft
	fig, axs = plt.subplots(3,1,figsize=(10,11))
	axs[0].axvline(x=3.5,color="red",linestyle=":")
	axs[0].axvline(x=7,color="red",linestyle=":")
	axs[0].set_title("FFT(new_case) - Power vs Period (days)")
	axs[0].plot(periods, power,"o")
	axs[0].plot([ x['period'] for x in peaks[:2] ],
		[ x['power'] for x in peaks[:2] ],"o", color="orange")
	axs[0].set_xlabel("Period (days)")
	axs[0].set_ylabel("Power")
	axs[0].set_xticks(np.arange(0, 18))
	axs[0].set_xlim([0,18])
	#plot the actual data
	#smooth things over 
	f = interpolate.interp1d(range(len(new_cases)), new_cases)
	plotx = np.linspace(0,len(new_cases)-1,2000)
	ploty = f(plotx)
	f_smooth = interpolate.interp1d(range(len(new_cases_smooth)), new_cases_smooth)
	ploty_smooth = f_smooth(plotx)

	#amp=(ploty-ploty_smooth).max()/2 # just use something reasonable?

	for idx in [0,1]: # plot the top two peaks
		period=peaks[idx]['period']
		phase=peaks[idx]['phase']
		amp=np.sqrt(peaks[idx]['power'])/8
		shift=peaks[idx]['shift']
		axs[1+idx].set_title("New %s (raw, smooth, reconstructed) - %smax period peak (%0.2f)" % (categories[category]['label'],"" if idx==0 else "second ",period))
		axs[1+idx].plot(plotx,ploty_smooth,label="smooth",color="orange")
		axs[1+idx].plot(plotx,
			ploty_smooth+amp*np.cos(phase+2*np.pi*plotx/period),label="smooth+cos",color='darkred',alpha=0.4)
		axs[1+idx].plot(range(new_cases.shape[0]),new_cases,label='raw new %s' % categories[category]['label'],color="darkgreen")
		axs[1+idx].legend()
		[ axs[1+idx].axvline(x=x_pos) for x_pos in np.arange( shift, plotx[-1], period) ]
		axs[1+idx].set_xlabel("Days")
		axs[1+idx].set_ylabel(categories[category]['label'])
	plt.suptitle(name + "(%s)" % weekday)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.savefig("%s_covid_%s_%s.png" % (prefix,categories[category]['label'],name))
	plt.close()


def nyt_to_delta(df):
	cases=0
	deaths=0
	rows=[]
	for index, row in df.iterrows():
		rows.append([row['date'],int(row['cases'])-cases,int(row['deaths'])-deaths])
		cases=int(row['cases'])
		deaths=int(row['deaths'])
	return pd.DataFrame(rows,columns=['Date_reported','New_cases','New_deaths'])	


def process_df(prefix,df,category):
	top_peaks=[]
	if 'Date_reported' in df:
		df=df.rename(columns=lambda x: x.strip())
		#World
		process_district("World",df.groupby('Date_reported',as_index=False).sum(),prefix,top_peaks,category)
		#regions
		for region in df.WHO_region.unique():
			process_district(region, df[ df['WHO_region'] == region ].groupby('Date_reported',as_index=False).sum(),prefix,top_peaks,category)
		for name,country_df in df.groupby("Country"):
			if name=='United States of America':
				name="US"
			if name=="The United Kingdom":
				name="UK"
			if name=="Russian Federation":
				name="Russia"
			if name=="United Arab Emirates":
				name="UAE"
			process_district(name,country_df,prefix,top_peaks,category)
	else:
		#assume nytimes data
		process_district("US",nyt_to_delta(df.groupby('date',as_index=False).sum()),prefix,top_peaks,category)
		if False and 'county' in df:
			county_and_state=set([ (r['county'],r['state']) for idx,r in df.iterrows() ])
			for county,state in county_and_state:
				county_df=nyt_to_delta(df[(df['state']==state) & (df['county']==county)])
				process_district(state+"_"+county,county_df,prefix,top_peaks,category)
		states = df['state'].unique()
		for state in states:
			state_df=nyt_to_delta(df[df['state']==state].groupby('date',as_index=False).sum())
			process_district(state,state_df,prefix,top_peaks,category)


	#fig, axs = plt.subplots(1,3,figsize=(14,11*(3/2)))
	m=1
	fig = plt.figure(figsize=(14/m,11*(3/2)/m))

	gs = fig.add_gridspec(2,2)
	ax1 = fig.add_subplot(gs[:, 0])
	ax2 = fig.add_subplot(gs[1, 1])
	ax3 = fig.add_subplot(gs[0, 1])
	axs = [ax1,ax2,ax3]
	texts=[]
	for name,weekday,these_peaks in top_peaks:
		xs, ys = list(zip(*[ (x['period'],x['power']) for x in these_peaks ]))
		axs[0].plot(xs[0],np.log2(ys[0]),label=name,marker='o',markersize=6)
		offset_y=0
		offset_x=0
		#texts.append( axs[0].annotate("%s(%s)" % (name,weekday[:3]), (xs[0]+offset_x,np.log2(ys[0])+offset_y),fontsize=10) )
		texts.append( axs[0].text(xs[0]+offset_x,np.log2(ys[0])+offset_y,"%s(%s)" % (name,weekday[:3]) , ha='center',  va='center'))
	axs[0].set_title("Max power period (COVID new %s)" % categories[category]['label'])
	axs[0].axvline(x=7,color="red",linestyle=":")
	axs[0].axvline(x=3.5,color="red",linestyle=":")
	axs[0].set_xlabel("Max power period in FFT (days)")
	axs[0].set_ylabel("Power (log2)")
	axs[0].set_xticks(range(12))
	axs[0].set_xlim([0,12])
	import calendar
	days = [ (i,name) for i,name in enumerate(calendar.day_name)] 
	days.sort()

	n, bins, patches = axs[1].hist( [ these_peaks[0]['period'] for these_peaks in [ x[2] for x in top_peaks] ]  ,  density=False, facecolor='g', alpha=0.75)
	axs[1].set_xticks(range(12))
	axs[1].set_xlim([0,12])
	axs[1].set_ylabel("Count")
	axs[1].set_xlabel("Max power period")
	axs[1].set_title("Count vs max power period")

	counts={}
	for i,day in days:
		counts[day]=0
		
	for name,weekday,peaks in top_peaks:
		counts[weekday]+=1
	#axs[2].bar(
	axs[2].bar( [ x[0] for x in days ], [ counts[x[1]] for x in days ])
	#axs[2].set_xticks( [ x[0] for x in days ], [ x[1] for x in days ] )
	axs[2].set_xticks( [ x[0] for x in days ])
	axs[2].set_xticklabels( [ x[1][:3] for x in days ] ,rotation=90)
	axs[2].set_ylabel("Count")
	axs[2].set_title("Count vs Max day of week")



	adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'),ax=axs[0])
	plt.savefig("%s_covid_%s.png" % (prefix,categories[category]['label']))
	plt.close()


#Download CSV from 
#https://covid19.who.int/
#https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/


categories = {
	"cases":{'col':'New_cases','start_threshold':500, 'end_threshold':100, 'label':'cases', 'min_days':60},
	"deaths":{'col':'New_deaths','start_threshold':300, 'end_threshold':20, 'label':'deaths', 'min_days':60},
	}

for (source,url) in [ 
	#("NYT-County","https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"),
	("NYT","https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"), 
	("WHO","https://covid19.who.int/WHO-COVID-19-global-data.csv") ]:
	s=requests.get(url).content
	df=pd.read_csv(io.StringIO(s.decode('utf-8')))
	for category in "cases","deaths":
		process_df(source,df,category)
