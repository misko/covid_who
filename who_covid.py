import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d,uniform_filter1d
from scipy.signal import find_peaks


#Download CSV from 
#https://covid19.who.int/
if len(sys.argv)!=2:
	print("%s covid_who.csv" % sys.argv[0])
	print("Please download covid csv from https://covid19.who.int/")
covid_csv=sys.argv[1]

top_peaks=[]
df=pd.read_csv(covid_csv)
for name,country_df in df.groupby("Country"):
	new_cases=country_df['New_cases'].values

	#restrict for countries with some minimum number of cases
	idx=0
	while idx<len(new_cases) and new_cases[idx]<1000:
		idx+=1
	if idx>=len(new_cases):
		continue
	new_cases=new_cases[idx:]
	if len(new_cases)<60:
		continue

	#different smoothing options, trying to remove high freq (low period) noise
	smooth_fn = lambda x : uniform_filter1d(x,10)
	#smooth_fn = lambda x : gaussian_filter1d(x,5)

	new_cases_smooth = smooth_fn(new_cases)
	sp = np.fft.rfft(new_cases-new_cases_smooth)
	power=np.abs(sp)**2
	angle=np.angle(sp)
	periods = 1.0/np.fft.rfftfreq(new_cases.shape[-1])

	peaks = [ { 'power':power[peak],'phase':angle[peak],'period':periods[peak] } for peak in power.argsort()[:][::-1] if power[peak]>1e4]

	top_peaks.append((name,peaks[:2]))

	#plot power fft
	fig, axs = plt.subplots(3,1,figsize=(10,11))
	axs[0].axvline(x=7,color="red")
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
		axs[1+idx].set_title("New cases (raw, smooth, reconstructed) - %smax period peak (%0.2f)" % ("" if idx==0 else "second ",period))
		axs[1+idx].plot(plotx,ploty_smooth,label="smooth",color="orange")
		axs[1+idx].plot(plotx,
			ploty_smooth+amp*np.cos(phase+2*np.pi*plotx/period),label="smooth+cos",color='darkred',alpha=0.4)
		axs[1+idx].plot(range(new_cases.shape[0]),new_cases,label='raw new cases',color="darkgreen")
		axs[1+idx].legend()
		[ axs[1+idx].axvline(x=x_pos) for x_pos in np.arange( -phase*(period/(2*np.pi)), plotx[-1], period) ]
		axs[1+idx].set_xlabel("Days")
		axs[1+idx].set_ylabel("Cases")
	plt.suptitle(name)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.savefig("covid_%s.png" % name)
	plt.close()


fig, ax = plt.subplots(figsize=(8,5))
for name,these_peaks in top_peaks:
	xs, ys = list(zip(*[ (x['period'],x['power']) for x in these_peaks ]))
	ax.plot(xs[0],np.log2(ys[0]),label=name,marker='x',markersize=6)
	#close your eyes now
	offset_y=-0.2
	offset_x=0.2
	if name=="China":
		offset_x+=0
		offset_y-=0
	if name=="Italy":
		offset_x-=1.2
	if name=="India":
		offset_x-=1.2
	if name=="Switzerland":
		offset_x-=2.5
	if name=="France":
		offset_x-=1.6
	if name=="Belarus":
		offset_y+=0.3
	if name=="United Arab Emirates":
		offset_y+=0.5
		offset_x-=1
	if name=="Belgium":
		offset_x-=1.8
	if name=="Saudi Arabia":
		offset_y-=0.15
	if name=="Mexico":
		offset_y-=0.0
	if name=="Peru":
		offset_x-=1.2
	if name=='Russian Federation':
		offset_x-=1.5
		offset_y+=0.55
	if name=='The United Kingdom':
		name="UK"
	#and open your eyes, dont look up
	ax.annotate(name, (xs[0]+offset_x,np.log2(ys[0])+offset_y),fontsize=10)
ax.set_title("Max power period in FFT (COVID new cases)")
ax.axvline(x=7)
ax.set_xlabel("Max power period in FFT (days)")
ax.set_ylabel("Power (log2)")
ax.legend(prop={'size': 6},loc="center right")
plt.savefig("covid.png")
plt.close()
