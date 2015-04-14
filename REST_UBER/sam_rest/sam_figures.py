
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
## agg backend is used to create plot as a .png file
#mpl.use('agg')

#monthly streak data
def GetSAMMonthlyStreakOutput(self, mongokey):
	#fake, change to mongoquery
	jan = rand(50) * 100
	feb = rand(50) * 100
	mar = rand(50) * 100
	apr = rand(50) * 100
	may = rand(50) * 100
	jun = rand(50) * 100
	jul = rand(50) * 100
	aug = rand(50) * 100
	sep = rand(50) * 100
	octo = rand(50) * 100
	nov = rand(50) * 100
	dec = rand(50) * 100

	sam = concatenate( (jan, feb, mar, apr, may, jun, jul, aug, sep, octo, nov, dec), 0 )
	sam.shape = (50, 12)

	return sam

#all streak data
def GetSAMAllStreakOutput(self, mongokey):
	#fake, change to mongo query
	jan = rand(50) * 100
	feb = rand(50) * 100
	mar = rand(50) * 100
	apr = rand(50) * 100
	may = rand(50) * 100
	jun = rand(50) * 100
	jul = rand(50) * 100
	aug = rand(50) * 100
	sep = rand(50) * 100
	octo = rand(50) * 100
	nov = rand(50) * 100
	dec = rand(50) * 100
	
	sam_vector = concatenate( (jan, feb, mar, apr, may, jun, jul, aug, sep, octo, nov, dec), 0 )

	return sam_vector

## figure 1
def GenerateSAMBoxplot(self, mongokey):

	# get sam monthly data array of streaks
	sam_vector = GetSAMMonthlyStreakOutput(mongokey)

	# Create a figure instance
	fig = plt.figure(1, figsize=(10, 6))

	# Create an axes instance
	ax = fig.add_subplot(111)
	ax1 = fig.add_subplot(111)

	# Create a boxplot
	bp = ax.boxplot(sam, notch=0, sym='+', vert=1, whis=1.5, patch_artist = True)
	plt.setp(bp['boxes'], color='black')
	plt.setp(bp['whiskers'], color='black')
	plt.setp(bp['fliers'], color='red', marker='+')
	colors = ['lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue',
		 'lightblue','lightblue','lightblue','lightblue','lightblue','lightblue']
	for patch, color in zip(bp['boxes'], colors):
	    patch.set_facecolor(color)

	## Custom x-axis labels
	monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	ax.set_xticklabels(monthNames)

	## Remove top axes and right axes ticks
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax1.set_xlabel('Month')
	ax1.set_ylabel('Days')

	# Add a horizontal grid to the plot - light in color
	ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
		      alpha=0.5)
	ax1.set_axisbelow(True)
	fig.suptitle("Monthly Average Exceedance Streak Distribution across HUCs")

	# Save the figure
	fig.savefig("streak_boxplot.png", bbox_inches = "tight")
	fig.canvas.set_window_title('Monthly Streak Average')
	fig.clf()


## figure 2
def GenerateSAMHistogram(self, mongokey):

	# get sam data vector of streaks
	sam_vector = GetSAMAllStreakOutput(mongokey)

	# Create a second figure instance
	fig2 = plt.figure(2, figsize=(10, 6))

	# Create an axes instance
	ax_2 = fig2.add_subplot(111)
	ax1_2 = fig2.add_subplot(111)

	# Add a horizontal grid to the plot - light in color
	ax1_2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
		      alpha=0.5)
	ax1_2.set_axisbelow(True)

	# Create a histogram
	hist_fig = ax_2.hist(sam_vector, facecolor="darkseagreen")

	## Remove top axes and right axes ticks
	ax_2.get_xaxis().tick_bottom()
	ax_2.get_yaxis().tick_left()
	ax1_2.set_xlabel('Days')
	ax1_2.set_ylabel('Frequency')

	#fig2.title('Streak Average across all Months and HUCs')
	ax_2.set_title('Streak Average across all Months and HUCs')

	# Save the figure
	fig2.savefig("streak_histogram.png", bbox_inches = "tight")

	fig2.clf()


