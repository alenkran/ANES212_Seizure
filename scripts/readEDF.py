################################################################################
# Description: Functions to read EDF files

# Converts the EDF files to panda dataframes with labeled columns along with
# labeling each time points are within a seizure or not.
################################################################################

from bs4 import BeautifulSoup
import numpy as np
import os
import pandas as pd
import pyedflib
import re
import requests
from urllib import urlopen, urlretrieve


def extract_summary_data(f, temp_d):
	summary = f.read()
	summary = summary.split('\n\n')
	temp_d['freq'] = float(summary[0].split(' ')[3])
	for line in summary:
		if 'File Name:' in line:
			temp = filter(None, line.split('\n'))
			key = temp[0].split(' ')[2]
			if len(temp) == 4:
				temp_d[key] = []
				continue
			temp = temp[4:]
			seizure_time = []
			for x,y in zip(temp[0::2], temp[1::2]):
				x = int(x.split(' ')[-2])
				y = int(y.split(' ')[-2])
				seizure_time.append((x,y))
			temp_d[key] = seizure_time

def read_edf(filename):
	f = pyedflib.EdfReader(filename)
	assert(np.all(f.getSampleFrequencies() == f.getSampleFrequencies()[0]))
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
		sigbufs[i, :] = f.readSignal(i)
	return sigbufs, f.getSignalLabels()
	
def label_edf(sigbufs, seizure_time, eeg_label, freq = 256.0):
	time = np.arange(sigbufs.shape[1])/freq
	seizure_id = np.zeros(sigbufs.shape[1])
	seizure_delay = np.zeros(sigbufs.shape[1])
	if seizure_time:
		prev_end = 0
		for x,y in seizure_time:
			start_idx = int(freq*x)
			end_idx = int(freq*y)+1 # Include the end point within the data
			seizure_id[start_idx:end_idx].fill(1)
			seizure_delay[prev_end:start_idx] = np.arange(start_idx-prev_end)[::-1]/freq
			seizure_delay[end_idx:].fill(np.inf)
			prev_end = end_idx
	else:
		seizure_delay += np.inf
	df = np.vstack([time, seizure_id, seizure_delay, sigbufs])
	df = df.T
	df = df.astype('float32')
	eeg_label = ['time', 'seizure', 'seizure_delay'] + eeg_label
	df = pd.DataFrame(df, columns=eeg_label)
	df['seizure'] = df['seizure'].astype('int8')
	return df

# Use the function like this
# df = read_single_edf('chb10_89.edf')
def read_single_edf(filename):
	""" Reads a single edf file into a labeld data frame 

	Use the function like
	> import readEDF 
	> df = read_single_edf('chb10_89.edf')
	"""
	temp = np.array(os.getcwd().lower().split('/'))
	idx_base = np.where(temp[::-1] == 'github')[0][0]
	if '/' not in filename:
		data_folder = '/'.join(os.getcwd().split('/')[:-idx_base]) + '/ANES212_data/'
		patient_folder = filename.split('_')[0]
		filename = data_folder + patient_folder + '/' + filename
	else:
		data_folder = '/'.join(os.getcwd().split('/')[:-idx_base]) + '/ANES212_data/'
		patient_folder = filename.split('/')[-2]
	
	# Extract label data for the filename (e.g. start and end of seizures if any)
	summary_file = data_folder + patient_folder + '/' + patient_folder + '-summary.txt'
	temp_d = {}
	with open(summary_file, 'rb') as f:
		extract_summary_data(f, temp_d)
	
	# Now read edf and label
	if '.edf' not in filename:
		filename = filename + '.edf'
	sigbufs, eeg_label = read_edf(filename)
	df = label_edf(sigbufs, temp_d[filename.split('/')[-1]], eeg_label)
	return df

def read_patient_edf(patient_list, save = False):
	""" Reads all the edf files of patients in patient_list 
	
	Returns a dictionary with each key being each edf filename
	"""

	# Correct input if not a list
	if isinstance(patient_list, str):
		patient_list = [patient_list]

	data_folder = '/'.join(os.getcwd().split('/')[:-1]) + '/ANES212_data/'
	patient_dict = {}
	for root, dirs, files in os.walk(data_folder):
		if any([patient_folder in root for patient_folder in patient_list]):
			temp_d = {} # Stores data from the summary txt file
			for filename in files:
				full_path = os.path.join(root, filename)
				if '.edf'in filename and '.seizures' not in filename:
					print(full_path)
					sigbufs, eeg_label = read_edf(full_path)
					df = label_edf(sigbufs, temp_d[filename], eeg_label)
					df_filename = os.path.join(root, filename.split('.')[0]) + '.csv'
					if save:
						df.to_csv(df_filename, sep='\t')
					patient_dict[filename] = df
				elif '-summary.txt' in filename:
					with open(full_path, 'rb') as f:
						extract_summary_data(f, temp_d)
	return patient_dict

def read_single_edf_raw(filename):
	""" Reads a single edf file. Do not have to specify the folder or .edf """
	temp = np.array(os.getcwd().lower().split('/'))
	idx_base = np.where(temp[::-1] == 'github')[0][0]
	if '/' not in filename:
		data_folder = '/'.join(os.getcwd().split('/')[:-idx_base]) + '/ANES212_data/'
		patient_folder = filename.split('_')[0]
		filename = data_folder + patient_folder + '/' + filename
	else:
		data_folder = '/'.join(os.getcwd().split('/')[:-idx_base]) + '/ANES212_data/'
		patient_folder = filename.split('/')[-2]

	# Now read edf and label
	if '.edf' not in filename:
		filename = filename + '.edf'
	return read_edf(filename)

def create_folder(directory):
    """ Creates directory if it doesn't not exist """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_href(url):
    """ Returns a list of all href in a url """
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    return soup.findAll('a', href=True)

def scrape_BCH_dateset():
	""" Downloads the entire BCH seizure dataset 
	
	Saves the dataset in /..ANES212_data/
	"""
	url = 'https://physionet.org/pn6/chbmit/'

	# Location for storing the data
	data_folder = '/'.join(os.getcwd().split('/')[:-1]) + '/ANES212_data/'
	create_folder(data_folder)

	# Compile a list of files and folders
	filelist = []
	folderlist = []
	for x in get_href(url):
	    temp = x['href']
	    if re.match(r'^\w+$', temp[0]) and 'http' not in temp and '.pdf' not in temp and '.org' not in temp:
	        if '/' in temp:
	            folderlist.append(temp)
	        else:
	            urlretrieve(url + temp, data_folder + temp)

	# For each folder download data
	for folder in folderlist: # Replace this
	    temp_url = url + folder
	    temp_data_folder = data_folder + folder
	    
	    # Create folder to store the data if it doesn't exist
	    create_folder(temp_data_folder)
	    
	    r = requests.get(temp_url)
	    soup = BeautifulSoup(r.text, "html.parser")
	    temp_filelist = []
	    for x in get_href(temp_url):
	        temp = x['href']
	        if re.match(r'^\w+$', temp[0]) and '.org' not in temp:
	            urlretrieve(temp_url + temp, temp_data_folder + temp)