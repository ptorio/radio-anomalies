# radio-anomalies

In order for the authorities of each country to automate surveillance tasks in the use of the radio spectrum, we present a Python app to represent radio anomalies from a dataset of received samples which monitors radio channels in time and frequency based on power detection. A receiver with an omnidirectional antenna records electric field amplitude samples on a set of channels belonging to the terrestrial or maritime mobile service. This method has been successfully used to detect unauthorized radio emissions.

The app is "represent_Power.py"
The app represents data from .csv files.
"SMM 2024-06-7 al 8.csv" and "SMT 2024-09-09 al 10.csv" are data files that can be used to run the app with.

"Figure SMM_analysis.pdf" is a screenshot of the app analyzing the data in file "SMM 2024-06-7 al 8.csv".
"SMM_Anomalies_vs_AveragePower.png" is a screenshot of the app comparing anomalies vs average power in all frequencies in file "SMM 2024-06-7 al 8.csv"

In order to set up Python and install the appropriate extensions on Windows, you will want to read the instructions in "requirementsToRunRepresentation_Power_onWindows.txt". For other operative systems as Linux, you will want to read the instructions in "requirementsToRunRepresentation_Power.txt".


Algorithm Description:

The analysis is performed in surveillance windows. The estimator to be used will be the Mean Square Field (MSF). This is a power estimator. On a surveillance window W of length Nw, it is calculated as:
MSF = (∑ xi^2) / Nw			
where xi are the electric field amplitude samples, which are summed over the Nw samples.

To establish a reference, we calculate the MSF over the total samples in each channel. The channel with the lowest MSF (named after MSFmin) is considered to be the one with the lowest activity, and it is taken as the reference noise power for all the others.

The Mean Square Field (MSF) activity in each surveillance window W is calculated as:
MSF_Activity  = 10 log(MSFw/MSFmin) dB

For this power activity detection method, the algorithm is the following:
	Locate the channel with the minimum mean square field (MSFmin).
	Form window-sized (Nw) batches centered on each sample.
	Calculate the 10 log(MSFw/MSFmin) dB ratio in each window.
	Set a threshold.
	Check each window; if 10 log10(MSFw/MSFmin) > threshold → Anomaly.

