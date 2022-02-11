# SINTEC project
Hypertension (the ‘silent killer’) is one of the main risk factors for cardiovascular diseases (CVDs), main cause of death worldwide. Its continuous monitoring can offer a valid tool for patients care, as blood pressure (BP) is a significant indicator of health and putting it together with other parameters, such as heart and breath rates, could strongly improve prevention of CVDs. In this work we investigate the cuff-less estimation of continuous BP through pulse transit time (PTT) and heart rate (HR) using regression techniques. Our approach is intended as the first step towards continuous BP estimation with a low error according to AAMI guidelines. The novelties introduced in this work are represented by the implementation of pre-processing and by the innovative method for features research and features processing to continuously monitor blood pressure in a non-invasive way. In fact, invasive methods are the only reliable methods for continuous monitoring, while non-invasive techniques recover the values in a discreet way. 
This approach can be considered the first step for the integration of this type of algorithms on wearable devices, in particular on the devices developed for SINTEC project.

## Folder organization
To test the code related to the SHIMMER database, an example is provide. Files are organized according the following description:
* **SINTEC_proj**: .py file containing the Python code.
* **Files**: zip folder containing:
   * **Prova_Sofia**: a .csv file with the timestamp values (1st column) and the control SBP (2nd column) and DBP (3rd column) values; 
   * **Prova_Sofia_Session1_Shimmer_6C0E_Calibrated_SD**: a .mat file with the selected PPG signal;
   * **Prova_Sofia_Session1_Shimmer_9404_Calibrated_SD**: a .mat file with the selected ECG signal.
