"""
   DWF Python Example
   Author:  Digilent, Inc.
   Modifier: Maya Viust
   Revision: 10/17/2013

   Requires:
       Python 2.7, numpy, matplotlib
       python-dateutil, pyparsing
"""
from ctypes import *
from dwfconstants import *
import matplotlib.pyplot as plt
import math, time, sys, threading
from multiprocessing import Queue, Process
from functions import *

global wsize
global win_num
global overlap

knn = None

wsize = 500
win_num = 0
overlap = 0.5
fft_threshold = 12

# create or clear testing file
open("windows.txt", "w").close()

cur_gesture = None
MOVING_AVERAGE_N = 20
MOVING_AVERAGE_WEIGHTS = np.logspace(1, 1.6, num=MOVING_AVERAGE_N)
STARTING_THRESHOLD = 12
last_sums = [STARTING_THRESHOLD]*MOVING_AVERAGE_N
wma = STARTING_THRESHOLD

def fft(window):
    global win_num, wsize, cur_gesture, last_sums, wma
    scaled_win = scale_vector(window, 500)
    fft_sum = rfft_sum(scaled_win)
    print('Window Len:', len(window), '\tWindow Num:', win_num, '\tFFT Sum:', fft_sum, '\tWMA Thresh:', wma)
    if fft_sum > fft_threshold:
        # append to cur_gesture
        if cur_gesture is None:
            cur_gesture = scaled_win
        else:
            cur_gesture = np.concatenate([cur_gesture, scaled_win[250:]])
            print(len(cur_gesture))
    else:
        if cur_gesture is not None:
            # refine gesture endpoints, scale it to 500 length, and make a prediction!
            cur_gesture = refine_gesture(cur_gesture)
            scaled_gesture = scale_vector(cur_gesture, 500)
            prediction = knn.predict([scaled_gesture])
            print("Gesture completed.\tLength: %d\tPredicted gesture: %s\t" % (len(cur_gesture), prediction[0]))
            cur_gesture = None  # reset for next gesture capture
        # calculate WMA
        last_sums.pop(0)
        last_sums.append(fft_sum)
        wma = np.average(last_sums, weights=MOVING_AVERAGE_WEIGHTS) # weighted moving average
        wma *= 1.2

    #file = open("windows.txt", "a+")
    #file.write("Start: {}\t{}\nEnd: {}\t{}\n\n".format(win_num, window[0], win_num + wsize, window[-1]))
    #file.write("Start: {}\nEnd: {}\n".format(win_num, win_num + wsize))
    #file.close()
    win_num += int(wsize * overlap)

def buildWindow(window_data):
    global win_num
    global wsize
    global overlap
    global rgdSamples0
    time.sleep(5)
    # curr_size = window_data.qsize()
    # print("search seg pre while")
    # while window_data.qsize() > 0:
    #     try:
    #         fft(window_data.get())
    #         time.sleep(0.5)
    #     except KeyboardInterrupt:
    #         print("caught KeyboardInterrupt in searchSegment")

    curr_start = 0
    end = wsize
    end = curr_start + end
    next_start = wsize * overlap
    curr_win = []
    next_win = []
    i = 0
    time.sleep(2)
    while True:
        try:
            if window_data.qsize() == 0:
                time.sleep(0.1)
            else:
                fft(window_data.get())
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("CTRL C close in buildWindow")
            break

    print("EXITTED LOOP. i == {0}".format(i))
    sys.exit()

if __name__ == '__main__':
    # train model
    mapping = get_data_mappings()
    X, y, knn = build_knn_model(mapping)
    test_classification(X, y, knn)

    # begin data acquisition
    if sys.platform.startswith("win"):
        dwf = cdll.dwf
    elif sys.platform.startswith("darwin"):
        dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
    else:
        dwf = cdll.LoadLibrary("libdwf.so")

    # DATA PROCESSING SETUP
    # list of data in windows
    window_data = Queue()

    # DATA PROCESSING FUNCS
    # temporary def for testing

    sst = Process(name='Bulid Window', target=buildWindow, args=(window_data,))


    #declare ctype variables
    idxChannel = c_int(1) # analog channel 2
    hdwf = c_int()
    sts = c_byte()
    hzAcq = c_double(500.0)
    nSamples = 200000
    rgdSamples = (c_double*nSamples)()
    cAvailable = c_int()
    cLost = c_int()
    cCorrupted = c_int()
    fLost = 0
    fCorrupted = 0

    #print DWF version
    version = create_string_buffer(16)
    dwf.FDwfGetVersion(version)
    print("DWF Version: %s" % (version.value))

    #open device
    print("Opening first device")
    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

    if hdwf.value == hdwfNone.value:
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(szerr.value)
        print("failed to open device")
        quit()

    print("Preparing to read sample...")

    #set up acquisition
    dwf.FDwfAnalogInChannelEnableSet(hdwf, idxChannel, c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, idxChannel, c_double(5))
    dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
    dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
    dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(nSamples/hzAcq.value)) # -1 infinite record length

    #wait at least 2 seconds for the offset to stabilize
    time.sleep(2)

    phzAcq = c_double()
    pnMin = c_int()
    pnMax = c_int()
    dwf.FDwfAnalogInFrequencyGet(hdwf, byref(phzAcq))
    dwf.FDwfAnalogInBufferSizeInfo(hdwf, byref(pnMin), byref(pnMax))
    print("Configured Hz: %.2f" % (phzAcq.value), "Buffer size: Min %d, Max %d" % (pnMin.value, pnMax.value))

    #begin acquisition
    dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))
    print("   recording now")
    sst.start()
    cSamples = 0
    curr_start = 0
    next_start = 250
    end = wsize
    curr_win = []
    next_win = []

    called = 1
    while True:
        try:
            dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
            if cSamples == 0 and (sts == DwfStateConfig or sts == DwfStatePrefill or sts == DwfStateArmed) :
                # Acquisition not yet started.
                continue

            dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))

            cSamples += cLost.value

            if cLost.value :
                fLost = 1
            if cCorrupted.value :
                fCorrupted = 1

            if cAvailable.value==0 :
                continue

            if cSamples+cAvailable.value > nSamples :
                cAvailable = c_int(nSamples-cSamples)

            dwf.FDwfAnalogInStatusData(hdwf, idxChannel, byref(rgdSamples, sizeof(c_double)*cSamples), cAvailable) # get channel 2 datat and cSamples < end:
            #     if cSamples >= next_start:
            #         next_win.append(rgdSamples[cSamples])
            #     curr_win.append(rgdSamples[cSamples])
            #
            if cSamples >= end:
                 curr_win = rgdSamples[curr_start:end]
                 #print "wd putting {}".format(called)
                 called += 1
                 window_data.put(curr_win)

            #
                 #nothing is getting copied, just reassigning pointers
            #     curr_win = next_win
            #     next_win = [rgdSamples[cSamples]]
            #
                 curr_start = int(end - wsize * overlap)
            #     next_start = end
                 end = int(curr_start + wsize)

            cSamples += cAvailable.value
        except KeyboardInterrupt:
            break


    print("Recording finished")
    if fLost:
        print("Samples were lost! Reduce frequency")
    if fCorrupted:
        print("Samples could be corrupted! Reduce frequency")

    #write_samples_to_csv(rgdSamples)
    #graph_samples(rgdSamples)
