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

def fft(window):
    global win_num
    global wsize
    global cur_gesture
    scaled_win = scale_vector(window, 500)
    fft_sum = rfft_sum(scaled_win)
    print(len(window), win_num, fft_sum)
    if fft_sum > fft_threshold:
        # append to cur_gesture
        if cur_gesture is None:
            cur_gesture = scaled_win
        else:
            cur_gesture = np.concatenate([cur_gesture, scaled_win[250:]])
    elif cur_gesture is not None:
        # classify cur_gesture and reset it
        cur_gesture = scale_vector(cur_gesture, 500)
        prediction = knn.predict([cur_gesture])
        print("Gesture completed.\tLength: %d\tPredicted gesture: %s\t" % (len(cur_gesture), prediction[0]))
        cur_gesture = None

    file = open("windows.txt", "a+")
    file.write("Start: {}\t{}\nEnd: {}\t{}\n\n".format(win_num, window[0], win_num + wsize, window[-1]))
    #file.write("Start: {}\nEnd: {}\n".format(win_num, win_num + wsize))
    file.close()
    win_num += int(wsize * overlap)

def buildWindow():
    global win_num
    global wsize
    global overlap
    global rgdSamples
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
    end = curr_start + end
    next_start = wsize * overlap
    curr_win = []
    next_win = []
    i = 0

    while i < len(rgdSamples):
        try:
            if (rgdSamples[i] != 0):
                curr_win.append(rgdSamples[i])
                if i >= next_start:
                    next_win.append(rgdSamples[i])
                elif i == end:
                    # save copy so curr_win pointer is not used in fft but stays in this while loop
                    window_data.put(curr_win)
                    curr_win = next_win
                    next_win = [rgdSamples[i]]
                    curr_start = int(end - wsize * overlap)
                    next_start = end
                    end = curr_start + wsize
                    fft(window_data.get())
                i += 1
            time.sleep(.002)
        except IndexError:
            print "index error"
            time.sleep(.01)
        except KeyboardInterrupt:
            print("caught KeyboardInterrupt in buildWindow")

    print("EXITTED LOOP. i == {0}".format(i))
    sys.exit()

if __name__ == '__main__':
    # train model
    mapping = get_data_mappings()
    X, y, knn = build_knn_model(mapping)


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

    sst = Process(name='Bulid Window', target=buildWindow, args=())


    #declare ctype variables
    hdwf = c_int()
    sts = c_byte()
    hzAcq = c_double(500)
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
    print("DWF Version: "+str(version.value))

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

    print("Generating sine wave...")
    dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_bool(True))
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), AnalogOutNodeCarrier, funcSine)
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(1))
    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(2))
    dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_bool(True))

    #set up acquisition
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(5))
    dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
    dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
    dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(nSamples/hzAcq.value)) # -1 infinite record length

    #wait at least 2 seconds for the offset to stabilize
    time.sleep(2)

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

            dwf.FDwfAnalogInStatusData(hdwf, c_int(1), byref(rgdSamples, sizeof(c_double)*cSamples), cAvailable) # get channel 2 data

            # if cSamples >= curr_start and cSamples < end:
            #     if cSamples >= next_start:
            #         next_win.append(rgdSamples[cSamples])
            #     curr_win.append(rgdSamples[cSamples])
            #
            # if cSamples >= end:
            #     curr_win.append(rgdSamples[cSamples])
            #     #print "wd putting {}".format(called)
            #     called += 1
            #     window_data.put(curr_win)
            #
            #     #nothing is getting copied, just reassigning pointers
            #     curr_win = next_win
            #     next_win = [rgdSamples[cSamples]]
            #
            #     curr_start = int(end - wsize * overlap)
            #     next_start = end
            #     end = int(curr_start + wsize)

            cSamples += cAvailable.value
        except KeyboardInterrupt:
            break


    print("Recording finished")
    if fLost:
        print("Samples were lost! Reduce frequency")
    if fCorrupted:
        print("Samples could be corrupted! Reduce frequency")

    f = open("record.csv", "w")

    print("writing data to csv file")
    zero_count = 0
    i = 0
    while zero_count < 3:
        if rgdSamples[i] == 0.0:
            zero_count += 1
        f.write("%s\n" % rgdSamples[i])
        i += 1
    f.close()

    print("writing data to list to be graphed")
    rgpy= []
    zero_count = 0
    i = 0
    while zero_count < 3:
        if rgdSamples[i] == 0.0:
            zero_count += 1
        rgpy.append(rgdSamples[i])
        i += 1

    print("plotting")
    plt.plot(rgpy)
    plt.show()
