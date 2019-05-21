from __future__ import print_function
"""
analyze_head_alignment_Siemens_NM.py

Analyze Head Alignment test for Siemens Symbia/Intevo auto QC

Author: Rob van Rooij
2019


NB: some variable names for arrays start with D_, S_, A_, or combinations such 
as DA_. These letters indicate the axes of the array for convenience, where the 
letters stand for:
    D: Detector
    S: pointSource
    A: Angle

For instance, with 2 detectors, 3 point sources and 120 angles, the shape of an
array, such as DSA_Xpos (which indicates the X-position of the point sources for
each detector at every angle) is (2, 3, 120)
"""

import numpy as np
import scipy.optimize
from scipy import ndimage
from skimage.feature import peak_local_max

__version__ = '20190417'
__author__ = 'rvrooij'


def gaus(x,a,x0,sigma,bgr):
    return a * np.exp(-(x - x0)**2 /(2*sigma**2)) + bgr


def cos360(x,a,p,b):
    return a * np.cos((x - p) * 2*np.pi/360.) + b


def guess_width_pix(data):
    thr = np.where(data > data.max()/2)[0]
    try:
        return (thr[-1] - thr[0]) / 2.355
    except IndexError:
        return 1.0


def fit_sources(data, pixsize, testing=False):
    peak_signal = data.sum(axis=2).sum(axis=0)
    peaks = peak_local_max(peak_signal, min_distance=10, threshold_abs=peak_signal.max()/2)
    dist = int(np.diff(sorted(peaks[:,0])).mean()/2)

    if testing:
        fig, ax = plt.subplots(1,len(peaks))

    SA_poptsX = []
    SA_poptsY = []
    for i, peak in enumerate(peaks[:,0]):
        SA_poptsX.append([])
        SA_poptsY.append([])
        
        cropped = data[:, peak-dist:peak+dist]
                        
        X_cropped = cropped.sum(axis=1)
        Y_cropped = cropped.sum(axis=2)
            
        X_coords = -(np.arange(cropped.shape[2]) - cropped.shape[2]/2 + 0.5) * pixsize
        Y_coords = (np.arange(cropped.shape[1]) - cropped.shape[1]/2 + 0.5) * pixsize
        
        if testing:
            ax[i].imshow(Y_cropped)
            ax[i].set_title("S%i"%i)
        
        for j, crop in enumerate(X_cropped):
            width = guess_width_pix(crop) * pixsize
            popt, pcov = scipy.optimize.curve_fit(gaus, X_coords, crop, p0=[crop.max(),X_coords[np.argmax(crop)],width,0])
            SA_poptsX[-1].append(popt)
                    
        for j, crop in enumerate(Y_cropped):
            width = guess_width_pix(crop) * pixsize
            popt, pcov = scipy.optimize.curve_fit(gaus, Y_coords, crop, p0=[crop.max(),0,width,0])
            SA_poptsY[-1].append(popt)

    if testing:
        plt.show()

    return np.array(SA_poptsX), np.array(SA_poptsY)


def analyze_pointsources(ds, testing=False):
    nWin = ds.NumberOfEnergyWindows
    nDet = ds.NumberOfDetectors
    nRot = ds.NumberOfRotations
    
    ordered = ds.pixel_array.reshape((nWin, nDet, nRot, -1, ds.Rows, ds.Columns))
    ordered = ordered.sum(axis=2)[0]

    rot = ds.RotationInformationSequence[0]
    direction = {
        'CW': -1,
        'CC': 1,
        }[rot.RotationDirection]
        
    pixsize = float(ds.PixelSpacing[0])

    DA_angles = []
    DSA_Xpos = []
    DSA_Ypos = []
    DSA_widths = []
    for i, det in enumerate(ds.DetectorInformationSequence):
        data = ordered[i]
        angles = np.array([det.StartAngle + direction * rot.AngularStep * n for n in range(rot.NumberOfFramesInRotation)])
        angles[angles<0] += 360.
        
        sorting = np.argsort(angles)
        data = data[sorting]
        angles = angles[sorting]

        SA_poptsX, SA_poptsY = fit_sources(data, pixsize, testing)
        
        SA_widths = np.sqrt( SA_poptsX[...,2] * SA_poptsY[...,2] ) * 2*np.sqrt(2*np.log(2))
        
        DSA_Xpos.append( SA_poptsX[...,1] )
        DSA_Ypos.append( SA_poptsY[...,1] )
        DSA_widths.append( SA_widths )
        DA_angles.append( angles )
    
    if testing:
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for det_idx in range(nDet):
            for src_idx, A_Xpos in enumerate(DSA_Xpos[det_idx]):
                plt.plot(DA_angles[det_idx], A_Xpos, color=cycle[src_idx], linestyle=['-','--',':'][det_idx], label="D%i,S%i"%(det_idx, src_idx))
        plt.title("Xpos"); plt.legend(); plt.show()
        
        for det_idx in range(nDet):
            for src_idx, A_Ypos in enumerate(DSA_Ypos[det_idx]):
                plt.plot(DA_angles[det_idx], A_Ypos, color=cycle[src_idx], linestyle=['-','--',':'][det_idx], label="D%i,S%i"%(det_idx, src_idx))
        plt.title("Ypos"); plt.legend(); plt.show()
        
        for det_idx in range(nDet):
            for src_idx, A_widths in enumerate(DSA_widths[det_idx]): 
                plt.plot(DA_angles[det_idx], A_widths, color=cycle[src_idx], linestyle=['-','--',':'][det_idx], label="D%i,S%i"%(det_idx, src_idx))
        plt.title("widths"); plt.legend(); plt.show()
    
    return np.array(DA_angles), np.array(DSA_Xpos), np.array(DSA_Ypos), np.array(DSA_widths)


def get_system_resolution(ds, DA_angles, DSA_Xpos, DSA_widths, testing=False):
    D_sys_res = []
    for det_idx, A_angles in enumerate(DA_angles):
        rotation_radius = float(ds.RotationInformationSequence[0].RadialPosition[0])
        collimator_thickness = ds[0x0055,0x107e].value
        crystal_thickness = ds[0x0033,0x1029].value
        dist20cm = 200 + collimator_thickness[det_idx] + crystal_thickness[det_idx]

        S_sys_res = []
        for src_idx, A_widths in enumerate(DSA_widths[det_idx]):
            source_offset = 0.5 * DSA_Xpos[det_idx, src_idx].ptp()
            
            if source_offset > 50.:                
                b_guess = A_widths.mean()
                a_guess = 0.5 * A_widths.ptp()
                p_guess = A_angles[A_widths.argmax()]
                                
                popt, pcov = scipy.optimize.curve_fit(cos360, A_angles, A_widths, p0=[a_guess,p_guess,b_guess])

                if testing:
                    plt.plot(A_angles, A_widths); plt.plot(A_angles, cos360(A_angles, *popt)); plt.show()

                Wmin, Wmax = popt[2]-abs(popt[0]), popt[2]+abs(popt[0])
                slope = (Wmax - Wmin) / (2*source_offset)
                res = Wmin + slope * (dist20cm - rotation_radius + source_offset)

                S_sys_res.append( res )
                
        D_sys_res.append( np.max(S_sys_res) )
    
    return D_sys_res


def analyze(ds, testing=False):
    DA_angles, DSA_Xpos, DSA_Ypos, DSA_widths = analyze_pointsources(ds, testing)
    
    D_sys_res = get_system_resolution(ds, DA_angles, DSA_Xpos, DSA_widths, testing)
    
    DS_Xpos = DSA_Xpos.mean(axis=2)
    DS_Ypos = DSA_Ypos.mean(axis=2)
    
    D_ctr_rot = DS_Xpos.mean(axis=1)
    ax_shift = (DS_Ypos[0]-DS_Ypos[1]).mean() / 2.
    
    results = [('AxShift', ax_shift)]
    for i, det in enumerate(ds.DetectorInformationSequence):
        results.append(("D%i_COR"%(i+1), D_ctr_rot[i]))
        results.append(("D%i_SysRes"%(i+1), D_sys_res[i]))

    return {'float': results}


def header(ds, params):
    col_ids = params["Collimator IDs"]

    results = []
    for i, det in enumerate(ds.DetectorInformationSequence):
        cid = det.CollimatorGridName
        
        for label, col_id in col_ids.items():
            if cid.startswith(col_id):
                cid = label
                break
    
        results.append(("D%i_CollimatorGridName"%(i+1), cid))

    return {'string': results}
    

if __name__ == "__main__":
    """
    Standalone test
    """

    try:
        import pydicom as dicom
    except ImportError:
        import dicom
    import matplotlib.pyplot as plt
        
    #dicom_path = "Head Alignment Calibration QC Study/NM Head Alignment Verification LEHR/NM000000.dcm"
    #dicom_path = "Head Alignment Calibration QC Study/NM Head Alignment Calibration 180 LEHR/NM000000.dcm"
    dicom_path = "MHR COR 180 Calibration 3 Points Study/NM HEGP/NM000000.dcm"
    #dicom_path = "430dcd75-8f57-4e39-b587-aa431c12c756.dcm"
    
    ds = dicom.read_file(dicom_path)
    
    results = analyze(ds, testing=False)
    for key, value in results.get('float', []):
        print(key, value)
    print()
    results = header(ds)
    for key, value in results.get('string', []):
        print(key, value)
        
    
