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
from skimage.transform import iradon
from skimage.filters import threshold_otsu

__version__ = '20231024'
__author__ = 'rvrooij'


def gauss(x, a, x0, sigma, bgr):
    return a * np.exp(-(x - x0)**2 /(2*sigma**2)) + bgr
    
def gauss2D(xz, a, x0, z0, sigma, bgr):
    x, z = xz
    g = a * np.exp(- ((x-x0)**2 + (z-z0)**2)/(2*sigma**2)) + bgr
    return g.ravel()

def guess_width_pix(data):
    thr = np.where(data > data.max()/2)[0]
    try:
        return max(1.0, (thr[-1] - thr[0]) / 2.355)
    except IndexError:
        return 1.0


def fit_sources(data, pixsize, angles, testing=False):
    # data(theta,y,x)
    # sum proj data first over x then theta 
    peak_signal = data.sum(axis=2).sum(axis=0)
    # find approximate y coordinates for all point sources 
    peaks = sorted(peak_local_max(peak_signal, min_distance=5, threshold_abs=peak_signal.max()/2)[:,0])
    # average distance between sources in y direction divided by 2 
    dist = int(np.diff(peaks).mean()/2)
    
    if testing:
        fig, ax = plt.subplots(2,len(peaks))

    # popt = optimal parameters
    SA_poptsX = []
    SA_poptsZ = []
        
    for i, peak in enumerate(peaks):
        SA_poptsX.append([])
        SA_poptsZ.append([])
        
        # projection cropped in y direction around each pointsource (peak +/- dist) 
        # cropped[theta,y,x] 
        cropped = data[:, peak-dist:peak+dist]
            
        # new coordinates. (0,0) is the exact center of the image (boundary of 4 pixels)
        X_coords = (np.arange(cropped.shape[2]) - cropped.shape[2]/2 + 0.5) * pixsize
        Z_coords = (np.arange(cropped.shape[1]) - cropped.shape[1]/2 + 0.5) * pixsize
        
        
        if 0: # 2D fitting
            X, Z = np.meshgrid(X_coords, Z_coords)        
            for img in cropped:
                ctr = np.unravel_index(np.argmax(img), img.shape)
                p0 = [img.max(), X_coords[ctr[1]], Z_coords[ctr[0]], 10, 0]

                popt, pcov = scipy.optimize.curve_fit(gauss2D, (X, Z), img.ravel(), p0=p0)

                SA_poptsX[-1].append([popt[0], popt[1], abs(popt[3]), popt[4]])
                SA_poptsZ[-1].append([popt[0], popt[2], abs(popt[3]), popt[4]])

        else: # 1D fitting (faster)
            X_cropped = cropped.sum(axis=1)
            Z_cropped = cropped.sum(axis=2)
                    
            # j=view, crop=1D profile in x-direction
            bounds = ((0, X_coords.min(), 1., 0), (2*X_cropped.max(), X_coords.max(), 50., 0.5*X_cropped.max()))
            p0 = [X_cropped[0].max(), X_coords[np.argmax(X_cropped[0])], guess_width_pix(X_cropped[0]) * pixsize, 0]
            for j, crop in enumerate(X_cropped):
                width = guess_width_pix(crop) * pixsize
                # fit 1D profile with gaussian
                # popt = [a, x0, sigma, bgr]
                popt, pcov = scipy.optimize.curve_fit(gauss, X_coords, crop, bounds=bounds, p0=p0)
                SA_poptsX[-1].append(popt)
                p0 = popt #Use previous fitresult as starting parameter
                      
            # j=view, crop=1D profile in y-direction        
            p0 = [Z_cropped[0].max(), 0, guess_width_pix(Z_cropped[0]) * pixsize, 0]
            for j, crop in enumerate(Z_cropped):
                popt, pcov = scipy.optimize.curve_fit(gauss, Z_coords, crop, p0=p0)
                SA_poptsZ[-1].append(popt)
                p0 = popt
        
            if testing:
                # Show the data which is to be fitted            
                ax[0,i].imshow(X_cropped)
                ax[0,i].set_title("S%i"%i)
                ax[1,i].imshow(Z_cropped)
                ax[1,i].set_title("S%i"%i)
        
    if testing:
        plt.show()

    return np.array(SA_poptsX), np.array(SA_poptsZ)


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
    DSA_Zpos = []
    DSA_widths = []
    for i, det in enumerate(ds.DetectorInformationSequence):
        data = ordered[i]
        angles = np.array([det.StartAngle + direction * rot.AngularStep * n for n in range(rot.NumberOfFramesInRotation)])
        angles[angles<0] += 360
        angles[angles>=360] -= 360
        
        sorting = np.argsort(angles)        
        data = data[sorting]
        angles = angles[sorting]

        SA_poptsX, SA_poptsZ = fit_sources(data, pixsize, angles, testing)
            
        SA_widths = np.sqrt( np.abs(SA_poptsX[...,2] * SA_poptsZ[...,2]) ) * 2*np.sqrt(2*np.log(2))
        
        DSA_Xpos.append( SA_poptsX[...,1] )
        DSA_Zpos.append( SA_poptsZ[...,1] )
        DSA_widths.append( SA_widths )
        DA_angles.append( angles )
    
    if testing:
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for det_idx in range(nDet):
            for src_idx, A_Xpos in enumerate(DSA_Xpos[det_idx]):
                plt.plot(DA_angles[det_idx], A_Xpos, color=cycle[src_idx], linestyle=['-','--',':'][det_idx], label="D%i,S%i"%(det_idx, src_idx))
        plt.title("Xpos"); plt.legend(); plt.show()
        
        for det_idx in range(nDet):
            for src_idx, A_Zpos in enumerate(DSA_Zpos[det_idx]):
                plt.plot(DA_angles[det_idx], A_Zpos, color=cycle[src_idx], linestyle=['-','--',':'][det_idx], label="D%i,S%i"%(det_idx, src_idx))
        plt.title("Zpos"); plt.legend(); plt.show()
        
        for det_idx in range(nDet):
            for src_idx, A_widths in enumerate(DSA_widths[det_idx]): 
                plt.plot(DA_angles[det_idx], A_widths, color=cycle[src_idx], linestyle=['-','--',':'][det_idx], label="D%i,S%i"%(det_idx, src_idx))

        plt.title("Widths"); plt.legend(); plt.show()
    
    return np.array(DA_angles), np.array(DSA_Xpos), np.array(DSA_Zpos), np.array(DSA_widths)


def get_source_location_XY(A_angles, A_Xpos):
    ### Find Y, X location of source using by intersecting orthogonal angles
    xvals, yvals = [], []
    for i in range(120):
        a0 = A_angles[i]
        a1 = A_angles[i-30]
        d0 = A_Xpos[i]
        d1 = A_Xpos[i-30]
        
        v0 = [-d0*np.cos(a0 * np.pi/180), d0*np.sin(a0 * np.pi/180)]
        v1 = [-d1*np.cos(a1 * np.pi/180), d1*np.sin(a1 * np.pi/180)]
        vt = [v0[0]+v1[0], v0[1]+v1[1]]
        
        xvals.append(vt[0])
        yvals.append(vt[1])            
    source_loc = [np.mean(xvals), np.mean(yvals)]        
    #plt.scatter(xvals, yvals)
    #plt.gca().set_aspect('equal')
    #plt.show()
    return source_loc


def get_system_resolution(ds, DA_angles, DSA_Xpos, DSA_widths, testing=False):
    D_sys_res = []
    for det_idx, A_angles in enumerate(DA_angles):
        rotation_radius = float(ds.RotationInformationSequence[0].RadialPosition[0])
        
        S_sys_res = []
        for src_idx, A_widths in enumerate(DSA_widths[det_idx]):
            source_loc = get_source_location_XY(A_angles, DSA_Xpos[det_idx, src_idx])
            #print(det_idx, src_idx, source_loc)
            
            distances = []
            for i in range(len(A_angles)):
                d = DSA_Xpos[det_idx, src_idx, i]
                a = A_angles[i]
                v = [-d*np.cos(a * np.pi/180), d*np.sin(a * np.pi/180)]

                det_vec = [rotation_radius*np.sin(a * np.pi/180), rotation_radius*np.cos(a * np.pi/180)]
                detection = [v[0]+det_vec[0], v[1]+det_vec[1]]
                distances.append(np.linalg.norm([detection[0]-source_loc[0], detection[1]-source_loc[1]]))
                
                '''
                #Just for debugging, show the source-detector distances
                if i % 14 == 0:
                    plt.plot([det_vec[0], detection[0]], [det_vec[1], detection[1]], 'k--')
                    plt.plot([source_loc[0], detection[0]], [source_loc[1], detection[1]], "k-")
                    plt.xlim(-230, 230)
                    plt.ylim(-230, 230)
                    plt.gca().set_aspect('equal')
                        
            plt.show()
            '''            
            
            if min(distances) < 200: # The MHR measurement has a source in the center which never reaches a 200 mm source-detector distance
                coef = np.polyfit(distances, A_widths, 1)
                poly1d_fn = np.poly1d(coef)
                S_sys_res.append( poly1d_fn(200) )
                
                if testing:
                    plt.plot(distances, A_widths, '.', label="S%i"%src_idx)
                    plt.plot(distances, poly1d_fn(distances), 'k-')
                
        D_sys_res.append( np.mean(S_sys_res) )
        
        if testing:
            plt.title("Detector %i"%det_idx)        
            plt.axhline(D_sys_res[-1], c="k", ls="--")        
            plt.ylabel("Width FWHM (mm)")
            plt.xlabel("Source-Detector distance (mm)")
            plt.grid()
            plt.legend()
            plt.show()

    return D_sys_res


def analyze(ds, testing=False):
    DA_angles, DSA_Xpos, DSA_Zpos, DSA_widths = analyze_pointsources(ds, testing)
    
    D_sys_res = get_system_resolution(ds, DA_angles, DSA_Xpos, DSA_widths, testing)
    
    DS_Xpos = DSA_Xpos.mean(axis=2)
    DS_Zpos = DSA_Zpos.mean(axis=2)
    
    D_ctr_rot = -DS_Xpos.mean(axis=1)
    ax_shift = (DS_Zpos[0]-DS_Zpos[1]).mean() / 2.
    
    results = [('AxShift', ax_shift)]
    for i, det in enumerate(ds.DetectorInformationSequence):
        results.append(("D%i_COR"%(i+1), D_ctr_rot[i]))
        results.append(("D%i_SysRes"%(i+1), D_sys_res[i]))

    return {'float': results}


def header(ds, params):
    col_ids = params.get("Collimator IDs")
    results = []
    
    if(col_ids):
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
        
    #dicom_path = "ded2555f-592f-4aff-ae20-65cf80c9af7c.dcm"
    dicom_path = "NM000000.dcm"
    #dicom_path = "../NM_COR_DATA/module_error.dcm"
    #dicom_path = "../NM_COR_DATA/1f9162c8-cd5f-4307-bad1-7f8e6b1999f4.dcm"
    
    ds = dicom.read_file(dicom_path)
    print(ds.StationName)
    print(ds.StudyDescription,"-", ds.SeriesDescription)
    print(ds.SeriesDate)
    
    results = analyze(ds, testing=True)
    for key, value in results.get('float', []):
        print(key, value)
    print()
    results = header(ds,{})
    for key, value in results.get('string', []):
        print(key, value)
        
    
