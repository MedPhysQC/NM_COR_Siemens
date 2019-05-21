#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# WAD-QC is open-source software and consists of a set of modules written in 
# python for the WAD-Software medical physics quality control software. 
# The WAD Software can be found on https://bitbucket.org/MedPhysNL/wadqc
# 
#
# Changelog:
# ./QCCBCT_wadwrapper.py -c Config/cbct_instrumentarium_op300.json -d TestSet/20171030 -r results_20171030.json
from __future__ import print_function

__version__ = '20190418'
__author__ = 'rvrooij'

import os
# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib # for acqdatetime

try:
    import pydicom as dicom
except ImportError:
    import dicom


##### Load analysis module
import NM_COR_Siemens as module


##### Real functions
def qc_series(data, results, params):
    
    ds = dicom.read_file(data.series_filelist[0][0] )
    analysis = module.analyze(ds)

    for key,value in analysis.get('float', []):
        results.addFloat(key, value)


def header_series(data, results, params):
    ds = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)
    analysis = module.header(ds, params)

    for key,val in analysis.get('float', []):
        results.addFloat(key, val)
    for key,val in analysis.get('string', []):
        results.addString(key, str(val)[:min(len(str(val)), 100)])

    # plugionversion is newly added in for this plugin since pywad2
    varname = 'pluginversion'
    results.addString(varname, str(module.__version__))

    
def acqdatetime_series(data, results):
    """
    Read acqdatetime from dicomheaders
    """
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)
    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt) 




if __name__ == "__main__":
    data, results, config = pyWADinput()

    # read runtime parameters for module
    for name,action in config['actions'].items():
    
        try:
            params = action['params']
        except KeyError:
            params = {}
    
        if name == 'acqdatetime':
            acqdatetime_series(data, results)

        elif name == 'header_series':
            header_series(data, results, params)
        
        elif name == 'qc_series':
            qc_series(data, results, params)

    #results.limits["minlowhighmax"]["mydynamicresult"] = [1,2,3,4]

    results.write()
