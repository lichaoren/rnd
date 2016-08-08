""" Example of reading float binary files with metadata file
"""

# # python import
import os, argparse, array
import struct

# # tornado import
from base import Volume

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--meta", required=True,
    help="File contains metadata")
ap.add_argument("-v1", "--bin", required=True,
    help="Input Volume 1")
# ap.add_argument("-v2", "--volume2", required=True,
#     help="Input Volume 2")
# ap.add_argument("-o", "--output", required=True,
#     help="Path to output volume")
args = vars(ap.parse_args())


# # construct volumes
vol = Volume()

# # load size from metadata
size = []
with open(args["meta"], "rb") as f:
    for line in f:
        if line.find("AXIS_N"):
            size = line.split()
            vol.setSize(size[2], size[3], size[4])
            print "Find size in meta file", size[2], size[3], size[4])
            break

# # # load data from volumes        
# read1 = read_binary_file(args["bin"])
# read2 = read_binary_file(args["volume2"])
# 
# fsize1 = read1[0]
# fsize2 = read2[0]
# 
# if (fsize1 == fsize2 and fsize1 > 0):
#     data1 = read1[1]
#     data2 = read2[1]
#     # # file volumes
#     for i in xrange(size[2]):
#         for j in xrange(size[3]):
#             for k in xrange(size[4]):
#                 idx = i*size[3]*size[4] + j*size[4] + k 
#                 val1 = data1[idx]
#                 val2 = data2[idx]
#                 vol.setValue(i, j, k, val1)
#                 vol2.setValue(i, j, k, val2)
#     vol.saveAs("./" + args["bin"].split("@@")[0] + ".fdm")
#     vol2.saveAs("./" + args["bin"].split("@@")[0] + ".fdm")
#                 
# else:
#     print 'Error reading file'
                    
fname = "./" + args["bin"].split("@@")[0] + ".fdm"          
with open('Project.voxet_Ves@@', 'rb') as inp:
    for i in xrange(size[2]):
        for j in xrange(size[3]):
            for k in xrange(size[4]):
                val = struct.unpack('f', by4(inp)) 
                vol.setValue(i, j, k, val)
    vol.saveAs(fname)
    print "fdm is saved at", fname
                    
# # read binary file into an array
def read_binary_file(filename):
    try:
        f = open(filename, 'rb')
        n = os.path.getsize(filename)
        data = array.array('f')
        data.read(f, n)
        f.close()
        fsize = data.__len__()
        return (fsize, data)
    
    except IOError:
        return (-1, [])
    
## read binary file 4 bytes at a time
def by4(f):
    rec = 'x'  # placeholder for the `while`
    while rec:
        rec = f.read(4)
        if rec: yield rec           
        
        
        
        
#!/usr/bin/python


# # python import
import os, argparse, array
import struct

# # tornado import
from base import Volume

metafile = "/data4/d3382shl/97_From_Client/06_0804_basin_model/cgg_model1/Project.voxet_trim.vo"
binfile = "/data4/d3382shl/97_From_Client/06_0804_basin_model/cgg_model1/Project.voxet_Ves@@"
output = "/data4/d3382shl/97_From_Client/06_0804_basin_model/cgg_model1/Project.voxet_Ves.fdm"

# # construct volumes
vol = Volume()
vol.load('/data2/devtest/tornado/gocadVox/ves.fdm')
ss = vol.getSize()
print ss

# # load size from metadata
with open(metafile, "r") as f:
    for line in f:
        sp = line.split()
        if "AXIS_N" in sp:
            size = line.split()
            vec = [int(size[1]), int(size[2]), int(size[3]), 0]
            vol.setSize(vec)
            print "Find size in meta file", size[1], size[2], size[3]
            break
                    
with open(binfile, 'rb') as inp:
    for i in xrange(vec[1]):
        for j in xrange(vec[2]):
            for k in xrange(vec[3]):
                val = struct.unpack('f', by4(inp)) 
                vol.setValue(i, j, k, val)
    vol.saveAs(output)
    print "fdm is saved at", output
                    
# # read binary file into an array
def read_binary_file(filename):
    try:
        f = open(filename, 'rb')
        n = os.path.getsize(filename)
        data = array.array('f')
        data.read(f, n)
        f.close()
        fsize = data.__len__()
        return (fsize, data)
    
    except IOError:
        return (-1, [])
    
## read binary file 4 bytes at a time
def by4(f):
    rec = 'x'  # placeholder for the `while`
    while rec:
        rec = f.read(4)
        if rec: yield rec           