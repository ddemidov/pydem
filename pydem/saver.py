# -*- coding: utf-8 -*-

import numpy as np
import h5py
from lxml import etree

class saver(object):
    def __init__(self, fname, type, mass, radius):
        self.fname = fname
        self.hdf = h5py.File('{0}.h5'.format(fname), 'w')

        self.hdf.create_dataset('type',   data=type)
        self.hdf.create_dataset('mass',   data=mass)
        self.hdf.create_dataset('radius', data=radius)

        self.time_ds = self.hdf.create_dataset('time', shape=(0,), maxshape=(None,))

        self.hdf.flush()

        xmf = etree.Element('Xdmf', nsmap={
            "xi" : "http://www.w3.org/2003/XInclude"
            })

        domain = etree.SubElement(xmf, 'Domain')
        time   = etree.SubElement(domain, 'Grid')
        time.attrib['Name'] = 'Time'
        time.attrib['GridType'] = 'Collection'
        time.attrib['CollectionType'] = 'Temporal'

        self.xmf  = xmf
        self.time = time
        self.step = 0

    def __del__(self):
        self.hdf.close()

    def __call__(self, t, coo, vel, acc):
        self.hdf.create_group('step-{0}'.format(self.step))

        self.hdf.create_dataset('step-{0}/coo'.format(self.step), data=coo)

        z = np.zeros((coo.shape[0], 1))
        self.hdf.create_dataset('step-{0}/vel'.format(self.step), data=np.hstack((vel, z)))
        self.hdf.create_dataset('step-{0}/acc'.format(self.step), data=np.hstack((acc, z)))

        grid = etree.SubElement(self.time, 'Grid')
        etree.SubElement(grid, 'Time').attrib['Value'] = '{0:f}'.format(t)

        grid.attrib['Name']     = 'step-{0}'.format(self.step)
        grid.attrib['GridType'] = 'Uniform'

        topo = etree.SubElement(grid, 'Topology')
        topo.attrib['TopologyType'] = 'Polyvertex'
        topo.attrib['NumberOfElements'] = '{0}'.format(coo.shape[0])

        geom = etree.SubElement(grid, 'Geometry')
        geom.attrib['GeometryType'] = 'XY'

        data = etree.SubElement(geom, 'DataItem')
        data.attrib['Format'] = 'HDF'
        data.attrib['NumberType'] = 'Float'
        data.attrib['Precision'] = '8'
        data.attrib['Dimensions'] = '{0} {1}'.format(coo.shape[0], coo.shape[1])
        data.text = '{0}.h5:/step-{1}/coo'.format(self.fname, self.step)

        for a in ('vel', 'acc'):
            attr = etree.SubElement(grid, 'Attribute')
            attr.attrib['Name'] = a
            attr.attrib['AttributeType'] = 'Vector'
            attr.attrib['Center'] = 'Node'

            data = etree.SubElement(attr, 'DataItem')
            data.attrib['Format'] = 'HDF'
            data.attrib['NumberType'] = 'Float'
            data.attrib['Precision'] = '8'
            data.attrib['Dimensions'] = '{0} 3'.format(coo.shape[0])
            data.text = '{0}.h5:/step-{1}/{2}'.format(self.fname, self.step, a)

        for a in ('mass', 'type', 'radius'):
            attr = etree.SubElement(grid, 'Attribute')
            attr.attrib['Name'] = a
            attr.attrib['AttributeType'] = 'Scalar'
            attr.attrib['Center'] = 'Node'

            data = etree.SubElement(attr, 'DataItem')
            data.attrib['Format'] = 'HDF'
            data.attrib['NumberType'] = 'Float'
            data.attrib['Precision'] = '8'
            data.attrib['Dimensions'] = '{0}'.format(coo.shape[0])
            data.text = '{0}.h5:/{1}'.format(self.fname, a)

        self.step += 1

        self.time_ds.resize((self.step,))
        self.time_ds[-1] = t

        self.hdf.flush()

        with open('{0}.xmf'.format(self.fname), 'wb') as f:
            f.write(etree.tostring(self.xmf,
                pretty_print=True, xml_declaration=True, encoding='utf-8',
                doctype='<!DOCTYPE Xdmf SYSTEM "Sdmf.dtd" []>'))
