# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:34:58 2023

@author: sasha
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from random import randint

import rpyc


import logging
logger = logging.getLogger(__name__)

from PyQt5.QtCore import QObject
from urllib.parse import urlparse
import ssl
import rpyc
from rpyc.utils.server import ThreadedServer
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
import os
import sys
from rpyc.utils.helpers import classpartial
import qdarkstyle
import numpy as np
import pickle


COLOR_LIST = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
                  '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                  '#000075', '#808080']


class RPyCServer(QObject):
    """ Contains a RPyC server that serves modules to remote computers. Runs in a QThread.
    """
    progress = QtCore.pyqtSignal(np.ndarray) # used to be "float"
    plotparams = QtCore.pyqtSignal(dict)

    def __init__(self, serviceClass, host, port):
        """
          @param class serviceClass: class that represents an RPyC service
          @param int port: port that hte RPyC server should listen on
        """
        super().__init__()
        self.serviceClass =classpartial(serviceClass, self.progress, self.plotparams)
        self.host = host
        self.port = port



    def run(self):
        """ Start the RPyC server
        """
        self.server = ThreadedServer(
        self.serviceClass,
        hostname=self.host,
        port=self.port,
        protocol_config={'allow_all_attrs': True})
        self.server.start()
    
    def stop(self):
        self.server.close()
        
             
        
class MyService(rpyc.Service):
        def __init__(self, progress, plotparams):
            self.progress = progress
            self.plotparams = plotparams
            
        def on_connect(self, conn):
            pass
        
        def exposed_update(self, number):
            number = pickle.loads(number)
            self.progress.emit(number)
            
        def exposed_plot_params(self, d):
            self.plotparams.emit(pickle.loads(d))
    

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
         super(MainWindow, self).__init__(*args, **kwargs)
         
         self.thread = QtCore.QThread()
         self.server = RPyCServer(MyService, 'localhost', 18861)
         self.server.moveToThread(self.thread)
         self.thread.started.connect(self.server.run)
         self.server.progress.connect(self.update_plot_data)
         self.server.plotparams.connect(self.update_plot_params)
         self.thread.start()
         font = QtGui.QFont()
         font.setFamily('OpenSans')
         
         self.plots = []
         # p = plot1Dwindow()
         # p.show()
         # self.plots.append(p)
         
         # p2 = plot2Dwindow()
         # p2.show()
         # self.plots.append(p2)

        
        
        
        
         # self.graphWidget = pg.PlotWidget()
         # self.setCentralWidget(self.graphWidget)
        
         # self.x =np.array([])  
         # # list(range(100))  # 100 time points
         # self.y = np.array([]) 
         # # [randint(0,100) for _ in range(100)]  # 100 data points
        
         # self.graphWidget.setBackground('#2A292B' )
        
         # pen = pg.mkPen(color='#FFFF66', width=1.5, style=QtCore.Qt.SolidLine)
         # self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen)
         # labelStyle = {'color': '#FF69B4', 'font-size': '14pt'}
         # titleStyle = {'color': '#FFDDF4', 'size': '18pt'}

         # self.graphWidget.setTitle('Title', **titleStyle)
         # self.graphWidget.setLabel('left', 'Value', **labelStyle)
         # self.graphWidget.setLabel('bottom', 'Time', **labelStyle)
         # plt = self.graphWidget.getPlotItem()
         
    def update_plot_data(self, d):
        if isinstance(self.plots[-1], plot1Dwindow):
            self.plots[-1].y = np.append(self.plots[-1].y, d)
            self.plots[-1].x = np.array([*range(0, len(self.plots[-1].x)+len(d),1)])
        else:
            self.plots[-1].update(d)
        # # self.x = self.x[1:].  # Remove the first y element.
        # self.x.append(len(self.x) + 1)  # Add a new value 1 higher than the last.

        # # self.y = self.y[1]  # Remove the first
        # self.y.append(d)  # Add a new random value.

            # self.plots[-1].data_line.setData(self.plots[-1].x, self.plots[-1].y)  # Update the data.
    
    def update_plot_params(self, d):
        print("updating plot params", d)
        if 'type' in d.keys():
            if d['type'] == '1d':
                p = plot1Dwindow()
                self.plots.append(p)
                p.show()
            elif d['type'] == '2d':
                p = plot2Dwindow(d)
                self.plots.append(p)
                p.show()
                
                
        
    
    def closeEvent(self, *args, **kwargs):
        super(QtWidgets.QMainWindow, self).closeEvent(*args, **kwargs)
        if self.thread.isRunning():
            self.server.stop()
            self.thread.terminate()

class plot1Dwindow(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
         super(QtWidgets.QWidget, self).__init__(*args, **kwargs)
         layout = QtWidgets.QHBoxLayout()
         
         self.graphWidget = pg.PlotWidget()
         layout.addWidget(self.graphWidget)
      
         self.x = np.linspace(0, 100, 100)
         self.y = np.random.random(100)
         self.graphWidget.setBackground('#2A292B' )

         
         self.properties = {'title': 'title',
                            'xlabel': 'xlabel',
                            'ylabel': 'ylabel',
                            'xunits': 'xuints',
                            'yunits': 'yunits'}
         
         pen = pg.mkPen(color='#FFFF66', width=1.5, style=QtCore.Qt.SolidLine)
         labelStyle = {'color': '#FF69B4', 'font-size': '14pt'}
         titleStyle = {'color': '#FFDDF4', 'size': '18pt'}
         
         self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen)
         
         self.graphWidget.setTitle(self.properties['title'], **titleStyle)
         self.graphWidget.setLabel('left', self.properties['ylabel'], **labelStyle)
         self.graphWidget.setLabel('bottom', self.properties['xlabel'], **labelStyle)
         self.data_line.setData(self.x, self.y)
         
         self.setLayout(layout)

class plot2Dwindow(QtWidgets.QWidget):
    def __init__(self, plotparams, *args, **kwargs):
         super(QtWidgets.QWidget, self).__init__(*args, **kwargs)
         self.plotparams = plotparams
         self.extents = [self.plotparams['x_range'][0], self.plotparams['x_range'][1], self.plotparams['y_range'][0], self.plotparams['y_range'][1]]
         self.npoints = [self.plotparams['npnts'][0], self.plotparams['npnts'][1]]
         self.setWindowTitle(self.plotparams['title'])
         
         layout = QtWidgets.QHBoxLayout()
         self.setGeometry(100, 100, 500, 400)
         self.setAutoFillBackground(True)
         
         self.view = pg.PlotItem(name='plot',title=self.plotparams['title'] )
         self.view.showAxis('top', show = True)
         self.view.showAxis('right', show = True)
         self.view.showAxis('left', self.plotparams['ylabel'])
         self.view.showAxis('bottom', self.plotparams['xlabel'])
         self.view.getAxis('left').setLabel('123')
         
         self.graphWidget = pg.ImageView(view = self.view)
         layout.addWidget(self.graphWidget)
         
         self.data = np.zeros([self.npoints[0], self.npoints[1]])
         self.data[:]=np.NAN
         
         self.graphWidget.ui.histogram.item.gradient.loadPreset('viridis')
         self.graphWidget.ui.roiBtn.hide()
         self.graphWidget.ui.menuBtn.hide()
         self.view.invertY(False)
         self.view.getViewBox().setBackgroundColor('#2A292B')
         # view = self.graphWidget.getView()
         # view.setRange((0,1), (0,1))
         
         self.idx = 0
         # if not np.isnan(self.data).all():
         #     self.graphWidget.setImage(self.data)
         self.setLayout(layout)
         
    def update(self, newdata):
        d2 = self.data.flatten()
        d2[self.idx:self.idx+len(newdata.flatten())] = newdata.flatten()
        self.data = d2.reshape(self.data.shape)
        self.idx += len(newdata.flatten())
        print(self.data)
        print(self.data.shape)
        print(self.data[0,:])
        self.graphWidget.setImage(self.data.T)


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # setup stylesheet
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # or in new API
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    


    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
    
    
    
plotparams2d = {'type': '2d',
                'title': '2d plot',
                'xlabel': 'x',
                'ylabel': 'y',
                'x_range': (-10, 10),
                'y_range': (-4, 3),
                'npnts': (10, 10)
                }


if __name__ == '__main__':
    main()
    
    
    
    # c = rpyc.connect("localhost", 18861)
    # for i in range(1000):
    #     c.root.update(pickle.dumps(np.random.rand(random.randint(4,100))))
    #     time.sleep(0.1)